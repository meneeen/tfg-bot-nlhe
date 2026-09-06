from treys import Card, Evaluator
from treys.lookup import LookupTable as _LookupTable

_EVALUATOR = Evaluator()

_HIGH_CARD_SCORE: list[float] = [
    1.0,   # 2
    1.5,   # 3
    2.0,   # 4
    2.5,   # 5
    3.0,   # 6
    3.5,   # 7
    4.0,   # 8
    4.5,   # 9
    5.0,   # T
    6.0,   # J
    7.0,   # Q
    8.0,   # K
    10.0,  # A
]

# Anclas de equity por clase: (equity_en_la_peor, equity_en_la_mejor) mano de
# esa clase. Son heurísticas calibradas para heads-up, no cálculos exactos.
_CLASS_EQUITY: dict[int, tuple[float, float]] = {
    0: (1.00, 1.00),  # Royal Flush     (la mejor mano posible: no hay rango)
    1: (0.98, 1.00),  # Straight Flush
    2: (0.95, 0.98),  # Four of a Kind
    3: (0.90, 0.95),  # Full House
    4: (0.83, 0.90),  # Flush
    5: (0.78, 0.83),  # Straight
    6: (0.68, 0.78),  # Three of a Kind
    7: (0.55, 0.68),  # Two Pair
    8: (0.30, 0.55),  # One Pair  (doses flojas .30 -> overpair AA .55)
    9: (0.12, 0.30),  # High Card
}


def _build_bands() -> dict[int, tuple[int, int, float, float]]:
    """
    Construye ``clase -> (mejor_score, peor_score, eq_peor, eq_mejor)``.

    Los rangos de score se derivan de ``LookupTable.MAX_TO_RANK_CLASS`` de
    treys, que da el score MÁXIMO de cada clase; el mínimo es el máximo de la
    clase anterior más uno. Así las bandas no pueden desalinearse de treys.

    Si treys introdujera una clase nueva, ``_CLASS_EQUITY[cls]`` lanzaría
    KeyError **al importar el módulo** — un fallo ruidoso e inmediato, en vez
    de un crash a las diez horas de entrenamiento.
    """
    bands: dict[int, tuple[int, int, float, float]] = {}
    prev_max = 0
    for score_max, cls in sorted(_LookupTable.MAX_TO_RANK_CLASS.items()):
        try:
            eq_worst, eq_best = _CLASS_EQUITY[cls]
        except KeyError as exc:
            raise KeyError(
                f"treys define la clase de mano {cls} "
                f"({_LookupTable.RANK_CLASS_TO_STRING.get(cls, '?')}) pero "
                f"_CLASS_EQUITY no tiene ancla de equity para ella. "
                f"Añádela a _CLASS_EQUITY en hand_strength.py."
            ) from exc
        bands[cls] = (prev_max + 1, score_max, eq_worst, eq_best)
        prev_max = score_max
    return bands


_RANK_CLASS_BANDS: dict[int, tuple[int, int, float, float]] = _build_bands()


def chen_score(hole_cards: list[int]) -> float:
    r0 = Card.get_rank_int(hole_cards[0])
    r1 = Card.get_rank_int(hole_cards[1])
    s0 = Card.get_suit_int(hole_cards[0])
    s1 = Card.get_suit_int(hole_cards[1])

    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)

    score = _HIGH_CARD_SCORE[hi]

    if hi == lo:  # pareja
        score = max(score * 2.0, 5.0)
    else:
        if s0 == s1:             # color
            score += 2.0

        gap = (hi - lo) - 1     # 0=conector, 1=un hueco, etc.
        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap >= 4:
            score -= 5.0

        # Bonus de potencial de escalera baja (ambas cartas ≤ J: rank int < 10 es Q)
        if gap <= 1 and hi < 10:
            score += 1.0

    return score


def preflop_strength(hole_cards: list[int]) -> float:
    return max(0.0, min(1.0, (chen_score(hole_cards) + 2.0) / 22.0))


def postflop_equity_estimate(hole_cards: list[int], board: list[int]) -> float:
    if len(board) < 3:
        raise ValueError(
            f"postflop_equity_estimate necesita al menos 3 cartas en el board, "
            f"recibidas: {len(board)}"
        )
    score      = _EVALUATOR.evaluate(board, hole_cards)
    rank_class = _EVALUATOR.get_rank_class(score)

    if rank_class not in _RANK_CLASS_BANDS:
        raise KeyError(
            f"Clase de mano {rank_class} sin banda de equity "
            f"(score={score}). Clases conocidas: {sorted(_RANK_CLASS_BANDS)}."
        )

    best, worst, eq_worst, eq_best = _RANK_CLASS_BANDS[rank_class]
    # frac = 1.0 en la mejor mano de la clase, 0.0 en la peor
    frac = (worst - score) / (worst - best) if worst > best else 1.0
    return eq_worst + frac * (eq_best - eq_worst)

def _straight_windows() -> list[set[int]]:
    return [set(range(s, s + 5)) for s in range(-1, 9)]

def _ranks_with_low_ace(cards: list[int]) -> set[int]:
    ranks = {Card.get_rank_int(c) for c in cards}
    if 12 in ranks:              # el as también juega bajo (A-2-3-4-5)
        ranks.add(-1)
    return ranks

def has_flush_draw(hole_cards: list[int], board: list[int]) -> bool:
    if len(board) >= 5:          # en el river ya no hay proyecto que valga
        return False

    def counts(cards: list[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for c in cards:
            s = Card.get_suit_int(c)
            out[s] = out.get(s, 0) + 1
        return out

    all_counts   = counts(hole_cards + board)
    board_counts = counts(board)

    for suit, n in all_counts.items():
        if n == 4 and board_counts.get(suit, 0) < 4:
            return True
    return False


def has_straight_draw(hole_cards: list[int], board: list[int]) -> bool:
    if len(board) >= 5:
        return False

    all_ranks   = _ranks_with_low_ace(hole_cards + board)
    board_ranks = _ranks_with_low_ace(board)
    windows     = _straight_windows()

    # una escalera YA HECHA no es un proyecto
    if any(len(all_ranks & w) == 5 for w in windows):
        return False

    # el proyecto debe desaparecer al quitar las hole cards
    return any(
        len(all_ranks & w) == 4 and len(board_ranks & w) < 4
        for w in windows
    )
