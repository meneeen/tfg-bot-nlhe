"""
Tests para hand_strength.

Cobertura:
1. Chen preflop        — rangos y orden entre manos conocidas.
2. Equity postflop     — continua y monótona (no colapsa manos distintas).
3. Proyectos (draws)   — el proyecto debe ser PROPIO, no de la mesa.
"""
import pytest
from treys import Card

from src.ai.hand_strength import (
    chen_score,
    preflop_strength,
    postflop_equity_estimate,
    has_flush_draw,
    has_straight_draw,
    _RANK_CLASS_BANDS,
)


def C(*names: str) -> list[int]:
    return [Card.new(n) for n in names]


# ═══════════════════════════ 1. PREFLOP (Chen) ════════════════════════════════

class TestChen:
    def test_aa_is_the_best_hand(self):
        assert chen_score(C("As", "Ah")) == 20.0

    def test_premium_beats_trash(self):
        assert chen_score(C("As", "Ah")) > chen_score(C("7s", "2h"))

    def test_suited_beats_offsuit(self):
        assert chen_score(C("As", "Ks")) > chen_score(C("As", "Kh"))

    def test_preflop_strength_is_normalized(self):
        for hole in [C("As", "Ah"), C("7s", "2h"), C("Jh", "Ts"), C("2c", "2d")]:
            assert 0.0 <= preflop_strength(hole) <= 1.0

    def test_aa_normalizes_to_one(self):
        assert preflop_strength(C("As", "Ah")) == pytest.approx(1.0)


# ═══════════════════════ 2. EQUITY POSTFLOP CONTINUA ══════════════════════════

class TestPostflopEquityIsContinuous:
    """
    Regresión del techo de información: la tabla vieja tenía 9 valores discretos
    y colapsaba manos muy distintas en el mismo número, así que ninguna política
    podía apostarlas de forma diferente.
    """

    BOARD = ("Kd", "7c", "2h", "9s", "4d")

    def test_overpair_beats_top_pair(self):
        board = C(*self.BOARD)
        aa = postflop_equity_estimate(C("Ah", "Ac"), board)   # overpair de ases
        kq = postflop_equity_estimate(C("Kh", "Qc"), board)   # top pair
        assert aa > kq, "AA overpair y KQ top pair volvieron a colapsar"

    def test_higher_trips_beat_lower_trips(self):
        board = C(*self.BOARD)
        t99 = postflop_equity_estimate(C("9h", "9c"), board)
        t44 = postflop_equity_estimate(C("4h", "4c"), board)
        assert t99 > t44

    def test_hand_classes_are_ordered(self):
        board = C(*self.BOARD)
        nothing  = postflop_equity_estimate(C("Jh", "3c"), board)  # carta alta
        top_pair = postflop_equity_estimate(C("Kh", "Qc"), board)  # pareja
        two_pair = postflop_equity_estimate(C("7h", "2c"), board)  # doble pareja
        trips    = postflop_equity_estimate(C("9h", "9c"), board)  # trío
        assert nothing < top_pair < two_pair < trips

    def test_equity_in_range(self):
        board = C(*self.BOARD)
        for hole in [C("Ah", "Ac"), C("Jh", "3c"), C("9h", "9c")]:
            assert 0.0 <= postflop_equity_estimate(hole, board) <= 1.0

    def test_needs_at_least_three_board_cards(self):
        with pytest.raises(ValueError):
            postflop_equity_estimate(C("Ah", "Ac"), C("Kd", "7c"))


# ═══════════════ 2b. TODAS LAS CLASES DE TREYS (regresión KeyError) ═══════════

class TestAllRankClasses:
    """
    Regresión del KeyError con escalera de color real.

    treys tiene DIEZ clases (0-9), no nueve: la clase 0 es Royal Flush, con
    score exactamente 1. Las bandas estaban escritas a mano y no la incluían,
    así que un royal reventaba con ``KeyError: 0``. Ocurre 1 de cada ~31.600
    manos: un entrenamiento de 500k steps (~333k manos) ve ~21 y peta seguro.
    """

    # Una mano real por cada clase de treys: (nombre, hole, board)
    HANDS = [
        ("royal flush",     ("As", "Ks"), ("Qs", "Js", "Ts")),
        ("straight flush",  ("9s", "8s"), ("7s", "6s", "5s")),
        ("four of a kind",  ("As", "Ah"), ("Ac", "Ad", "5s")),
        ("full house",      ("As", "Ah"), ("Ac", "Kd", "Ks")),
        ("flush",           ("As", "9s"), ("2s", "5s", "7s")),
        ("straight",        ("9c", "8d"), ("7s", "6h", "5c")),
        ("three of a kind", ("As", "Ah"), ("Ac", "7d", "2s")),
        ("two pair",        ("As", "Kh"), ("Ac", "Kd", "2s")),
        ("pair",            ("As", "Kh"), ("Ac", "7d", "2s")),
        ("high card",       ("As", "Kh"), ("9c", "7d", "2s")),
    ]

    def test_royal_flush_does_not_crash(self):
        """El caso exacto que reventaba: score 1 -> clase 0."""
        equity = postflop_equity_estimate(C("As", "Ks"), C("Qs", "Js", "Ts"))
        assert equity == pytest.approx(1.0)

    @pytest.mark.parametrize("name,hole,board", HANDS, ids=[h[0] for h in HANDS])
    def test_every_rank_class_is_covered(self, name, hole, board):
        """Ninguna de las 10 clases de treys debe lanzar KeyError."""
        equity = postflop_equity_estimate(C(*hole), C(*board))
        assert 0.0 <= equity <= 1.0, f"{name}: equity fuera de rango ({equity})"

    def test_all_ten_treys_classes_are_reached(self):
        """Las manos de arriba cubren de verdad las 10 clases, no repiten."""
        from treys import Evaluator
        ev = Evaluator()
        classes = {
            ev.get_rank_class(ev.evaluate(C(*board), C(*hole)))
            for _, hole, board in self.HANDS
        }
        assert classes == set(range(10)), (
            f"Las manos de prueba no cubren las 10 clases: falta {set(range(10)) - classes}"
        )

    def test_royal_flush_is_the_strongest_hand(self):
        royal = postflop_equity_estimate(C("As", "Ks"), C("Qs", "Js", "Ts"))
        sflush = postflop_equity_estimate(C("9s", "8s"), C("7s", "6s", "5s"))
        quads = postflop_equity_estimate(C("As", "Ah"), C("Ac", "Ad", "5s"))
        assert royal >= sflush > quads


class TestRankClassBands:
    """Las bandas se derivan de treys, no se escriben a mano."""

    def test_bands_match_treys_exactly(self):
        from treys.lookup import LookupTable as L
        expected: dict[int, tuple[int, int]] = {}
        prev_max = 0
        for score_max, cls in sorted(L.MAX_TO_RANK_CLASS.items()):
            expected[cls] = (prev_max + 1, score_max)
            prev_max = score_max

        actual = {c: (b[0], b[1]) for c, b in _RANK_CLASS_BANDS.items()}
        assert actual == expected, "Las bandas se han desalineado de treys"

    def test_all_ten_classes_present(self):
        assert set(_RANK_CLASS_BANDS) == set(range(10))

    def test_bands_are_contiguous_and_cover_every_score(self):
        prev = 0
        for cls in sorted(_RANK_CLASS_BANDS):
            best, worst, _, _ = _RANK_CLASS_BANDS[cls]
            assert best == prev + 1, f"Hueco antes de la clase {cls}"
            prev = worst
        assert prev == 7462, "Las bandas no llegan hasta la peor mano posible"

    def test_class_zero_is_royal_flush_score_one(self):
        best, worst, eq_worst, eq_best = _RANK_CLASS_BANDS[0]
        assert (best, worst) == (1, 1)
        assert eq_worst == eq_best == 1.0

    def test_class_one_starts_at_score_two(self):
        """El score 1 es del royal (clase 0), no del straight flush."""
        best, _, _, _ = _RANK_CLASS_BANDS[1]
        assert best == 2


# ═══════════════════════════ 3. PROYECTOS ═════════════════════════════════════

class TestFlushDraw:
    """El proyecto de color debe ser PROPIO: al menos una hole card del palo."""

    def test_board_only_draw_is_not_yours(self):
        # 4 corazones en la mesa, ninguno en la mano: el color se lo puede ligar
        # cualquiera, así que no es un proyecto tuyo ni justifica semi-farol.
        assert not has_flush_draw(C("Ks", "Qc"), C("Ah", "7h", "2h", "9h"))

    def test_one_hole_card_of_the_suit_counts(self):
        assert has_flush_draw(C("Kh", "Qc"), C("Ah", "7h", "2h"))

    def test_two_hole_cards_of_the_suit_count(self):
        assert has_flush_draw(C("Kh", "Qh"), C("Ah", "7h", "2c"))

    def test_made_flush_is_not_a_draw(self):
        # 4 en mesa + 1 en mano = color hecho, no proyecto
        assert not has_flush_draw(C("Kh", "Qc"), C("Ah", "7h", "2h", "9h"))

    def test_no_draw_at_all(self):
        assert not has_flush_draw(C("Ks", "Qc"), C("Ah", "7d", "2s"))

    def test_river_has_no_draws(self):
        assert not has_flush_draw(C("Kh", "Qc"), C("Ah", "7h", "2h", "9s", "3d"))


class TestStraightDraw:
    """El proyecto de escalera debe desaparecer si quitas tus dos cartas."""

    def test_board_only_draw_is_not_yours(self):
        # 9-T-J-Q en la mesa: el proyecto es de los dos, no tuyo.
        assert not has_straight_draw(C("2s", "3h"), C("9c", "Tc", "Jd", "Qh"))

    def test_open_ended_with_both_hole_cards(self):
        assert has_straight_draw(C("Jh", "Ts"), C("9c", "8d", "2h"))

    def test_gutshot_counts(self):
        assert has_straight_draw(C("Jh", "Ts"), C("Qc", "9d", "2h"))

    def test_one_hole_card_completing_the_window_counts(self):
        # Q + board 9-T-J = Q-J-T-9, abierta (necesita 8 o K). Es proyecto propio.
        assert has_straight_draw(C("Qh", "2s"), C("9c", "Td", "Jh"))

    def test_wheel_draw_with_low_ace(self):
        # A-2 con 3-4 en mesa: proyecto a la rueda (A-2-3-4-5)
        assert has_straight_draw(C("Ah", "2s"), C("3c", "4d", "Kh"))

    def test_made_straight_is_not_a_draw(self):
        # JT en 9-8-7 es escalera hecha; sin este descarte se contaría doble
        assert not has_straight_draw(C("Jh", "Ts"), C("9c", "8d", "7h"))

    def test_no_draw_at_all(self):
        assert not has_straight_draw(C("Kh", "7s"), C("9c", "2d", "5h"))

    def test_river_has_no_draws(self):
        assert not has_straight_draw(C("Jh", "Ts"), C("9c", "8d", "2h", "3s", "4c"))
