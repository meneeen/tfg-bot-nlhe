"""
Tests del harness de evaluacion (src/ai/match.py).

Cobertura de las metricas anadidas a MatchResult:
  - AF postflop: ignora las acciones preflop y devuelve None sin calls postflop.
  - WTSD: usa como denominador las manos que vieron flop (no las manos totales).
  - deltas_bb: se guarda una entrada por mano (len == hands).
  - ci95 / significant: intervalo de confianza y marca de significancia.

Las metricas de clasificacion (postflop vs preflop, denominador de WTSD) se
prueban alimentando _record_hand_metrics con logs sinteticos: asi el resultado
no depende de las cartas de una partida concreta. Ademas, un test de integracion
con agentes deterministas comprueba len(deltas_bb) == hands sobre play_match.

Ejecutar con:
    pytest tests/ai/test_match.py -v
"""
import pytest

from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent
from src.ai.match import MatchResult, play_match
from src.engine.controller import Street


# ─────────────────────────── agente determinista ─────────────────────────────

class _CallStation(PokerAgent):
    """Nunca sube ni se retira: CHECK sin apuesta, CALL con apuesta.

    Dos de estas estaciones de pago producen una partida totalmente
    determinista: nadie apuesta postflop (solo checks), asi que no hay ni raises
    ni calls postflop, y toda mano llega a showdown viendo el flop.
    """

    def decide_action(self, state: GameState) -> AgentAction:
        if state.to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.CALL)


class _BetOrCall(PokerAgent):
    """Postflop apuesta el mínimo si le pasan la acción; iguala si hay apuesta.
    Preflop solo check/call. Nunca se retira.

    En un duelo de dos de estos, cada calle postflop produce un raise (del que
    actúa primero) y un call (del otro). Como la posición alterna entre manos,
    AMBOS jugadores acumulan raises Y calls postflop -> af_a y af_b definidos.
    """

    def decide_action(self, state: GameState) -> AgentAction:
        if state.to_call == 0:
            if state.street != Street.PREFLOP:
                return AgentAction(ActionType.RAISE, state.min_raise_to)
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.CALL)


def _result(board_len: int, by_fold: bool) -> dict:
    """result dict minimo con lo que consume _record_hand_metrics."""
    return {"board": list(range(board_len)), "by_fold": by_fold}


# ─────────────────────────── AF postflop ─────────────────────────────────────

def test_af_ignores_preflop_actions():
    res = MatchResult("A", "B", hands=1, big_blind=50)
    # Tuplas de log: (nombre, acción, importe, calle, índice_jugador). La
    # atribución es por índice (0=A, 1=B), no por nombre.
    log = [
        ("A", "raise", 150, "PREFLOP", 0),   # preflop -> debe ignorarse
        ("B", "call", 150, "PREFLOP", 1),    # preflop -> debe ignorarse
        ("A", "raise", 100, "FLOP", 0),      # A: raise postflop
        ("B", "call", 100, "FLOP", 1),       # B: call postflop
        ("A", "call", 50, "TURN", 0),        # A: call postflop
        ("B", "raise", 120, "TURN", 1),      # B: raise postflop
        ("B", "raise", 200, "RIVER", 1),     # B: raise postflop
    ]
    res._record_hand_metrics(log, _result(5, by_fold=False), delta_a=100)

    # A postflop: 1 raise / 1 call -> 1.0   (si contara el raise preflop seria 2.0)
    assert res.af_a == 1.0
    # B postflop: 2 raises / 1 call -> 2.0  (si contara el call preflop seria 1.0)
    assert res.af_b == 2.0


def test_af_is_none_without_postflop_calls():
    res = MatchResult("A", "B", hands=1, big_blind=50)
    log = [
        ("A", "raise", 150, "PREFLOP", 0),
        ("B", "call", 150, "PREFLOP", 1),    # hay call, pero PREFLOP
        ("A", "raise", 100, "FLOP", 0),      # raise postflop, cero calls postflop
        ("B", "fold", 0, "FLOP", 1),
    ]
    res._record_hand_metrics(log, _result(3, by_fold=True), delta_a=250)

    # Denominador 0 -> None (no infinito), para poder serializar a JSON.
    assert res.af_a is None      # raises=1, calls=0
    assert res.af_b is None      # raises=0, calls=0


def test_af_serializes_to_json():
    import json
    res = MatchResult("A", "B", hands=1, big_blind=50)
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=0)
    # None es serializable (se vuelve null); un float('inf') reventaria.
    assert json.dumps({"af_a": res.af_a, "af_b": res.af_b}) == '{"af_a": null, "af_b": null}'


# ─────────────────────────────── WTSD ────────────────────────────────────────

def test_wtsd_denominator_is_hands_that_saw_flop():
    res = MatchResult("A", "B", hands=4, big_blind=50)
    hands = [
        _result(5, by_fold=False),   # vio flop + showdown
        _result(5, by_fold=True),    # vio flop, fold en el river -> sin showdown
        _result(0, by_fold=True),    # fold preflop -> ni flop ni showdown
        _result(3, by_fold=False),   # vio flop + showdown (flop justo)
    ]
    for r in hands:
        res._record_hand_metrics([], r, delta_a=0)

    assert res.n_saw_flop == 3       # manos 1, 2 y 4
    assert res.n_showdowns == 2      # manos 1 y 4
    # WTSD usa el flop como denominador (3), no las manos totales (4).
    assert res.wtsd == pytest.approx(2 / 3)
    assert res.wtsd != pytest.approx(res.n_showdowns / res.hands)


def test_wtsd_is_none_when_no_hand_saw_flop():
    res = MatchResult("A", "B", hands=2, big_blind=50)
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=0)
    res._record_hand_metrics([], _result(2, by_fold=True), delta_a=0)  # 2 cartas: no es flop
    assert res.n_saw_flop == 0
    assert res.wtsd is None


# ─────────────────────────────── deltas_bb ───────────────────────────────────

def test_deltas_bb_records_one_entry_per_hand():
    res = MatchResult("A", "B", hands=2, big_blind=50)
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=100)
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=-50)
    # delta_fichas / BIG_BLIND
    assert res.deltas_bb == [2.0, -1.0]
    assert len(res.deltas_bb) == 2


def test_deltas_bb_length_equals_hands_in_real_match():
    res = play_match(_CallStation(), _CallStation(), "A", "B", hands=10, seed=1)
    assert len(res.deltas_bb) == 10
    assert res.hands == 10


# ─────────────────────── intervalo de confianza ──────────────────────────────

def test_ci95_zero_variance_is_significant():
    res = MatchResult("A", "B", hands=3, big_blind=50, net_chips_a=150)
    for _ in range(3):
        res._record_hand_metrics([], _result(0, by_fold=True), delta_a=50)
    # Resultado constante -> varianza 0 -> IC de ancho 0.
    assert res.ci95 == 0.0
    assert res.bb_per_100 == pytest.approx(100.0)
    assert res.significant is True


def test_not_significant_when_within_margin():
    res = MatchResult("A", "B", hands=2, big_blind=50, net_chips_a=0)
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=100)
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=-100)
    # Neto 0 -> bb/100 = 0, pero hay dispersion -> ci95 > 0 -> no significativo.
    assert res.bb_per_100 == 0.0
    assert res.ci95 > 0.0
    assert res.significant is False


# ─────────────────────────── flop / preflop ──────────────────────────────────

def test_saw_flop_flags_and_preflop_winners():
    res = MatchResult("A", "B", hands=3, big_blind=50)
    # mano 1: vio flop (board completo), gana A en el showdown
    res._record_hand_metrics([], _result(5, by_fold=False), delta_a=100)
    # mano 2: NO vio flop (board vacío), fold preflop -> gana B
    res._record_hand_metrics([], _result(0, by_fold=True), delta_a=-50)
    # mano 3: NO vio flop (2 cartas < 3), fold preflop -> gana A
    res._record_hand_metrics([], _result(2, by_fold=True), delta_a=20)

    assert res.saw_flop_flags == [True, False, False]
    assert res.n_saw_flop == 1
    # 2 manos acabaron preflop: una la gana A (mano 3), otra B (mano 2)
    assert res.wins_preflop_a == 1
    assert res.wins_preflop_b == 1


# ─────────────────────── integracion determinista ────────────────────────────

def test_two_calling_stations_never_aggress_postflop():
    res = play_match(_CallStation(), _CallStation(), "A", "B", hands=20, seed=7)

    assert len(res.deltas_bb) == 20
    # Nadie sube: cero raises y cero calls postflop -> AF indefinido en ambos,
    # aunque SI hay calls preflop cada mano (confirma que preflop se ignora).
    assert res.af_a is None
    assert res.af_b is None
    assert res.actions_a.get("call", 0) + res.actions_b.get("call", 0) > 0
    # Dos estaciones de pago llegan siempre a showdown viendo el flop.
    assert res.n_saw_flop == 20
    assert res.n_showdowns == 20
    assert res.wtsd == 1.0


def test_mirror_match_attributes_actions_to_both_players():
    """Duelo espejo: MISMO nombre en A y B. Con el bug de atribución por nombre,
    todas las acciones se contaban al jugador A y af_b salía None. Con la
    atribución por índice, ambos deben registrar calls postflop."""
    res = play_match(_BetOrCall(), _BetOrCall(), "Espejo", "Espejo",
                     hands=30, seed=3)

    assert len(res.deltas_bb) == 30
    # Ambos jugadores acumulan calls Y raises postflop (posición alternante).
    assert res.calls_postflop_a > 0, "El jugador A no registró calls postflop"
    assert res.calls_postflop_b > 0, "El jugador B no registró calls postflop (bug por nombre)"
    assert res.af_a is not None
    assert res.af_b is not None
