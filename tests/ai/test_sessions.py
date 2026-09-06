"""
Tests del modo "partida al KO" (play_session / play_many_sessions).

A diferencia de play_match, aquí los stacks se ARRASTRAN de mano en mano, que es
justo donde puede colarse una fuga de fichas: cada test que juega una partida
comprueba también la conservación.

Ejecutar con:
    pytest tests/ai/test_sessions.py -v
"""
import pytest

from src.ai.base_agent import ActionType, AgentAction, PokerAgent
from src.ai.match import (
    SessionResult, play_many_sessions, play_session, session_summary,
)
from src.ai.random_agent import RandomAgent
from src.ai.rule_based_agent import RuleBasedAgent
from src.ai.rule_profiles import get_profile
from src.engine.controller import GameController, STARTING_STACK

BB = GameController.BIG_BLIND


class _SiempreFold(PokerAgent):
    """Se retira siempre que puede. Sanea a CHECK si no hay nada que pagar."""

    def decide_action(self, state):
        if state.to_call == 0:
            return AgentAction(ActionType.CHECK)
        return AgentAction(ActionType.FOLD)


class _SiempreAllIn(PokerAgent):
    def decide_action(self, state):
        return AgentAction(ActionType.ALL_IN)


def _reglas(perfil, seed=0):
    return RuleBasedAgent(get_profile(perfil), seed=seed)


# ───────────────────── 1. la partida siempre termina ─────────────────────────

@pytest.mark.parametrize("seed", range(20))
def test_una_partida_siempre_termina(seed):
    """Con dos RandomAgent, ninguna partida se queda colgada."""
    res = play_session(RandomAgent(seed=seed), RandomAgent(seed=seed + 100),
                       "A", "B", seed=seed, max_hands=200)
    assert res.hands <= 200
    # O hubo KO (alguien por debajo de la ciega grande) o se agotó el tope.
    acabo_por_ko = min(res.final_stack_a, res.final_stack_b) < BB
    assert acabo_por_ko or res.hands == 200
    assert (res.winner is None) == (not acabo_por_ko)


def test_ninguna_partida_supera_max_hands():
    for seed in range(30):
        res = play_session(RandomAgent(seed=seed), RandomAgent(seed=seed + 50),
                           "A", "B", seed=seed, max_hands=25)
        assert res.hands <= 25


# ─────────────────────── 2. conservación de fichas ───────────────────────────

@pytest.mark.parametrize("seed", range(15))
def test_conservacion_de_fichas_en_una_partida(seed):
    res = play_session(RandomAgent(seed=seed), RandomAgent(seed=seed + 7),
                       "A", "B", seed=seed)
    total = 2 * STARTING_STACK
    assert res.final_stack_a + res.final_stack_b == total
    # Y en TODAS las manos intermedias, no solo al final.
    for i, (sa, sb) in enumerate(zip(res.stacks_a, res.stacks_b)):
        assert sa + sb == total, f"fuga en la mano {i}"


def test_las_series_de_stack_tienen_una_entrada_por_mano_mas_la_inicial():
    res = play_session(_reglas("tag", 1), _reglas("nit", 2), "A", "B", seed=3)
    assert len(res.stacks_a) == res.hands + 1
    assert len(res.stacks_b) == res.hands + 1
    assert res.stacks_a[0] == res.stacks_b[0] == STARTING_STACK
    assert res.stacks_a[-1] == res.final_stack_a
    assert res.stacks_b[-1] == res.final_stack_b


def test_los_stacks_se_arrastran_no_se_resetean():
    """La diferencia con play_match: el stack de una mano parte del anterior.

    Se usa la pareja determinista (foldea siempre vs all-in siempre) porque dos
    agentes de reglas agresivos resuelven la partida en una sola mano y no
    llegan a mostrar el arrastre.
    """
    res = play_session(_SiempreFold(), _SiempreAllIn(), "A", "B", seed=11)
    assert res.hands > 1
    # Si se reseteasen, todas las entradas serían STARTING_STACK.
    assert any(s != STARTING_STACK for s in res.stacks_a[1:])
    # El que foldea solo pierde: la serie es estrictamente decreciente.
    assert all(b < a for a, b in zip(res.stacks_a, res.stacks_a[1:]))


# ──────────────────── 3. caso determinista: el que foldea ────────────────────

def test_el_que_siempre_foldea_pierde_en_un_numero_predecible_de_manos():
    """Un agente que se retira siempre pierde su ciega cada mano.

    Con stack 1000 y ciegas 20/50, alterna pagar SB (20) y BB (50) — el rival
    se lleva la ciega sin resistencia. Como el botón alterna, cada DOS manos
    pierde 70 fichas. Termina cuando baja de la ciega grande (50).
    """
    res = play_session(_SiempreFold(), _SiempreAllIn(), "Fold", "AllIn",
                       seed=1, max_hands=200)

    assert res.winner == "AllIn"
    assert res.final_stack_a < BB
    assert res.final_stack_a + res.final_stack_b == 2 * STARTING_STACK

    # 1000 = 14 * 70 + 20: tras 14 pares de manos le quedan 20 fichas, por
    # debajo de la ciega grande, y la partida acaba. Se fija el valor exacto
    # para que un cambio en las ciegas o en la rotación del botón rompa el test
    # en vez de pasar desapercibido.
    assert res.hands == 28
    assert res.final_stack_a == 20


def test_el_que_foldea_pierde_exactamente_su_ciega_cada_mano():
    res = play_session(_SiempreFold(), _SiempreAllIn(), "Fold", "AllIn", seed=2)
    perdidas = [res.stacks_a[i] - res.stacks_a[i + 1]
                for i in range(len(res.stacks_a) - 1)]
    # Cada mano pierde exactamente una ciega (pequeña o grande), salvo la
    # última, en la que puede quedarse con menos de la ciega que le tocaba.
    assert set(perdidas[:-1]) <= {GameController.SMALL_BLIND, BB}


# ───────────────────── 4. marcador de play_many_sessions ─────────────────────

@pytest.mark.parametrize("n", [2, 10, 50])
def test_el_marcador_suma_el_numero_de_partidas(n):
    res = play_many_sessions(RandomAgent(seed=1), RandomAgent(seed=2),
                             "A", "B", n_sessions=n, seed=42, verbose=False)
    assert res.wins_a + res.wins_b + res.draws == res.n_sessions
    assert len(res.durations) == res.n_sessions


def test_n_sessions_impar_se_redondea_al_par_siguiente():
    # Las partidas van en parejas espejo, así que un número impar no se puede
    # jugar sin dejar una sin su espejo.
    res = play_many_sessions(RandomAgent(seed=1), RandomAgent(seed=2),
                             "A", "B", n_sessions=7, seed=42, verbose=False)
    assert res.n_sessions == 8
    assert res.wins_a + res.wins_b + res.draws == 8


def test_n_sessions_invalido():
    with pytest.raises(ValueError, match="n_sessions"):
        play_many_sessions(RandomAgent(seed=1), RandomAgent(seed=2),
                           "A", "B", n_sessions=0, verbose=False)


def test_nombres_iguales_fallan():
    # El marcador se atribuye por nombre: con nombres iguales todo iría a A.
    with pytest.raises(ValueError, match="nombres"):
        play_many_sessions(RandomAgent(seed=1), RandomAgent(seed=2),
                           "X", "X", n_sessions=2, verbose=False)


# ───────────────────────── 5. reproducibilidad ───────────────────────────────

def test_la_misma_semilla_da_el_mismo_marcador():
    def tanda():
        return play_many_sessions(_reglas("tag", 1), _reglas("nit", 2),
                                  "tag", "nit", n_sessions=20, seed=99,
                                  verbose=False)
    a, b = tanda(), tanda()
    assert (a.wins_a, a.wins_b, a.draws) == (b.wins_a, b.wins_b, b.draws)
    assert a.durations == b.durations


def test_semillas_distintas_dan_partidas_distintas():
    a = play_many_sessions(RandomAgent(seed=1), RandomAgent(seed=2), "A", "B",
                           n_sessions=20, seed=1, verbose=False)
    b = play_many_sessions(RandomAgent(seed=1), RandomAgent(seed=2), "A", "B",
                           n_sessions=20, seed=2, verbose=False)
    assert a.durations != b.durations


# ──────────────────── 6. parejas espejo (sesgo de posición) ──────────────────

def test_las_partidas_van_en_parejas_con_la_misma_semilla():
    """Cada semilla se juega dos veces intercambiando asientos.

    Con dos agentes IDÉNTICOS y deterministas, el espejo garantiza que lo que
    gana uno en la primera lo gana el otro en la segunda: el marcador queda
    exactamente empatado, sin sesgo de posición residual.
    """
    res = play_many_sessions(_reglas("nit", 5), _reglas("nit", 5),
                             "A", "B", n_sessions=40, seed=7, verbose=False)
    assert res.wins_a == res.wins_b


# ─────────────────────────── 7. informe legible ──────────────────────────────

def test_session_summary_menciona_lo_esencial():
    res = play_many_sessions(_reglas("tag", 1), _reglas("nit", 2), "tag", "nit",
                             n_sessions=10, seed=3, verbose=False)
    txt = session_summary(res)
    assert "tag" in txt and "nit" in txt
    assert "Marcador" in txt
    assert str(res.wins_a) in txt and str(res.wins_b) in txt


def test_las_duraciones_se_conservan_aunque_no_se_reporten():
    """La duración no se reporta ni se guarda, pero la serie sigue en memoria:
    es lo que permite comprobar que una tanda es reproducible."""
    res = play_many_sessions(_reglas("tag", 1), _reglas("nit", 2), "tag", "nit",
                             n_sessions=20, seed=4, verbose=False)
    assert len(res.durations) == res.n_sessions
    assert all(d >= 1 for d in res.durations)


def test_partida_sin_jugar_no_rompe_las_propiedades():
    vacio = SessionResult("A", "B", None, 0, 0, 0, [], [])
    assert vacio.by_limit is True
