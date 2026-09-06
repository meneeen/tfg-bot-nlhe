"""
Tests para RuleBasedAgent.

Cobertura:
1. Invariantes de validez  — ninguna acción viola las restricciones del motor.
2. Estrategia preflop      — decisiones correctas para manos conocidas.
3. Estrategia postflop     — pot odds y equity aplicados correctamente.
4. Casos límite            — short stack, stack=0, min > max raise, etc.

Ejecutar con:
    pytest tests/ai/test_rule_based_agent.py -v
"""
import pytest
from treys import Card

from src.ai.base_agent import AgentAction, ActionType, GameState, Position
from src.ai.rule_based_agent import RuleBasedAgent
from src.engine.controller import Street

# ──────────────────────────────── fixture ────────────────────────────────────

# Sin farol ni semi-farol: el agente vuelve a ser DETERMINISTA, que es lo que
# necesitan las aserciones de estrategia de abajo ("mano floja -> check/fold").
# El baseline real sí farolea (ver test_baseline_is_not_a_passive_nit).
AGENT = RuleBasedAgent(bluff_freq=0.0, semi_bluff_freq=0.0)


def make_state(
    hole_cards: tuple[int, ...] | None = None,
    community_cards: tuple[int, ...] = (),
    pot: int = 60,
    my_stack: int = 980,
    opp_stack: int = 980,
    my_bet_this_street: int = 20,
    to_call: int = 0,
    min_raise_to: int = 40,
    max_raise_to: int = 1000,
    position: Position = Position.BTN,
    street: Street = Street.PREFLOP,
    action_history: tuple = (),
    big_blind: int = 20,
    small_blind: int = 10,
) -> GameState:
    """Crea un GameState con valores por defecto razonables para tests."""
    if hole_cards is None:
        hole_cards = (Card.new("As"), Card.new("Kh"))
    return GameState(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot=pot,
        my_stack=my_stack,
        opp_stack=opp_stack,
        my_bet_this_street=my_bet_this_street,
        to_call=to_call,
        min_raise_to=min_raise_to,
        max_raise_to=max_raise_to,
        position=position,
        street=street,
        action_history=action_history,
        big_blind=big_blind,
        small_blind=small_blind,
    )


def assert_valid(action: AgentAction, state: GameState) -> None:
    """Verifica que la acción cumple todas las restricciones del motor."""
    t = action.action_type

    # El importe nunca es negativo
    assert action.amount >= 0, f"Importe negativo: {action.amount}"

    # RAISE debe estar en el rango legal
    if t == ActionType.RAISE:
        assert action.amount >= state.min_raise_to, (
            f"RAISE {action.amount} < min_raise_to {state.min_raise_to}"
        )
        assert action.amount <= state.max_raise_to, (
            f"RAISE {action.amount} > max_raise_to {state.max_raise_to}"
        )

    # CHECK no es válido cuando hay algo que pagar
    if t == ActionType.CHECK:
        assert state.to_call == 0, (
            f"CHECK con to_call={state.to_call} > 0"
        )

    # FOLD cuando se puede pasar gratis es legal pero nunca óptimo;
    # el agente garantiza que no lo hace.
    if t == ActionType.FOLD:
        assert state.to_call > 0, (
            "FOLD con to_call == 0 (debería ser CHECK)"
        )

    # ALL_IN requiere tener fichas más allá del call
    if t == ActionType.ALL_IN:
        assert state.my_stack > state.to_call, (
            f"ALL_IN sin fichas para subir: stack={state.my_stack} to_call={state.to_call}"
        )


# ═══════════════════════════ 1. INVARIANTES DE VALIDEZ ═══════════════════════

class TestActionValidity:
    """Todas las acciones devueltas deben ser legales en cualquier estado."""

    # ── parámetros de escenarios variados ────────────────────────────────────

    PREFLOP_SCENARIOS = [
        # (hole_cards, to_call, my_stack, min_raise_to, max_raise_to, position)
        # Mano premium, apertura normal
        ((Card.new("As"), Card.new("Ah")), 10, 990, 40, 1000, Position.BTN),
        # Mano débil, BTN sin fichas para subir
        ((Card.new("7s"), Card.new("2h")), 10, 20, 40, 30, Position.BTN),
        # BB enfrentando subida, stack justo para call
        ((Card.new("Ks"), Card.new("Qh")), 60, 60, 120, 120, Position.BB),
        # Sin nada que pagar (check option BB)
        ((Card.new("Ts"), Card.new("9h")), 0, 980, 40, 1000, Position.BB),
        # Stack = 0 (ya all-in, no debería actuar pero por robustez)
        ((Card.new("As"), Card.new("Kh")), 0, 0, 40, 0, Position.BTN),
    ]

    POSTFLOP_BOARDS = [
        # (hole_cards, board, to_call, my_stack, pot)
        # Full house (fuerte): apostar
        (
            (Card.new("As"), Card.new("Ah")),
            (Card.new("Ac"), Card.new("Kc"), Card.new("Kd")),
            0, 900, 200,
        ),
        # High card en flop peligroso: check o fold
        (
            (Card.new("7s"), Card.new("2h")),
            (Card.new("Ac"), Card.new("Kd"), Card.new("Qh")),
            100, 900, 300,
        ),
        # Pot odds marginal: mano media vs apuesta media
        (
            (Card.new("Jh"), Card.new("Th")),
            (Card.new("9h"), Card.new("8c"), Card.new("2d")),
            50, 950, 200,
        ),
    ]

    @pytest.mark.parametrize(
        "hole_cards,to_call,my_stack,min_raise_to,max_raise_to,pos",
        PREFLOP_SCENARIOS,
    )
    def test_preflop_action_valid(
        self, hole_cards, to_call, my_stack, min_raise_to, max_raise_to, pos
    ):
        state = make_state(
            hole_cards=hole_cards,
            to_call=to_call,
            my_stack=my_stack,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
            position=pos,
        )
        action = AGENT.decide_action(state)
        assert_valid(action, state)

    @pytest.mark.parametrize(
        "hole_cards,board,to_call,my_stack,pot", POSTFLOP_BOARDS
    )
    @pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
    def test_postflop_action_valid(
        self, hole_cards, board, to_call, my_stack, pot, street
    ):
        # River necesita 5 cartas; usamos las mismas 3 para flop/turn/river
        # (el evaluador acepta board de 3 a 5 cartas)
        state = make_state(
            hole_cards=hole_cards,
            community_cards=tuple(board),
            to_call=to_call,
            my_stack=my_stack,
            pot=pot,
            min_raise_to=max(20, to_call * 2) if to_call > 0 else 20,
            max_raise_to=my_stack + 20,
            street=street,
        )
        action = AGENT.decide_action(state)
        assert_valid(action, state)


# ═══════════════════════════ 2. ESTRATEGIA PREFLOP ════════════════════════════

class TestPreflopStrategy:
    """El agente toma decisiones coherentes con su estrategia declarada."""

    def test_premium_hand_btn_raises(self):
        """AA en BTN debe abrir con raise."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Ah")),
            to_call=10,
            my_stack=990,
            min_raise_to=40,
            max_raise_to=1000,
            position=Position.BTN,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.RAISE, ActionType.ALL_IN)

    def test_trash_hand_btn_folds(self):
        """72o en BTN frente a la ciega debe retirarse."""
        state = make_state(
            hole_cards=(Card.new("7s"), Card.new("2h")),
            to_call=10,
            my_stack=990,
            min_raise_to=40,
            max_raise_to=1000,
            position=Position.BTN,
        )
        action = AGENT.decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_bb_checks_vs_limp_weak_hand(self):
        """Mano débil en BB sin subida: check (no fold gratis)."""
        state = make_state(
            hole_cards=(Card.new("7s"), Card.new("2h")),
            to_call=0,
            my_stack=980,
            min_raise_to=40,
            max_raise_to=1000,
            position=Position.BB,
        )
        action = AGENT.decide_action(state)
        assert action.action_type == ActionType.CHECK

    def test_bb_raises_premium_vs_limp(self):
        """AA en BB con limp del BTN: ISO raise."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Ah")),
            to_call=0,
            my_stack=980,
            min_raise_to=40,
            max_raise_to=1000,
            position=Position.BB,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.RAISE, ActionType.ALL_IN)

    def test_bb_threebet_vs_raise_premium(self):
        """QQ en BB enfrentando raise del BTN: 3-bet."""
        state = make_state(
            hole_cards=(Card.new("Qs"), Card.new("Qh")),
            to_call=60,          # BTN subió a 3 BB
            my_stack=940,
            min_raise_to=120,
            max_raise_to=1000,
            position=Position.BB,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.RAISE, ActionType.ALL_IN)

    def test_bb_folds_trash_vs_raise(self):
        """72o en BB enfrentando raise sustancial: fold."""
        state = make_state(
            hole_cards=(Card.new("7s"), Card.new("2h")),
            to_call=60,
            my_stack=940,
            min_raise_to=120,
            max_raise_to=1000,
            position=Position.BB,
        )
        action = AGENT.decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_raise_amount_minimum_open(self):
        """El raise de apertura debe ser al menos min_raise_to."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Ah")),
            to_call=10,
            my_stack=990,
            min_raise_to=40,
            max_raise_to=1000,
            position=Position.BTN,
        )
        action = AGENT.decide_action(state)
        if action.action_type == ActionType.RAISE:
            assert action.amount >= state.min_raise_to


# ═══════════════════════════ 3. ESTRATEGIA POSTFLOP ══════════════════════════

class TestPostflopStrategy:

    # Tablero de flop y river para reutilizar
    BOARD_FLUSH = (Card.new("2h"), Card.new("7h"), Card.new("Jh"))   # Flush draw possible
    BOARD_RAGS  = (Card.new("2c"), Card.new("7d"), Card.new("9s"))

    def test_strong_hand_bets_when_checked_to(self):
        """Full house en flop sin apuesta previa: el agente apuesta."""
        hole  = (Card.new("As"), Card.new("Ah"))
        board = (Card.new("Ac"), Card.new("Kc"), Card.new("Kd"))   # Full house ases
        state = make_state(
            hole_cards=hole,
            community_cards=board,
            to_call=0,
            my_stack=900,
            pot=200,
            min_raise_to=40,
            max_raise_to=920,
            street=Street.FLOP,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.RAISE, ActionType.ALL_IN)

    def test_weak_hand_checks_free(self):
        """High card con tablero conectado y sin apuesta: check (nunca fold gratis)."""
        hole  = (Card.new("7s"), Card.new("2h"))
        board = (Card.new("Ac"), Card.new("Kd"), Card.new("Qh"))   # High card solamente
        state = make_state(
            hole_cards=hole,
            community_cards=board,
            to_call=0,
            my_stack=900,
            pot=100,
            min_raise_to=40,
            max_raise_to=920,
            street=Street.FLOP,
        )
        action = AGENT.decide_action(state)
        assert action.action_type == ActionType.CHECK

    def test_weak_hand_folds_bad_pot_odds(self):
        """High card con pot odds pésimas (apuesta grande): fold."""
        hole  = (Card.new("7s"), Card.new("2h"))
        board = (Card.new("Ac"), Card.new("Kd"), Card.new("Qh"))
        # Pot 100, to_call 200 → pot odds = 200/300 ≈ 0.67; equity ~0.22 → fold
        state = make_state(
            hole_cards=hole,
            community_cards=board,
            to_call=200,
            my_stack=800,
            pot=100,
            min_raise_to=400,
            max_raise_to=820,
            street=Street.FLOP,
        )
        action = AGENT.decide_action(state)
        assert action.action_type == ActionType.FOLD

    def test_strong_hand_raises_facing_bet(self):
        """Four of a kind enfrentando una apuesta: debe subir."""
        hole  = (Card.new("As"), Card.new("Ah"))
        board = (Card.new("Ac"), Card.new("Ad"), Card.new("2c"))   # Póker de ases
        state = make_state(
            hole_cards=hole,
            community_cards=board,
            to_call=50,
            my_stack=900,
            pot=200,
            min_raise_to=100,
            max_raise_to=950,
            street=Street.FLOP,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.RAISE, ActionType.ALL_IN)

    def test_decent_hand_calls_good_pot_odds(self):
        """Par con pot odds favorables: call."""
        hole  = (Card.new("As"), Card.new("Ah"))   # Par de ases, equity ~0.38
        board = (Card.new("Kc"), Card.new("7d"), Card.new("2h"))
        # Pot 200, to_call 20 → pot odds = 20/220 ≈ 0.09; equity ~0.38 > pot_odds → call/raise
        state = make_state(
            hole_cards=hole,
            community_cards=board,
            to_call=20,
            my_stack=980,
            pot=200,
            min_raise_to=40,
            max_raise_to=1000,
            street=Street.FLOP,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN)


# ═══════════════════════════ 4. CASOS LÍMITE ══════════════════════════════════

class TestEdgeCases:

    def test_short_stack_no_raise_below_min(self):
        """Con stack corto (max_raise_to < min_raise_to) devuelve ALL_IN, no RAISE."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Ah")),
            to_call=0,
            my_stack=15,
            my_bet_this_street=0,
            min_raise_to=20,   # mínimo legal
            max_raise_to=15,   # stack no llega al mínimo
            position=Position.BTN,
        )
        action = AGENT.decide_action(state)
        # Si decide subir, debe ser ALL_IN (nunca RAISE < min)
        if action.action_type == ActionType.RAISE:
            assert action.amount >= state.min_raise_to
        assert_valid(action, state)

    def test_raise_clamped_to_max(self):
        """Un raise calculado mayor que max_raise_to se convierte en ALL_IN."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Ah")),
            to_call=0,
            my_stack=50,
            my_bet_this_street=0,
            min_raise_to=20,
            max_raise_to=50,   # all-in forzado si se quiere subir
            pot=300,           # 50 % pot = 150 >> max_raise_to
            position=Position.BTN,
            street=Street.FLOP,
            community_cards=(Card.new("Ac"), Card.new("Kd"), Card.new("Qh")),
        )
        action = AGENT.decide_action(state)
        assert_valid(action, state)
        if action.action_type == ActionType.RAISE:
            assert action.amount <= state.max_raise_to

    def test_stack_equal_to_call_cannot_raise(self):
        """Cuando stack == to_call, solo quedan call (all-in) o fold."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Kh")),
            to_call=100,
            my_stack=100,   # call deja stack en 0; no se puede subir
            min_raise_to=200,
            max_raise_to=100,  # all-in = just the call
            position=Position.BB,
        )
        action = AGENT.decide_action(state)
        assert action.action_type in (ActionType.CALL, ActionType.FOLD)
        assert_valid(action, state)

    def test_no_fold_when_free(self):
        """El agente nunca hace fold cuando puede pasar gratis."""
        for pos in Position:
            state = make_state(
                hole_cards=(Card.new("7s"), Card.new("2h")),  # peor mano posible
                to_call=0,
                my_stack=500,
                position=pos,
            )
            action = AGENT.decide_action(state)
            assert action.action_type != ActionType.FOLD, (
                f"FOLD con to_call=0 en posición {pos}"
            )

    def test_no_check_when_must_call(self):
        """El agente nunca devuelve CHECK cuando hay una apuesta que pagar."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Kh")),
            to_call=50,
            my_stack=950,
            min_raise_to=100,
            max_raise_to=1000,
        )
        action = AGENT.decide_action(state)
        assert action.action_type != ActionType.CHECK

    def test_no_negative_raise_amount(self):
        """El campo amount de cualquier acción nunca es negativo."""
        state = make_state(
            hole_cards=(Card.new("As"), Card.new("Ah")),
            to_call=0,
            my_stack=1000,
            min_raise_to=20,
            max_raise_to=1000,
        )
        action = AGENT.decide_action(state)
        assert action.amount >= 0

    @pytest.mark.parametrize("street", list(Street))
    def test_valid_action_every_street(self, street: Street):
        """Acción siempre válida independientemente de la calle."""
        board = (Card.new("Ac"), Card.new("Kd"), Card.new("Qh"))
        state = make_state(
            hole_cards=(Card.new("Jh"), Card.new("Th")),
            community_cards=board if street != Street.PREFLOP else (),
            to_call=30,
            my_stack=970,
            pot=100,
            min_raise_to=60,
            max_raise_to=1000,
            position=Position.BB,
            street=street,
        )
        action = AGENT.decide_action(state)
        assert_valid(action, state)


# ══════════════════ TestBaselineIsNotANit ════════════════════════════════════

class TestBaselineIsNotANit:
    """
    Regresión del baseline pasivo.

    Con el umbral antiguo (_EQUITY_VALUE_BET = 0.65, por encima del trío en la
    tabla discreta de equity) el agente hacía CHECK el 82 % de las veces
    postflop y solo apostaba con escalera o mejor. Un rival así no vale como
    baseline: ganarle no demuestra nada, y enseña al RL que "el rival nunca
    apuesta", estrategia que se hunde en cuanto llega el self-play.
    """

    def test_agent_bets_postflop_often_enough(self):
        from src.engine.controller import GameController, HAND_DONE
        from src.ai.agent_player import AgentPlayer
        from collections import Counter

        a = AgentPlayer("A", 1000, RuleBasedAgent(seed=1))
        b = AgentPlayer("B", 1000, RuleBasedAgent(seed=2))
        gc = GameController(a, b, seed=99)
        a.bind(gc); b.bind(gc)

        counts = Counter()
        for _ in range(400):
            a.stack = b.stack = 1000
            gc.start_hand()
            while gc.step() != HAND_DONE:
                pass
            for (_, act, _, street, _idx) in gc.action_log:
                if street != "PREFLOP":
                    counts[act] += 1

        total = sum(counts.values())
        assert total > 0, "No hubo acciones postflop"
        check_rate = counts["check"] / total
        raise_rate = counts["raise"] / total

        # Banda por AMBOS lados: un maníaco que sube el 72 % es tan mal baseline
        # como el nit que hacía check el 82 %. Solo que explotable al revés.
        assert check_rate < 0.70, (
            f"El baseline es demasiado pasivo (nit): check {check_rate:.1%}"
        )
        assert 0.15 < raise_rate < 0.55, (
            f"Agresión fuera de banda: raise {raise_rate:.1%} (esperado 15-55 %)"
        )

    def test_bluff_frequencies_are_configurable(self):
        nit = RuleBasedAgent(bluff_freq=0.0, semi_bluff_freq=0.0)
        assert nit.bluff_freq == 0.0 and nit.semi_bluff_freq == 0.0

    def test_seeded_agent_is_reproducible(self):
        s = make_state(
            hole_cards=(Card.new("Jh"), Card.new("Th")),
            community_cards=(Card.new("9h"), Card.new("8c"), Card.new("2d")),
            street=Street.FLOP,
            to_call=0,
        )
        a1 = [RuleBasedAgent(seed=7).decide_action(s) for _ in range(5)]
        a2 = [RuleBasedAgent(seed=7).decide_action(s) for _ in range(5)]
        assert a1 == a2
