"""
Tests de regresión para GameController.

Cubren en particular los tres bugs de all-in que se corrigieron:

  Bug 1 — _betting_complete() colgaba con all-ins short-stack (apuestas
          desiguales): un jugador all-in por menos nunca igualaba max_bet.
  Bug 2 — _begin_betting_round() pedía acción en calles posteriores aunque
          todos los activos estuvieran ya all-in.
  Bug 3 — _apply_action() preguntaba a un jugador all-in (stack 0, to_call 0)
          si quería pasar/retirarse, cuando no había acción posible.

Además: conservación de fichas (invariante global), devolución de apuesta no
igualada, resolución por fold y reparto en empate.
"""
import random

from src.engine.controller import (
    GameController, Street, WAIT_HUMAN, HAND_DONE,
)
from src.engine.player import Player, HumanPlayer, Action

# Los tests derivan de las constantes en vez de hardcodear 10/20: así
# sobreviven a un cambio de estructura de ciegas.
SB = GameController.SMALL_BLIND
BB = GameController.BIG_BLIND


# ─────────────────────────── jugadores de test ───────────────────────────────

class ScriptedPlayer(Player):
    """
    Jugador no interactivo que ejecuta una lista fija de acciones.

    Cada elemento de ``actions`` es una tupla ``(Action, amount)``.  Para ir
    all-in con RAISE basta pasar un ``amount`` muy grande: el motor lo capa
    automáticamente al stack disponible.

    Cuando se agotan las acciones programadas, usa un default seguro:
    CHECK si no hay apuesta que igualar, CALL en otro caso.
    """

    def __init__(self, name: str, stack: int, actions=None):
        super().__init__(name, stack)
        self.actions = list(actions or [])

    def decide(self, game_state: dict):
        to_call = game_state["to_call"]
        if self.actions:
            return self.actions.pop(0)
        return (Action.CHECK, 0) if to_call == 0 else (Action.CALL, to_call)


class RandomPlayer(Player):
    """
    Jugador que elige acciones legales al azar. **Fixture de test, no un agente**:
    no vive en ``src/`` porque no es una estrategia, es una fuente de caos.

    Existe para estresar el motor con líneas absurdas (subidas descabelladas,
    all-ins a destiempo, folds gratuitos). Un baseline sensato como
    RuleBasedAgent nunca produciría esas secuencias, y son justamente las que
    hacen valiosa la invariante de conservación de fichas.
    """

    def __init__(self, name: str, stack: int, seed: int | None = None):
        super().__init__(name, stack)
        self._rng = random.Random(seed)

    def decide(self, game_state: dict):
        to_call   = game_state["to_call"]
        min_raise = game_state["min_raise"]
        roll      = self._rng.random()

        if to_call == 0:
            if roll < 0.25:
                return Action.RAISE, self._rng.randint(min_raise, min_raise * 4)
            return Action.CHECK, 0

        if roll < 0.12:
            return Action.FOLD, 0
        if roll < 0.80:
            return Action.CALL, to_call
        return Action.RAISE, self._rng.randint(min_raise, min_raise * 4)


ALL_IN = (Action.RAISE, 10 ** 9)   # RAISE capado a all-in por el motor


def run_to_completion(gc: GameController, max_steps: int = 500) -> list[str]:
    """Ejecuta step() hasta HAND_DONE y devuelve la traza de estados."""
    trace = []
    for _ in range(max_steps):
        st = gc.step()
        trace.append(st)
        if st == HAND_DONE:
            return trace
    raise AssertionError(f"La mano no terminó en {max_steps} pasos (bucle infinito?)")


def total_chips(gc: GameController) -> int:
    return gc.player.stack + gc.agent.stack + gc.pot


# ═══════════════════════ TestChipConservation ════════════════════════════════

class TestChipConservation:
    """El total de fichas (stacks + bote) es invariante durante toda la mano."""

    def test_conserved_across_many_random_hands(self):
        a = RandomPlayer("A", 1000, seed=2024)
        b = RandomPlayer("B", 1000, seed=7)
        gc = GameController(a, b, seed=2024)

        for _ in range(300):
            a.stack = b.stack = 1000        # cada mano independiente
            gc.start_hand()
            run_to_completion(gc)
            assert a.stack + b.stack == 2000, "Fichas no conservadas tras la mano"

    def test_conserved_at_every_step(self):
        # stacks + bote es invariante DURANTE la mano. Al terminar, _award pasa
        # el bote a un stack (gc.pot queda obsoleto hasta la siguiente mano),
        # así que el estado terminal se comprueba sobre la suma de stacks.
        p = ScriptedPlayer("P", 1000, [ALL_IN])
        a = ScriptedPlayer("A", 1000, [(Action.CALL, 0)])
        gc = GameController(p, a, seed=1)
        gc.start_hand()
        expected = total_chips(gc)
        for _ in range(500):
            st = gc.step()
            if st == HAND_DONE:
                assert p.stack + a.stack == expected, "Fichas no conservadas al final"
                break
            assert total_chips(gc) == expected, "Fichas descuadradas en un step"


# ═══════════════════════ TestAllInMechanics ══════════════════════════════════

class TestAllInMechanics:
    def test_preflop_allin_and_call_reaches_showdown(self):
        # Bug 1 + Bug 2: all-in preflop + call debe correr el board hasta river.
        p = ScriptedPlayer("P", 1000, [ALL_IN])          # SB sube all-in
        a = ScriptedPlayer("A", 1000, [(Action.CALL, 0)])  # BB iguala all-in
        gc = GameController(p, a, seed=5)
        gc.start_hand()
        run_to_completion(gc)

        assert gc.hand_over
        assert len(gc.board) == 5, "El board debe completarse en un all-in a la ciega"
        assert p.stack + a.stack == 2000

    def test_shortstack_allin_does_not_hang(self):
        # Bug 1: stacks desiguales -> apuestas desiguales -> antes colgaba.
        p = ScriptedPlayer("P", 100,  [ALL_IN])            # SB corto all-in a 100
        a = ScriptedPlayer("A", 1000, [(Action.CALL, 0)])  # BB iguala 100
        gc = GameController(p, a, seed=5)
        gc.start_hand()
        run_to_completion(gc)                 # no debe lanzar (no hay bucle infinito)

        assert gc.hand_over
        assert p.stack + a.stack == 1100

    def test_no_action_prompt_after_players_are_allin(self):
        # Bug 2 + Bug 3: tras all-in completo, las calles siguientes NO piden
        # acción; el motor corre el board solo hasta el showdown.
        p = ScriptedPlayer("P", 1000, [ALL_IN])
        a = ScriptedPlayer("A", 1000, [(Action.CALL, 0)])
        gc = GameController(p, a, seed=9)
        gc.start_hand()
        trace = run_to_completion(gc)

        assert WAIT_HUMAN not in trace
        assert gc.hand_over

    def test_human_allin_is_never_prompted_again(self):
        # Bug 3 en su forma original: un HUMANO all-in no debe volver a recibir
        # WAIT_HUMAN en calles posteriores.
        human = HumanPlayer("H", 1000)
        a     = ScriptedPlayer("A", 1000, [(Action.CALL, 0)])
        gc    = GameController(human, a, seed=3)   # dealer=human=SB, actúa 1º
        gc.start_hand()

        # Primer paso: le toca al humano (SB preflop)
        assert gc.step() == WAIT_HUMAN
        gc.submit_human_action(*ALL_IN)            # humano all-in

        # A partir de aquí no debe haber más WAIT_HUMAN
        for _ in range(500):
            st = gc.step()
            assert st != WAIT_HUMAN, "El humano all-in fue interpelado de nuevo"
            if st == HAND_DONE:
                break
        assert gc.hand_over
        assert len(gc.board) == 5


# ═══════════════════════ TestReturnUncalled ══════════════════════════════════

class TestReturnUncalled:
    def test_uncalled_portion_is_returned(self):
        # SB con stack grande sube all-in (1000); BB corto solo puede igualar 100.
        # La diferencia no igualada (900) debe volver al SB.
        p = ScriptedPlayer("P", 1000, [ALL_IN])            # SB all-in a 1000
        a = ScriptedPlayer("A", 100,  [(Action.CALL, 0)])  # BB all-in a 100
        gc = GameController(p, a, seed=11)
        gc.start_hand()
        run_to_completion(gc)

        assert gc.hand_over
        # El bote efectivo se limita a 2×100 = 200; el resto vuelve al SB.
        assert p.stack + a.stack == 1100
        winner_stack = max(p.stack, a.stack)
        # Quien gane el enfrentamiento se lleva como mucho 200 sobre 900 de base.
        assert winner_stack >= 900, "La apuesta no igualada no se devolvió"


# ═══════════════════════ TestFold ════════════════════════════════════════════

class TestFold:
    def test_fold_gives_pot_to_other(self):
        # SB se retira preflop de inmediato.
        p = ScriptedPlayer("P", 1000, [(Action.FOLD, 0)])
        a = ScriptedPlayer("A", 1000)
        gc = GameController(p, a, seed=1)
        gc.start_hand()
        run_to_completion(gc)

        assert gc.hand_over
        assert gc.result["by_fold"] is True
        assert gc.result["winner"] == "A"
        # SB pierde solo la ciega pequeña (10); BB la recupera.
        assert p.stack == 1000 - SB
        assert a.stack == 1000 + SB

    def test_fold_does_not_reveal_agent_cards(self):
        p = ScriptedPlayer("P", 1000, [(Action.FOLD, 0)])
        a = ScriptedPlayer("A", 1000)
        gc = GameController(p, a, seed=1)
        gc.start_hand()
        run_to_completion(gc)
        assert gc.result["reveal_agent"] is False


# ═══════════════════════ TestAward ═══════════════════════════════════════════

class TestAward:
    """Prueba directa de _award (reparto), sin depender de cartas concretas."""

    def _fresh(self):
        p = ScriptedPlayer("P", 0)
        a = ScriptedPlayer("A", 0)
        return GameController(p, a), p, a

    def test_winner_takes_whole_pot(self):
        gc, p, a = self._fresh()
        gc.pot = 300
        gc._award(p)
        assert p.stack == 300 and a.stack == 0

    def test_tie_splits_pot_evenly(self):
        gc, p, a = self._fresh()
        gc.pot = 200
        gc._award(None)
        assert p.stack == 100 and a.stack == 100

    def test_tie_odd_chip_is_conserved(self):
        gc, p, a = self._fresh()
        gc.pot = 201
        gc._award(None)
        # La ficha impar va a uno de los dos, pero el total se conserva.
        assert p.stack + a.stack == 201
        assert abs(p.stack - a.stack) == 1


# ═══════════════════════ TestBettingComplete ═════════════════════════════════

class TestBettingComplete:
    def test_check_check_completes_preflop_when_bb_checks(self):
        # SB paga (limp) y BB pasa: la ronda preflop debe cerrarse y avanzar.
        p = ScriptedPlayer("P", 1000, [(Action.CALL, 0)])   # SB iguala la ciega
        a = ScriptedPlayer("A", 1000, [(Action.CHECK, 0)])  # BB pasa
        gc = GameController(p, a, seed=7)
        gc.start_hand()

        # Avanzar hasta que salga el flop
        for _ in range(50):
            gc.step()
            if gc.street != Street.PREFLOP:
                break
        assert gc.street == Street.FLOP
        assert len(gc.board) == 3


# ═══════════════════════ TestMinRaiseRule ════════════════════════════════════

class TestMinRaiseRule:
    """
    Regla real de NLHE: min_raise_to = max_bet + tamaño del ÚLTIMO incremento,
    no max_bet + BIG_BLIND.

    Con la regla vieja, tras una subida a 300 el re-raise mínimo salía 320: un
    agente podía re-subir por 20 fichas sobre una subida enorme, y el RL habría
    aprendido a explotarlo.
    """

    def _btn_raises_to(self, target: int, stacks=(1000, 1000)):
        """Prepara una mano donde el SB/BTN sube a `target` y devuelve (gc, bb)."""
        sb = ScriptedPlayer("SB", stacks[0], [(Action.RAISE, target)])
        bb = ScriptedPlayer("BB", stacks[1])
        gc = GameController(sb, bb, seed=1)
        gc.dealer = sb                    # SB actúa primero preflop
        gc.start_hand()
        gc.step()                         # SB sube
        return gc, bb

    def test_preflop_opening_min_raise_is_two_bb(self):
        sb = ScriptedPlayer("SB", 1000)
        bb = ScriptedPlayer("BB", 1000)
        gc = GameController(sb, bb, seed=1)
        gc.start_hand()
        # max_bet = BB, último incremento = BB -> mínimo = 2 BB
        assert gc.min_raise_to == 2 * BB

    def test_min_reraise_uses_last_increment_not_big_blind(self):
        gc, _ = self._btn_raises_to(300)
        assert gc.max_bet == 300
        assert gc.last_raise_size == 300 - BB       # incremento real de la subida
        assert gc.min_raise_to == 300 + (300 - BB), (  # max_bet + último incremento
            f"Re-subida mínima incorrecta: {gc.min_raise_to}. Con la regla vieja "
            f"habría salido {300 + BB}, permitiendo re-subir por solo una ciega."
        )

    def test_min_raise_chains_across_reraises(self):
        # SB sube a 300 -> BB re-sube a 600 (incr 300) -> min 900
        sb = ScriptedPlayer("SB", 5000, [(Action.RAISE, 300)])
        bb = ScriptedPlayer("BB", 5000, [(Action.RAISE, 600)])
        gc = GameController(sb, bb, seed=1)
        gc.dealer = sb
        gc.start_hand()
        gc.step()                                  # SB -> 300
        gc.step()                                  # BB -> 600
        assert gc.max_bet == 600
        assert gc.last_raise_size == 300           # 600 - 300
        assert gc.min_raise_to == 900              # 600 + 300

    def test_postflop_min_bet_is_one_bb(self):
        sb = ScriptedPlayer("SB", 1000, [(Action.CALL, 0)])
        bb = ScriptedPlayer("BB", 1000, [(Action.CHECK, 0)])
        gc = GameController(sb, bb, seed=7)
        gc.start_hand()
        for _ in range(50):
            gc.step()
            if gc.street != Street.PREFLOP:
                break
        # Nueva calle: max_bet = 0, incremento de referencia = 1 BB
        assert gc.max_bet == 0
        assert gc.min_raise_to == BB

    def test_last_raise_size_resets_each_street(self):
        # Subida gorda preflop; al llegar al flop el listón vuelve a 1 BB.
        sb = ScriptedPlayer("SB", 5000, [(Action.RAISE, 300), (Action.CHECK, 0)])
        bb = ScriptedPlayer("BB", 5000, [(Action.CALL, 0), (Action.CHECK, 0)])
        gc = GameController(sb, bb, seed=7)
        gc.dealer = sb
        gc.start_hand()
        gc.step()                                  # SB -> 300
        assert gc.last_raise_size == 300 - BB
        for _ in range(50):
            gc.step()
            if gc.street != Street.PREFLOP:
                break
        assert gc.last_raise_size == BB, "El incremento no se reinició en la nueva calle"
        assert gc.min_raise_to == BB


# ═══════════════════════ TestIncompleteAllIn ═════════════════════════════════

class TestIncompleteAllIn:
    """
    Un all-in por MENOS de una subida completa no reabre la acción: quien ya
    había igualado solo puede pagar la diferencia o retirarse, no re-subir.
    """

    def _setup_incomplete_allin(self):
        """
        SB sube a 300. El BB solo tiene 400: su all-in a 400 es un incremento
        de 100, menor que el de la subida del SB, o sea una subida INCOMPLETA.
        """
        sb = ScriptedPlayer("SB", 2000, [(Action.RAISE, 300)])
        bb = ScriptedPlayer("BB", 400,  [ALL_IN])
        gc = GameController(sb, bb, seed=1)
        gc.dealer = sb
        gc.start_hand()
        gc.step()                                  # SB sube a 300
        gc.step()                                  # BB all-in a 400 (incompleto)
        return gc, sb, bb

    def test_incomplete_allin_is_detected(self):
        gc, _, bb = self._setup_incomplete_allin()
        assert bb.stack == 0 and bb.bet == 400
        assert gc.max_bet == 400
        # El incremento (100) es menor que el de la subida completa -> se cierra
        assert gc.raise_closed is True
        # Y NO se reinició el listón: sigue siendo el de la subida completa
        assert gc.last_raise_size == 300 - BB

    def test_incomplete_allin_does_not_reopen_action(self):
        gc, sb, _ = self._setup_incomplete_allin()
        # El SB ya había actuado (subió). El all-in incompleto no debe
        # devolverle el derecho a re-subir.
        assert gc.acted[sb] is True, (
            "El all-in incompleto reabrió la acción para quien ya había subido"
        )

    def test_reraise_over_incomplete_allin_is_coerced_to_call(self):
        gc, sb, bb = self._setup_incomplete_allin()
        # El SB intenta re-subir a 1500: el motor debe tratarlo como CALL.
        gc._apply_action(sb, Action.RAISE, 1500)
        assert sb.bet == 400, (
            f"El SB re-subió a {sb.bet} sobre un all-in incompleto; "
            f"debería haberse limitado a igualar 400"
        )
        assert gc.max_bet == 400

    def test_hand_completes_and_chips_conserved(self):
        gc, sb, bb = self._setup_incomplete_allin()
        run_to_completion(gc)
        assert gc.hand_over
        assert sb.stack + bb.stack == 2400, "Fichas no conservadas"

    def test_full_allin_still_reopens_action(self):
        # Contraste: si el all-in SÍ es una subida completa, sí reabre la acción.
        sb = ScriptedPlayer("SB", 2000, [(Action.RAISE, 300)])
        bb = ScriptedPlayer("BB", 1000, [ALL_IN])   # all-in a 1000: incremento grande
        gc = GameController(sb, bb, seed=1)
        gc.dealer = sb
        gc.start_hand()
        gc.step()                                   # SB -> 300
        gc.step()                                   # BB all-in -> 1000 (completo)
        assert gc.raise_closed is False
        assert gc.last_raise_size == 1000 - 300
        assert gc.acted[sb] is False, "Una subida completa debe reabrir la acción"


# ═══════════════════════ TestIdenticalNames ══════════════════════════════════

class TestIdenticalNames:
    """Regresión: self.acted se indexaba por NOMBRE. Con dos jugadores del mismo
    nombre (duelo espejo) el dict colapsaba a una sola clave y la ronda de
    apuestas se cerraba en cuanto hablaba UNO de los dos."""

    def test_acted_has_two_entries_with_identical_names(self):
        a = ScriptedPlayer("X", 1000)
        b = ScriptedPlayer("X", 1000)
        gc = GameController(a, b, seed=9)
        gc.dealer = a
        gc.start_hand()
        assert len(gc.acted) == 2, f"acted colapsó a {len(gc.acted)} clave(s)"
        assert set(gc.acted.keys()) == {a, b}

    def test_round_not_closed_until_both_act_with_identical_names(self):
        # SB (a) iguala, BB (b) pasa. Ambos se llaman "X".
        a = ScriptedPlayer("X", 1000, [(Action.CALL, BB - SB)])
        b = ScriptedPlayer("X", 1000, [(Action.CHECK, 0)])
        gc = GameController(a, b, seed=9)
        gc.dealer = a
        gc.start_hand()

        assert gc._betting_complete() is False   # nadie ha actuado aún
        gc.step()                                # actúa SOLO a (iguala)
        assert gc._betting_complete() is False, (
            "La ronda se cerró tras actuar un solo jugador (acted por nombre)"
        )
        gc.step()                                # actúa b (pasa)
        assert gc._betting_complete() is True     # ahora sí, ambos han actuado

    def test_chips_conserved_with_identical_names(self):
        a = RandomPlayer("Dup", 1000, seed=1)
        b = RandomPlayer("Dup", 1000, seed=2)
        gc = GameController(a, b, seed=123)
        for i in range(200):
            a.stack = b.stack = 1000
            gc.dealer = a if i % 2 == 0 else b
            gc.start_hand()
            run_to_completion(gc)
            assert a.stack + b.stack == 2000, f"Fichas no conservadas en la mano {i}"
