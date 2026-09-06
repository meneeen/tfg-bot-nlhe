import random
from enum import Enum
from .card import PokerDeck, HandEvaluator
from .player import Player, Action


class Street(Enum):
    PREFLOP = 0
    FLOP    = 1
    TURN    = 2
    RIVER   = 3


# Estados que devuelve step()
WAIT_HUMAN = "WAIT_HUMAN"
CONTINUE   = "CONTINUE"
HAND_DONE  = "HAND_DONE"

STARTING_STACK = 1000


class GameController:
    BIG_BLIND   = 50
    SMALL_BLIND = 20

    def __init__(self, player: Player, agent: Player, seed: int = None):
        self.player    = player
        self.agent     = agent
        self.players    = [player, agent]
        self.evaluator = HandEvaluator()

        self.dealer    = player          # en HU el dealer paga SB y actúa 1º preflop
        self.hand_num  = 0

        self._rng = random.Random(seed) if seed is not None else None

        # Estado de mano (se rellena en start_hand)
        self.deck            = None
        self.board           = []
        self.pot             = 0
        self.street          = Street.PREFLOP
        self.max_bet         = 0
        self.to_act          = None
        self.acted           = {}
        self.betting_done    = False
        
        # Tamaño del último incremento de apuesta en la calle actual.
        self.last_raise_size = self.BIG_BLIND
        
        # True cuando un all-in por MENOS de una subida completa cierra la
        # acción: quien ya había igualado solo puede pagar la diferencia.
        self.raise_closed    = False
        
        self.hand_over       = True      # arranca "terminada" -> GUI llama start_hand
        self.awaiting_human  = False
        self.last_action_str = ""
        self.action_log      = []
        self.result          = None

    # ───────────────────────── utilidades ─────────────────────────

    def seed(self, seed: int = None):
        self._rng = random.Random(seed) if seed is not None else None

    def _next_deck(self) -> PokerDeck:
        if self._rng is None:
            return PokerDeck()
        return PokerDeck(self._rng.randrange(2 ** 31))

    def _other(self, p: Player) -> Player:
        return self.agent if p is self.player else self.player

    def _apply_bet(self, player: Player, amount: int):
        amount = max(0, min(amount, player.stack))   # nunca más de lo que tiene
        player.stack -= amount
        player.bet   += amount
        self.pot     += amount

    def _log(self, actor: Player, action: Action, amount):
        # El 5º campo es el índice del jugador (0/1). Se necesita para atribuir
        # cada acción sin ambigüedad: dos jugadores pueden compartir nombre (p.ej.
        # un duelo espejo), pero nunca su posición en self.players.
        idx = 0 if actor is self.players[0] else 1
        self.action_log.append((actor.name, action.value, amount, self.street.name, idx))

    @property
    def min_raise_to(self) -> int:
        #Importe TOTAL mínimo al que se puede subir en la calle actual.
        return self.max_bet + self.last_raise_size

    def _build_state(self, actor: Player) -> dict:
        return {
            "board"     : list(self.board),
            "pot"       : self.pot,
            "to_call"   : self.max_bet - actor.bet,
            "min_raise" : self.min_raise_to,   # 'raise to' mínimo
            "street"    : self.street,
            "my_stack"  : actor.stack,
            "opp_stack" : self._other(actor).stack,
        }

    # ─────────────────────── inicio de mano ───────────────────────

    def start_hand(self):
        self.hand_num   += 1
        self.pot         = 0
        self.board       = []
        self.action_log  = []
        self.result      = None
        self.hand_over   = False
        self.last_action_str = ""

        self.deck = self._next_deck()
        for p in self.players:
            p.reset_for_hand()
            p.hole_cards = self.deck.draw(2)

        # Ciegas (heads-up: dealer = SB)
        sb = self.dealer
        bb = self._other(self.dealer)
        self._apply_bet(sb, self.SMALL_BLIND)
        self._apply_bet(bb, self.BIG_BLIND)
        self.last_action_str = "Ciegas publicadas"

        self.street = Street.PREFLOP
        self._begin_betting_round()

    def _begin_betting_round(self):
        self.betting_done = False
        # Indexado por el OBJETO jugador, no por nombre: dos jugadores con el
        # mismo nombre colapsarían a una sola clave y la ronda se cerraría en
        # cuanto hablase uno (rompiendo los duelos espejo).
        self.acted   = {p: False for p in self.players}
        self.max_bet = max(p.bet for p in self.players)

        self.last_raise_size = self.BIG_BLIND
        self.raise_closed    = False
        # Preflop actúa primero el dealer (SB); postflop el que NO es dealer
        if self.street == Street.PREFLOP:
            self.to_act = self.dealer
        else:
            self.to_act = self._other(self.dealer)
        # Si todos los activos ya están all-in no hay más apuestas posibles
        active = [p for p in self.players if not p.folded]
        if all(p.stack == 0 for p in active):
            self.betting_done = True

    # ──────────────────────────── step ────────────────────────────

    def step(self) -> str:
        if self.hand_over:
            return HAND_DONE
        if self.betting_done:
            return self._advance_street()
        return self._step_betting()

    def _step_betting(self) -> str:
        if self._betting_complete():
            self.betting_done = True
            return CONTINUE

        actor = self.to_act
        if actor.is_human:
            self.awaiting_human = True
            return WAIT_HUMAN

        action, amount = actor.decide(self._build_state(actor))
        self._apply_action(actor, action, amount)
        return HAND_DONE if self.hand_over else CONTINUE

    def submit_human_action(self, action: Action, amount: int = 0):
        """La GUI llama aquí cuando el humano pulsa un botón."""
        if not self.awaiting_human:
            return
        self.awaiting_human = False
        self._apply_action(self.to_act, action, amount)

    # ─────────────────────── aplicar acciones ─────────────────────

    def _apply_action(self, actor: Player, action: Action, amount: int):
        other   = self._other(actor)
        to_call = self.max_bet - actor.bet

        # Jugador all-in que por error intenta actuar: marcar y salir
        if actor.stack == 0 and to_call == 0:
            self.acted[actor] = True
            if not self.hand_over:
                self.to_act = other
            return

        # Si pide RAISE pero no le llega ni para igualar -> se trata como CALL/all-in
        if action == Action.RAISE and actor.stack <= to_call:
            action = Action.CALL

        # Acción cerrada por un all-in incompleto: el rival ya no puede re-subir,
        # solo pagar la diferencia. (En HU subir sobre un all-in no tendría efecto
        # de todos modos: _return_uncalled devolvería el exceso.)
        if action == Action.RAISE and self.raise_closed:
            action = Action.CALL

        # Si el jugador se retira, la mano termina inmediatamente y el rival gana
        if action == Action.FOLD:
            actor.folded = True
            self._log(actor, action, 0)
            self.last_action_str = f"{actor.name} se retira"
            self._finalize(winner=other, by_fold=True)
            return

        # Igualar o pasar: se paga lo que haga falta (o todo el stack si es all-in)
        if action in (Action.CALL, Action.CHECK):
            pay = min(to_call, actor.stack)
            self._apply_bet(actor, pay)
            self.acted[actor] = True
            self._log(actor, action, pay)
            if to_call == 0:
                self.last_action_str = f"{actor.name} pasa"
            else:
                allin = " (all-in)" if actor.stack == 0 else ""
                self.last_action_str = f"{actor.name} iguala {pay}{allin}"

        # Subida: se paga lo que haga falta para llegar al total indicado (o todo el stack si es all-in)
        elif action == Action.RAISE:
            # `amount` = total al que se sube en esta calle ('raise to')
            prev_max = self.max_bet
            min_to   = self.min_raise_to              # max_bet + último incremento
            allin_to = actor.bet + actor.stack
            target   = max(amount, min_to)
            target   = min(target, allin_to)          # tope = all-in
            pay      = target - actor.bet
            self._apply_bet(actor, pay)
            self.max_bet = actor.bet

            increment = self.max_bet - prev_max
            if increment >= self.last_raise_size:
                # Subida COMPLETA: fija el nuevo listón y reabre la acción.
                self.last_raise_size = increment
                self.acted = {actor: True, other: False}
            else:
                # Subida INCOMPLETA (solo posible yendo all-in por menos de una
                # subida entera): NO reabre la acción. Quien ya había igualado
                # no puede re-subir, solo pagar la diferencia o retirarse.
                self.raise_closed = True
                self.acted[actor] = True

            self._log(actor, action, actor.bet)
            allin = " (all-in)" if actor.stack == 0 else ""
            self.last_action_str = f"{actor.name} sube a {actor.bet}{allin}"

        # Pasa el turno (en HU siempre al otro) si la mano sigue
        if not self.hand_over:
            self.to_act = other

    def _betting_complete(self) -> bool:
        active = [p for p in self.players if not p.folded]
        if len(active) <= 1:
            return True
        # Si todos están all-in no queda acción posible
        if all(p.stack == 0 for p in active):
            return True
        both_acted = all(self.acted[p] for p in active) # ambos han actuado una vez?
        all_resolved = all(p.stack == 0 or p.bet == self.max_bet for p in active) # dinero igualado?
        return both_acted and all_resolved

    # ──────────────────── transición ─────────────────────

    def _advance_street(self) -> str:
        self._return_uncalled()        # devuelve apuesta no igualada (all-in corto)

        if self.street == Street.RIVER:
            return self._goto_showdown()

        for p in self.players:
            p.bet = 0

        if self.street == Street.PREFLOP:
            self.street = Street.FLOP
            self.board += self.deck.draw(3)
        elif self.street == Street.FLOP:
            self.street = Street.TURN
            self.board += self.deck.draw(1)
        elif self.street == Street.TURN:
            self.street = Street.RIVER
            self.board += self.deck.draw(1)

        self.last_action_str = f"Reparto: {self.street.name}"
        self._begin_betting_round()
        return CONTINUE

    def _return_uncalled(self):
        active = [p for p in self.players if not p.folded]
        if len(active) < 2:
            return
        hi = max(active, key=lambda p: p.bet)
        lo = min(active, key=lambda p: p.bet)
        diff = hi.bet - lo.bet
        if diff > 0:
            hi.stack += diff
            hi.bet   -= diff
            self.pot -= diff

    # ───────────────────────── resolución ─────────────────────────

    def _goto_showdown(self) -> str: # muestra cartas y decide ganador
        s_player = self.evaluator.evaluate(self.board, self.player.hole_cards)
        s_agent  = self.evaluator.evaluate(self.board, self.agent.hole_cards)
        if s_player < s_agent:      # menor = mejor en treys
            winner = self.player
        elif s_agent < s_player:
            winner = self.agent
        else:
            winner = None           # empate
        self._finalize(winner, by_fold=False,
                       names=(self.evaluator.rank_to_string(s_player),
                              self.evaluator.rank_to_string(s_agent)))
        return HAND_DONE

    def _award(self, winner): # reparte el bote al ganador o lo divide en caso de empate
        if winner is None:
            half = self.pot // 2
            self.player.stack += half
            self.agent.stack  += self.pot - half   # el resto impar va a uno (conserva fichas)
        else:
            winner.stack += self.pot

    def _finalize(self, winner, by_fold: bool, names=None): # actualiza self.result y marca la mano como terminada
        awarded_pot = self.pot          # se guarda para el resumen antes de vaciarlo
        self._award(winner)
        self.pot            = 0         # el bote ya se repartió: evita estado obsoleto
        self.hand_over      = True
        self.awaiting_human = False
        self.result = {
            "hand_num"     : self.hand_num,
            "winner"       : winner.name if winner else "Empate",
            "by_fold"      : by_fold,
            "reveal_agent" : not by_fold,     # solo se enseñan cartas en showdown
            "pot"          : awarded_pot,
            "board"        : list(self.board),
            "player_cards" : list(self.player.hole_cards),
            "agent_cards"  : list(self.agent.hole_cards),
            "player_hand"  : names[0] if names else None,
            "agent_hand"   : names[1] if names else None,
            "actions"      : list(self.action_log),
        }
        # Rota el botón para la siguiente mano
        self.dealer = self._other(self.dealer)

    # ───────────────────── ayudas para la GUI ─────────────────────

    @property
    def human_to_call(self) -> int:
        return self.max_bet - self.player.bet

    @property
    def human_min_raise(self) -> int:
        return self.min_raise_to

    @property
    def human_all_in(self) -> int:
        return self.player.bet + self.player.stack

    def session_over(self) -> bool:
        return self.player.stack <= 0 or self.agent.stack <= 0
