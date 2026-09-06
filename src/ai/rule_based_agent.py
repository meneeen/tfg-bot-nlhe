"""
Agente de reglas parametrizable.

Este módulo contiene UNA lógica de decisión — qué se consulta, en qué orden y
con qué prioridad — y ningún número. Los umbrales viven en rule_profiles.py:
cambiando el perfil, el mismo esqueleto juega como un nit, un maníaco o una
calling station.

    RuleBasedAgent()                    # perfil baseline (histórico)
    RuleBasedAgent("tag")               # por nombre
    RuleBasedAgent(NIT, seed=7)         # por objeto RuleProfile
    RuleBasedAgent("lag", bluff_freq=0) # perfil con un umbral sobrescrito
"""

from __future__ import annotations

import random

from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent, Position
from src.ai.hand_strength import (
    chen_score, postflop_equity_estimate, has_flush_draw, has_straight_draw,
)
from src.ai.rule_profiles import DEFAULT_PROFILE, RuleProfile, get_profile
from src.engine.controller import Street


class RuleBasedAgent(PokerAgent):

    def __init__(
        self,
        profile: RuleProfile | str | None = None,
        bluff_freq: float | None = None,
        semi_bluff_freq: float | None = None,
        seed: int | None = None,
    ) -> None:
        base = self._resolve(profile)
        overrides = {
            k: v
            for k, v in (("bluff_freq", bluff_freq),
                         ("semi_bluff_freq", semi_bluff_freq))
            if v is not None
        }
        self.profile = base.variante(**overrides) if overrides else base
        self._rng    = random.Random(seed)

    @staticmethod
    def _resolve(profile: RuleProfile | str | None) -> RuleProfile:
        if profile is None:
            return DEFAULT_PROFILE
        if isinstance(profile, RuleProfile):
            return profile
        return get_profile(profile)

    @property
    def bluff_freq(self) -> float:
        return self.profile.bluff_freq

    @property
    def semi_bluff_freq(self) -> float:
        return self.profile.semi_bluff_freq

    def __repr__(self) -> str:
        return f"RuleBasedAgent({self.profile.name!r})"

    def decide_action(self, state: GameState) -> AgentAction:
        raw = (
            self._preflop(state)
            if state.street == Street.PREFLOP
            else self._postflop(state)
        )
        return self._sanitize(raw, state)

    # ──────────────────────────── preflop ────────────────────────────────────

    def _preflop(self, state: GameState) -> AgentAction:
        score = chen_score(list(state.hole_cards))
        return (
            self._preflop_btn(state, score)
            if state.position == Position.BTN
            else self._preflop_bb(state, score)
        )

    def _preflop_btn(self, state: GameState, score: float) -> AgentAction:
        p = self.profile
        if score >= p.btn_open_raise and self._can_raise(state):
            return self._raise_bb(state, p.open_bb_mult)
        if score >= p.btn_limp:
            return AgentAction(ActionType.CALL)    # limp → sanitize lo pasa a CHECK si aplica
        return AgentAction(ActionType.FOLD)

    def _preflop_bb(self, state: GameState, score: float) -> AgentAction:
        p = self.profile

        if state.to_call == 0:
            # Nadie subió (limp del BTN o simplemente el check de BB)
            if score >= p.btn_open_raise and self._can_raise(state):
                return self._raise_bb(state, p.open_bb_mult)
            return AgentAction(ActionType.CHECK)

        # Enfrentando una subida del BTN
        if score >= p.bb_threebet and self._can_raise(state):
            return self._raise_bb(state, p.threebet_mult)
        if score >= p.bb_call_raise:
            return AgentAction(ActionType.CALL)
        return AgentAction(ActionType.FOLD)

    # ──────────────────────────── postflop ───────────────────────────────────

    def _postflop(self, state: GameState) -> AgentAction:
        hole, board = list(state.hole_cards), list(state.community_cards)
        equity = postflop_equity_estimate(hole, board)
        drawing = has_flush_draw(hole, board) or has_straight_draw(hole, board)

        if state.to_call == 0:
            return self._postflop_check_or_bet(state, equity, drawing)
        return self._postflop_call_or_raise(state, equity, drawing)

    def _postflop_check_or_bet(
        self, state: GameState, equity: float, drawing: bool
    ) -> AgentAction:
        """Nadie ha apostado: valor, semi-farol, farol puro… o check."""
        if not self._can_raise(state):
            return AgentAction(ActionType.CHECK)

        p          = self.profile
        value      = equity >= p.equity_value_bet
        semi_bluff = drawing and self._rng.random() < p.semi_bluff_freq
        bluff      = self._rng.random() < p.bluff_freq

        if value or semi_bluff or bluff:
            return self._bet_pot_fraction(state)
        return AgentAction(ActionType.CHECK)

    def _postflop_call_or_raise(
        self, state: GameState, equity: float, drawing: bool
    ) -> AgentAction:
        """Hay apuesta del rival: raise, call o fold."""
        p = self.profile

        # Re-subir exige más que apostar de primeras (ver equity_raise_vs_bet)
        if equity >= p.equity_raise_vs_bet and self._can_raise(state):
            return self._bet_pot_fraction(state)

        # Descuento por rango: quien apuesta no tiene un rango aleatorio.
        effective = max(equity * p.facing_bet_discount, p.call_equity_floor)

        # A un proyecto se le atribuye equity implícita: si lo liga, cobra el
        # bote futuro, así que paga con más frecuencia de la que dictaría su
        # fuerza actual (que es casi nula).
        if drawing:
            effective = max(effective, p.draw_call_equity)

        pot_odds = state.to_call / (state.pot + state.to_call)
        if effective >= pot_odds:
            return AgentAction(ActionType.CALL)
        return AgentAction(ActionType.FOLD)

    def _bet_pot_fraction(self, state: GameState) -> AgentAction:
        bet = max(int(state.pot * self.profile.bet_pot_fraction), state.min_raise_to)
        return self._clamp_raise(state, bet)

    # ──────────────────────── utilidades internas ─────────────────────────────

    def _can_raise(self, state: GameState) -> bool:
        return state.my_stack > state.to_call

    def _raise_bb(self, state: GameState, bb_mult: float) -> AgentAction:
        return self._clamp_raise(state, int(state.big_blind * bb_mult))

    @staticmethod
    def _clamp_raise(state: GameState, amount: int) -> AgentAction:
        if state.max_raise_to <= state.min_raise_to:
            return AgentAction(ActionType.ALL_IN)
        amount = max(amount, state.min_raise_to)
        if amount >= state.max_raise_to:
            return AgentAction(ActionType.ALL_IN)
        return AgentAction(ActionType.RAISE, amount)

    @staticmethod
    def _sanitize(action: AgentAction, state: GameState) -> AgentAction:
        t = action.action_type

        if t == ActionType.FOLD and state.to_call == 0:
            return AgentAction(ActionType.CHECK)

        if t == ActionType.CHECK and state.to_call > 0:
            return AgentAction(ActionType.CALL)

        if t == ActionType.RAISE:
            return RuleBasedAgent._clamp_raise(state, action.amount)

        return action
