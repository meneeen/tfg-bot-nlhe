from __future__ import annotations

import random

from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent
# Reutiliza el mismo _sanitize que el resto de agentes: garantiza que una acción
# propuesta por la política (p.ej. un RAISE cuando ya no hay stack) se convierta
# en una jugada legal en vez de reventar el motor.
from src.ai.llm_agent import _sanitize


class RandomAgent(PokerAgent):
    """
    Agente aleatorio: suelo de comparación del harness de evaluación.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide_action(self, state: GameState) -> AgentAction:
        return _sanitize(self._policy(state), state)

    def _policy(self, state: GameState) -> AgentAction:
        r = self._rng.random()
        if state.to_call == 0:
            if r < 0.75:
                return AgentAction(ActionType.CHECK)
            return AgentAction(ActionType.RAISE, state.min_raise_to)

        # Hay una apuesta que igualar.
        if r < 0.12:
            return AgentAction(ActionType.FOLD)
        if r < 0.80:                       # 0.12 .. 0.80  ->  68 %
            return AgentAction(ActionType.CALL)
        return AgentAction(ActionType.RAISE, state.min_raise_to)
