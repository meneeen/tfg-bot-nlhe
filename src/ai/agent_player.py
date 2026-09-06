from __future__ import annotations

from typing import TYPE_CHECKING

from src.engine.player import Player, Action
from src.ai.base_agent import AgentAction, ActionType, GameState, PokerAgent

if TYPE_CHECKING:
    from src.engine.controller import GameController


def to_engine_action(action: AgentAction, state: GameState) -> tuple[Action, int]:
    t = action.action_type
    if t == ActionType.FOLD:   return Action.FOLD,  0
    if t == ActionType.CHECK:  return Action.CHECK, 0
    if t == ActionType.CALL:   return Action.CALL,  state.to_call
    if t == ActionType.RAISE:  return Action.RAISE, action.amount
    if t == ActionType.ALL_IN: return Action.RAISE, state.max_raise_to
    return Action.FOLD, 0


class AgentPlayer(Player):
    def __init__(self, name: str, stack: int, agent: PokerAgent) -> None:
        super().__init__(name, stack)
        self._agent: PokerAgent = agent
        self._gc: "GameController | None" = None

    def bind(self, gc: "GameController") -> None:
        self._gc = gc

    def set_agent(self, agent: PokerAgent) -> None:
        self._agent = agent

    def decide(self, game_state: dict) -> tuple[Action, int]:
        assert self._gc is not None, (
            "AgentPlayer: llama a bind(gc) antes de decidir."
        )
        state  = GameState.from_engine_state(self._gc, self)
        action = self._agent.decide_action(state)
        return to_engine_action(action, state)
