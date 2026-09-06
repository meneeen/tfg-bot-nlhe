from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src.engine.controller import Street

if TYPE_CHECKING:
    from src.engine.controller import GameController
    from src.engine.player import Player
    
Card = int

class Position(Enum):

    BTN = "btn"  # dealer / small blind
    BB  = "bb"   # big blind


class ActionType(Enum):

    FOLD   = "fold"
    CHECK  = "check"
    CALL   = "call"
    RAISE  = "raise"
    ALL_IN = "all_in"


_ENGINE_ACTION_TO_TYPE: dict[str, ActionType] = {
    "fold"  : ActionType.FOLD,
    "check" : ActionType.CHECK,
    "call"  : ActionType.CALL,
    "raise" : ActionType.RAISE,
}

@dataclass(frozen=True)
class ActionRecord:

    player_name : str
    action_type : ActionType
    amount      : int    # fichas apostadas/subidas; 0 para FOLD y CHECK
    street      : Street


@dataclass(frozen=True)
class AgentAction:

    action_type : ActionType
    amount      : int = 0


@dataclass(frozen=True)
class GameState:

    # cartas

    hole_cards      : tuple[Card, ...]

    community_cards : tuple[Card, ...]

    # fichas

    pot               : int
    my_stack          : int
    opp_stack         : int
    my_bet_this_street : int

    # situación de apuestas

    to_call      : int
    min_raise_to : int
    max_raise_to : int

    # contexto de la mano

    position       : Position
    street         : Street
    action_history : tuple[ActionRecord, ...]

    # estructura de la partida

    big_blind  : int
    small_blind: int

    # adaptador desde el motor

    @staticmethod
    def from_engine_state(gc: "GameController", actor: "Player") -> "GameState":
        
        from src.engine.controller import GameController as _GC
        other = gc.agent if actor is gc.player else gc.player

        return GameState(
            hole_cards         = tuple(actor.hole_cards),
            community_cards    = tuple(gc.board),
            pot                = gc.pot,
            my_stack           = actor.stack,
            opp_stack          = other.stack,
            my_bet_this_street = actor.bet,
            to_call            = gc.max_bet - actor.bet,
            min_raise_to       = gc.min_raise_to,
            max_raise_to       = actor.bet + actor.stack,
            position           = Position.BTN if actor is gc.dealer else Position.BB,
            street             = gc.street,
            action_history     = GameState._parse_action_log(gc.action_log),
            big_blind          = _GC.BIG_BLIND,
            small_blind        = _GC.SMALL_BLIND,
        )

    @staticmethod
    def _parse_action_log(
        log: list[tuple[str, str, int, str, int]],
    ) -> tuple[ActionRecord, ...]:

        records: list[ActionRecord] = []
        for player_name, action_str, amount, street_name, _idx in log:
            records.append(ActionRecord(
                player_name = player_name,
                action_type = _ENGINE_ACTION_TO_TYPE.get(action_str, ActionType.RAISE),
                amount      = amount,
                street      = Street[street_name],
            ))
        return tuple(records)
    

class PokerAgent(ABC):

    @abstractmethod
    def decide_action(self, state: GameState) -> AgentAction:
        ...