from enum import Enum

class Action(Enum):
    FOLD  = "fold"
    CALL  = "call"
    RAISE = "raise"
    CHECK = "check"


class Player:
    def __init__(self, name: str, stack: int):
        self.name       = name
        self.stack      = stack
        self.hole_cards = []
        self.bet        = 0      # fichas apostadas en la calle actual
        self.folded     = False
        self.is_human   = False

    def reset_for_hand(self):
        self.hole_cards = []
        self.bet        = 0
        self.folded     = False

    def decide(self, game_state: dict):
        raise NotImplementedError

    def __repr__(self):
        return f"Player({self.name}, stack={self.stack})"


class HumanPlayer(Player):
    def __init__(self, name: str, stack: int):
        super().__init__(name, stack)
        self.is_human = True
