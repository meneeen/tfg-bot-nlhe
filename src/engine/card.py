from treys import Card, Deck, Evaluator


class PokerDeck:
    def __init__(self, seed: int = None):
        self._deck = Deck(seed=seed)

    def draw(self, n: int = 1) -> list:
        cards = self._deck.draw(n)
        return cards if isinstance(cards, list) else [cards]


class HandEvaluator:
    def __init__(self):
        self._ev = Evaluator()

    def evaluate(self, board: list, hole_cards: list) -> int:
        return self._ev.evaluate(board, hole_cards)

    def rank_to_string(self, score: int) -> str:
        return self._ev.class_to_string(self._ev.get_rank_class(score))


def card_to_str(card: int) -> str:
    return Card.int_to_str(card)
