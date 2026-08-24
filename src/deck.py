import tomllib
import numpy as np
from pathlib import Path

class Deck:
    CONFIG_PATH = Path(__file__).parent.parent / 'config.toml'
    cards_dict  = None

    def __init__(self, n_players, seed):
        self.n_players = n_players
        self._set_config()
        self._set_deck()
        self.rng = np.random.default_rng(seed)
        self.n_suit_cards = self.d_conf['n_cards_per_suit'] * self.d_conf['n_suits']
    
    def _set_deck(self):
        n_cards = self.d_conf['n_suits'] * self.d_conf['n_cards_per_suit']
        self.special_card_range = {}
        for card_type, c in self.d_conf['special_cards'].items():
            count = c['count']
            self.special_card_range[card_type] = (n_cards, n_cards + count - 1)
            n_cards += count
        self.deck = np.arange(n_cards, dtype=np.uint8)
    
    def _shuffle_deck(self):
        self.rng.shuffle(self.deck)
    
    def _set_config(self):
        with self.CONFIG_PATH.open('rb') as file:
            self.d_conf = tomllib.load(file)['deck']

    def _generate_cards_dict(self):
        self.cards_dict = {}

        # suited cards
        for val in range(self.n_suit_cards):
            rank = val % self.d_conf['n_cards_per_suit'] + 1
            suit = val // self.d_conf['n_cards_per_suit']
            self.cards_dict[val] = str(rank) + self.d_conf['suits'][suit]

        # special cards
        val = self.n_suit_cards
        for card_type, c in self.d_conf['special_cards'].items():
            for _ in range(c['count']):
                self.cards_dict[val] = card_type
                val += 1

    def _check_card(self, card):
        if not isinstance(card, (int, np.integer)) or isinstance(card, bool):
            raise ValueError(f"Card must be an integer, got {type(card).__name__}.")
        card = int(card)
        if card < 0 or card >= len(self.deck):
            raise ValueError(f"Card {card} does not exist in the deck. Valid card IDs are 0-{len(self.deck) - 1}.")
        return card

    def _check_suit(self, suit):
        if not isinstance(suit, (int, np.integer)) or isinstance(suit, bool):
            raise ValueError(f"Suit must be an integer, got {type(suit).__name__}.")
        suit = int(suit)
        if suit < self.fool_suit or suit > self.wizard_suit:
            raise ValueError(f"Suit {suit} does not exist. Valid suit IDs are 0-{self.d_conf['n_suits'] - 1}.")
        return suit

    def _check_n_cards(self, n_cards):
        if not isinstance(n_cards, (int, np.integer)) or isinstance(n_cards, bool):
            raise ValueError(f"n_cards must be an integer, got {type(n_cards).__name__}.")
        n_cards = int(n_cards)
        if n_cards < 0:
            raise ValueError("n_cards cannot be negative.")
        return n_cards

    def _check_hand(self, player_hand):
        if not isinstance(player_hand, np.ndarray):
            raise ValueError("player_hand must be a NumPy array.")
        if player_hand.ndim != 1:
            raise ValueError("player_hand must be a one-dimensional vector.")
        if len(player_hand) != len(self.deck):
            raise ValueError(
                f"Invalid hand vector length. Expected {len(self.deck)}, got {len(player_hand)}.")
        if not np.all((player_hand == 0) | (player_hand == 1)):
            raise ValueError("player_hand must contain only 0 and 1.")

    def cards_string_repr(self, cards):
        if self.cards_dict is None:
            self._generate_cards_dict()
        if isinstance(cards, (int, np.integer)):
            card = self._check_card(cards)
            return self.cards_dict[card]
        self._check_hand(cards)
        return [self.cards_dict[card] for card in np.flatnonzero(cards)]

    def is_wizard(self, card):
        card = self._check_card(card)
        wizard_start, wizard_end = self.special_card_range['Z']
        return wizard_start <= card <= wizard_end

    def get_suit(self, card, trump):
        card = self._check_card(card)
        if card < self.n_suit_cards:
            return card // self.d_conf['n_cards_per_suit']
        if self.is_wizard(card):
            trump = self._check_suit(trump)
            return trump
        fool_start, fool_end = self.special_card_range['N']
        if fool_start <= card <= fool_end:
            return self.fool_suit
        raise ValueError(f"Card {card} could not be assigned a suit!")

    def player_has_suit(self, player_hand, suit, trump):
        self._check_hand(player_hand)
        suit = self._check_suit(suit)
        trump = self._check_suit(trump)
        wizard_start, wizard_end = self.special_card_range['Z']
        # Wizard suit
        if suit == self.wizard_suit:
            return np.any(player_hand[wizard_start:wizard_end + 1])
        n = self.d_conf['n_cards_per_suit']
        offset = suit * n
        has_suit = np.any(player_hand[offset:offset + n])
        # Normal non trump suit
        if suit != trump:
            return has_suit
        # Normal trump suit: wizards also count
        has_wizard = np.any(player_hand[wizard_start:wizard_end + 1])
        return has_suit or has_wizard

    def deal_hands(self, n_cards, shuffle=True):
        n_cards = self._check_n_cards(n_cards)
        total_cards = n_cards * self.n_players
        if total_cards > len(self.deck):
            raise ValueError(f"Cannot deal {n_cards} cards to {self.n_players} players given a deck of size {len(self.deck)}.")
        if shuffle:
            self._shuffle_deck()

        dealt_cards = self.deck[:total_cards]
        vectors = np.zeros((self.n_players, len(self.deck)), dtype=np.uint8)
        vectors[np.arange(self.n_players).repeat(n_cards), dealt_cards] = 1
        return vectors

    def reveal_trump_card(self, n_cards):
        n_cards = self._check_n_cards(n_cards)
        n_cards_dealt = n_cards * self.n_players
        if n_cards_dealt >= len(self.deck):
            return self.wizard_suit
        return self.deck[n_cards_dealt]

    @property
    def wizard_suit(self):
        return self.d_conf['n_suits']
    
    @property
    def fool_suit(self):
        return -1