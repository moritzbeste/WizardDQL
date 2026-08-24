import tomllib
import numpy as np
import random
import warnings
from pathlib import Path

from src.deck import Deck

class Game:
    CONFIG_PATH = Path(__file__).parent.parent / 'config.toml'

    def __init__(self, n_players=3):
        self.n_players = n_players
        self._set_config()
        self.reset_game()
        self.deck = Deck(n_players=n_players, seed=self.g_conf["seed"])
    
    def _set_config(self):
        with self.CONFIG_PATH.open('rb') as file:
            self.g_conf = tomllib.load(file)['game']
    
    def reset_game(self):
        self.priority     = random.randrange(self.n_players)
        self.round_number = 1
        self.round        = None
        self.trump_card   = None
    
    def setup_round(self):
        self.round        = _Round(n_players=self.n_players, deck=self.deck, round_number=self.round_number, priority=self.priority)
        self.round_number = self.round_number + 1
        self.priority     = (self.priority + 1) % self.n_players 


class _Round:
    trick = None

    def __init__(self, n_players, deck, round_number, priority):
        self.n_players    = n_players
        self.deck         = deck
        self.round_number = round_number
        self.trick_number = 1
        self.priority     = priority
        
        self._setup()
    
    def _setup(self):
        self.player_hands = self.deck.deal_hands(self.round_number, shuffle=True)
        self.trump        = self.deck.reveal_trump_card(self.round_number)
        self.bids         = [None for _ in range(self.n_players)]
        self.tricks_won   = [   0 for _ in range(self.n_players)]
    
    def _check_player_index(self, player_index, check_priority):
        if not isinstance(player_index, (int, np.integer)) or isinstance(player_index, bool):
            raise ValueError(f"player_index must be an integer, got {type(player_index).__name__}.")
        if player_index < 0 or player_index >= self.n_players:
            raise ValueError(f"Player with index {player_index} does not exist. There are {self.n_players} players.")
        if check_priority and player_index != self.priority:
            raise ValueError(f"Player with index {player_index} does not have priority! Player {self.priority} has priority")
    
    def _check_bid(self, bid):
        if not isinstance(bid, (int, np.integer)) or isinstance(bid, bool):
            raise ValueError(f"bid must be an integer, got {type(bid).__name__}.")
        bid = int(bid)
        if bid > self.round_number:
            warnings.warn(f"The bid {bid} > {self.round_number} was submitted by player {self.priority}")
        return bid
    
    def _winning_card_index(self, cards, i1, i2):
        _, c1, s1 = cards[i1]
        _, c2, s2 = cards[i2]
        # a wizard cannot be beaten
        # if both are wizards, the earlier one wins
        if self.deck.is_wizard(c1):
            return i1
        # a fool cannot beat anything
        if s2 == self.deck.fool_suit:
            return i1
        # same suit: higher card wins
        if s1 == s2:
            # important case to consider:
            # since wizards are treated as trump suit, we can get a case where s1 == s2
            # but s2 is a wizard. This is okay, since the values associated with all 
            # wizards are higher than any suit card. Furthermore, it is important to consider
            # that the values also encode the suit. So Blue goes from 0-12, Green goes from 
            # 13-25 etc. However since the suits are identical this does not matter
            if c2 > c1:
                return i2
            return i1
        # different suits
        # wizard beats everything except an earlier wizard,
        # which was handled above
        if self.deck.is_wizard(c2):
            return i2
        # trump beats non trump
        if s1 == self.trump:
            return i1
        if s2 == self.trump:
            return i2
        # leading suit beats cards that do not follow suit
        if s1 == self.trick.first_suit:
            return i1
        return i2

    def get_hand_reprs(self):
        return [self.deck.cards_string_repr(hand) for hand in self.player_hands]
    
    def get_hand(self, player_index):
        self._check_player_index(player_index=player_index, check_priority=False)
        return self.player_hands[player_index]
    
    def bid(self, player_index, bid):
        self._check_player_index(player_index=player_index, check_priority=True)
        bid                     = self._check_bid(bid)
        self.bids[player_index] = bid
        self.priority           = (self.priority + 1) % self.n_players
    
    def start_trick(self):
        if any(bid is None for bid in self.bids):
            raise ValueError(f"Not all bids have been set! Bids: {self.bids}")
        self.trick = _Trick(n_players=self.n_players, fool_suit=self.deck.fool_suit, wizard_suit=self.deck.wizard_suit)
        self.trick_number = self.trick_number + 1
    
    def play_card(self, player_index, card):
        if self.trick is None:
            self.start_trick()
        self._check_player_index(player_index=player_index, check_priority=True)
        # check if player has the played card
        if not self.player_hands[player_index][card] == 1:
            raise ValueError(f"Player with index {player_index} does not have the card {card} ({self.deck.cards_string_repr(card)})!")
        # check if player is allowed to play the card
        played_suit = self.deck.get_suit(card)
        if (
            played_suit != self.deck.fool_suit
            and self.trick.first_suit is not None
            and played_suit != self.trick.first_suit
            and self.deck.player_has_suit(self.player_hands[player_index], self.trick.first_suit, self.trump)
            ):
            raise ValueError(f"Player with index {player_index} is not allowed to play the card {card} ({self.deck.cards_string_repr(card)})!")
        self.trick.play_card(player_index=player_index, card=card, suit=self.deck.get_suit(card))
        # update card position in players hands
        self.player_hands[:, card] = -1 # -1 indicates already played
        self.priority = (self.priority + 1) % self.n_players
    
    def evaluate_trick(self):
        if self.trick is None:
            raise ValueError("There is no active trick.")
        cards = self.trick.get_cards()
        if len(cards) != self.n_players:
            raise ValueError(f"Cannot evaluate an incomplete trick. Expected {self.n_players} cards, got {len(cards)}.")
        highest = 0
        for i in range(1, len(cards), 1):
            highest = self._winning_card_index(cards, highest, i)
        self.trick = None
        winner = cards[highest][0]
        self.tricks_won[winner] += 1
        self.priority = winner
        return winner


class _Trick:
    def __init__(self, n_players, fool_suit, wizard_suit):
        self.n_players   = n_players
        self.fool_suit   = fool_suit
        self.wizard_suit = wizard_suit
        self.cards       = []
        self.first_suit  = None

    def play_card(self, player_index, card, suit):
        if len(self.cards) >= self.n_players:
            raise ValueError("Cannot play a card when the trick is already full!")
        if self.first_suit is None and suit != self.fool_suit and suit != self.wizard_suit:
            # card decides first suit
            self.first_suit = suit
        self.cards.append((player_index, card, suit))
    
    def get_cards():
        return self.cards
