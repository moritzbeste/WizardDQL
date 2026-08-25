import tomllib
import numpy as np
import random
import warnings
from pathlib import Path
from enum import Enum

from src.deck import Deck

class Game:

    # ====================================================================================================
    # CLASS CONSTANTS AND STATE
    # ====================================================================================================

    CONFIG_PATH = Path(__file__).parent.parent / 'config.toml'

    class State(Enum):
        SETUP    = auto()
        ROUND    = auto()
        FINISHED = auto()

    # ====================================================================================================
    # INITIALIZATION AND SETUP
    # ====================================================================================================

    def __init__(self, n_players=3):
        self.n_players = n_players
        self.state = self.State.SETUP
        self._set_config()
        seed = self.g_conf['seed_priority']
        self.rng = np.random.default_rng(seed)
        self.scores = np.zeros(self.n_players, dtype=np.float32)
        self.deck = Deck(n_players=n_players, seed=self.g_conf['seed_deck'])
        self.total_rounds = len(self.deck.deck) // self.n_players
        self.reset_game()
        self.min_players, self.max_players = self.get_player_range()
        self._advance_state()
    
    def _set_config(self):
        with self.CONFIG_PATH.open('rb') as file:
            self.g_conf = tomllib.load(file)['game']

    def reset_game(self):
        self.priority     = self.rng.integers(self.n_players)
        self.round_number = 1
        self.round        = None
        self.scores[:]    = 0.0
        self.rounds_left = self.total_rounds

    # ====================================================================================================
    # STATE MANAGEMENT
    # ====================================================================================================

    def _advance_state(self):
        while True:
            match self.state:
                case self.State.SETUP:
                    self.setup_round()
                    self.state = self.State.ROUND
                case self.State.ROUND:
                    if self.round is None:
                        if self.rounds_left == 0:
                            self.state = self.State.FINISHED
                        else:
                            self.setup_round()
                    else:
                        return
                case self.State.FINISHED:
                    return

    # ====================================================================================================
    # INTERNAL ROUND MANAGEMENT
    # ====================================================================================================

    def setup_round(self):
        self.priority     = (self.priority + 1) % self.n_players 
        self.round        = _Round(n_players=self.n_players, deck=self.deck, round_number=self.round_number, priority=self.priority)
    
    def finish_round(self):
        bids   = self.round.bids
        tricks = self.round.tricks_won
        for i in range(self.n_players):
            if bids[i] == tricks[i]:
                self.scores[i] += self.g_conf['points_for_guessing_correctly'] + tricks[i] * self.g_conf['points_for_successful_trick']
            else: 
                self.scores[i] += abs(bids[i] - tricks[i]) * self.g_conf['points_for_unsuccessfuly_trick']
        self.round = None
        self.round_number = self.round_number + 1
        self.rounds_left  = self.rounds_left  - 1
        self._advance_state()

    # ====================================================================================================
    # INFORMATION AND QUERIES
    # ====================================================================================================

    def get_player_range(self):
        return (self.g_conf['min_players'], self.g_conf['max_players'])

    def get_hand(self, player_index):
        if self.round is None:
            raise ValueError("The round has not started!")
        return self.round.get_hand(player_index=player_index)

    def get_winner(self):
        if self.rounds_left != 0:
            return None
        return np.argmax(self.scores)


class _Round:

    # ====================================================================================================
    # STATE
    # ====================================================================================================

    class State(Enum):
        SETUP         = auto()
        CHOOSE_TRUMP  = auto()
        BIDDING       = auto()
        TRICK         = auto()
        FINISHED      = auto()

    # ====================================================================================================
    # INITIALIZATION AND SETUP
    # ====================================================================================================

    def __init__(self, n_players, deck, round_number, priority):
        self.n_players    = n_players
        self.state        = self.State.SETUP
        self.trick        = None
        self.deck         = deck
        self.round_number = round_number
        self.trick_number = 0
        self.priority     = priority
        self._setup()
        self._advance_state()
    
    def _setup(self):
        self.player_hands     = self.deck.deal_hands(self.round_number, shuffle=True)
        self.trump            = self.deck.reveal_trump_suit(self.round_number)
        self.bids             = [None for _ in range(self.n_players)]
        self.tricks_won       = [   0 for _ in range(self.n_players)]
        self.played_cards     = []
        self.completed_tricks = []

    # ====================================================================================================
    # STATE MANAGEMENT
    # ====================================================================================================

    # advance to the next state which requires player action
    def _advance_state(self):
        while True:
            match self.state:
                case self.State.SETUP:
                    if self.trump is None:
                        self.state = self.State.CHOOSE_TRUMP
                    else:
                        self.state = self.State.BIDDING
                case self.State.CHOOSE_TRUMP:
                    if self.trump is None:
                        return
                    self.state = self.State.BIDDING
                case self.State.BIDDING:
                    if any(bid is None for bid in self.bids):
                        return
                    self.state = self.State.TRICK
                    self._start_trick()
                case self.State.TRICK:
                    if not self.trick.trick_completed():
                        return
                    self._evaluate_trick()
                    if self.trick_number == self.round_number:
                        self.state = self.State.FINISHED
                    else:
                        self._start_trick()
                case self.State.FINISHED:
                    return

    # ====================================================================================================
    # CHECKING AND VALIDATION
    # ====================================================================================================

    def _check_player_index(self, player_index, check_priority):
        if not isinstance(player_index, (int, np.integer)) or isinstance(player_index, bool):
            raise ValueError(f"player_index must be an integer, got {type(player_index).__name__}.")
        if player_index < 0 or player_index >= self.n_players:
            raise ValueError(f"Player with index {player_index} does not exist. There are {self.n_players} players.")
        if check_priority and player_index != self.priority:
            raise ValueError(f"Player with index {player_index} does not have priority! Player {self.priority} has priority")
    
    def _check_bid(self, bid):
        if not self.state == self.State.BIDDING:
            raise ValueError(f"The round is not in the bidding state. It is in state {self.state.name}")
        if not isinstance(bid, (int, np.integer)) or isinstance(bid, bool):
            raise ValueError(f"bid must be an integer, got {type(bid).__name__}.")
        if self.trump is None:
            raise ValueError("Cannot Start Bidding Before Trump Suit is Decided!")
        bid = int(bid)
        if bid < 0:
            raise ValueError(f"Cannot make a negative bid but got {bid} submitted by player {self.priority}!")
        if bid > self.round_number:
            warnings.warn(f"The bid {bid} > {self.round_number} was submitted by player {self.priority}")
        return bid
    
    def _check_trick(self):
        if self.trick is None:
            raise ValueError("There is no active trick.")
        if len(self.cards) >= self.n_players:
            raise ValueError("Cannot play a card when the trick is already full!")
    
    def _check_state(self, desired_state):
        if self.state != desired_state:
            raise ValueError(f"The round is not in state {desired_state}. It is in state {self.state}!")

    # ====================================================================================================
    # INTERNAL TRICK MANAGEMENT
    # ====================================================================================================

    def _start_trick(self):
        self._check_state(self.State.TRICK)
        if self.trick is not None:
            raise ValueError(f"There exists an active trick. Cannot create a new one!")
        self.trick_number = self.trick_number + 1
        self.trick = _Trick(n_players=self.n_players, fool_suit=self.deck.fool_suit, wizard_suit=self.deck.wizard_suit)

    def _evaluate_trick(self):
        self._check_state
        self._check_trick()
        cards = self.trick.get_cards()
        if len(cards) != self.n_players:
            raise ValueError(f"Cannot evaluate an incomplete trick. Expected {self.n_players} cards, got {len(cards)}.")
        highest = 0
        for i in range(1, len(cards), 1):
            highest = self._winning_card_index(cards, highest, i)
        self.completed_tricks.append(self.trick)
        self.trick = None
        winner = cards[highest][0]
        self.tricks_won[winner] += 1
        self.priority = winner
        return winner

    # ====================================================================================================
    # INTERNAL GAME LOGIC
    # ====================================================================================================

    def _winning_card_index(self, cards, i1, i2):
        _, c1, s1 = cards[i1]
        _, c2, s2 = cards[i2]

        # ====================================================================================================
        # the rules encoded here differ slightly from the original rules
        # they are house rules that I prefer
        # a more readable explanation of the house rules is provided in README.md
        # ====================================================================================================

        # the first wizard always wins
        if self.deck.is_wizard(c1):
            return i1
        # a fool as the second card cannot win
        if s2 == self.deck.fool_suit:
            return i1
        # we know: 
        # - i2 is not a fool
        # -- therefore if i1 is a fool, i2 wins because anything other than a fool beats a fool
        if s1 == self.deck.fool_suit:
            return i2
        if s1 == s2:
            # we know:
            # - s1 == s2
            # - numerical compaison of wizards and fools is unsafe because precedence matters over encoding
            # -- i1 is not a wizard
            # -- i1, i2 are not fools
            # --- therefore the cases
            # ---- s1 == s2 == wizards
            # ---- s1 == s2 == fools
            # --- are no longer possible
            # - wizards have trump suit
            # -- possible case: s1 is trump, s2 is wizard
            # --- since wizards are encoded with a higher value the comparison is numerically safe
            if c2 > c1:
                return i2
            return i1
        # we know:
        # - s1 != s2
        # -- therefore the winner is decided based on suit
        # - i1 is not a wizard
        # -- therefore i2 could be a wizard in which case it wins
        if self.deck.is_wizard(c2):
            return i2
        # - i2 is not a wizard
        # - trump wins over non trump
        if s1 == self.trump:
            return i1
        if s2 == self.trump:
            return i2
        # - s1, s2 are not the trump suit
        # - i1 is the current winner
        # -- therefore, s1 must be the leading suit
        # - since s1 != s2, s2 is not the leading suit
        # -- therefore, i1 wins
        return i1

    # ====================================================================================================
    # INFORMATION AND GETTERS
    # ====================================================================================================

    def get_hand_reprs(self):
        return [self.deck.cards_string_repr(hand) for hand in self.player_hands]
    
    def get_hand(self, player_index):
        self._check_player_index(player_index=player_index, check_priority=False)
        return self.player_hands[player_index]
    
    def get_bids(self):
        return self.bids

    def can_play_card(self, player_index, card):
        self._check_trick()
        self._check_state(self.State.TRICK)
        self._check_player_index(player_index=player_index, check_priority=True)
        # check if player has the played card
        if not self.player_hands[player_index][card] == 1:
            return False
        # check if player is allowed to play the card
        played_suit = self.deck.get_effective_suit(card, self.trump)
        if (
            played_suit != self.deck.fool_suit
            and self.trick.first_suit is not None
            and played_suit != self.trick.first_suit
            and self.deck.player_has_suit(self.player_hands[player_index], self.trick.first_suit, self.trump)
            ):
            return False
        return True

    # ====================================================================================================
    # PLAYER ACTIONS
    # ====================================================================================================

    # used for a player to choose the trump suit
    def pick_trump_color(self, player_index, suit):
        self._check_player_index(player_index=player_index, check_priority=True)
        suit = self.deck.check_suit(suit)
        self.trump = suit
        self._advance_state()

    def bid(self, player_index, bid):
        self._check_player_index(player_index=player_index, check_priority=True)
        self._check_bid(self, bid)
        bid                     = self._check_bid(bid)
        self.bids[player_index] = bid
        self.priority           = (self.priority + 1) % self.n_players
        self._advance_state()
    
    def play_card(self, player_index, card):
        self._check_state(self.State.TRICK)
        if not self.can_play_card(player_index, card):
            raise ValueError(f"Player with index {player_index} cannot play card {card}!")
        self.trick.play_card(player_index=player_index, card=card, suit=played_suit)
        # update card position in players hands
        self.player_hands[:, card] = -1 # -1 indicates already played
        self.played_cards.append((self.priority, card))
        self.priority = (self.priority + 1) % self.n_players
        self._advance_state()


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
    
    def get_cards(self):
        return self.cards
    
    def trick_completed(self):
        return len(self.cards) == self.n_players
