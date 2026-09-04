from abc import ABC, abstractmethod
import numpy as np

# Abstract method for agent utility classes
class AgentUtility(ABC):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        self.manager = manager

    @property
    def game(self):
        return self.manager.game

    @property
    def player_index(self):
        return self.manager.player_index

    @property
    def a_conf(self):
        return self.manager.a_conf

    @property
    def g_conf(self):
        return self.manager.g_conf

    @property
    def absolute_to_relative(self):
        return self.manager.absolute_to_relative
    
    @property
    def relative_to_absolute(self):
        return self.manager.relative_to_absolute
    
    # ====================================================================================================
    # REWARD
    # ====================================================================================================

    def _current_round_scores(self):
        player_tricks = self.game.round.tricks_won[self.relative_to_absolute]
        correct = self.player_bids == player_tricks
        return np.where(
            correct,
            player_tricks * self.g_conf['points_for_successful_trick'] + self.g_conf['points_for_guessing_correctly'],
            self.g_conf['points_for_unsuccessfuly_trick'] * np.abs(self.player_bids - player_tricks)
        )

    def reward(self):
        scores = self.game.scores[self.relative_to_absolute]
        differences = scores[0] - np.delete(scores, 0)
        round_scores = self._current_round_scores()
        round_differences = round_scores[0] - np.delete(round_scores, 0)
        return (
            self.a_conf['score_difference_weight'] * np.sum(differences) + 
            self.a_conf['round_scores_weight'] * np.sum(round_differences)
        )

    # ====================================================================================================
    # STATE REPRESENTATION
    # ====================================================================================================

    @abstractmethod
    def generate_state_rerpesentation(self):
        pass

    def input_output_length(self):
        state = self.generate_state_rerpesentation()
        mask  = self._get_legal_move_mask()
        return (len(state), len(mask))

    # ====================================================================================================
    # CARD AND TRICK ENCODING
    # ====================================================================================================

    def _encode_hand(self):
        current_hand = self.game.get_hand(self.player_index)
        encoded_hand = np.zeros((2, len(self.game.deck.deck)), dtype=np.float32)
        encoded_hand[0, current_hand ==  1] = 1
        encoded_hand[1, current_hand ==  0] = 1
        return encoded_hand
    
    def _encode_trump(self):
        trump = self.game.round.trump
        encoded_trump = np.zeros(self.game.deck.d_conf["n_suits"] + 1, dtype=np.float32) # + 1 for wizard suit 
        return encoded_trump
    
    # ====================================================================================================
    # PLAYER ENCODING
    # ====================================================================================================

    def _encode_bids(self):
        max_players = self.game.max_players
        encoded_bids = np.zeros((max_players, 2), dtype=np.float32)
        bids = self.game.round.bids
        for player, bid in enumerate(bids):
            if bid is None:
                encoded_bids[player, 1] = 1
            else:
                encoded_bids[player, 0] = bid
        return encoded_bids[self.relative_to_absolute]

    def _encode_player_mask(self):
        max_players = self.game.max_players
        n_players   = self.game.n_players
        player_mask = np.zeros(max_players, dtype=np.float32)
        player_mask[:n_players] = 1
        return player_mask

    def _encode_priority(self):
        max_players = self.game.max_players
        priority_encoding = np.zeros(max_players, dtype=np.float32)
        priority_index = self.game.round.priority
        relative_priority = self.absolute_to_relative[priority_index]
        priority_encoding[relative_priority] = 1
        return priority_encoding

    def _encode_score(self):
        scores = np.asarray(self.game.scores, dtype=np.float32)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            encoded_scores = np.zeros_like(scores)
        else:
            encoded_scores = (scores - min_score) / (max_score - min_score)
        return encoded_scores[self.relative_to_absolute]
    
    # ====================================================================================================
    # GAME PROGRESS ENCODING
    # ====================================================================================================

    def _encode_fraction_rounds(self):
        encoded_fraction_rounds    = np.zeros(1, dtype=np.float32)
        encoded_fraction_rounds[0] = self.game.round_number / self.game.total_rounds
        return encoded_fraction_rounds

    def _encode_total_n_rounds(self):
        encoded_total_rounds    = np.zeros(1, dtype=np.float32)
        encoded_total_rounds[0] = self.game.total_rounds
        return encoded_total_rounds

    def _encode_rounds_left(self):
        encoded_rounds_left    = np.zeros(1, dtype=np.float32)
        encoded_rounds_left[0] = (self.game.total_rounds - self.game.round_number) / self.game.total_rounds
        return encoded_rounds_left
    
    # ====================================================================================================
    # GET MOVE
    # ====================================================================================================

    @abstractmethod
    def _get_legal_move_mask(self):
        pass

    def get_move(self, q_values):
        epsilon = self.a_conf['epsilon']
        mask = self._get_legal_move_mask()
        legal_moves = np.where(mask == 1)[0]
        if np.random.random() < epsilon:
            return np.random.choice(legal_moves)
        masked_q_values = np.where(mask, q_values, -np.inf)
        return np.argmax(masked_q_values)
    