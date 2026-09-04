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
    