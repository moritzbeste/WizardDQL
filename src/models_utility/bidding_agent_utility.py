import numpy as np
import torch

from src.models_utility.agent_utility import AgentUtility

class BiddingAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager)

    # ====================================================================================================
    # STATE REPRESENTATION
    # ====================================================================================================

    def generate_state_rerpesentation(self):
        # one hot encoding of the state of each card 
        # - (in agents hand, already played, unknown (in opponents hand or in the deck))
        # - shape: n_states * n_cards (3, 60)

        # player mask
        # - shape: max_players (6,)

        # relative one hot encoding of player with priority
        # - shape: max_players (6,)

        # one hot encoding of trump
        # - cases: B, G, R, Y, W
        # - shape: n_cases (5,)

        # normalized encoding of score
        # - shape: max_players (6,)

        # round number
        # - shape: scalar (1,)

        # bids made by each player
        # - shape: max_players (6,)

        # total number of rounds
        # - shape: scalar (1,)

        # rounds left
        # - shape: scalar (1,)

        encoded_hand            = self._encode_hand()
        encoded_player_mask     = self._encode_player_mask()
        encoded_priority        = self._encode_priority()
        encoded_trump           = self._encode_trump()
        encoded_scores          = self._encode_score()
        encoded_fraction_rounds = self._encode_fraction_rounds()
        encoded_bids            = self._encode_bids()
        encoded_n_rounds        = self._encode_total_n_rounds()
        encoded_rounds_left     = self._encode_rounds_left()

        state = np.concatenate([
            encoded_hand.flatten(),
            encoded_player_mask,
            encoded_priority,
            encoded_trump,
            encoded_scores,
            encoded_fraction_rounds,
            encoded_bids.flatten(),
            encoded_n_rounds,
            encoded_rounds_left,
        ])

        return torch.from_numpy(state)

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self):
        mask = np.zeros(self.game.total_rounds, dtype=np.int8)
        mask[:self.game.round_number] = 1
        return mask
