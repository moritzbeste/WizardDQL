import numpy as np
import torch

from src.utility import get_config

from src.models_utility.agent_utility import AgentUtility

class TrumpAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager=manager)
        self.d_conf = get_config('deck')

    # ====================================================================================================
    # STATE REPRESENTATION
    # ====================================================================================================

    def _generate_state_rerpesentation(self, game_index, player_index):
        # one hot encoding of the state of each card 
        # - (in agents hand, already played, unknown (in opponents hand or in the deck))
        # - shape: n_states * n_cards (3, 60)

        # player mask
        # - shape: max_players (6,)

        # normalized encoding of score
        # - shape: max_players (6,)

        # round number
        # - shape: scalar (1,)

        # total number of rounds
        # - shape: scalar (1,)

        # rounds left
        # - shape: scalar (1,)

        encoded_hand            = self._encode_hand(game_index=game_index, player_index=player_index)
        encoded_player_mask     = self._encode_player_mask(game_index=game_index)
        encoded_scores          = self._encode_score(game_index=game_index, player_index=player_index)
        encoded_fraction_rounds = self._encode_fraction_rounds(game_index=game_index)
        encoded_rounds          = self._encode_total_rounds(game_index=game_index)
        encoded_rounds_left     = self._encode_rounds_left(game_index=game_index)

        state = np.concatenate([
            encoded_hand.flatten(),
            encoded_player_mask,
            encoded_scores,
            encoded_fraction_rounds,
            encoded_rounds,
            encoded_rounds_left,
        ])

        return torch.from_numpy(state)

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self, _game_index, _player_index):
        return np.ones(self.d_conf['n_suits'], dtype=np.int8)
