import numpy as np
import torch

from src.models_utility.agent_utility import AgentUtility

class BiddingAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager=manager)

    # ====================================================================================================
    # STATE REPRESENTATION
    # ====================================================================================================

    def _generate_state_rerpesentation(self, game_index, player_index):
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

        encoded_hand            = self._encode_hand(game_index=game_index, player_index=player_index)
        encoded_player_mask     = self._encode_player_mask(game_index=game_index)
        encoded_priority        = self._encode_priority(game_index=game_index, player_index=player_index)
        encoded_trump           = self._encode_trump(game_index=game_index)
        encoded_scores          = self._encode_score(game_index=game_index, player_index=player_index)
        encoded_fraction_rounds = self._encode_fraction_rounds(game_index=game_index)
        encoded_bids            = self._encode_bids(game_index=game_index, player_index=player_index)
        encoded_rounds          = self._encode_total_rounds(game_index=game_index)
        encoded_rounds_left     = self._encode_rounds_left(game_index=game_index)

        state = np.concatenate([
            encoded_hand.flatten(),
            encoded_player_mask,
            encoded_priority,
            encoded_trump,
            encoded_scores,
            encoded_fraction_rounds,
            encoded_bids.flatten(),
            encoded_rounds,
            encoded_rounds_left,
        ])

        return torch.from_numpy(state)

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self, game_index, _player_index):
        mask = np.zeros(self._game(game_index).total_rounds, dtype=np.int8)
        mask[:self._game(game_index).round_number] = 1
        return mask
