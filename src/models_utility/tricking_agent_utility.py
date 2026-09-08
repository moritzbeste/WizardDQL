import numpy as np
import torch

from src.models_utility.agent_utility import AgentUtility

class TrickingAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager=manager)

    def _compute_player_weights(self, game_index, player_index):
        scores = self._game(game_index).scores[self._relative_to_absolute(player_index=player_index)]
        temp = self.a_conf['weight_temperature']
        weights = np.exp((scores - np.max(scores)) / temp)
        weights /= np.sum(weights)
        return weights

    # ====================================================================================================
    # REWARD
    # ====================================================================================================
    
    # overwrite
    def reward(self, game_index, player_index):
        round_scores = self._current_round_scores()
        player_weights = _compute_player_weights(game_index, player_index)
        return round_scores[0] * self.player_weights[0] - np.sum(np.delete(round_scores, 0) * np.delete(self.player_weights, 0))

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

        # relative one hot encoding of played cards eg
        # - [0, ..., 1, ..., 0], [0, ..., 0], [0, ..., 0]
        # - shape: max_players * n_cards (6, 60)
        # - in the case that the first player has played their card and the other two have not
        # - (position of agent in that list should remain fixed regardless of seat)

        # encoding of when cards were played
        # - shape: n_cards (60,)

        # one hot encoding of trump
        # - cases: B, G, R, Y, W
        # - shape: n_cases (5,)

        # one hot encoding of current leading suit
        # - cases: None, B, G, R, Y, W
        # - shape: n_cases (6,)

        # normalized encoding of score
        # - shape: max_players (6,)

        # round number
        # - shape: scalar (1,)

        # number of tricks remaining
        # - shape: scalar (1,)

        # bids made by each player
        # - shape: max_players (6,)

        # tricks won by each player in current round
        # - shape: max_players (6,)

        # total number of rounds
        # - shape: scalar (1,)

        # rounds left
        # - shape: scalar (1,)

        encoded_hand            = self._encode_hand(game_index=game_index, player_index=player_index)
        encoded_player_mask     = self._encode_player_mask(game_index=game_index)
        encoded_priority        = self._encode_priority(game_index=game_index, player_index=player_index)
        encoded_played          = self._encode_played_cards(game_index=game_index)
        encoded_recency         = self._encode_card_recency(game_index=game_index)
        encoded_trump           = self._encode_trump(game_index=game_index)
        encoded_leading_suit    = self._encode_leading_suit(game_index=game_index)
        encoded_scores          = self._encode_score(game_index=game_index, player_index=player_index)
        encoded_fraction_rounds = self._encode_fraction_rounds(game_index=game_index)
        encoded_fraction_tricks = self._encode_fraction_tricks(game_index=game_index)
        encoded_bids            = self._encode_bids(game_index=game_index, player_index=player_index)
        encoded_wins            = self._encode_tricks_won(game_index=game_index, player_index=player_index)
        encoded_rounds          = self._encode_total_rounds(game_index=game_index)
        encoded_rounds_left     = self._encode_rounds_left(game_index=game_index)

        state = np.concatenate([
            encoded_hand.flatten(),
            encoded_player_mask,
            encoded_priority,
            encoded_played.flatten(),
            encoded_recency,
            encoded_trump,
            encoded_leading_suit,
            encoded_scores,
            encoded_fraction_rounds,
            encoded_fraction_tricks,
            encoded_bids.flatten(),
            encoded_wins,
            encoded_rounds,
            encoded_rounds_left,
        ])

        return torch.from_numpy(state)
    
    # overwrite
    def input_output_length(self):
        state = self.generate_state_rerpesentation()
        output_size = len(self.manager.deck.deck)
        return (len(state), output_size)

    # ====================================================================================================
    # CARD AND TRICK ENCODING
    # ====================================================================================================

    # overwrite
    def _encode_hand(self, game_index, player_index):
        current_hand = self._game(game_index).get_hand(player_index)
        encoded_hand = np.zeros((3, len(self._game(game_index).deck.deck)), dtype=np.float32)
        encoded_hand[0, current_hand ==  1] = 1
        encoded_hand[1, current_hand == -1] = 1
        encoded_hand[2, current_hand ==  0] = 1
        return encoded_hand

    def _encode_played_cards(self, game_index):
        max_players = self._game(game_index).max_players
        encoded_played = np.zeros((max_players, len(self._game(game_index).deck.deck)), dtype=np.float32)
        for player, card in self._game(game_index).round.played_cards:
            relative_player = self.absolute_to_relative[player]
            encoded_played[relative_player, card] = 1
        return encoded_played

    def _encode_card_recency(self, game_index):
        n_cards = len(self._game(game_index).deck.deck)
        encoded_recency = np.zeros(n_cards, dtype=np.float32)
        for trick_number, trick in enumerate(self._game(game_index).round.completed_tricks, start=1):
            for card in trick.cards:
                encoded_recency[card] = 1.0 - trick_number / self._game(game_index).round.trick_number
        return encoded_recency

    def _encode_leading_suit(self, game_index):
        leading_suit = self._game(game_index).round.trick.leading_suit if self._game(game_index).round.trick is not None else None
        encoded_leading_suit = np.zeros(self._game(game_index).deck.d_conf["n_suits"] + 2, dtype=np.float32) # + 1 for wizard suit, + 1 for None
        if leading_suit is None:
            encoded_leading_suit[-1] = 1
        else:
            encoded_leading_suit[leading_suit] = 1
        return encoded_leading_suit

    # ====================================================================================================
    # PLAYER ENCODING
    # ====================================================================================================

    def _encode_tricks_won(self, game_index, player_index):
        max_players = self._game(game_index).max_players
        encoded_wins = np.zeros(max_players, dtype=np.float32)
        wins = self._game(game_index).round.tricks_won
        encoded_wins[:len(wins)] = wins
        return encoded_wins[self._relative_to_absolute(player_index=player_index)]

    # ====================================================================================================
    # ROUND PROGRESS ENCODING
    # ====================================================================================================

    def _encode_fraction_tricks(self, game_index):
        encoded_fraction_tricks    = np.zeros(1, dtype=np.float32)
        encoded_fraction_tricks[0] = (self._game(game_index).round_number - self._game(game_index).round.trick_number) / self._game(game_index).round_number
        return encoded_fraction_tricks

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self, game_index, player_index):
        hand = self._game(game_index).round.player_hands[player_index]
        cards = np.where(hand == 1)[0]
        mask = np.zeros(len(hand), dtype=np.int8)
        for card in cards:
            mask[card] = self._game(game_index).round.can_play_card(self.player_index, card)
        return mask
