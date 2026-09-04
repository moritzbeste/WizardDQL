import numpy as np
import torch

from src.models_utility.agent_utility import AgentUtility

class TrickingAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager)
        self.player_weights = self._compute_player_weights()
        self.player_bids = np.array(self.game.round.bids)[self.relative_to_absolute]

    def _compute_player_weights(self):
        scores = self.game.scores[self.relative_to_absolute]
        temp = self.a_conf['weight_temperature']
        weights = np.exp((scores - np.max(scores)) / temp)
        weights /= np.sum(weights)
        return weights

    # ====================================================================================================
    # REWARD
    # ====================================================================================================
    
    # overwrite
    def reward(self):
        round_scores = self._current_round_scores()
        return round_scores[0] * self.player_weights[0] - np.sum(np.delete(round_scores, 0) * np.delete(self.player_weights, 0))

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

        encoded_hand            = self._encode_hand()
        encoded_player_mask     = self._encode_player_mask()
        encoded_priority        = self._encode_priority()
        encoded_played          = self._encode_played_cards()
        encoded_recency         = self._encode_card_recency()
        encoded_trump           = self._encode_trump()
        encoded_leading_suit    = self._encode_leading_suit()
        encoded_scores          = self._encode_score()
        encoded_fraction_rounds = self._encode_fraction_rounds()
        encoded_fraction_tricks = self._encode_fraction_tricks()
        encoded_bids            = self._encode_bids()
        encoded_wins            = self._encode_tricks_won()
        encoded_n_rounds        = self._encode_total_n_rounds()
        encoded_rounds_left     = self._encode_rounds_left()

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
            encoded_n_rounds,
            encoded_rounds_left,
        ])

        return torch.from_numpy(state)
    
    # overwrite
    def input_output_length(self):
        state = self.generate_state_rerpesentation()
        output_size = len(self.game.deck.deck)
        return (len(state), output_size)

    # ====================================================================================================
    # CARD AND TRICK ENCODING
    # ====================================================================================================

    # overwrite
    def _encode_hand(self):
        current_hand = self.game.get_hand(self.player_index)
        encoded_hand = np.zeros((3, len(self.game.deck.deck)), dtype=np.float32)
        encoded_hand[0, current_hand ==  1] = 1
        encoded_hand[1, current_hand == -1] = 1
        encoded_hand[2, current_hand ==  0] = 1
        return encoded_hand

    def _encode_played_cards(self):
        max_players = self.game.max_players
        encoded_played = np.zeros((max_players, len(self.game.deck.deck)), dtype=np.float32)
        for player, card in self.game.round.played_cards:
            relative_player = self.absolute_to_relative[player]
            encoded_played[relative_player, card] = 1
        return encoded_played

    def _encode_card_recency(self):
        n_cards = len(self.game.deck.deck)
        encoded_recency = np.zeros(n_cards, dtype=np.float32)
        for trick_number, trick in enumerate(self.game.round.completed_tricks, start=1):
            for card in trick.cards:
                encoded_recency[card] = 1.0 - trick_number / self.game.round.trick_number
        return encoded_recency

    def _encode_leading_suit(self):
        leading_suit = self.game.round.trick.leading_suit if self.game.round.trick is not None else None
        encoded_leading_suit = np.zeros(self.game.deck.d_conf["n_suits"] + 2, dtype=np.float32) # + 1 for wizard suit, + 1 for None
        if leading_suit is None:
            encoded_leading_suit[-1] = 1
        else:
            encoded_leading_suit[leading_suit] = 1
        return encoded_leading_suit

    # ====================================================================================================
    # PLAYER ENCODING
    # ====================================================================================================

    def _encode_tricks_won(self):
        max_players = self.game.max_players
        encoded_wins = np.zeros(max_players, dtype=np.float32)
        wins = self.game.round.tricks_won
        encoded_wins[:len(wins)] = wins
        return encoded_wins[self.relative_to_absolute]

    # ====================================================================================================
    # ROUND PROGRESS ENCODING
    # ====================================================================================================

    def _encode_fraction_tricks(self):
        encoded_fraction_tricks    = np.zeros(1, dtype=np.float32)
        encoded_fraction_tricks[0] = (self.game.round_number - self.game.round.trick_number) / self.game.round_number
        return encoded_fraction_tricks

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self):
        hand = self.game.round.player_hands[self.player_index]
        cards = np.where(hand == 1)[0]
        mask = np.zeros(len(hand), dtype=np.int8)
        for card in cards:
            mask[card] = self.game.round.can_play_card(self.player_index, card)
        return mask
