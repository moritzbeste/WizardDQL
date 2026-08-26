import numpy as np

class TrickingAgentUtility:

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, utility):
        self.utility = utility
        self.player_weights = self._compute_player_weights()
        self.player_bids = np.array(self.game.round.get_bids())[self.relative_to_absolute]
        self.player_tricks = np.zeros(self.game.n_players)

    @property
    def game(self):
        return self.utility.game

    @property
    def player_index(self):
        return self.utility.player_index

    @property
    def a_conf(self):
        return self.utility.a_conf

    @property
    def g_conf(self):
        return self.utility.g_conf

    @property
    def absolute_to_relative(self):
        return self.utility.absolute_to_relative
    
    @property
    def relative_to_absolute(self):
        return self.utility.relative_to_absolute

    def _compute_player_weights(self):
        scores = self.game.scores[self.relative_to_absolute]
        temp = self.a_conf['weight_temperature']
        weights = np.exp((scores - np.max(scores)) / temp)
        weights /= np.sum(weights)
        return weights

    # ====================================================================================================
    # REWARD
    # ====================================================================================================

    def _current_round_scores(self):
        correct = self.player_bids == self.player_tricks
        return np.where(
            correct,
            self.player_tricks * self.g_conf['points_for_successful_trick'] + self.g_conf['points_for_guessing_correctly'],
            self.g_conf['points_for_unsuccessfuly_trick'] * np.abs(self.player_bids - self.player_tricks)
        )
    
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

        # relative lead for each player
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
        encoded_leads           = self._encode_leads()
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
            encoded_leads,
            encoded_fraction_rounds,
            encoded_fraction_tricks,
            encoded_bids.flatten(),
            encoded_wins,
            encoded_n_rounds,
            encoded_rounds_left,
        ])

        return torch.from_numpy(state)

    # ====================================================================================================
    # CARD AND TRICK ENCODING
    # ====================================================================================================

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

    def _encode_trump(self):
        trump = self.game.round.trump
        encoded_trump = np.zeros(self.game.deck.d_conf["n_suits"] + 1, dtype=np.float32) # + 1 for wizard suit 
        return encoded_trump

    def _encode_leading_suit(self):
        leading_suit = self.game.round.trick.first_suit if self.game.round.trick is not None else None
        encoded_leading_suit = np.zeros(self.game.deck.d_conf["n_suits"] + 2, dtype=np.float32) # + 1 for wizard suit, + 1 for None
        if leading_suit is None:
            encoded_leading_suit[-1] = 1
        else:
            encoded_leading_suit[leading_suit] = 1
        return encoded_leading_suit

    # ====================================================================================================
    # PLAYER ENCODING
    # ====================================================================================================

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
    
    def _encode_leads(self):
        leads = np.array([self._compute_lead(p) for p in range(self.game.n_players)], dtype=np.float32)
        return leads[self.relative_to_absolute]

    def _encode_bids(self):
        max_players = self.game.max_players
        encoded_bids = np.zeros((max_players, 2), dtype=np.float32)
        bids = self.game.round.get_bids()
        for player, bid in enumerate(bids):
            if bid is None:
                encoded_bids[player, 1] = 1
            else:
                encoded_bids[player, 0] = bid
        return encoded_bids[self.relative_to_absolute]

    def _encode_tricks_won(self):
        max_players = self.game.max_players
        encoded_wins = np.zeros(max_players, dtype=np.float32)
        wins = self.game.round.tricks_won
        encoded_wins[:len(wins)] = wins
        return encoded_wins[self.relative_to_absolute]

    # ====================================================================================================
    # GAME PROGRESS ENCODING
    # ====================================================================================================

    def _encode_fraction_rounds(self):
        encoded_fraction_rounds    = np.zeros(1, dtype=np.float32)
        encoded_fraction_rounds[0] = self.game.round_number / self.game.total_rounds
        return encoded_fraction_rounds

    def _encode_fraction_tricks(self):
        encoded_fraction_tricks    = np.zeros(1, dtype=np.float32)
        encoded_fraction_tricks[0] = (self.game.round_number - self.game.round.trick_number) / self.game.round_number
        return encoded_fraction_tricks

    def _encode_total_n_rounds(self):
        encoded_total_rounds    = np.zeros(1, dtype=np.float32)
        encoded_total_rounds[0] = self.game.total_rounds
        return encoded_total_rounds

    def _encode_rounds_left(self):
        encoded_rounds_left    = np.zeros(1, dtype=np.float32)
        encoded_rounds_left[0] = self.game.rounds_left / self.game.total_rounds
        return encoded_rounds_left
        