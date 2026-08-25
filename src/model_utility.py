import numpy as np
from scipy.special import logsumexp
import torch
from pathlib import Path
import tomllib


class AgentUtility:

    # ====================================================================================================
    # CLASS CONSTANTS
    # ====================================================================================================

    CONFIG_PATH = Path(__file__).parent.parent / 'config.toml'

    # ====================================================================================================
    # INITIALIZATION AND SETUP
    # ====================================================================================================

    def __init__(self, game, player_index):
        self._set_a_config()
        self.game = game
        self.player_index = player_index
        relative_indices = [(self.player_index + i) % self.game.n_players for i in range(self.game.n_players)]
        self.absolute_to_relative = np.empty(self.game.n_players, dtype=int)
        for relative, absolute in enumerate(relative_indices):
            self.absolute_to_relative[absolute] = relative
        self.previous_lead = 0.0
        self.target_lead = 0.0

    def _set_a_config(self):
        with self.CONFIG_PATH.open('rb') as file:
            self.a_conf = tomllib.load(file)['agent']


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

        print(state.shape)

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
        return encoded_scores[self.absolute_to_relative]
    
    def _encode_leads(self):
        leads = np.array([self._compute_lead(p) for p in range(self.game.n_players)], dtype=np.float32)
        return leads[self.absolute_to_relative]

    def _encode_bids(self):
        max_players = self.game.max_players
        encoded_bids = np.zeros((max_players, 2), dtype=np.float32)
        bids = self.game.round.get_bids()
        for player, bid in enumerate(bids):
            if bid is None:
                encoded_bids[1, player] = 1
            else:
                encoded_bids[0, player] = bid
        return encoded_bids[self.absolute_to_relative]

    def _encode_tricks_won(self):
        max_players = self.game.max_players
        encoded_wins = np.zeros(max_players, dtype=np.float32)
        wins = self.game.round.tricks_won
        encoded_wins[:len(wins)] = wins
        return encoded_wins[self.absolute_to_relative]

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

    # ====================================================================================================
    # INTERNAL REWARD COMPUTATION
    # ====================================================================================================

    def _growth_target(self):
        n_players = self.game.n_players
        base_lead_growth_per_trick = self.a_conf['base_lead_growth_per_trick']
        alpha = self.a_conf['alpha']
        game_round = self.game.round_number - 1
        return base_lead_growth_per_trick * game_round * alpha / n_players
    
    def _asymmetric_reward(self, error, temperature, penalty):
        x = error / temperature
        if x >= 0:
            return x + 0.75 * x ** 2
        else:
            return x - penalty * x ** 2

    def _r_dense(self, lead):
        n_players = self.game.n_players
        # local reward
        target_lead_growth = self._growth_target()
        lead_growth = lead - self.previous_lead
        growth_error = lead_growth - target_lead_growth
        local_reward = self._asymmetric_reward(growth_error, self.a_conf['growth_temperature'], self.a_conf['growth_penalty'])
        # global reward
        self.target_lead += target_lead_growth
        target_error = lead - self.target_lead
        global_reward = self._asymmetric_reward(target_error, self.a_conf['position_temperature'], self.a_conf['position_penalty'])
        progress = (self.game.round_number - 1) / self.game.total_rounds
        position_weight = self.a_conf['position_weight'] * (0.25 + 0.75 * progress**2)
        return self.a_conf['growth_weight'] * local_reward + position_weight * global_reward

    def _compute_lead(self, player_index):
        scores = self.game.scores
        differences = scores[player_index] - np.delete(scores, player_index)
        beta = self.a_conf['softmin_sharpness']
        # softmin
        lead = -(logsumexp(-beta * differences) - np.log(len(differences))) / beta
        return lead

    # ====================================================================================================
    # REWARD
    # ====================================================================================================

    def reward(self):
        # compute lead
        lead = self._compute_lead(player_index=self.player_index)
        # dense reward
        reward_dense = self._r_dense(lead)
        # terminal reward
        W = self.a_conf['win_reward']
        winner = self.game.get_winner()
        if winner is None:
            reward_terminal = 0.0
            self.previous_lead = lead
        else:
            if winner == self.player_index:
                reward_terminal = W
            else:
                reward_terminal = -W
            # game is over
            # reset lead history
            self.target_lead = 0.0
            self.previous_lead = 0.0
        return reward_dense + reward_terminal