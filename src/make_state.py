import numpy as np
import torch

class AgentState:
    def __init__(self, game, player_index):
        self.game = game
        self.player_index = player_index
        self.relative_indices = [(self.player_index + i) % self.game.n_players for i in range(self.game.n_players)]

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

        encoded_hand         = self._encode_hand()
        encoded_player_mask  = self._encode_player_mask()
        encoded_priority     = self._encode_priority()
        encoded_played       = self._encode_played_cards()
        encoded_trump        = self._encode_trump()
        encoded_leading_suit = self._encode_leading_suit()
        encoded_scores       = self._encode_score()
        encoded_round_number = self._encode_round_number()
        encoded_n_tricks     = self._encode_number_tricks()
        encoded_bids         = self._encode_bids()
        encoded_wins         = self._encode_tricks_won()

        state = np.concatenate([
            encoded_hand.flatten(),
            encoded_player_mask,
            encoded_priority,
            encoded_played.flatten(),
            encoded_trump,
            encoded_leading_suit,
            encoded_scores,
            encoded_round_number,
            encoded_n_tricks,
            encoded_bids.flatten(),
            encoded_wins,
        ])

        return torch.from_numpy(state)


    def _encode_hand(self):
        current_hand = self.game.get_hand(self.player_index)
        encoded_hand = np.zeros((3, len(self.game.deck.deck)), dtype=np.float32)
        encoded_hand[0, current_hand ==  1] = 1
        encoded_hand[1, current_hand == -1] = 1
        encoded_hand[2, current_hand ==  0] = 1
        return encoded_hand

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
        relative_priority = self.relative_indices[priority_index]
        priority_encoding[relative_priority] = 1
        return priority_encoding

    def _encode_played_cards(self):
        max_players = self.game.max_players
        encoded_played = np.zeros((max_players, len(self.game.deck.deck)), dtype=np.float32)
        for player, card in self.game.round.played_cards:
            relative_player = self.relative_indices[player]
            encoded_played[relative_player, card] = 1
        return encoded_played
    
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
    
    def _encode_score(self):
        scores = np.asarray(self.game.scores, dtype=np.float32)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            encoded_scores = np.zeros_like(scores)
        else:
            encoded_scores = (scores - min_score) / (max_score - min_score)
        return encoded_scores[self.relative_indices]
        
    def _encode_round_number(self):
        encoded_round_number    = np.zeros(1, dtype=np.float32)
        encoded_round_number[0] = self.game.round_number
        return encoded_round_number

    def _encode_number_tricks(self):
        encoded_n_tricks    = np.zeros(1, dtype=np.float32)
        encoded_n_tricks[0] = self.game.round_number - self.game.round.trick_number
        return encoded_n_tricks
    
    def _encode_bids(self):
        max_players = self.game.max_players
        encoded_bids = np.zeros((max_players, 2), dtype=np.float32)
        bids = self.game.round.get_bids()
        if bids is not None:
            encoded_bids[0, :self.game.n_players] = bids
        else:
            encoded_bids[1, :self.game.n_players] = 1
        return encoded_bids

    def _encode_tricks_won(self):
        max_players = self.game.max_players
        encoded_wins = np.zeros(max_players, dtype=np.float32)
        wins = self.game.round.tricks_won
        encoded_wins[:len(wins)] = wins
        encoded_wins[self.relative_indices]
        return encoded_wins
