from abc import ABC, abstractmethod
import numpy as np

# Abstract method for agent utility classes
class AgentUtility(ABC):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        self.manager = manager

    def _game(self, index):
        return self.manager.games[index]
    
    def _relative_to_absolute(self, player_index):
        return self.manager.relative_to_absolute[player_index]
    
    def _absolute_to_relative(self, player_index):
        return self.manager.absolute_to_relative[player_index]

    @property
    def a_conf(self):
        return self.manager.a_conf

    @property
    def g_conf(self):
        return self.manager.g_conf
    
    # ====================================================================================================
    # REWARD
    # ====================================================================================================

    def _current_round_scores(self, game_index, player_index):
        player_tricks = self._game(game_index).round.tricks_won[self._relative_to_absolute(player_index=player_index)]
        correct = self.player_bids == player_tricks
        return np.where(
            correct,
            player_tricks * self.g_conf['points_for_successful_trick'] + self.g_conf['points_for_guessing_correctly'],
            self.g_conf['points_for_unsuccessfuly_trick'] * np.abs(self.player_bids - player_tricks)
        )

    def reward(self, game_index, player_index):
        scores = self._game(game_index).scores[self._relative_to_absolute(player_index=player_index)]
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
    def _generate_state_rerpesentation(self, game_index, player_index):
        pass

    def input_output_length(self):
        state = self._generate_state_rerpesentation(game_index=0)
        mask  = self._get_legal_move_mask(game_index=0) # input, output lengths are constant
        return len(state), len(mask)

    # ====================================================================================================
    # CARD AND TRICK ENCODING
    # ====================================================================================================

    def _encode_hand(self, game_index, player_index):
        current_hand = self._game(game_index).get_hand(player_index)
        encoded_hand = np.zeros((2, len(self._game(game_index).deck.deck)), dtype=np.float32)
        encoded_hand[0, current_hand ==  1] = 1
        encoded_hand[1, current_hand ==  0] = 1
        return encoded_hand
    
    def _encode_trump(self, game_index):
        trump = self._game(game_index).round.trump
        encoded_trump = np.zeros(self._game(game_index).deck.d_conf["n_suits"] + 1, dtype=np.float32) # + 1 for wizard suit 
        return encoded_trump
    
    # ====================================================================================================
    # PLAYER ENCODING
    # ====================================================================================================

    def _encode_bids(self, game_index, player_index):
        max_players = self._game(game_index).max_players
        encoded_bids = np.zeros((max_players, 2), dtype=np.float32)
        bids = self._game(game_index).round.bids
        for player, bid in enumerate(bids):
            if bid is None:
                encoded_bids[player, 1] = 1
            else:
                encoded_bids[player, 0] = bid
        return encoded_bids[self._relative_to_absolute(player_index=player_index)]

    def _encode_player_mask(self, game_index):
        max_players = self._game(game_index).max_players
        n_players   = self._game(game_index).n_players
        player_mask = np.zeros(max_players, dtype=np.float32)
        player_mask[:n_players] = 1
        return player_mask

    def _encode_priority(self, game_index, player_index):
        max_players = self._game(game_index).max_players
        priority_encoding = np.zeros(max_players, dtype=np.float32)
        priority_index = self._game(game_index).round.priority
        relative_priority = self._absolute_to_relative(player_index=player_index)[priority_index]
        priority_encoding[relative_priority] = 1
        return priority_encoding

    def _encode_score(self, game_index, player_index):
        scores = np.asarray(self._game(game_index).scores, dtype=np.float32)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            encoded_scores = np.zeros_like(scores)
        else:
            encoded_scores = (scores - min_score) / (max_score - min_score)
        return encoded_scores[self._relative_to_absolute(player_index=player_index)]
    
    # ====================================================================================================
    # GAME PROGRESS ENCODING
    # ====================================================================================================

    def _encode_fraction_rounds(self, game_index):
        encoded_fraction_rounds    = np.zeros(1, dtype=np.float32)
        encoded_fraction_rounds[0] = self._game(game_index).round_number / self._game(game_index).total_rounds
        return encoded_fraction_rounds

    def _encode_total_rounds(self, game_index):
        encoded_total_rounds    = np.zeros(1, dtype=np.float32)
        encoded_total_rounds[0] = self._game(game_index).total_rounds
        return encoded_total_rounds

    def _encode_rounds_left(self, game_index):
        encoded_rounds_left    = np.zeros(1, dtype=np.float32)
        encoded_rounds_left[0] = (self._game(game_index).total_rounds - self._game(game_index).round_number) / self._game(game_index).total_rounds
        return encoded_rounds_left
    
    # ====================================================================================================
    # GET MOVE
    # ====================================================================================================

    @abstractmethod
    def _get_legal_move_mask(self, game_index, player_index):
        pass

    def get_moves(self, priorities, epsilon, game_indices=None):
        if game_indices is None:
            game_indices = range(len(priorities))
        pairs  = list(zip(game_indices, priorities))
        states = np.stack([self._generate_state_rerpesentation(g, p) for g, p in pairs])
        masks  = np.stack([self._get_legal_move_mask(g, p) for g, p in pairs])
        states = torch.from_numpy(states).float()
        with torch.no_grad():
            q_values = self.model(states).numpy()
        actions = []
        for q, mask in zip(q_values, masks):
            legal_moves = np.where(mask == 1)[0]
            if np.random.random() < epsilon:
                action = np.random.choice(legal_moves)
            else:
                action = np.argmax(np.where(mask, q, -np.inf))
            actions.append(action)
        return actions
