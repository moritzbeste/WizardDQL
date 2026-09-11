import numpy as np
from pathlib import Path
import tomllib
from dataclasses import dataclass

from src.utility import get_config

from src.models_utility.tricking_agent_utility import TrickingAgentUtility
from src.models_utility.bidding_agent_utility  import BiddingAgentUtility
from src.models_utility.trump_agent_utility    import TrumpAgentUtility

from src.game import Game, RoundState
from src.deck import Deck

@dataclass
class PlayerAgents:
    trump: TrumpAgentUtility
    bidding: BiddingAgentUtility
    tricking: TrickingAgentUtility


class AgentManagement:

    # ====================================================================================================
    # CLASS CONSTANTS
    # ====================================================================================================

    CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.toml'

    # ====================================================================================================
    # INITIALIZATION AND SETUP
    # ====================================================================================================

    def __init__(self, n_players=3, n_games=1):
        self.games_info = {}
        self.n_players  = n_players
        self.n_games    = n_games
        self.epsilon    = 1.0

        self.a_conf      = get_config('agent')
        self.g_conf      = get_config('game')
        self.max_players = self.g_conf['max_players']

        self.deck = Deck(n_players=n_players, seed=self.g_conf['seed_deck'])
        
        self.games = np.array(
            [Game(n_players=self.n_players, deck=self.deck)
            for _ in range(self.n_games)], dtype=object)

        self.agents = PlayerAgents(
            trump=TrumpAgentUtility(self),
            bidding=BiddingAgentUtility(self), 
            tricking=TrickingAgentUtility(self))
        self.relative_to_absolute = []
        self.absolute_to_relative = []
        for i in range(n_players):
            rta, atr = self._get_index_conversion(player_index=i)
            self.relative_to_absolute.append(rta)
            self.absolute_to_relative.append(atr)
    
    def _get_index_conversion(self, player_index):
        relative_to_absolute = np.full(self.max_players, -1, dtype=int)
        for relative in range(self.n_players):
            relative_to_absolute[relative] = (player_index + relative) % self.n_players
        absolute_to_relative = np.empty(self.n_players, dtype=int)
        for relative in range(self.n_players):
            absolute = relative_to_absolute[relative]
            absolute_to_relative[absolute] = relative
        return relative_to_absolute, absolute_to_relative
    
    # ====================================================================================================
    # GAME MANAGEMENT
    # ====================================================================================================

    def _get_priorities(self, indices=None):
        if indices is None:
            return np.array([game.round.priority for game in self.games], dtype=np.int16)
        else:
            return np.array([game.round.priority for game in self.games[indices]], dtype=np.int16)

    def _get_games_requiring_trump(self):
        states  = np.array([game.round.state for game in self.games], dtype=np.int16)
        indices = np.flatnonzero(states == RoundState.CHOOSE_TRUMP)
        return indices
    
    def _choose_trumps(self):
        game_indices = self._get_games_requiring_trump()
        priorities = self._get_priorities(indices=game_indices)
        chosen_trumps = self.agents.trump.get_moves(priorities=priorities, epsilon=self.epsilon, game_indices=game_indices)
        for game_index, player, chosen_trump in zip(game_indices, priorities, chosen_trumps):
            game = self.games[game_index]
            game.round.pick_trump_color(player_index=player, suit=chosen_trump)
        
    def _choose_bids(self):
        priorities = self._get_priorities() # indices = None to get priorities for all games
        chosen_bids = self.agents.bidding.get_moves(priorities=priorities, epsilon=self.epsilon)
        for game_index, (player, chosen_bid) in enumerate(zip(priorities, chosen_bids)):
            game = self.games[game_index]
            game.round.bid(player_index=player, bid=chosen_bid)
    
    def _orchestrate_all_bids(self):
        for _ in range(self.n_players):
            self._choose_bids()
    
    def _choose_tricks(self):
        priorities = self._get_priorities()
        chosen_tricks = self.agents.tricking.get_moves(priorities=priorities, epsilon=self.epsilon)
        for game_index, (player, chosen_bid) in enumerate(zip(priorities, chosen_bids)):
            game = self.games[game_index]
            game.round.play_card(player_index=player, bid=chosen_bid)
        
    def _orchestrate_all_tricks(self):
        # all games are syncronized and have the same number of tricks corresponding to the round
        n_tricks = self.games[0].round_number
        for _ in range(n_tricks):
            for _ in range(self.n_players):
                self._choose_tricks()
    
    def _orchestrate_full_rounds(self):
        # choose trumps for games that require a trump
        self._choose_trumps() # all games are now synced
        self._orchestrate_all_bids()
        self._orchestrate_all_tricks()
    
    def orchestrate_full_games(self):
        n_rounds = self.games[0].total_rounds
        for _ in range(n_rounds):
            self._orchestrate_full_rounds()
