import numpy as np
from pathlib import Path
import tomllib
from dataclasses import dataclass

from src.utility import get_config

from src.models_utility.tricking_agent_utility import TrickingAgentUtility
from src.models_utility.bidding_agent_utility  import BiddingAgentUtility
from src.models_utility.trump_agent_utility    import TrumpAgentUtility

from src.game import Game
from sec.deck import Deck

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

        self.a_conf = get_config('agent')
        self.g_conf = get_config('game')

        self.deck = Deck(n_players=n_players, seed=self.g_conf['seed_deck'])

        self.games = [
            Game(n_players=self.n_players, deck=self.deck)
            for _ in range(self.n_games)]

        self.players = [
            PlayerAgents(
                trump=TrumpAgentUtility(self, player_index=i),
                bidding=BiddingAgentUtility(self, player_index=i),
                tricking=TrickingAgentUtility(self, player_index=i),
            ) for i in range(self.n_players)]
    
    def _get_agent(self, player_index, agent):
        return getattr(self.players[player_index], agent)
    
    # ====================================================================================================
    # GAME MANAGEMENT
    # ====================================================================================================

    def _retrieve_moves(self, game_index, agent_type):
        game = self.games[game_index]
        moves = np.zeros(self.n_players)
        for player in range(self.n_players):
            agent = self._get_agent(player, agent_type)
            moves[player] = agent.get_move(game, player, self.epsilon)
        return moves
