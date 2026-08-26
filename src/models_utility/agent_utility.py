import numpy as np
from pathlib import Path
import tomllib

from src.models_utility.tricking_agent_utility import TrickingAgentUtility
from src.models_utility.bidding_agent_utility  import BiddingAgentUtility

class AgentUtility:

    # ====================================================================================================
    # CLASS CONSTANTS
    # ====================================================================================================

    CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.toml'

    # ====================================================================================================
    # INITIALIZATION AND SETUP
    # ====================================================================================================

    def __init__(self, game, player_index):
        self.game = game
        self.player_index = player_index
        self._set_index_conversion()
        self._set_a_config()
        self._set_g_config()

        self.tricking_agent = TrickingAgentUtility(self)
        self.bidding_agent  = BiddingAgentUtility(self)
        
    def _set_index_conversion(self):
        self.relative_to_absolute = np.array([(self.player_index + i) % self.game.n_players for i in range(self.game.n_players)])
        self.absolute_to_relative = np.empty(self.game.n_players, dtype=int)
        for relative, absolute in enumerate(self.relative_to_absolute):
            self.absolute_to_relative[absolute] = relative

    def _set_a_config(self):
        with self.CONFIG_PATH.open('rb') as file:
            self.a_conf = tomllib.load(file)['agent']

    def _set_g_config(self):
        with self.CONFIG_PATH.open('rb') as file:
            self.g_conf = tomllib.load(file)['game']
