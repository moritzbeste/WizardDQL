import numpy as np
from pathlib import Path
import tomllib

from src.utility import get_config

from src.models_utility.tricking_agent_utility import TrickingAgentUtility
from src.models_utility.bidding_agent_utility  import BiddingAgentUtility
from src.models_utility.trump_agent_utility    import TrumpAgentUtility

class AgentManagement:

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
        self.a_conf = get_config('agent')
        self.g_conf = get_config('game')

        self.tricking_agent = TrickingAgentUtility(self)
        self.bidding_agent  = BiddingAgentUtility(self)
        self.trump_agent    = TrumpAgentUtility(self)
        
    def _set_index_conversion(self):
        self.relative_to_absolute = np.array([(self.player_index + i) % self.game.n_players for i in range(self.game.n_players)])
        self.absolute_to_relative = np.empty(self.game.n_players, dtype=int)
        for relative, absolute in enumerate(self.relative_to_absolute):
            self.absolute_to_relative[absolute] = relative
