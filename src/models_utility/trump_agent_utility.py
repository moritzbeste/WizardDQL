import numpy as np
import torch

from src.utility import get_config

from src.models_utility.agent_utility import AgentUtility

class TrumpAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager)
        self.d_conf = get_config('deck')

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self):
        return np.ones(self.d_conf['n_suits'])
