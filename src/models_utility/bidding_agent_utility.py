import numpy as np
import torch

from src.models_utility.agent_utility import AgentUtility

class BiddingAgentUtility(AgentUtility):

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, manager):
        super().__init__(manager)

    # ====================================================================================================
    # MOVE MASK
    # ====================================================================================================

    def _get_legal_move_mask(self):
        mask = np.zeros(self.game.total_rounds, dtype=np.int8)
        mask[:self.game.round_number] = 1
        return mask
