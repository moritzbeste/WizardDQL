import numpy as np
import torch

class BiddingAgentUtility:

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, utility):
        self.utility = utility

    @property
    def game(self):
        return self.utility.game

    @property
    def player_index(self):
        return self.utility.player_index

    @property
    def absolute_to_relative(self):
        return self.utility.absolute_to_relative
    
    @property
    def relative_to_absolute(self):
        return self.utility.relative_to_absolute

    # ====================================================================================================
    # REWARD
    # ====================================================================================================