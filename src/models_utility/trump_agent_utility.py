import numpy as np
import torch

class TrumpAgentUtility:

    # ====================================================================================================
    # INITIALIZATION
    # ====================================================================================================

    def __init__(self, utility):
        self.utility = utility