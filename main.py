from src.game import Game
from src.models_utility.agent_utility import AgentUtility
import time
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()
    print(game.round.state)

    agent = AgentUtility(game, 2)
    
    game.scores[:] = [100,200,350]
    game.round.bids = [2, 3, 4]
    