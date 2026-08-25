from src.game import Game
from src.model_utility import AgentUtility
import time
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()
    print(game.round.state)

    agent = AgentUtility(game, 2)
    
    game.scores[:] = [4, 2, 1]
    agent.generate_state_rerpesentation()
    print(agent.reward())
