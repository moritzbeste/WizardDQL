from src.game import Game
from src.model_utility import AgentUtility
import time
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()

    agent = AgentUtility(game, 2)
    game.scores[:] = [0, 0, 0]
    agent.generate_state_rerpesentation()
    print(agent.reward())
