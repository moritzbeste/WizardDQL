from src.game import Game
from src.models_utility.agent_management import AgentManagement
import time
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()
    print(game.round.player_hands)
    game.scores[:] = [10, 20, 0]

    agent = AgentManagement(game, 0)
    print(agent.bidding_agent.input_output_length())
    