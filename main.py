from src.game import Game
from src.models_utility.agent_management import AgentManagement
import time
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()

    agent = AgentManagement(n_games=100)
    agent.agents.bidding._generate_state_rerpesentation(0, 0)
    