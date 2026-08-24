from src.game import Game
from src.make_state import AgentState
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()

    agent = AgentState(game, 0)
    agent.generate_state_rerpesentation()
    