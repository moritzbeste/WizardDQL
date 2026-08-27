from src.game import Game
from src.models_utility.agent_utility import AgentUtility
import time
import numpy as np

if __name__ == "__main__":
    game = Game(n_players=3)
    game.setup_round()
    print(game.round.player_hands)

    agent = AgentUtility(game, 1)
    game.round.bid(1, 1)
    game.round.bid(2, 1)
    game.round.bid(0, 1)
    print(game.round.state)
    game.round.trick.leading_suit = 3
    game.round.trump = 0
    print(game.round.trump)
    
    print(agent.tricking_agent._get_legal_move_mask())
    print(agent.tricking_agent.game.round.get_vector_reprs(agent.tricking_agent.game.round.player_hands[1]))
    print(agent.tricking_agent.game.round.get_vector_reprs(agent.tricking_agent._get_legal_move_mask()))
    