import random

from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.pieces import PIECES


def test_random_agent_selects_one_of_the_legal_moves() -> None:
    agent = RandomAgent(rng=random.Random(1))
    state = GameState.new_game()
    legal_moves = [
        Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]),
        Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]),
    ]

    assert agent.select_move(state, legal_moves) in legal_moves


def test_random_agent_returns_none_when_no_legal_moves_exist() -> None:
    agent = RandomAgent(rng=random.Random(1))
    state = GameState.new_game()

    assert agent.select_move(state, []) is None
