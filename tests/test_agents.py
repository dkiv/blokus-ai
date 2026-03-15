import random

from blokus_ai.agents.blocking_agent import BlockingAgent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent
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


def test_largest_first_agent_prefers_move_with_most_cells() -> None:
    agent = LargestFirstAgent()
    state = GameState.new_game()
    legal_moves = [
        Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]),
        Move(player=0, piece_name="I2", origin=(0, 0), cells=PIECES["I2"]),
    ]

    assert agent.select_move(state, legal_moves) == legal_moves[1]


def test_blocking_agent_prefers_move_that_reduces_opponent_frontier() -> None:
    agent = BlockingAgent()
    state = GameState(
        current_player=0,
        remaining_pieces={
            0: frozenset({"I1"}),
            1: frozenset({"I1"}),
            2: frozenset(),
            3: frozenset(),
        },
    )
    state.board.place({(5, 5)}, player=0)
    state.board.place({(7, 7)}, player=1)

    block_move = Move(player=0, piece_name="I1", origin=(6, 6), cells=PIECES["I1"])
    non_block_move = Move(player=0, piece_name="I1", origin=(4, 4), cells=PIECES["I1"])

    assert agent.select_move(state, [non_block_move, block_move]) == block_move


def test_weighted_blocking_agent_uses_weight_to_trade_off_tiles_and_blocking() -> None:
    state = GameState(
        current_player=0,
        remaining_pieces={
            0: frozenset({"I1", "I2"}),
            1: frozenset({"I1"}),
            2: frozenset(),
            3: frozenset(),
        },
    )
    state.board.place({(5, 5)}, player=0)
    state.board.place({(7, 7)}, player=1)

    more_tiles = Move(
        player=0,
        piece_name="I2",
        origin=(3, 4),
        cells=frozenset({(0, 0), (1, 0)}),
    )
    more_blocking = Move(player=0, piece_name="I1", origin=(6, 6), cells=PIECES["I1"])

    low_weight_agent = WeightedBlockingAgent(blocked_corner_weight=0.2)
    high_weight_agent = WeightedBlockingAgent(blocked_corner_weight=1.1)

    assert low_weight_agent.select_move(state, [more_tiles, more_blocking]) == more_tiles
    assert high_weight_agent.select_move(state, [more_tiles, more_blocking]) == more_blocking
