import random

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent
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


def test_adaptive_weighted_blocking_agent_prioritizes_blocking_high_ranked_opponents() -> None:
    agent = AdaptiveWeightedBlockingAgent(
        early_block_weight=1.0,
        late_block_weight=1.0,
        rank_weights=(2.0, 0.5, 0.1),
    )
    state = GameState(
        current_player=0,
        remaining_pieces={
            0: frozenset({"I1"}),
            1: frozenset({"I1"}),
            2: frozenset({"I1"}),
            3: frozenset(),
        },
    )
    state.board.place({(5, 5)}, player=0)
    state.board.place({(7, 7), (8, 8)}, player=1)
    state.board.place({(7, 3)}, player=2)

    block_leader = Move(player=0, piece_name="I1", origin=(6, 6), cells=PIECES["I1"])
    block_trailer = Move(player=0, piece_name="I1", origin=(6, 4), cells=PIECES["I1"])

    assert agent.select_move(state, [block_trailer, block_leader]) == block_leader


def test_adaptive_weighted_blocking_agent_increases_blocking_pressure_later_in_game() -> None:
    early_state = GameState(
        current_player=0,
        remaining_pieces={
            0: frozenset({"I1", "I2"}),
            1: frozenset({"I1"}),
            2: frozenset(),
            3: frozenset(),
        },
    )
    early_state.board.place({(5, 5)}, player=0)
    early_state.board.place({(7, 7)}, player=1)

    late_state = GameState(
        current_player=0,
        remaining_pieces=dict(early_state.remaining_pieces),
        board=early_state.board.clone(),
    )
    filler_cells = {
        (row, col)
        for row in range(10)
        for col in range(10)
        if (row, col) not in {(5, 5), (7, 7), (3, 4), (4, 4), (6, 6)}
    }
    late_state.board.place(filler_cells, player=3)

    more_tiles = Move(
        player=0,
        piece_name="I2",
        origin=(3, 4),
        cells=frozenset({(0, 0), (1, 0)}),
    )
    more_blocking = Move(player=0, piece_name="I1", origin=(6, 6), cells=PIECES["I1"])

    agent = AdaptiveWeightedBlockingAgent(
        early_block_weight=0.1,
        late_block_weight=2.0,
        rank_weights=(1.0, 0.6, 0.3),
    )

    assert agent.select_move(early_state, [more_tiles, more_blocking]) == more_tiles
    assert agent.select_move(late_state, [more_tiles, more_blocking]) == more_blocking
