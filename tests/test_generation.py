from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves
from blokus_ai.core.pieces import PIECES


def test_generate_legal_moves_for_initial_i1_piece_returns_single_corner_move() -> None:
    state = GameState(
        current_player=0,
        remaining_pieces={0: frozenset({"I1"}), 1: frozenset(), 2: frozenset(), 3: frozenset()},
    )

    moves = generate_legal_moves(state)

    assert moves == [Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"])]


def test_generate_legal_moves_for_initial_i2_piece_returns_both_corner_orientations() -> None:
    state = GameState(
        current_player=0,
        remaining_pieces={0: frozenset({"I2"}), 1: frozenset(), 2: frozenset(), 3: frozenset()},
    )

    moves = generate_legal_moves(state)

    assert set(moves) == {
        Move(player=0, piece_name="I2", origin=(0, 0), cells=frozenset({(0, 0), (1, 0)})),
        Move(player=0, piece_name="I2", origin=(0, 0), cells=frozenset({(0, 0), (0, 1)})),
    }


def test_generate_legal_moves_for_second_turn_uses_corner_contact_rule() -> None:
    state = GameState(
        current_player=0,
        remaining_pieces={
            0: frozenset({"I2"}),
            1: frozenset({"I2"}),
            2: frozenset(),
            3: frozenset(),
        },
    )
    state.board.place(PIECES["I1"], player=0)
    state.board.place({(0, 19)}, player=1)

    moves = generate_legal_moves(state)

    assert set(moves) == {
        Move(player=0, piece_name="I2", origin=(1, 1), cells=frozenset({(0, 0), (1, 0)})),
        Move(player=0, piece_name="I2", origin=(1, 1), cells=frozenset({(0, 0), (0, 1)})),
    }


def test_generate_legal_moves_respects_explicit_player_argument() -> None:
    state = GameState(
        current_player=0,
        remaining_pieces={
            0: frozenset(),
            1: frozenset({"I1"}),
            2: frozenset(),
            3: frozenset(),
        },
    )

    moves = generate_legal_moves(state, player=1)

    assert moves == [Move(player=1, piece_name="I1", origin=(0, 19), cells=PIECES["I1"])]
