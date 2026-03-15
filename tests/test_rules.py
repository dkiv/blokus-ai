import pytest

from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.pieces import PIECES
from blokus_ai.core.rules import is_legal_move


def test_first_move_is_legal_when_it_covers_the_correct_corner() -> None:
    state = GameState.new_game()
    move = Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"])

    assert is_legal_move(state, move)


def test_first_move_is_illegal_when_it_misses_the_correct_corner() -> None:
    state = GameState.new_game()
    move = Move(player=0, piece_name="I1", origin=(1, 1), cells=PIECES["I1"])

    assert not is_legal_move(state, move)


def test_overlap_is_rejected() -> None:
    state = GameState.new_game()
    state = state.apply_move(Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]))
    move = Move(player=1, piece_name="I1", origin=(0, 0), cells=PIECES["I1"])

    assert not is_legal_move(state, move)


def test_out_of_bounds_is_rejected() -> None:
    state = GameState.new_game()
    move = Move(player=0, piece_name="I2", origin=(-1, 0), cells=PIECES["I2"])

    assert not is_legal_move(state, move)


def test_edge_contact_with_same_player_is_illegal() -> None:
    state = GameState.new_game()
    state = state.apply_move(Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]))
    state = state.apply_move(Move(player=1, piece_name="I1", origin=(0, 19), cells=PIECES["I1"]))
    move = Move(player=0, piece_name="I2", origin=(1, 0), cells=PIECES["I2"])

    assert not is_legal_move(state, move)


def test_corner_contact_with_same_player_is_legal() -> None:
    state = GameState.new_game()
    state = state.apply_move(Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]))
    state = state.apply_move(Move(player=1, piece_name="I1", origin=(0, 19), cells=PIECES["I1"]))
    move = Move(player=0, piece_name="I2", origin=(1, 1), cells=PIECES["I2"])

    assert is_legal_move(state, move)


def test_non_first_move_without_corner_contact_is_illegal() -> None:
    state = GameState.new_game()
    state = state.apply_move(Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]))
    state = state.apply_move(Move(player=1, piece_name="I1", origin=(0, 19), cells=PIECES["I1"]))
    move = Move(player=0, piece_name="I2", origin=(5, 5), cells=PIECES["I2"])

    assert not is_legal_move(state, move)


def test_apply_move_raises_for_illegal_move() -> None:
    state = GameState.new_game()

    with pytest.raises(ValueError, match="starting corner"):
        state.apply_move(Move(player=0, piece_name="I1", origin=(1, 1), cells=PIECES["I1"]))
