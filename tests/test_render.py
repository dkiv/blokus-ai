from blokus_ai.core.game_state import GameState
from blokus_ai.ui.ascii_renderer import render_board


def test_render_board_shows_empty_board_with_coordinates() -> None:
    state = GameState.new_game()

    rendered = render_board(state)
    lines = rendered.splitlines()

    assert lines[0].startswith("   00 01 02")
    assert lines[1].startswith("00 . . .")
    assert len(lines) == 21


def test_render_board_shows_player_symbols() -> None:
    state = GameState.new_game()
    state.board.place({(0, 0), (1, 1)}, player=0)
    state.board.place({(0, 19)}, player=1)

    rendered = render_board(state)
    lines = rendered.splitlines()

    assert lines[1].startswith("00 A")
    assert lines[1].endswith("B")
    assert "01 . A" in lines[2]
