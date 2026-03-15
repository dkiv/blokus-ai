from blokus_ai.experiments.self_play import play_random_game
from blokus_ai.ui.ascii_renderer import render_board


def test_random_self_play_is_reproducible_with_seed() -> None:
    result = play_random_game(seed=7, max_turns=4, print_boards=False)

    assert result.turn_count == 4
    assert len(result.moves) == 4
    assert result.passes == 0
    assert result.final_state.current_player == 0


def test_random_self_play_populates_board() -> None:
    result = play_random_game(seed=3, max_turns=2, print_boards=False)

    rendered = render_board(result.final_state)

    assert "A" in rendered
    assert "B" in rendered
