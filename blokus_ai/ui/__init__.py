"""UI helpers for displaying Blokus state."""

from .ascii_renderer import EMPTY_SYMBOL, PLAYER_SYMBOLS, render_board


def run_move_replay_viewer(*args, **kwargs):
    from .pygame_viewer import run_move_replay_viewer as _run_move_replay_viewer

    return _run_move_replay_viewer(*args, **kwargs)


def run_agent_match_viewer(*args, **kwargs):
    from .pygame_viewer import run_agent_match_viewer as _run_agent_match_viewer

    return _run_agent_match_viewer(*args, **kwargs)


def run_random_self_play_viewer(*args, **kwargs):
    from .pygame_viewer import run_random_self_play_viewer as _run_random_self_play_viewer

    return _run_random_self_play_viewer(*args, **kwargs)

__all__ = [
    "EMPTY_SYMBOL",
    "PLAYER_SYMBOLS",
    "render_board",
    "run_agent_match_viewer",
    "run_move_replay_viewer",
    "run_random_self_play_viewer",
]
