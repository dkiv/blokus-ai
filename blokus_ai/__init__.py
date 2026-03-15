"""Top-level exports for the Blokus engine package."""

from .agents import Agent, RandomAgent
from .core import (
    ALL_PIECES,
    BOARD_SIZE,
    Board,
    Coordinate,
    GameState,
    Move,
    PIECES,
    PIECE_TRANSFORMS,
    STARTING_CORNERS,
    Shape,
    generate_legal_moves,
    is_legal_move,
    validate_move,
)
from .experiments import (
    SelfPlayResult,
    SelfPlaySession,
    TournamentResult,
    play_game,
    play_random_game,
    run_tournament,
)
from .ui import render_board


def run_move_replay_viewer(*args, **kwargs):
    from .ui import run_move_replay_viewer as _run_move_replay_viewer

    return _run_move_replay_viewer(*args, **kwargs)


def run_random_self_play_viewer(*args, **kwargs):
    from .ui import run_random_self_play_viewer as _run_random_self_play_viewer

    return _run_random_self_play_viewer(*args, **kwargs)

__all__ = [
    "ALL_PIECES",
    "Agent",
    "BOARD_SIZE",
    "Board",
    "Coordinate",
    "GameState",
    "Move",
    "PIECE_TRANSFORMS",
    "PIECES",
    "RandomAgent",
    "STARTING_CORNERS",
    "SelfPlayResult",
    "SelfPlaySession",
    "Shape",
    "TournamentResult",
    "generate_legal_moves",
    "is_legal_move",
    "play_game",
    "play_random_game",
    "render_board",
    "run_tournament",
    "run_move_replay_viewer",
    "run_random_self_play_viewer",
    "validate_move",
]
