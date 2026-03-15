"""Top-level exports for the Blokus engine package."""

from importlib import import_module

from .agents import (
    AdaptiveWeightedBlockingAgent,
    Agent,
    BlockingAgent,
    LargestFirstAgent,
    RandomAgent,
    WeightedBlockingAgent,
)
from .core import (
    ALL_PIECES,
    BOARD_SIZE,
    Board,
    Coordinate,
    frontier_targets,
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
from .ui import render_board


def run_move_replay_viewer(*args, **kwargs):
    from .ui import run_move_replay_viewer as _run_move_replay_viewer

    return _run_move_replay_viewer(*args, **kwargs)


def run_agent_match_viewer(*args, **kwargs):
    from .ui import run_agent_match_viewer as _run_agent_match_viewer

    return _run_agent_match_viewer(*args, **kwargs)


def run_random_self_play_viewer(*args, **kwargs):
    from .ui import run_random_self_play_viewer as _run_random_self_play_viewer

    return _run_random_self_play_viewer(*args, **kwargs)


_EXPERIMENT_EXPORTS = {
    "BenchmarkResult",
    "SelfPlayResult",
    "SelfPlaySession",
    "TournamentResult",
    "benchmark_games",
    "play_game",
    "play_random_game",
    "run_tournament",
}


def __getattr__(name: str):
    if name not in _EXPERIMENT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    experiments = import_module(".experiments", __name__)
    value = getattr(experiments, name)
    globals()[name] = value
    return value

__all__ = [
    "ALL_PIECES",
    "Agent",
    "AdaptiveWeightedBlockingAgent",
    "BlockingAgent",
    "BOARD_SIZE",
    "BenchmarkResult",
    "Board",
    "Coordinate",
    "GameState",
    "LargestFirstAgent",
    "Move",
    "PIECE_TRANSFORMS",
    "PIECES",
    "RandomAgent",
    "STARTING_CORNERS",
    "SelfPlayResult",
    "SelfPlaySession",
    "Shape",
    "TournamentResult",
    "WeightedBlockingAgent",
    "generate_legal_moves",
    "frontier_targets",
    "is_legal_move",
    "benchmark_games",
    "play_game",
    "play_random_game",
    "render_board",
    "run_tournament",
    "run_move_replay_viewer",
    "run_agent_match_viewer",
    "run_random_self_play_viewer",
    "validate_move",
]
