"""Experiment runners and demos."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark import BenchmarkResult, benchmark_games
    from .self_play import SelfPlayResult, SelfPlaySession, play_game, play_random_game
    from .tournament import TournamentResult, run_tournament

__all__ = [
    "BenchmarkResult",
    "SelfPlayResult",
    "SelfPlaySession",
    "TournamentResult",
    "benchmark_games",
    "play_game",
    "play_random_game",
    "run_tournament",
]

_MODULE_BY_EXPORT = {
    "BenchmarkResult": ".benchmark",
    "benchmark_games": ".benchmark",
    "SelfPlayResult": ".self_play",
    "SelfPlaySession": ".self_play",
    "play_game": ".self_play",
    "play_random_game": ".self_play",
    "TournamentResult": ".tournament",
    "run_tournament": ".tournament",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_EXPORT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
