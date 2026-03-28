"""Experiment runners and demos."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark import BenchmarkResult, benchmark_games
    from .rl_self_play import (
        ImitationStats,
        RLEvalStats,
        RLSelfPlayStats,
        evaluate_policy,
        train_actor_critic,
        warm_start_policy,
    )
    from .self_play import SelfPlayResult, SelfPlaySession, play_game, play_random_game
    from .tournament import TournamentResult, run_tournament

__all__ = [
    "BenchmarkResult",
    "ImitationStats",
    "RLEvalStats",
    "RLSelfPlayStats",
    "SelfPlayResult",
    "SelfPlaySession",
    "TournamentResult",
    "benchmark_games",
    "evaluate_policy",
    "play_game",
    "play_random_game",
    "run_tournament",
    "train_actor_critic",
    "warm_start_policy",
]

_MODULE_BY_EXPORT = {
    "BenchmarkResult": ".benchmark",
    "benchmark_games": ".benchmark",
    "ImitationStats": ".rl_self_play",
    "RLEvalStats": ".rl_self_play",
    "RLSelfPlayStats": ".rl_self_play",
    "evaluate_policy": ".rl_self_play",
    "SelfPlayResult": ".self_play",
    "SelfPlaySession": ".self_play",
    "play_game": ".self_play",
    "play_random_game": ".self_play",
    "TournamentResult": ".tournament",
    "train_actor_critic": ".rl_self_play",
    "warm_start_policy": ".rl_self_play",
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
