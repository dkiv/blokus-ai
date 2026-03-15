"""Experiment runners and demos."""

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
