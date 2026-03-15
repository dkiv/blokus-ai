"""Experiment runners and demos."""

from .self_play import SelfPlayResult, SelfPlaySession, play_game, play_random_game
from .tournament import TournamentResult, run_tournament

__all__ = [
    "SelfPlayResult",
    "SelfPlaySession",
    "TournamentResult",
    "play_game",
    "play_random_game",
    "run_tournament",
]
