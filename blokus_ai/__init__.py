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
from .experiments import SelfPlayResult, play_random_game
from .ui import render_board

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
    "Shape",
    "generate_legal_moves",
    "is_legal_move",
    "play_random_game",
    "render_board",
    "validate_move",
]
