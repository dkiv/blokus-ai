"""Core Blokus engine primitives."""

from .board import BOARD_SIZE, Board
from .coords import Coordinate, Shape
from .game_state import GameState
from .move import Move
from .move_generation import count_legal_moves, frontier_targets, generate_legal_moves
from .pieces import ALL_PIECES, PIECES, PIECE_TRANSFORMS
from .rules import STARTING_CORNERS, is_legal_move, validate_move

__all__ = [
    "ALL_PIECES",
    "BOARD_SIZE",
    "Board",
    "count_legal_moves",
    "Coordinate",
    "frontier_targets",
    "GameState",
    "Move",
    "PIECE_TRANSFORMS",
    "PIECES",
    "STARTING_CORNERS",
    "Shape",
    "generate_legal_moves",
    "is_legal_move",
    "validate_move",
]
