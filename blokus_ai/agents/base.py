"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move


class Agent(ABC):
    """Base class for move-selection agents."""

    @abstractmethod
    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        """Choose one legal move, or ``None`` to pass when no legal moves exist."""
