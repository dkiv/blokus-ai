"""Base agent protocol."""

from __future__ import annotations

from typing import Protocol

from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move


class Agent(Protocol):
    """Minimal protocol for choosing a move from a game state."""

    def select_move(self, state: GameState) -> Move | None:
        """Return a move for the state's current player, or None to pass."""
