"""Simple heuristic agent that prefers larger pieces."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Agent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move


@dataclass
class LargestFirstAgent(Agent):
    """Pick the legal move that places the most cells, with deterministic tie-breaks."""

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None

        return max(
            legal_moves,
            key=lambda move: (
                len(move.cells),
                move.piece_name,
                -move.origin[0],
                -move.origin[1],
                tuple(sorted(move.cells)),
            ),
        )
