"""Heuristic agent that tries to reduce opponent frontier targets."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Agent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import frontier_targets


@dataclass
class BlockingAgent(Agent):
    """Prefer moves that leave opponents with fewer frontier targets."""

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None

        opponents = sorted(player for player in state.remaining_pieces if player != state.current_player)

        return min(
            legal_moves,
            key=lambda move: self._score_move(state, move, opponents),
        )

    def _score_move(
        self,
        state: GameState,
        move: Move,
        opponents: list[int],
    ) -> tuple[int, int, int, str, tuple[int, int], tuple[tuple[int, int], ...]]:
        next_state = state.apply_move(move)
        opponent_frontier_total = sum(len(frontier_targets(next_state, player)) for player in opponents)
        own_frontier = len(frontier_targets(next_state, move.player))

        return (
            opponent_frontier_total,
            -len(move.cells),
            -own_frontier,
            move.piece_name,
            move.origin,
            tuple(sorted(move.cells)),
        )
