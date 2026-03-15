"""Heuristic agent that balances piece size against blocked opponent frontier."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Agent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import frontier_targets


@dataclass
class WeightedBlockingAgent(Agent):
    """Maximize tiles placed plus a weighted bonus for blocked opponent frontier targets."""

    blocked_corner_weight: float = 1.0 / 3.0

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None

        opponents = sorted(player for player in state.remaining_pieces if player != state.current_player)
        opponent_frontier_before = {
            player: len(frontier_targets(state, player)) for player in opponents
        }

        return max(
            legal_moves,
            key=lambda move: self._score_move(state, move, opponents, opponent_frontier_before),
        )

    def _score_move(
        self,
        state: GameState,
        move: Move,
        opponents: list[int],
        opponent_frontier_before: dict[int, int],
    ) -> tuple[float, int, str, tuple[int, int], tuple[tuple[int, int], ...]]:
        next_state = state.apply_move(move)
        blocked_corners = sum(
            opponent_frontier_before[player] - len(frontier_targets(next_state, player))
            for player in opponents
        )
        score = len(move.cells) + self.blocked_corner_weight * blocked_corners

        return (
            score,
            len(move.cells),
            move.piece_name,
            (-move.origin[0], -move.origin[1]),
            tuple(sorted(move.cells)),
        )
