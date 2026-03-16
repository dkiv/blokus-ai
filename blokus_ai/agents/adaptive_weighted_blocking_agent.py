"""Heuristic agent with phase-dependent, rank-aware blocking pressure."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Agent
from blokus_ai.core.board import BOARD_SIZE
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import frontier_targets


@dataclass
class AdaptiveWeightedBlockingAgent(Agent):
    """Balance tiles placed with blocking, increasing pressure later in the game."""

    early_block_weight: float = 0.136
    late_block_weight: float = 0.709
    rank_weights: tuple[float, ...] = field(default_factory=lambda: (2.355, 0.607, 0.122))

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None

        opponents = sorted(player for player in state.remaining_pieces if player != state.current_player)
        opponent_frontier_before = {
            player: len(frontier_targets(state, player)) for player in opponents
        }
        opponent_rank_weights = self._opponent_rank_weights(state, opponents)
        block_weight = self._phase_block_weight(state)

        return max(
            legal_moves,
            key=lambda move: self._score_move(
                state,
                move,
                opponents,
                opponent_frontier_before,
                opponent_rank_weights,
                block_weight,
            ),
        )

    def _score_move(
        self,
        state: GameState,
        move: Move,
        opponents: list[int],
        opponent_frontier_before: dict[int, int],
        opponent_rank_weights: dict[int, float],
        block_weight: float,
    ) -> tuple[float, float, int, str, tuple[int, int], tuple[tuple[int, int], ...]]:
        next_state = state.apply_move(move)
        weighted_blocked_frontier = sum(
            opponent_rank_weights[player]
            * (opponent_frontier_before[player] - len(frontier_targets(next_state, player)))
            for player in opponents
        )
        score = len(move.cells) + block_weight * weighted_blocked_frontier

        return (
            score,
            weighted_blocked_frontier,
            len(move.cells),
            move.piece_name,
            (-move.origin[0], -move.origin[1]),
            tuple(sorted(move.cells)),
        )

    def _phase_block_weight(self, state: GameState) -> float:
        total_cells = BOARD_SIZE * BOARD_SIZE
        progress = state.board.occupied_count / total_cells
        return self.early_block_weight + progress * (self.late_block_weight - self.early_block_weight)

    def _opponent_rank_weights(
        self,
        state: GameState,
        opponents: list[int],
    ) -> dict[int, float]:
        scores = state.scores()
        ranked_opponents = sorted(opponents, key=lambda player: (-scores[player], player))
        weights: dict[int, float] = {}
        for rank, player in enumerate(ranked_opponents):
            if rank < len(self.rank_weights):
                weights[player] = self.rank_weights[rank]
            else:
                weights[player] = self.rank_weights[-1]
        return weights
