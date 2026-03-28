"""Heuristic agent that balances blocking, expansion, mobility, and piece urgency."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Agent
from blokus_ai.core.board import BOARD_SIZE
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import count_legal_moves, frontier_targets
from blokus_ai.core.pieces import PIECE_TRANSFORMS

_PIECE_AWKWARDNESS = {
    piece_name: (len(transforms[0]) - 1) + (8 - len(transforms)) / 8.0
    for piece_name, transforms in PIECE_TRANSFORMS.items()
}


@dataclass
class StrategicHeuristicAgent(Agent):
    """Score moves with a blend of size, blocking, frontier growth, and mobility."""

    piece_weight: float = 1.0
    early_block_weight: float = 0.15
    late_block_weight: float = 1.05
    own_frontier_weight: float = 0.18
    mobility_weight: float = 0.04
    early_piece_pressure_weight: float = 0.55
    late_piece_pressure_weight: float = 0.1
    rank_weights: tuple[float, ...] = field(default_factory=lambda: (1.5, 0.8, 0.3))
    mobility_limit: int = 48

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None

        opponents = sorted(player for player in state.remaining_pieces if player != state.current_player)
        opponent_frontier_before = {
            player: len(frontier_targets(state, player)) for player in opponents
        }
        opponent_rank_weights = self._opponent_rank_weights(state, opponents)
        progress = self._game_progress(state)
        block_weight = self._interpolate(self.early_block_weight, self.late_block_weight, progress)
        piece_pressure_weight = self._interpolate(
            self.early_piece_pressure_weight,
            self.late_piece_pressure_weight,
            progress,
        )

        return max(
            legal_moves,
            key=lambda move: self._score_move(
                state=state,
                move=move,
                opponents=opponents,
                opponent_frontier_before=opponent_frontier_before,
                opponent_rank_weights=opponent_rank_weights,
                block_weight=block_weight,
                piece_pressure_weight=piece_pressure_weight,
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
        piece_pressure_weight: float,
    ) -> tuple[float, float, int, int, float, str, tuple[int, int], tuple[tuple[int, int], ...]]:
        next_state = state.apply_move(move)
        weighted_blocked_frontier = sum(
            opponent_rank_weights[player]
            * (opponent_frontier_before[player] - len(frontier_targets(next_state, player)))
            for player in opponents
        )
        own_frontier = len(frontier_targets(next_state, move.player))
        own_mobility = count_legal_moves(
            next_state,
            player=move.player,
            limit=self.mobility_limit,
        )
        piece_pressure = _PIECE_AWKWARDNESS[move.piece_name]
        score = (
            self.piece_weight * len(move.cells)
            + block_weight * weighted_blocked_frontier
            + self.own_frontier_weight * own_frontier
            + self.mobility_weight * own_mobility
            + piece_pressure_weight * piece_pressure
        )

        return (
            score,
            weighted_blocked_frontier,
            own_mobility,
            own_frontier,
            len(move.cells),
            move.piece_name,
            (-move.origin[0], -move.origin[1]),
            tuple(sorted(move.cells)),
        )

    def _game_progress(self, state: GameState) -> float:
        total_cells = BOARD_SIZE * BOARD_SIZE
        return state.board.occupied_count / total_cells

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

    def _interpolate(self, early: float, late: float, progress: float) -> float:
        return early + progress * (late - early)
