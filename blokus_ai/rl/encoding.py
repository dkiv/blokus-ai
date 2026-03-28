"""RL-friendly state and action encoders built from raw board geometry."""

from __future__ import annotations

from dataclasses import dataclass

from blokus_ai.core.board import BOARD_SIZE
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves
from blokus_ai.core.pieces import PIECES

_PIECE_NAMES = tuple(sorted(PIECES))
_PIECE_INDEX = {piece_name: index for index, piece_name in enumerate(_PIECE_NAMES)}


@dataclass(frozen=True)
class RLObservation:
    """Current position encoded from the acting player's perspective."""

    current_player: int
    board_planes: tuple[tuple[tuple[int, ...], ...], ...]
    remaining_piece_planes: tuple[tuple[int, ...], ...]
    scores: tuple[int, ...]
    turn_index: int


@dataclass(frozen=True)
class RLCandidateMove:
    """One legal move with a raw geometric encoding suitable for scoring."""

    move: Move
    placed_mask: tuple[tuple[int, ...], ...]
    piece_vector: tuple[int, ...]
    origin: tuple[float, float]
    cell_count: int
    transform_extent: tuple[int, int]


def encode_observation(state: GameState) -> RLObservation:
    """Encode the current state using player-relative channels."""
    player_order = _relative_player_order(state)
    board_planes = tuple(
        tuple(
            tuple(1 if state.board.grid[row][col] == player else 0 for col in range(BOARD_SIZE))
            for row in range(BOARD_SIZE)
        )
        for player in player_order
    )
    remaining_piece_planes = tuple(
        tuple(1 if piece_name in state.remaining_pieces[player] else 0 for piece_name in _PIECE_NAMES)
        for player in player_order
    )
    scores = tuple(state.scores()[player] for player in player_order)

    return RLObservation(
        current_player=state.current_player,
        board_planes=board_planes,
        remaining_piece_planes=remaining_piece_planes,
        scores=scores,
        turn_index=sum(scores),
    )


def encode_candidate_moves(
    state: GameState,
    legal_moves: list[Move] | None = None,
) -> list[RLCandidateMove]:
    """Encode all legal moves for the current player."""
    legal_moves = (
        generate_legal_moves(state, player=state.current_player)
        if legal_moves is None
        else legal_moves
    )
    return [encode_candidate_move(move) for move in legal_moves]


def encode_candidate_move(move: Move) -> RLCandidateMove:
    """Encode a legal move without relying on handcrafted strategic heuristics."""
    piece_vector = [0] * len(_PIECE_NAMES)
    piece_vector[_PIECE_INDEX[move.piece_name]] = 1

    placed_mask = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for row, col in move.placed_cells:
        placed_mask[row][col] = 1

    max_row = max(cell_row for cell_row, _ in move.cells)
    max_col = max(cell_col for _, cell_col in move.cells)

    return RLCandidateMove(
        move=move,
        placed_mask=tuple(tuple(row) for row in placed_mask),
        piece_vector=tuple(piece_vector),
        origin=(
            move.origin[0] / (BOARD_SIZE - 1),
            move.origin[1] / (BOARD_SIZE - 1),
        ),
        cell_count=len(move.cells),
        transform_extent=(max_row + 1, max_col + 1),
    )


def piece_names() -> tuple[str, ...]:
    """Stable piece ordering used by the RL encoders."""
    return _PIECE_NAMES


def _relative_player_order(state: GameState) -> tuple[int, ...]:
    players = sorted(state.remaining_pieces)
    current_index = players.index(state.current_player)
    return tuple(players[(current_index + offset) % len(players)] for offset in range(len(players)))
