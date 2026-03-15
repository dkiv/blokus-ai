"""Move validation for standard 4-player Blokus."""

from __future__ import annotations

from .board import BOARD_SIZE, Board
from .coords import Coordinate
from .game_state import GameState
from .move import Move
from .pieces import PIECES
from .transforms import canonical

STARTING_CORNERS: dict[int, Coordinate] = {
    0: (0, 0),
    1: (0, BOARD_SIZE - 1),
    2: (BOARD_SIZE - 1, BOARD_SIZE - 1),
    3: (BOARD_SIZE - 1, 0),
}


def is_legal_move(state: GameState, move: Move, player: int | None = None) -> bool:
    """Return True when a move satisfies Blokus placement rules."""
    try:
        validate_move(state, move, player=player)
    except ValueError:
        return False
    return True


def validate_move(state: GameState, move: Move, player: int | None = None) -> None:
    """Raise ValueError if a move is illegal."""
    active_player = move.player if player is None else player
    _validate_player_and_piece(state, move, active_player)

    placed_cells = move.placed_cells
    board = state.board

    if not placed_cells:
        raise ValueError("A move must place at least one cell.")

    out_of_bounds = [cell for cell in placed_cells if not board.in_bounds(cell)]
    if out_of_bounds:
        raise ValueError(f"Move places cells out of bounds: {out_of_bounds}")

    occupied = [cell for cell in placed_cells if not board.is_empty(cell)]
    if occupied:
        raise ValueError(f"Move overlaps occupied cells: {occupied}")

    if _has_same_player_edge_contact(board, placed_cells, active_player):
        raise ValueError("Move cannot share an edge with the player's existing pieces.")

    if _is_first_move(board, active_player):
        starting_corner = STARTING_CORNERS[active_player]
        if starting_corner not in placed_cells:
            raise ValueError(
                f"First move for player {active_player} must cover starting corner {starting_corner}."
            )
        return

    if not _has_same_player_corner_contact(board, placed_cells, active_player):
        raise ValueError("Move must touch one of the player's existing pieces at a corner.")


def _validate_player_and_piece(state: GameState, move: Move, player: int) -> None:
    if player != move.player:
        raise ValueError("Move player does not match the validated player.")

    if player not in state.remaining_pieces:
        raise ValueError(f"Unknown player {player}.")

    available = state.remaining_pieces[player]
    if move.piece_name not in available:
        raise ValueError(f"Piece {move.piece_name} is not available for player {player}.")

    if move.piece_name not in PIECES:
        raise ValueError(f"Unknown piece {move.piece_name}.")

    if canonical(move.cells) != canonical(PIECES[move.piece_name]):
        raise ValueError(f"Cells do not match the shape of piece {move.piece_name}.")


def _is_first_move(board: Board, player: int) -> bool:
    return not any(value == player for value in board.occupied_cells().values())


def _has_same_player_edge_contact(
    board: Board, placed_cells: set[Coordinate] | frozenset[Coordinate], player: int
) -> bool:
    for cell in placed_cells:
        for neighbor in board.edge_neighbors(cell):
            if neighbor in placed_cells:
                continue
            if board.get(neighbor) == player:
                return True
    return False


def _has_same_player_corner_contact(
    board: Board, placed_cells: set[Coordinate] | frozenset[Coordinate], player: int
) -> bool:
    for cell in placed_cells:
        for neighbor in board.corner_neighbors(cell):
            if neighbor in placed_cells:
                continue
            if board.get(neighbor) == player:
                return True
    return False
