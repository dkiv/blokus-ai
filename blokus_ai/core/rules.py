"""Move validation for standard 4-player Blokus."""

from __future__ import annotations

from .board import BOARD_SIZE, Board
from .coords import Coordinate
from .game_state import GameState
from .move import Move
from .pieces import PIECES, PIECE_TRANSFORMS

STARTING_CORNERS: dict[int, Coordinate] = {
    0: (0, 0),
    1: (0, BOARD_SIZE - 1),
    2: (BOARD_SIZE - 1, BOARD_SIZE - 1),
    3: (BOARD_SIZE - 1, 0),
}


def is_legal_move(state: GameState, move: Move, player: int | None = None) -> bool:
    """Return True when a move satisfies Blokus placement rules."""
    return _move_legality_error(state, move, player=player) is None


def validate_move(state: GameState, move: Move, player: int | None = None) -> None:
    """Raise ValueError if a move is illegal."""
    error = _move_legality_error(state, move, player=player)
    if error is not None:
        raise ValueError(error)


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

    if move.cells not in PIECE_TRANSFORMS[move.piece_name]:
        raise ValueError(f"Cells do not match the shape of piece {move.piece_name}.")


def _is_first_move(board: Board, player: int) -> bool:
    return board.player_counts.get(player, 0) == 0


def _has_same_player_edge_contact(
    board: Board, placed_cells: set[Coordinate] | frozenset[Coordinate], player: int
) -> bool:
    grid = board.grid
    size = board.size

    for row, col in placed_cells:
        if row > 0 and (row - 1, col) not in placed_cells and grid[row - 1][col] == player:
            return True
        if row + 1 < size and (row + 1, col) not in placed_cells and grid[row + 1][col] == player:
            return True
        if col > 0 and (row, col - 1) not in placed_cells and grid[row][col - 1] == player:
            return True
        if col + 1 < size and (row, col + 1) not in placed_cells and grid[row][col + 1] == player:
            return True
    return False


def _has_same_player_corner_contact(
    board: Board, placed_cells: set[Coordinate] | frozenset[Coordinate], player: int
) -> bool:
    grid = board.grid
    size = board.size

    for row, col in placed_cells:
        if row > 0 and col > 0 and (row - 1, col - 1) not in placed_cells and grid[row - 1][col - 1] == player:
            return True
        if row > 0 and col + 1 < size and (row - 1, col + 1) not in placed_cells and grid[row - 1][col + 1] == player:
            return True
        if row + 1 < size and col > 0 and (row + 1, col - 1) not in placed_cells and grid[row + 1][col - 1] == player:
            return True
        if row + 1 < size and col + 1 < size and (row + 1, col + 1) not in placed_cells and grid[row + 1][col + 1] == player:
            return True
    return False


def _move_legality_error(state: GameState, move: Move, player: int | None = None) -> str | None:
    active_player = move.player if player is None else player

    try:
        _validate_player_and_piece(state, move, active_player)
    except ValueError as error:
        return str(error)

    return _placement_legality_error(state.board, move.placed_cells, active_player)


def _placement_legality_error(
    board: Board,
    placed_cells: set[Coordinate] | frozenset[Coordinate],
    player: int,
) -> str | None:
    if not placed_cells:
        return "A move must place at least one cell."

    grid = board.grid
    size = board.size

    for row, col in placed_cells:
        if row < 0 or row >= size or col < 0 or col >= size:
            return f"Move places cells out of bounds: {[(row, col)]}"
        if grid[row][col] is not None:
            return f"Move overlaps occupied cells: {[(row, col)]}"

    if _has_same_player_edge_contact(board, placed_cells, player):
        return "Move cannot share an edge with the player's existing pieces."

    if _is_first_move(board, player):
        starting_corner = STARTING_CORNERS[player]
        if starting_corner not in placed_cells:
            return f"First move for player {player} must cover starting corner {starting_corner}."
        return None

    if not _has_same_player_corner_contact(board, placed_cells, player):
        return "Move must touch one of the player's existing pieces at a corner."

    return None
