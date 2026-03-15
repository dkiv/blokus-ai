"""Simple legal move generation for Blokus."""

from __future__ import annotations

from .game_state import GameState
from .move import Move
from .pieces import PIECE_TRANSFORMS
from .rules import STARTING_CORNERS, is_legal_move


def generate_legal_moves(state: GameState, player: int | None = None) -> list[Move]:
    """Return all legal moves for the target player."""
    active_player = state.current_player if player is None else player

    if active_player not in state.remaining_pieces:
        return []

    legal_moves: list[Move] = []
    candidate_origins = _candidate_origins(state, active_player)

    for piece_name in sorted(state.remaining_pieces[active_player]):
        for cells in PIECE_TRANSFORMS[piece_name]:
            seen_origins: set[tuple[int, int]] = set()
            for origin in _origins_for_targets(cells, candidate_origins):
                if origin in seen_origins:
                    continue
                seen_origins.add(origin)
                move = Move(
                    player=active_player,
                    piece_name=piece_name,
                    origin=origin,
                    cells=cells,
                )
                if is_legal_move(state, move, player=active_player):
                    legal_moves.append(move)

    return legal_moves


def _candidate_origins(state: GameState, player: int) -> tuple[tuple[int, int], ...]:
    board = state.board
    if board.player_counts.get(player, 0) == 0:
        return (STARTING_CORNERS[player],)

    grid = board.grid
    size = board.size
    targets: set[tuple[int, int]] = set()

    for row, col in board.occupied_by_player(player):
        for target_row, target_col in (
            (row - 1, col - 1),
            (row - 1, col + 1),
            (row + 1, col - 1),
            (row + 1, col + 1),
        ):
            if not (0 <= target_row < size and 0 <= target_col < size):
                continue
            if grid[target_row][target_col] is not None:
                continue
            if _has_same_player_edge_neighbor(grid, size, target_row, target_col, player):
                continue
            targets.add((target_row, target_col))

    return tuple(sorted(targets))


def _origins_for_targets(
    cells: frozenset[tuple[int, int]],
    targets: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    origins: list[tuple[int, int]] = []
    for target_row, target_col in targets:
        for cell_row, cell_col in sorted(cells):
            origins.append((target_row - cell_row, target_col - cell_col))
    return tuple(origins)


def _has_same_player_edge_neighbor(
    grid: list[list[int | None]],
    size: int,
    row: int,
    col: int,
    player: int,
) -> bool:
    return (
        (row > 0 and grid[row - 1][col] == player)
        or (row + 1 < size and grid[row + 1][col] == player)
        or (col > 0 and grid[row][col - 1] == player)
        or (col + 1 < size and grid[row][col + 1] == player)
    )
