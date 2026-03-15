"""Board model for a standard 20x20 Blokus board."""

from __future__ import annotations

from dataclasses import dataclass, field

from .coords import Coordinate

BOARD_SIZE = 20
EDGE_OFFSETS: tuple[Coordinate, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNER_OFFSETS: tuple[Coordinate, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class Board:
    """A mutable 20x20 grid storing player occupancy."""

    size: int = BOARD_SIZE
    grid: list[list[int | None]] = field(default_factory=list)
    occupied_count: int = 0
    player_counts: dict[int, int] = field(default_factory=dict)
    cells_by_player: dict[int, set[Coordinate]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size != BOARD_SIZE:
            raise ValueError(f"Blokus boards must be {BOARD_SIZE}x{BOARD_SIZE}.")

        if not self.grid:
            self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
            self.occupied_count = 0
            self.player_counts = {}
            self.cells_by_player = {}
            return

        if len(self.grid) != self.size or any(len(row) != self.size for row in self.grid):
            raise ValueError("Board grid must match the configured board size.")

        if self.occupied_count == 0 and not self.player_counts and not self.cells_by_player:
            self._rebuild_metadata()

    def in_bounds(self, cell: Coordinate) -> bool:
        row, col = cell
        return 0 <= row < self.size and 0 <= col < self.size

    def get(self, cell: Coordinate) -> int | None:
        if not self.in_bounds(cell):
            raise IndexError(f"Cell {cell} is out of bounds.")
        row, col = cell
        return self.grid[row][col]

    def is_empty(self, cell: Coordinate) -> bool:
        return self.get(cell) is None

    def edge_neighbors(self, cell: Coordinate) -> tuple[Coordinate, ...]:
        row, col = cell
        return tuple(
            neighbor
            for neighbor in ((row + d_row, col + d_col) for d_row, d_col in EDGE_OFFSETS)
            if self.in_bounds(neighbor)
        )

    def corner_neighbors(self, cell: Coordinate) -> tuple[Coordinate, ...]:
        row, col = cell
        return tuple(
            neighbor
            for neighbor in ((row + d_row, col + d_col) for d_row, d_col in CORNER_OFFSETS)
            if self.in_bounds(neighbor)
        )

    def place(self, cells: set[Coordinate] | frozenset[Coordinate], player: int) -> None:
        out_of_bounds = [cell for cell in cells if not self.in_bounds(cell)]
        if out_of_bounds:
            raise ValueError(f"Cannot place cells out of bounds: {out_of_bounds}")

        occupied = [cell for cell in cells if not self.is_empty(cell)]
        if occupied:
            raise ValueError(f"Cannot place on occupied cells: {occupied}")

        for row, col in cells:
            self.grid[row][col] = player
            self.occupied_count += 1
            self.player_counts[player] = self.player_counts.get(player, 0) + 1
            self.cells_by_player.setdefault(player, set()).add((row, col))

    def clone(self) -> Board:
        return Board(
            size=self.size,
            grid=[row[:] for row in self.grid],
            occupied_count=self.occupied_count,
            player_counts=dict(self.player_counts),
            cells_by_player={
                player: set(cells) for player, cells in self.cells_by_player.items()
            },
        )

    def occupied_cells(self) -> dict[Coordinate, int]:
        occupied: dict[Coordinate, int] = {}
        for player, cells in self.cells_by_player.items():
            for cell in cells:
                occupied[cell] = player
        return occupied

    def occupied_by_player(self, player: int) -> frozenset[Coordinate]:
        return frozenset(self.cells_by_player.get(player, set()))

    def _rebuild_metadata(self) -> None:
        self.occupied_count = 0
        self.player_counts = {}
        self.cells_by_player = {}

        for row_index, row in enumerate(self.grid):
            for col_index, value in enumerate(row):
                if value is None:
                    continue
                self.occupied_count += 1
                self.player_counts[value] = self.player_counts.get(value, 0) + 1
                self.cells_by_player.setdefault(value, set()).add((row_index, col_index))
