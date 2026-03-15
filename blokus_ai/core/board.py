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

    def __post_init__(self) -> None:
        if self.size != BOARD_SIZE:
            raise ValueError(f"Blokus boards must be {BOARD_SIZE}x{BOARD_SIZE}.")

        if not self.grid:
            self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
            return

        if len(self.grid) != self.size or any(len(row) != self.size for row in self.grid):
            raise ValueError("Board grid must match the configured board size.")

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

    def clone(self) -> Board:
        return Board(size=self.size, grid=[row[:] for row in self.grid])

    def occupied_cells(self) -> dict[Coordinate, int]:
        occupied: dict[Coordinate, int] = {}
        for row_index, row in enumerate(self.grid):
            for col_index, value in enumerate(row):
                if value is not None:
                    occupied[(row_index, col_index)] = value
        return occupied
