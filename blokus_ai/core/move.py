"""Move value object."""

from __future__ import annotations

from dataclasses import dataclass

from .coords import Coordinate, Shape


@dataclass(frozen=True)
class Move:
    """A single move: piece choice, transformed shape, and anchor placement."""

    player: int
    piece_name: str
    origin: Coordinate
    cells: Shape

    @property
    def placed_cells(self) -> Shape:
        origin_row, origin_col = self.origin
        return frozenset((origin_row + row, origin_col + col) for row, col in self.cells)
