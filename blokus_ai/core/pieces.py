"""Blokus piece definitions as normalized coordinate sets."""

from __future__ import annotations

from .coords import Shape
from .transforms import normalize, unique_transformations


def _shape(*cells: tuple[int, int]) -> Shape:
    return normalize(set(cells))


PIECES: dict[str, Shape] = {
    "I1": _shape((0, 0)),
    "I2": _shape((0, 0), (1, 0)),
    "I3": _shape((0, 0), (1, 0), (2, 0)),
    "V3": _shape((0, 0), (1, 0), (1, 1)),
    "I4": _shape((0, 0), (1, 0), (2, 0), (3, 0)),
    "O4": _shape((0, 0), (0, 1), (1, 0), (1, 1)),
    "T4": _shape((0, 0), (0, 1), (0, 2), (1, 1)),
    "L4": _shape((0, 0), (1, 0), (2, 0), (2, 1)),
    "Z4": _shape((0, 0), (0, 1), (1, 1), (1, 2)),
    "I5": _shape((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    "L5": _shape((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    "Y5": _shape((0, 0), (1, 0), (2, 0), (3, 0), (2, 1)),
    "P5": _shape((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    "T5": _shape((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "V5": _shape((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    "N5": _shape((0, 0), (1, 0), (2, 0), (2, 1), (3, 1)),
    "Z5": _shape((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
    "W5": _shape((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    "U5": _shape((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    "F5": _shape((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    "X5": _shape((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
}

PIECE_TRANSFORMS: dict[str, tuple[Shape, ...]] = {
    name: unique_transformations(shape) for name, shape in PIECES.items()
}

ALL_PIECES: frozenset[str] = frozenset(PIECES)
