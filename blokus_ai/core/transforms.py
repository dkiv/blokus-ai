"""Utilities for transforming piece coordinate sets."""

from __future__ import annotations

from .coords import Coordinate, Shape


def normalize(shape: Shape | set[Coordinate]) -> Shape:
    """Translate a shape so its minimum row and column are both zero."""
    if not shape:
        return frozenset()

    min_row = min(row for row, _ in shape)
    min_col = min(col for _, col in shape)
    return frozenset((row - min_row, col - min_col) for row, col in shape)


def rotate_clockwise(shape: Shape | set[Coordinate]) -> Shape:
    """Rotate a shape 90 degrees clockwise around the origin, then normalize it."""
    rotated = {(col, -row) for row, col in shape}
    return normalize(rotated)


def reflect_horizontal(shape: Shape | set[Coordinate]) -> Shape:
    """Mirror a shape across the vertical axis, then normalize it."""
    reflected = {(row, -col) for row, col in shape}
    return normalize(reflected)


def unique_transformations(shape: Shape | set[Coordinate]) -> tuple[Shape, ...]:
    """Return all distinct rotations and reflections of a shape."""
    current = normalize(shape)
    variants: set[Shape] = set()

    for _ in range(4):
        variants.add(current)
        variants.add(reflect_horizontal(current))
        current = rotate_clockwise(current)

    return tuple(sorted(variants, key=_shape_sort_key))


def canonical(shape: Shape | set[Coordinate]) -> Shape:
    """Return a stable canonical representation across all symmetries."""
    return min(unique_transformations(shape), key=_shape_sort_key)


def _shape_sort_key(shape: Shape) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(shape))
