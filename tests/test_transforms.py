from blokus_ai.core.transforms import (
    canonical,
    normalize,
    reflect_horizontal,
    rotate_clockwise,
    unique_transformations,
)


def test_normalize_translates_shape_to_origin() -> None:
    shape = frozenset({(2, 3), (2, 4), (3, 3)})

    assert normalize(shape) == frozenset({(0, 0), (0, 1), (1, 0)})


def test_rotate_clockwise_normalizes_result() -> None:
    shape = frozenset({(0, 0), (1, 0), (1, 1)})

    assert rotate_clockwise(shape) == frozenset({(0, 0), (0, 1), (1, 0)})


def test_reflect_horizontal_normalizes_result() -> None:
    shape = frozenset({(0, 0), (1, 0), (1, 1)})

    assert reflect_horizontal(shape) == frozenset({(0, 1), (1, 0), (1, 1)})


def test_unique_transformations_collapses_symmetric_shapes() -> None:
    square = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    line = frozenset({(0, 0), (1, 0), (2, 0)})

    assert len(unique_transformations(square)) == 1
    assert len(unique_transformations(line)) == 2


def test_canonical_is_stable_across_rotations_and_reflections() -> None:
    shape = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})

    variants = unique_transformations(shape)
    canonicals = {canonical(variant) for variant in variants}

    assert len(canonicals) == 1
