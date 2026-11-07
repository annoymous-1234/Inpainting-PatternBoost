from __future__ import annotations

import numpy as np

from geometry import Point
from patternboost.state import points_to_canonical_state_ids, state_ids_to_points


GRID_D4_NON_IDENTITY_TRANSFORMS = (
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
    "reflect_vertical_midline",
    "reflect_horizontal_midline",
    "rotate90",
    "rotate180",
    "rotate270",
)


def transform_point(point: Point, n: int, transform_name: str) -> Point:
    """Transform one grid point by a named square-grid symmetry.

    ``point`` is an ``(x, y)`` lattice coordinate on an ``n x n`` grid and
    ``transform_name`` is one of ``GRID_D4_NON_IDENTITY_TRANSFORMS``. The
    function returns a transformed ``(x, y)`` coordinate, does not mutate
    inputs, and assumes the input point is within grid bounds.
    """

    x, y = point
    if not 0 <= x < n or not 0 <= y < n:
        raise ValueError(f"point out of bounds for n={n}: {point}")
    if transform_name == "reflect_main_diagonal":
        return (y, x)
    if transform_name == "reflect_anti_diagonal":
        return (n - 1 - y, n - 1 - x)
    if transform_name == "reflect_vertical_midline":
        return (n - 1 - x, y)
    if transform_name == "reflect_horizontal_midline":
        return (x, n - 1 - y)
    if transform_name == "rotate90":
        return (n - 1 - y, x)
    if transform_name == "rotate180":
        return (n - 1 - x, n - 1 - y)
    if transform_name == "rotate270":
        return (y, n - 1 - x)
    raise ValueError(f"unsupported grid symmetry transform: {transform_name}")


def transform_points(points: list[Point] | set[Point], n: int, transform_name: str) -> list[Point]:
    """Transform a point collection by one square-grid symmetry.

    ``points`` contains ``(x, y)`` lattice coordinates on an ``n x n`` grid and
    ``transform_name`` selects the symmetry. The function returns sorted
    transformed points, does not mutate inputs, and assumes duplicate points
    should be collapsed by set semantics after transformation.
    """

    return sorted({transform_point(point, n, transform_name) for point in points})


def generate_grid_symmetry_points(points: list[Point] | set[Point], n: int) -> list[tuple[str, list[Point]]]:
    """Generate all non-identity square-grid symmetry point sets.

    ``points`` contains ``(x, y)`` lattice coordinates on an ``n x n`` grid.
    The function returns ``(transform_name, transformed_points)`` pairs for the
    fixed seven non-identity D4 symmetries, does not mutate inputs, and assumes
    callers want named outputs for downstream provenance.
    """

    return [
        (transform_name, transform_points(points, n, transform_name))
        for transform_name in GRID_D4_NON_IDENTITY_TRANSFORMS
    ]


def generate_grid_symmetry_state_ids(slot_state_ids: np.ndarray, n: int) -> list[tuple[str, np.ndarray]]:
    """Generate non-identity symmetry states from a PatternBoost state vector.

    ``slot_state_ids`` is an integer array-like value of shape ``[2n]`` where
    ``0`` is masked and nonzero values store ``y + 1``. The function returns
    ``(transform_name, state_ids)`` pairs where each ``state_ids`` is an
    ``int64`` NumPy array of shape ``[2n]``. It does not mutate the source
    state and assumes the transformed point sets fit the two-slots-per-column
    representation, as legal no-three-in-line states do.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64)
    if state.shape != (2 * n,):
        raise ValueError(f"slot_state_ids must have shape [{2 * n}]")
    points = state_ids_to_points(state)
    return [
        (transform_name, points_to_canonical_state_ids(transformed_points, n))
        for transform_name, transformed_points in generate_grid_symmetry_points(points, n)
    ]
