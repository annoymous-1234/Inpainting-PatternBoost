from __future__ import annotations

import hashlib

import numpy as np

from geometry import Point


def filled_count(slot_state_ids: np.ndarray) -> int:
    """Return the number of filled slots in one state vector.

    ``slot_state_ids`` is an integer array-like value of shape ``[2n]`` with
    ``0`` for masked slots and ``y + 1`` for filled slots. The function returns
    a Python ``int``, does not mutate the input, and assumes the vector uses the
    project slot encoding.
    """

    return int(np.count_nonzero(np.asarray(slot_state_ids, dtype=np.int64)))


def state_ids_to_points(slot_state_ids: np.ndarray) -> list[Point]:
    """Decode slot-state ids into sorted lattice points.

    ``slot_state_ids`` is an integer array-like value of shape ``[2n]``. The
    function returns sorted ``(x, y)`` points, does not mutate the input, and
    assumes each slot belongs to column ``slot_id // 2`` with nonzero values
    storing ``y + 1``.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64)
    points: list[Point] = []
    for slot_id, state_id in enumerate(state.tolist()):
        if state_id == 0:
            continue
        points.append((slot_id // 2, int(state_id) - 1))
    return sorted(points)


def points_to_canonical_state_ids(points: list[Point] | set[Point], n: int) -> np.ndarray:
    """Encode points into canonical two-slots-per-column state ids.

    ``points`` is a legal or candidate point collection on an ``n x n`` grid.
    The function returns an ``int64`` array of shape ``[2n]`` with rows sorted
    within each column, does not mutate the input, and assumes no column
    contains more than two distinct rows.
    """

    columns: dict[int, list[int]] = {column: [] for column in range(n)}
    for point in points:
        x, y = point
        if not 0 <= x < n or not 0 <= y < n:
            raise ValueError(f"point out of bounds for n={n}: {point}")
        columns[x].append(y)

    state = np.zeros(2 * n, dtype=np.int64)
    for column in range(n):
        rows = sorted(columns[column])
        if len(rows) != len(set(rows)):
            raise ValueError(f"column {column} contains duplicate row assignments")
        if len(rows) > 2:
            raise ValueError(f"column {column} contains more than two points")
        for pair_index, row in enumerate(rows):
            state[2 * column + pair_index] = int(row) + 1
    return state


def canonicalize_state_ids(slot_state_ids: np.ndarray, n: int) -> np.ndarray:
    """Return the canonical encoding of the points represented by ``slot_state_ids``.

    ``slot_state_ids`` is an integer vector of shape ``[2n]``. The function
    decodes its filled points and re-encodes them with sorted rows per column,
    returns a new ``int64`` array, and does not mutate the input. It assumes the
    represented point multiset fits the two-slots-per-column layout.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64)
    if state.shape != (2 * n,):
        raise ValueError(f"slot_state_ids must have shape [{2 * n}]")
    return points_to_canonical_state_ids(state_ids_to_points(state), n)


def canonical_hash(slot_state_ids: np.ndarray, n: int) -> str:
    """Return a stable hash for a canonicalized slot-state vector.

    ``slot_state_ids`` is an integer vector of shape ``[2n]`` and ``n`` is the
    grid size. The function returns a hex SHA-256 digest, does not mutate the
    input, and assumes canonicalization is the desired equivalence relation.
    """

    canonical = canonicalize_state_ids(slot_state_ids, n)
    return hashlib.sha256(canonical.astype(np.int64, copy=False).tobytes()).hexdigest()


def state_ids_to_env_reset_args(slot_state_ids: np.ndarray) -> tuple[list[Point], dict[Point, int]]:
    """Convert slot-state ids into ``NoThreeInLineEnv.reset`` arguments.

    ``slot_state_ids`` is an integer vector where nonzero values store filled
    rows as ``y + 1``. The function returns ``(init_points, init_assignment)``,
    does not mutate the input, and assumes the current slot ids should be
    preserved for the mutable generation environment.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64)
    init_points: list[Point] = []
    init_assignment: dict[Point, int] = {}
    for slot_id, state_id in enumerate(state.tolist()):
        if state_id == 0:
            continue
        point = (slot_id // 2, int(state_id) - 1)
        init_points.append(point)
        init_assignment[point] = slot_id
    return sorted(init_points), init_assignment
