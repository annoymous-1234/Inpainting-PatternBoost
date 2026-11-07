from __future__ import annotations

from geometry import Point

from .types import ExecutedAction, SearchState


def filled_count(slot_state_ids) -> int:
    """Count filled slots in one unified slot-state vector.

    ``slot_state_ids`` is an integer array-like of shape ``[2n]`` with values
    ``0`` for ``[MASK]`` and ``y + 1`` for filled slots. The function returns a
    Python ``int``, does not mutate the input, and assumes the unified token
    encoding used by the project model.
    """

    return int((slot_state_ids != 0).sum())


def masked_count(slot_state_ids) -> int:
    """Count masked slots in one unified slot-state vector."""

    return int((slot_state_ids == 0).sum())


def slot_state_ids_to_points_and_assignment(slot_state_ids) -> tuple[list[Point], dict[Point, int]]:
    """Decode ``slot_state_ids`` into selected points and owning slots.

    ``slot_state_ids`` is an integer array-like of shape ``[2n]``. The
    function returns ``(selected_points, point_to_slot)``, does not mutate the
    input, and assumes each filled slot belongs to column ``slot_id // 2``.
    """

    points: list[Point] = []
    point_to_slot: dict[Point, int] = {}
    for slot_id, state_id in enumerate(slot_state_ids.tolist()):
        if state_id == 0:
            continue
        point = (slot_id // 2, state_id - 1)
        points.append(point)
        point_to_slot[point] = slot_id
    return points, point_to_slot


def apply_executed_action_to_slot_state(slot_state_ids, executed_action: ExecutedAction, n: int):
    """Apply one executed action to a copied slot-state vector.

    ``slot_state_ids`` is an integer array-like of shape ``[2n]``,
    ``executed_action`` contains the repair-filtered slot/value assignments, and
    ``n`` is the grid size. The function returns a new ``int64`` array of shape
    ``[2n]``, does not mutate the input state, and assumes
    ``allow_existing_deletions=False`` so only new fills are applied.
    """

    import numpy as np

    next_state = np.asarray(slot_state_ids, dtype=np.int64).copy()
    if next_state.shape != (2 * n,):
        raise ValueError(f"slot_state_ids must have shape [{2 * n}]")
    for slot_id in executed_action.active_slots:
        if not 0 <= slot_id < 2 * n:
            raise ValueError(f"slot id out of range: {slot_id}")
        y_value = executed_action.values[slot_id]
        if not 0 <= y_value < n:
            raise ValueError(f"row value out of range for slot {slot_id}: {y_value}")
        next_state[slot_id] = y_value + 1
    return next_state


def build_empty_root_state(n: int) -> SearchState:
    """Return the empty-board search state for an ``n x n`` problem."""

    import numpy as np

    return SearchState(slot_state_ids=np.zeros(2 * n, dtype=np.int64), episode_step=0)


def slot_state_ids_from_env(env) -> object:
    """Encode one live environment state into unified slot tokens.

    ``env`` is a ``NoThreeInLineEnv``-compatible object with ``num_slots`` and
    ``slot_states``. The function returns a new ``int64`` array of shape
    ``[num_slots]``, does not mutate the environment, and assumes filled slots
    store their owned point in ``slot_state.point``.
    """

    import numpy as np

    slot_state_ids = np.zeros(env.num_slots, dtype=np.int64)
    for slot_id in range(env.num_slots):
        slot_state = env.slot_states[slot_id]
        if slot_state.status != "FILLED" or slot_state.point is None:
            continue
        slot_state_ids[slot_id] = int(slot_state.point[1]) + 1
    return slot_state_ids


def search_state_from_env(env) -> SearchState:
    """Build one canonical search state from a live environment snapshot.

    ``env`` is a ``NoThreeInLineEnv``-compatible object whose ``timestep``
    counts executed real rollout steps. The function returns a ``SearchState``,
    does not mutate the environment, and assumes remask operations do not
    advance ``env.timestep``.
    """

    return SearchState(slot_state_ids=slot_state_ids_from_env(env), episode_step=int(env.timestep))
