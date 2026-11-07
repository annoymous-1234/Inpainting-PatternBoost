from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from env import NoThreeInLineEnv
from supervised.validation import check_bounds, check_duplicates, check_no_three_collinear

from .state import filled_count, slot_state_ids_to_points_and_assignment


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record as a line to a UTF-8 JSONL file."""

    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def serialize_slot_state(slot_state_ids: np.ndarray, n: int) -> dict[str, Any]:
    """Serialize one canonical slot state into JSON-friendly fields.

    ``slot_state_ids`` is an ``int64`` array of shape ``[2n]``. The returned
    mapping contains the raw token ids, sorted selected lattice points, and the
    derived filled-slot count. The function does not mutate the input state.
    """

    state_array = np.asarray(slot_state_ids, dtype=np.int64)
    points, _ = slot_state_ids_to_points_and_assignment(state_array)
    return {
        "slot_state_ids": [int(value) for value in state_array.tolist()],
        "selected_points": [[int(x), int(y)] for x, y in sorted(points)],
        "filled_count": int(filled_count(state_array)),
    }


def validate_candidate_state(slot_state_ids: np.ndarray, n: int) -> dict[str, Any]:
    """Validate one candidate state by geometry first, then env consistency.

    ``slot_state_ids`` is an integer array of shape ``[2n]``. The function
    checks token range validity, duplicate points, bounds, and no-three-line
    geometry, then reconstructs a ``NoThreeInLineEnv`` to verify observation,
    ``point_to_slot``, and per-slot state consistency. The returned mapping is
    JSON-friendly and contains ``is_valid`` plus ``validation_errors``.
    """

    state_array = np.asarray(slot_state_ids, dtype=np.int64)
    errors: list[str] = []
    if state_array.shape != (2 * n,):
        errors.append(f"slot_state_ids must have shape [{2 * n}]")
        serialized = {
            "slot_state_ids": [int(value) for value in state_array.reshape(-1).tolist()],
            "selected_points": [],
            "filled_count": 0,
        }
        return {
            **serialized,
            "is_valid": False,
            "validation_errors": errors,
        }

    if np.any(state_array < 0) or np.any(state_array > n):
        errors.append(f"slot_state_ids values must lie in [0, {n}]")

    points, point_to_slot = slot_state_ids_to_points_and_assignment(state_array)
    errors.extend(check_duplicates(points))
    errors.extend(check_bounds(points, n))
    errors.extend(check_no_three_collinear(points))

    if not errors:
        env = NoThreeInLineEnv(n=n, num_slots=2 * n)
        obs = env.reset(init_points=points, init_slot_assignment=point_to_slot)
        if obs["selected_points"] != sorted(points):
            errors.append("env observation does not match the decoded point set")
        if env.point_to_slot != point_to_slot:
            errors.append("env point_to_slot mapping does not match decoded slot ownership")
        for slot_id, state_id in enumerate(state_array.tolist()):
            slot_state = env.slot_states[slot_id]
            if state_id == 0:
                if slot_state.status != "MASKED" or slot_state.point is not None:
                    errors.append(f"env slot {slot_id} is inconsistent with a masked token")
            else:
                expected_point = (slot_id // 2, state_id - 1)
                if slot_state.status != "FILLED" or slot_state.point != expected_point:
                    errors.append(f"env slot {slot_id} is inconsistent with filled token {state_id}")

    serialized = serialize_slot_state(state_array, n)
    return {
        **serialized,
        "is_valid": not errors,
        "validation_errors": errors,
    }
