from __future__ import annotations

import numpy as np

from env import NoThreeInLineEnv
from proposal_eval import compute_existing_legal_rows_by_column

from .state import state_ids_to_points


def legal_rows_for_masked_slots(
    env: NoThreeInLineEnv,
    slot_ids: list[int],
) -> dict[int, np.ndarray]:
    """Return legal row masks for the columns touched by ``slot_ids``.

    ``env`` supplies the current committed points and active-line backend, and
    ``slot_ids`` are candidate slot ids. The function returns
    ``column -> bool[n]`` masks, does not mutate the environment, and computes
    each touched column once for efficient decode-step scoring.
    """

    columns = sorted({int(slot_id // 2) for slot_id in slot_ids})
    return compute_existing_legal_rows_by_column(
        existing_points=env.backend.get_selected_points(),
        n=env.n,
        candidate_columns=columns,
        backend=env.backend,
    )


def legal_counts_by_slot(slot_ids: list[int], legal_rows_by_column: dict[int, np.ndarray]) -> dict[int, int]:
    """Count legal candidate rows for each slot id.

    ``slot_ids`` are slot identifiers and ``legal_rows_by_column`` maps columns
    to boolean row masks. The function returns ``slot_id -> count``, does not
    mutate inputs, and assumes every slot's column appears in the mask mapping.
    """

    return {
        int(slot_id): int(np.count_nonzero(legal_rows_by_column[int(slot_id // 2)]))
        for slot_id in slot_ids
    }


def replaceability_by_slot(slot_state_ids: np.ndarray, n: int) -> dict[int, int]:
    """Return replaceability counts for every filled slot in a state.

    ``slot_state_ids`` is an integer vector of shape ``[2n]`` and ``n`` is the
    grid size. For each filled slot, the function removes that slot's point and
    counts legal rows in the slot's own column. It returns ``slot_id -> count``,
    does not mutate the input, and assumes the represented state is legal.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64)
    filled_slots = np.flatnonzero(state != 0)
    all_points = state_ids_to_points(state)
    replaceability: dict[int, int] = {}
    for slot_id in filled_slots:
        slot_id_int = int(slot_id)
        point = (slot_id_int // 2, int(state[slot_id_int]) - 1)
        reduced_points = [candidate for candidate in all_points if candidate != point]
        column = slot_id_int // 2
        # Rebuild only the removed-point neighborhood so generation remask stays
        # independent from the mutable environment state.
        legal_rows = compute_existing_legal_rows_by_column(
            existing_points=reduced_points,
            n=n,
            candidate_columns=[column],
        )
        replaceability[slot_id_int] = int(np.count_nonzero(legal_rows[column]))
    return replaceability
