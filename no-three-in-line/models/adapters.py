from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .decoding import logits_to_proposal_slots


def slot_states_to_state_ids(slot_states: Mapping[int, Any], n: int) -> Tensor:
    """Convert environment slot states into unified discrete token ids.

    ``slot_states`` is the environment's ``dict[int, SlotState]`` mapping and
    ``n`` is the grid size. The function returns a CPU ``LongTensor`` of shape
    ``[2n]`` whose values are ``0`` for masked slots and ``y + 1`` for filled
    slots. It does not mutate the input mapping and assumes slot ids follow the
    column-pair convention ``slot_id // 2 == x``.
    """
    num_tokens = 2 * n
    if len(slot_states) != num_tokens:
        raise ValueError(f"expected exactly {num_tokens} slot states, got {len(slot_states)}")

    state_ids = torch.zeros(num_tokens, dtype=torch.long)
    for slot_id in range(num_tokens):
        if slot_id not in slot_states:
            raise ValueError(f"missing slot state for slot_id={slot_id}")

        slot_state = slot_states[slot_id]
        actual_slot_id = getattr(slot_state, "slot_id", None)
        status = getattr(slot_state, "status", None)
        point = getattr(slot_state, "point", None)

        if actual_slot_id != slot_id:
            raise ValueError(f"slot state key {slot_id} does not match slot_state.slot_id={actual_slot_id}")

        if status == "MASKED":
            if point is not None:
                raise ValueError(f"masked slot {slot_id} must have point=None")
            state_ids[slot_id] = 0
            continue

        if status != "FILLED":
            raise ValueError(f"invalid slot status for slot {slot_id}: {status}")
        if point is None:
            raise ValueError(f"filled slot {slot_id} must have a point")

        x, y = point
        if slot_id // 2 != x:
            raise ValueError(f"slot_id {slot_id} is inconsistent with point x-coordinate {x}")
        if not (0 <= y < n):
            raise ValueError(f"point y-coordinate out of range for slot {slot_id}: {y}")

        state_ids[slot_id] = y + 1

    return state_ids


def batch_slot_states_to_state_ids(
    slot_states_list: Sequence[Mapping[int, Any]],
    n: int,
) -> Tensor:
    """Batch :func:`slot_states_to_state_ids` over multiple environment states.

    ``slot_states_list`` is a sequence of environment slot-state mappings and
    ``n`` is the shared grid size. The function returns a CPU ``LongTensor`` of
    shape ``[B, 2n]``, does not mutate the inputs, and assumes every batch item
    uses the same slot layout.
    """
    return torch.stack([slot_states_to_state_ids(slot_states, n) for slot_states in slot_states_list], dim=0)


def current_point_count_from_slot_states(slot_states: Mapping[int, Any]) -> int:
    """Count how many slots are currently filled in one environment state.

    ``slot_states`` is the environment slot-state mapping. The function returns
    an integer count, does not mutate the input mapping, and assumes filled
    slots use the status string ``"FILLED"``.
    """
    count = 0
    for slot_state in slot_states.values():
        if getattr(slot_state, "status", None) == "FILLED":
            count += 1
    return count


def batch_current_point_counts(slot_states_list: Sequence[Mapping[int, Any]]) -> Tensor:
    """Batch filled-slot counts into a float tensor for the global token.

    ``slot_states_list`` is a sequence of environment slot-state mappings. The
    function returns a CPU ``float32`` tensor of shape ``[B]``, does not mutate
    the inputs, and assumes each batch item follows the same slot-state
    conventions as the environment.
    """
    return torch.tensor(
        [current_point_count_from_slot_states(slot_states) for slot_states in slot_states_list],
        dtype=torch.float32,
    )
