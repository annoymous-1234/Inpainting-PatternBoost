from __future__ import annotations

import random
from typing import Any

import numpy as np

from env import NoThreeInLineEnv
from proposal_eval import CandidatePoint, ProposalSlot, compute_existing_legal_rows_by_column


def _rng_choice(sequence: list[int], rng: Any) -> int:
    """Return one item sampled from ``sequence`` using Python or NumPy RNGs."""

    if hasattr(rng, "choice"):
        return int(rng.choice(sequence))
    return int(rng.choice(sequence))


def _rng_shuffle(sequence: list[int], rng: Any) -> None:
    """Shuffle ``sequence`` in place using Python or NumPy RNGs."""

    if hasattr(rng, "shuffle"):
        rng.shuffle(sequence)
    else:
        random.shuffle(sequence)


def _masked_slot_ids(env: NoThreeInLineEnv) -> list[int]:
    """Return currently masked environment slot ids in ascending order."""

    return [
        int(slot_id)
        for slot_id, slot_state in env.slot_states.items()
        if slot_state.status == "MASKED"
    ]


def _legal_rows_by_column(env: NoThreeInLineEnv, slot_ids: list[int]) -> dict[int, np.ndarray]:
    """Return legal rows for the columns touched by ``slot_ids``."""

    columns = sorted({slot_id // 2 for slot_id in slot_ids})
    return compute_existing_legal_rows_by_column(
        existing_points=env.backend.get_selected_points(),
        n=env.n,
        candidate_columns=columns,
        backend=env.backend,
    )


def greedy_saturate(
    env: NoThreeInLineEnv,
    rng: Any,
    max_steps: int | None = None,
    slot_order: str = "mrv",
    y_order: str = "random",
) -> dict[str, object]:
    """Greedily fill legal masked slots until no immediate fill is available.

    ``env`` is mutated through normal ``env.step`` calls, ``rng`` provides
    deterministic random choices, ``max_steps`` optionally caps fill attempts,
    and the order arguments choose MRV/random slot order plus random/ascending
    row order. The function returns accepted points and step count, mutates only
    the environment, and assumes each step proposes one legal candidate.
    """

    if slot_order not in {"mrv", "random"}:
        raise ValueError("slot_order must be 'mrv' or 'random'")
    if y_order not in {"random", "ascending"}:
        raise ValueError("y_order must be 'random' or 'ascending'")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be >= 0 when set")

    steps_used = 0
    accepted_points: list[tuple[int, int]] = []
    step_limit = env.num_slots if max_steps is None else max_steps

    while steps_used < step_limit:
        masked_slots = _masked_slot_ids(env)
        if not masked_slots:
            break

        legal_rows = _legal_rows_by_column(env, masked_slots)
        feasible_slots = [
            slot_id
            for slot_id in masked_slots
            if int(np.count_nonzero(legal_rows[slot_id // 2])) > 0
        ]
        if not feasible_slots:
            break

        if slot_order == "mrv":
            feasible_slots.sort(key=lambda slot_id: (int(np.count_nonzero(legal_rows[slot_id // 2])), slot_id))
            chosen_slot = feasible_slots[0]
        else:
            chosen_slot = _rng_choice(feasible_slots, rng)

        candidate_rows = [int(row) for row in np.flatnonzero(legal_rows[chosen_slot // 2]).tolist()]
        if y_order == "random":
            _rng_shuffle(candidate_rows, rng)
        chosen_y = candidate_rows[0]

        _, _, _, info = env.step(
            [
                ProposalSlot(
                    slot_id=chosen_slot,
                    candidates=[
                        CandidatePoint(
                            x=chosen_slot // 2,
                            y=chosen_y,
                            score=1.0,
                            slot_id=chosen_slot,
                        )
                    ],
                )
            ]
        )
        steps_used += 1
        if not info["accepted_proposal_points"]:
            break
        accepted_points.extend(info["accepted_proposal_points"])

    return {
        "steps_used": steps_used,
        "accepted_points": sorted(accepted_points),
        "final_point_count": int(env.observe()["num_selected"]),
    }
