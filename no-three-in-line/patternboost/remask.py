from __future__ import annotations

import numpy as np

from .legal import replaceability_by_slot


def random_generation_remask(
    slot_state_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Mask a random subset of filled slots for candidate generation.

    ``slot_state_ids`` is an integer vector of shape ``[2n]`` and ``ratio``
    controls the filled-slot fraction to remove. The function returns a copied
    remasked state and sorted removed slot ids, does not mutate the input, and
    assumes only filled slots are eligible.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64).copy()
    filled_slots = np.flatnonzero(state != 0)
    num_remask = int(round(float(ratio) * filled_slots.size))
    if num_remask <= 0:
        return state, []
    chosen = rng.choice(filled_slots, size=min(num_remask, filled_slots.size), replace=False)
    remasked_slots = sorted(int(slot_id) for slot_id in chosen.tolist())
    state[remasked_slots] = 0
    return state, remasked_slots


def replaceability_generation_remask(
    slot_state_ids: np.ndarray,
    n: int,
    ratio: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Mask high-replaceability filled slots for candidate generation.

    ``slot_state_ids`` is an integer vector of shape ``[2n]`` and ``ratio``
    controls how many filled slots are removed. ``rng`` is accepted for API
    symmetry but only affects deterministic tie jitter when provided. The
    function returns a copied remasked state and removed slot ids, does not
    mutate the input, and assumes high replaceability is preferred.
    """

    state = np.asarray(slot_state_ids, dtype=np.int64).copy()
    filled_slots = np.flatnonzero(state != 0)
    num_remask = int(round(float(ratio) * filled_slots.size))
    if num_remask <= 0:
        return state, []
    replaceability = replaceability_by_slot(state, n)
    jitter = {
        int(slot_id): 0.0 if rng is None else float(rng.normal(0.0, 1e-6))
        for slot_id in filled_slots
    }
    ranked_slots = sorted(
        (int(slot_id) for slot_id in filled_slots),
        key=lambda slot_id: (-(replaceability.get(slot_id, 0) + jitter[slot_id]), slot_id),
    )
    remasked_slots = sorted(ranked_slots[: min(num_remask, len(ranked_slots))])
    state[remasked_slots] = 0
    return state, remasked_slots


def generation_remask(
    slot_state_ids: np.ndarray,
    n: int,
    ratio: float,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Dispatch PatternBoost generation remask by configured mode.

    ``mode`` is ``"random"`` or ``"replaceability"``. The function returns a
    copied remasked state and removed slot ids, does not mutate the input, and
    assumes validation has restricted mode names.
    """

    if mode == "random":
        return random_generation_remask(slot_state_ids, ratio, rng)
    if mode == "replaceability":
        return replaceability_generation_remask(slot_state_ids, n, ratio, rng)
    raise ValueError(f"unsupported generation remask mode: {mode}")
