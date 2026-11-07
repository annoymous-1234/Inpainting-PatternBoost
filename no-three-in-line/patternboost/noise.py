from __future__ import annotations

import numpy as np
import torch

from supervised.losses import compute_token_weights
from supervised.types import SupervisedSample

from .state import filled_count


def sample_mask_ratio(mask_ratios: list[float], rng: np.random.Generator) -> float:
    """Sample one configured training-noise mask ratio.

    ``mask_ratios`` contains probabilities in ``[0, 1]`` and ``rng`` supplies
    deterministic sampling. The function returns a float, does not mutate the
    list, and assumes validation has already ensured the list is non-empty.
    """

    return float(mask_ratios[int(rng.integers(0, len(mask_ratios)))])


def random_noise_slots(
    clean_state_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> list[int]:
    """Select filled slots to mask for PatternBoost denoising training.

    ``clean_state_ids`` has shape ``[2n]`` and ``ratio`` controls the fraction
    of filled slots masked. The function returns sorted slot ids, does not
    mutate the clean state, and assumes only filled slots may be noised.
    """

    state = np.asarray(clean_state_ids, dtype=np.int64)
    filled_slots = np.flatnonzero(state != 0)
    num_mask = int(round(float(ratio) * filled_slots.size))
    if num_mask <= 0:
        return []
    chosen = rng.choice(filled_slots, size=min(num_mask, filled_slots.size), replace=False)
    return sorted(int(slot_id) for slot_id in chosen.tolist())


def build_noised_state(
    clean_state_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Return a noised copy of one clean state plus masked slot ids.

    ``clean_state_ids`` is an integer vector of shape ``[2n]`` and ``ratio``
    controls the filled-slot mask fraction. The function returns
    ``(input_state_ids, masked_slot_ids)``, does not mutate the clean state, and
    assumes ``0`` is the only mask token.
    """

    input_state_ids = np.asarray(clean_state_ids, dtype=np.int64).copy()
    masked_slot_ids = random_noise_slots(input_state_ids, ratio, rng)
    input_state_ids[masked_slot_ids] = 0
    return input_state_ids, masked_slot_ids


def build_supervised_sample_from_clean(
    clean_state_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    loss_config,
) -> SupervisedSample:
    """Build one PatternBoost sample in the supervised-training format.

    ``clean_state_ids`` is the canonical clean state, ``ratio`` selects the
    noise level, ``rng`` samples masked filled slots, and ``loss_config``
    supplies token weights. The function returns a ``SupervisedSample``, does
    not mutate the clean state, and supervises only masked non-empty targets.
    """

    input_state_ids, masked_slot_ids = build_noised_state(clean_state_ids, ratio, rng)
    input_tensor = torch.tensor(input_state_ids, dtype=torch.long)
    target_tensor = torch.tensor(np.asarray(clean_state_ids, dtype=np.int64), dtype=torch.long)
    supervision_mask = torch.logical_and(input_tensor.eq(0), target_tensor.ne(0)).to(dtype=torch.float32)
    return SupervisedSample(
        input_state_ids=input_tensor,
        target_state_ids=target_tensor,
        step_id=torch.tensor(0, dtype=torch.long),
        current_point_count=torch.tensor(float(filled_count(input_state_ids)), dtype=torch.float32),
        token_weights=compute_token_weights(target_tensor, loss_config),
        supervision_mask=supervision_mask,
        task_type="full",
        metadata={
            "task_type": "patternboost_full",
            "mask_ratio": float(ratio),
            "masked_slot_ids": list(masked_slot_ids),
            "current_point_count": filled_count(input_state_ids),
        },
    )
