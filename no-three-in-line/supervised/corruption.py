from __future__ import annotations

import random

from .config import CorruptionConfig
from .types import CorruptionResult, EncodedSolutionRecord


def mask_ratio_for_step(step_id: int, total_steps: int, q_min: float, q_max: float) -> float:
    """Return the step-dependent mask ratio used for corruption."""

    return q_min + (q_max - q_min) * step_id / max(total_steps - 1, 1)


def sample_step_id(rng: random.Random, config: CorruptionConfig) -> int:
    """Sample a denoising step id from the configured training range."""

    return rng.randint(config.sample_step_min, config.total_steps - 1)


def corrupt_clean_state_at_step(
    record: EncodedSolutionRecord,
    step_id: int,
    rng: random.Random,
    config: CorruptionConfig,
) -> CorruptionResult:
    """Corrupt one clean encoded solution at an explicit denoising step.

    ``record`` stores the clean slot state, ``step_id`` chooses the corruption
    level in ``[0, total_steps - 1]``, and ``rng`` drives any random masking
    decisions. The function returns a new ``CorruptionResult``, does not mutate
    the encoded record, and assumes only originally filled slots may be masked.
    """
    if not 0 <= step_id < config.total_steps:
        raise ValueError(f"step_id must be in [0, {config.total_steps - 1}]")

    q_t = mask_ratio_for_step(step_id, config.total_steps, config.q_min, config.q_max)
    filled_slot_ids = list(record.filled_slot_ids)
    num_filled = len(filled_slot_ids)

    if config.corruption_mode == "exact_ratio":
        num_mask = round(q_t * num_filled)
        masked_slot_ids = sorted(rng.sample(filled_slot_ids, num_mask)) if num_mask > 0 else []
    elif config.corruption_mode == "bernoulli":
        masked_slot_ids = sorted(slot_id for slot_id in filled_slot_ids if rng.random() < q_t)
    else:
        raise ValueError(f"unsupported corruption mode: {config.corruption_mode}")

    input_state_ids = list(record.clean_state_ids)
    for slot_id in masked_slot_ids:
        input_state_ids[slot_id] = 0

    current_point_count = sum(1 for state_id in input_state_ids if state_id != 0)
    masked_ratio_actual = len(masked_slot_ids) / max(num_filled, 1)
    return CorruptionResult(
        input_state_ids=input_state_ids,
        masked_slot_ids=masked_slot_ids,
        missing_slot_ids=list(masked_slot_ids),
        step_id=step_id,
        q_t=q_t,
        current_point_count=current_point_count,
        masked_ratio_actual=masked_ratio_actual,
    )


def corrupt_clean_state(
    record: EncodedSolutionRecord,
    rng: random.Random,
    config: CorruptionConfig,
) -> CorruptionResult:
    """Corrupt one clean encoded solution into a sampled noisy input state.

    ``record`` stores the clean slot state and ``rng`` samples the corruption
    step and any random masking choices. The function returns a
    ``CorruptionResult``, does not mutate the encoded record, and assumes the
    configured corruption mode only masks originally filled slots.
    """
    return corrupt_clean_state_at_step(record, sample_step_id(rng, config), rng, config)
