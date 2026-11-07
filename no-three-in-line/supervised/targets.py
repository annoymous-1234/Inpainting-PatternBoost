from __future__ import annotations

import random

from .config import PartialTargetConfig
from .types import CorruptionResult, EncodedSolutionRecord


def build_full_target(record: EncodedSolutionRecord) -> list[int]:
    """Return the full clean-state target for one encoded solution."""

    return list(record.clean_state_ids)


def restore_count_for_step(
    step_id: int,
    missing_count: int,
    config: PartialTargetConfig,
    total_steps: int,
) -> int:
    """Return how many missing slots to restore for a partial target."""

    if config.restore_schedule_type == "fixed":
        return min(config.restore_k_fixed, missing_count)

    progress = step_id / max(total_steps - 1, 1)
    raw_k = config.restore_k_min + progress * (config.restore_k_max - config.restore_k_min)
    restore_k = round(raw_k)
    restore_k = max(config.restore_k_min, min(config.restore_k_max, restore_k))
    return min(restore_k, missing_count)


def choose_order_id(
    record: EncodedSolutionRecord,
    rng: random.Random,
    use_multiple_orders: bool,
) -> int:
    """Choose which stored fixed slot order to use for a partial target."""

    if not record.slot_orders:
        raise ValueError("encoded solution does not contain any slot orders")
    if not use_multiple_orders or len(record.slot_orders) == 1:
        return 0
    return rng.randrange(len(record.slot_orders))


def select_restore_slots(
    missing_slot_ids: list[int],
    slot_order: list[int],
    restore_k_eff: int,
) -> list[int]:
    """Select restore slots by scanning a stored slot order."""

    missing_set = set(missing_slot_ids)
    restore_slots: list[int] = []
    for slot_id in slot_order:
        if slot_id in missing_set:
            restore_slots.append(slot_id)
            if len(restore_slots) >= restore_k_eff:
                break
    return restore_slots


def build_partial_target(
    record: EncodedSolutionRecord,
    corruption_result: CorruptionResult,
    order_id: int,
    config: PartialTargetConfig,
    total_steps: int,
) -> tuple[list[int], dict[str, object]]:
    """Build a one-step-more-denoised partial target from a noisy state."""

    missing_slot_ids = sorted(
        slot_id
        for slot_id, clean_state_id in enumerate(record.clean_state_ids)
        if clean_state_id != 0 and corruption_result.input_state_ids[slot_id] == 0
    )
    restore_k_eff = restore_count_for_step(
        step_id=corruption_result.step_id,
        missing_count=len(missing_slot_ids),
        config=config,
        total_steps=total_steps,
    )
    restore_slots = select_restore_slots(missing_slot_ids, record.slot_orders[order_id], restore_k_eff)

    target_state_ids = list(corruption_result.input_state_ids)
    for slot_id in restore_slots:
        target_state_ids[slot_id] = record.clean_state_ids[slot_id]

    metadata = {
        "order_id": order_id,
        "restore_k_eff": restore_k_eff,
    }
    return target_state_ids, metadata
