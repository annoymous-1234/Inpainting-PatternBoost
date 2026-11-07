from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TopPoolEntry:
    """Store one canonical legal construction in the PatternBoost pool.

    ``slot_state_ids`` is an ``int64`` array of shape ``[2n]`` where ``0`` is
    ``[MASK]`` and nonzero values are ``y + 1``. ``score`` is the filled-slot
    count, ``canonical_hash`` deduplicates canonical states, and ``metadata``
    carries optional provenance. The dataclass does not mutate external state.
    """

    slot_state_ids: np.ndarray
    score: int
    canonical_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeStepStats:
    """Record diagnostics from one PatternBoost iterative decode step.

    The fields store the step index, attempted proposal width, selected slots,
    accepted points, rejected points, and the point count before and after the
    environment step. The dataclass is a reporting container and does not
    mutate model or environment state.
    """

    step_index: int
    quota: int
    selected_slot_ids: list[int]
    accepted_points: list[tuple[int, int]]
    rejected_points: list[tuple[int, int]]
    point_count_before: int
    point_count_after: int


@dataclass
class GenerationResult:
    """Store the result of generating one PatternBoost candidate.

    ``seed_hash`` identifies the pool seed, ``remask_mode`` and ``mask_ratio``
    describe the generation perturbation, ``slot_state_ids`` stores the final
    canonical state, and ``step_stats`` contains decode diagnostics. The result
    does not own mutable environment state.
    """

    seed_hash: str
    remask_mode: str
    mask_ratio: float
    slot_state_ids: np.ndarray
    score: int
    inserted: bool
    seed_score: int = 0
    step_stats: list[DecodeStepStats] = field(default_factory=list)
