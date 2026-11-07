"""Typed containers for circle-packing PatternBoost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CircleTopPoolEntry:
    """One full center layout retained in TopPool with LP-derived score cache."""

    centers: np.ndarray
    radii: np.ndarray
    circles: np.ndarray
    active_mask: np.ndarray
    score: float
    lp_status: str
    canonical_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditSet:
    """One PatternBoost edit-set sample."""

    edit_mask: np.ndarray
    policy: str
    ratio: float
    query_point: tuple[float, float] | None = None


@dataclass
class PatternBoostCandidate:
    """Generated and optionally polished PatternBoost candidate."""

    centers: np.ndarray
    radii: np.ndarray | None
    circles: np.ndarray
    active_mask: np.ndarray
    raw_score: float | None
    polish_score: float | None = None
    lp_status: str | None = None
    legal: bool = False
    full_active: bool = False
    inserted: bool = False
    inference_center_distances: np.ndarray | None = None
    inference_center_distance_count: int = 0
    inference_center_distance_mean: float | None = None
    inference_center_distance_max: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Diagnostics for one generated PatternBoost candidate."""

    seed_hash: str
    generation_mode: str
    edit_set: EditSet
    candidate: PatternBoostCandidate
    records: list[Any] = field(default_factory=list)


@dataclass
class GenerationStats:
    """Aggregate metrics for one PatternBoost generation."""

    generation: int
    best_before: float
    best_after: float
    num_candidates: int
    num_inserted: int
    num_solver_polished: int
