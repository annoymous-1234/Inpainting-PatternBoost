"""Typed containers for environment state, snapshots, and rollout records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EnvSnapshot:
    """Serializable environment snapshot for branching and rollback.

    Attributes
    ----------
    active_mask:
        Boolean array of shape `[K]`. `True` marks an active slot.
    centers:
        Float array of shape `[K, 2]`. Each row stores `(x, y)` for one slot.
    step_idx:
        Number of completed environment steps.
    radii/circles/layout_score/lp_status:
        Terminal LP cache fields. They are `None` until all slots are active.
    rng_state:
        Bit-generator state dictionary for deterministic restore.
    """

    active_mask: np.ndarray
    centers: np.ndarray
    step_idx: int
    radii: np.ndarray | None
    circles: np.ndarray | None
    layout_score: float | None
    lp_status: str | None
    rng_state: dict[str, Any]


@dataclass
class RoundCache:
    """Temporary candidate cache used inside one environment step.

    This cache only exists during the greedy activation loop for one step.

    Attributes
    ----------
    cand_ids:
        Integer array of shape `[M]` holding slot ids for currently masked candidates.
    cand_centers:
        Float array of shape `[M, 2]` with candidate centers from the resolved proposal.
    alive:
        Boolean array of shape `[M]`. `True` means this candidate is still eligible in the
        current commit loop.
    """

    cand_ids: np.ndarray
    cand_centers: np.ndarray
    alive: np.ndarray


@dataclass
class CommitDecision:
    """One accepted candidate produced by the commit loop."""

    slot_id: int
    center: np.ndarray


@dataclass
class Transition:
    """One rollout transition collected from the environment."""

    observation: dict[str, Any]
    proposal: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]
