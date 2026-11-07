from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SearchState:
    """Represent one canonical AlphaZero search state.

    ``slot_state_ids`` is an ``int64`` array of shape ``[2n]`` whose values are
    ``0`` for ``[MASK]`` and ``y + 1`` for filled slots. ``episode_step`` is
    the number of executed joint actions in the real episode prefix. The
    dataclass stores only immutable state-by-convention and does not embed the
    mutable environment backend.
    """

    slot_state_ids: np.ndarray
    episode_step: int


@dataclass
class SearchModelOutput:
    """Hold one AlphaZero model evaluation for a search node.

    ``unified_logits`` is a ``float32`` array of shape ``[2n, n + 1]`` in raw
    logit space, and ``value`` is a scalar in ``[0, 1]`` after the AlphaZero
    adapter applies ``sigmoid`` to the model value head.
    """

    unified_logits: np.ndarray
    value: float


@dataclass
class JointAction:
    """Represent one raw sampled joint action before environment repair.

    ``active_slots`` is an ordered list of slot ids whose length is usually the
    configured root or non-root action width, and ``values`` maps each active
    slot to one proposed row ``y`` in ``[0, n - 1]``. The action is stored
    exactly as sampled: no deduplication, no canonicalization, and no sorting is
    imposed by the dataclass.
    """

    active_slots: list[int]
    values: dict[int, int]


@dataclass
class ExecutedAction:
    """Represent the repair-filtered action that was actually applied.

    ``active_slots`` is the ordered list of slots that survived duplicate
    resolution and line-conflict repair, and ``values`` stores the executed row
    for each surviving slot. The executed action may be empty when repair
    rejects every proposed slot.
    """

    active_slots: list[int]
    values: dict[int, int]


@dataclass
class SimulationResult:
    """Store the pure-function transition result for one joint action.

    ``next_state`` is the child search state, ``executed_action`` is the
    repair-filtered action, ``raw_reward_points`` is the number of accepted
    slots, and ``step_reward`` is the normalized step reward in ``[0, 1]``.
    ``done`` reports whether a natural terminal condition fired, and
    ``stop_reason`` is one of the documented terminal strings or ``None``.
    ``aux_info`` mirrors the low-level resolve/evaluate/repair outputs for
    debugging.
    """

    next_state: SearchState
    executed_action: ExecutedAction
    raw_reward_points: int
    step_reward: float
    done: bool
    stop_reason: str | None
    aux_info: dict[str, Any]


@dataclass
class SearchNode:
    """Represent one node in the sampled-action search tree.

    The node caches the AlphaZero model output, sampled action subset, visit
    statistics, legality masks, and child transitions for one ``SearchState``.
    Root nodes use ``candidate_slots`` and ``root_action_gumbels`` while
    non-root nodes use ``fixed_slots``.
    """

    state: SearchState
    expanded: bool = False
    unified_logits: np.ndarray | None = None
    value: float | None = None
    candidate_slots: list[int] = field(default_factory=list)
    fixed_slots: list[int] = field(default_factory=list)
    legal_rows_by_column: dict[int, np.ndarray] | None = None
    legal_row_counts_by_slot: dict[int, int] | None = None
    feasible_masked_slots: list[int] | None = None
    value_pools: dict[int, list[int]] = field(default_factory=dict)
    sampled_actions: list[JointAction] = field(default_factory=list)
    prior_logits: np.ndarray | None = None
    visit_counts: np.ndarray | None = None
    value_sums: np.ndarray | None = None
    child_nodes: dict[int, SearchNode] = field(default_factory=dict)
    executed_actions: list[ExecutedAction] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    stop_reasons: list[str | None] = field(default_factory=list)
    root_action_gumbels: np.ndarray | None = None


@dataclass
class RootSearchResult:
    """Store the full root-search output needed for acting and training."""

    root_node: SearchNode
    improved_policy: np.ndarray
    best_action_index: int
    completed_q: np.ndarray
    root_mcts_conflict_counts: dict[int, float] = field(default_factory=dict)


@dataclass
class EpisodeTransition:
    """Store one self-play root target before scalar value targets are attached."""

    slot_state_ids: np.ndarray
    episode_step: int
    candidate_slot_ids: list[int]
    fill_targets: list[float]
    value_pool_ids: list[list[int]]
    value_targets_on_support: list[list[float]]
    value_weights: list[float]
    step_reward: float
    terminal_score: float = 0.0
    terminal_reward: float = 0.0


@dataclass
class AlphaZeroSample:
    """Store one learner sample with policy marginals and scalar value target."""

    slot_state_ids: np.ndarray
    episode_step: int
    candidate_slot_ids: list[int]
    fill_targets: list[float]
    value_pool_ids: list[list[int]]
    value_targets_on_support: list[list[float]]
    value_weights: list[float]
    value_target: float
