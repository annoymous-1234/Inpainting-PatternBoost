from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from proposal_eval import (
    CandidatePoint,
    ProposalSlot,
    compute_existing_legal_rows_by_column,
    collect_existing_slot_line_conflict_counts,
    search_repaired_proposals,
)

from .config import AlphaZeroConfig, reward_mixture_weights
from .model_adapter import evaluate_search_state
from .state import (
    apply_executed_action_to_slot_state,
    filled_count,
    slot_state_ids_to_points_and_assignment,
)
from .targets import terminal_reward_from_points
from .types import (
    ExecutedAction,
    JointAction,
    RootSearchResult,
    SearchModelOutput,
    SearchNode,
    SearchState,
    SimulationResult,
)


def sample_gumbel(shape: Sequence[int], rng: np.random.Generator, eps: float = 1e-9) -> np.ndarray:
    """Sample ``Gumbel(0, 1)`` noise as a ``float64`` NumPy array."""

    uniform = rng.uniform(low=eps, high=1.0 - eps, size=shape)
    return -np.log(-np.log(uniform))


def gumbel_topk(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return top-``k`` indices after adding fresh Gumbel noise to raw scores."""

    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    stable_scores = scores - np.max(scores)
    perturbed = stable_scores + sample_gumbel(stable_scores.shape, rng)
    return np.argsort(-perturbed)[:k].astype(np.int64)


def gumbel_argmax(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Return one argmax index after adding fresh Gumbel noise to raw scores."""

    return int(gumbel_topk(scores, 1, rng)[0])


def _conditional_value_entropies(value_logits: np.ndarray) -> np.ndarray:
    """Return conditional entropies ``H(y | fill)`` in nats for slot logits.

    ``value_logits`` has shape ``[num_slots, n]`` or ``[n]`` and contains only
    non-``[MASK]`` logits. The returned array has shape ``[num_slots]`` or is a
    scalar-shaped NumPy array, and each entry is the entropy of the full
    conditional value distribution computed with a softmax over all ``y`` rows.
    """

    logits = np.asarray(value_logits, dtype=np.float64)
    stable = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(stable)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=-1)


def compute_slot_select_scores(
    mask_logits: np.ndarray,
    value_logits: np.ndarray,
    legal_row_counts: np.ndarray,
    tau: float,
    coef_mask_logit: float,
    coef_scarcity: float,
    coef_entropy: float,
) -> np.ndarray:
    """Return raw slot-selection scores with legality-aware optional bonuses.

    ``mask_logits`` has shape ``[num_slots]``, ``value_logits`` has shape
    ``[num_slots, n]``, and ``legal_row_counts`` has shape ``[num_slots]``.
    Scores are computed as:
    ``w_mask * (-mask_logit / tau) + w_scarcity * (1 / num_legal_rows) - w_entropy * H(y | fill)``.
    The entropy term always uses the full non-mask conditional distribution over
    all rows, not the legality-filtered support.
    """

    safe_legal_counts = np.maximum(np.asarray(legal_row_counts, dtype=np.float64), 1.0)
    scores = (
        float(coef_mask_logit) * (-np.asarray(mask_logits, dtype=np.float64) / float(tau))
        + float(coef_scarcity) * (1.0 / safe_legal_counts)
        - float(coef_entropy) * _conditional_value_entropies(value_logits)
    )
    return np.asarray(scores, dtype=np.float64)


def _compute_legality_cache(
    slot_state_ids: np.ndarray,
    n: int,
    use_existing_action_mask: bool,
) -> tuple[dict[int, np.ndarray], dict[int, int], list[int]]:
    """Compute node-local legality cache from one unified slot-state vector."""

    slot_state_ids = np.asarray(slot_state_ids, dtype=np.int64)
    masked_slot_ids = np.flatnonzero(slot_state_ids == 0)
    candidate_columns = sorted({int(slot_id // 2) for slot_id in masked_slot_ids})
    if use_existing_action_mask:
        existing_points, _ = slot_state_ids_to_points_and_assignment(slot_state_ids)
        legal_rows_by_column = compute_existing_legal_rows_by_column(
            existing_points=set(existing_points),
            n=n,
            candidate_columns=candidate_columns,
        )
    else:
        legal_rows_by_column = {
            int(column): np.ones(n, dtype=bool)
            for column in candidate_columns
        }

    legal_row_counts_by_slot: dict[int, int] = {}
    feasible_masked_slots: list[int] = []
    for slot_id in masked_slot_ids:
        column = int(slot_id // 2)
        legal_count = int(np.count_nonzero(legal_rows_by_column[column]))
        legal_row_counts_by_slot[int(slot_id)] = legal_count
        if legal_count > 0:
            feasible_masked_slots.append(int(slot_id))
    return legal_rows_by_column, legal_row_counts_by_slot, feasible_masked_slots


def _ensure_node_legality_cache(node: SearchNode, config: AlphaZeroConfig) -> None:
    """Populate one node's legality cache once and reuse it thereafter."""

    if (
        node.legal_rows_by_column is not None
        and node.legal_row_counts_by_slot is not None
        and node.feasible_masked_slots is not None
    ):
        return
    (
        node.legal_rows_by_column,
        node.legal_row_counts_by_slot,
        node.feasible_masked_slots,
    ) = _compute_legality_cache(
        np.asarray(node.state.slot_state_ids, dtype=np.int64),
        config.n,
        config.search.use_existing_action_mask,
    )


def count_feasible_masked_slots(slot_state_ids: np.ndarray, config: AlphaZeroConfig) -> int:
    """Count masked slots that still have at least one existing-legal row."""

    _, _, feasible_masked_slots = _compute_legality_cache(
        np.asarray(slot_state_ids, dtype=np.int64),
        config.n,
        config.search.use_existing_action_mask,
    )
    return len(feasible_masked_slots)


def sample_root_candidate_slots(
    mask_logits: np.ndarray,
    value_logits: np.ndarray,
    feasible_slot_ids: np.ndarray,
    legal_row_counts_by_slot: dict[int, int],
    J_root_prime: int,
    tau_slot_root: float,
    config: AlphaZeroConfig,
    rng: np.random.Generator,
) -> list[int]:
    """Sample the root slot candidate pool with Gumbel top-``J_root_prime``."""

    if feasible_slot_ids.size == 0:
        return []
    legal_row_counts = np.asarray(
        [legal_row_counts_by_slot[int(slot_id)] for slot_id in feasible_slot_ids],
        dtype=np.float64,
    )
    slot_scores = compute_slot_select_scores(
        mask_logits=mask_logits[feasible_slot_ids],
        value_logits=value_logits[feasible_slot_ids],
        legal_row_counts=legal_row_counts,
        tau=tau_slot_root,
        coef_mask_logit=config.search.slot_score_coef_mask_logit,
        coef_scarcity=config.search.slot_score_coef_scarcity,
        coef_entropy=config.search.slot_score_coef_entropy,
    )
    topk_local = gumbel_topk(slot_scores, min(J_root_prime, feasible_slot_ids.size), rng)
    return [int(feasible_slot_ids[index]) for index in topk_local]


def sample_value_pools_for_slots(
    value_logits: np.ndarray,
    slot_ids: Sequence[int],
    L: int,
    tau_val: float,
    legal_rows_by_column: dict[int, np.ndarray] | None,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    """Sample per-slot value candidate pools with Gumbel top-``L`` over rows."""

    pools: dict[int, list[int]] = {}
    for slot_id in slot_ids:
        if legal_rows_by_column is not None:
            legal_rows = np.flatnonzero(np.asarray(legal_rows_by_column[int(slot_id // 2)], dtype=bool))
        else:
            legal_rows = np.arange(value_logits.shape[1], dtype=np.int64)
        if legal_rows.size == 0:
            pools[int(slot_id)] = []
            continue
        value_scores = np.asarray(value_logits[slot_id, legal_rows], dtype=np.float64) / float(tau_val)
        topk = gumbel_topk(value_scores, min(L, value_scores.shape[0]), rng)
        pools[int(slot_id)] = [int(legal_rows[index]) for index in topk]
    return pools


def sample_root_joint_actions(
    candidate_slots: Sequence[int],
    value_pools: dict[int, list[int]],
    mask_logits: np.ndarray,
    value_logits: np.ndarray,
    legal_row_counts_by_slot: dict[int, int],
    J_root: int,
    K_root: int,
    tau_slot_root: float,
    tau_val_root: float,
    config: AlphaZeroConfig,
    rng: np.random.Generator,
) -> list[JointAction]:
    """Sample raw root joint actions from the restricted slot/value pools."""

    candidate_slots_array = np.asarray(candidate_slots, dtype=np.int64)
    actions: list[JointAction] = []
    if candidate_slots_array.size == 0:
        return actions

    legal_row_counts = np.asarray(
        [legal_row_counts_by_slot[int(slot_id)] for slot_id in candidate_slots_array],
        dtype=np.float64,
    )
    slot_scores = compute_slot_select_scores(
        mask_logits=mask_logits[candidate_slots_array],
        value_logits=value_logits[candidate_slots_array],
        legal_row_counts=legal_row_counts,
        tau=tau_slot_root,
        coef_mask_logit=config.search.slot_score_coef_mask_logit,
        coef_scarcity=config.search.slot_score_coef_scarcity,
        coef_entropy=config.search.slot_score_coef_entropy,
    )
    for _ in range(K_root):
        selected_local = gumbel_topk(slot_scores, min(J_root, candidate_slots_array.size), rng)
        active_slots = [int(candidate_slots_array[index]) for index in selected_local]
        values: dict[int, int] = {}
        for slot_id in active_slots:
            candidate_rows = np.asarray(value_pools[slot_id], dtype=np.int64)
            if candidate_rows.size == 0:
                values = {}
                break
            row_scores = np.asarray(value_logits[slot_id, candidate_rows], dtype=np.float64) / float(tau_val_root)
            values[slot_id] = int(candidate_rows[gumbel_argmax(row_scores, rng)])
        if len(values) == len(active_slots):
            actions.append(JointAction(active_slots=active_slots, values=values))
    return actions


def compute_root_action_prior_logits(
    actions: Sequence[JointAction],
    mask_logits: np.ndarray,
    value_logits: np.ndarray,
) -> np.ndarray:
    """Compute raw root action prior logits on the sampled action subset.

    The returned array has shape ``[K]`` with entries
    ``sum_i [value_logit_i(y_i) - mask_logit_i]``. The scores remain in raw
    logit space and are not normalized.
    """

    priors = []
    for action in actions:
        prior = 0.0
        for slot_id in action.active_slots:
            prior += float(value_logits[slot_id, action.values[slot_id]] - mask_logits[slot_id])
        priors.append(prior)
    return np.asarray(priors, dtype=np.float64)


def compute_nonroot_action_prior_logits(actions: Sequence[JointAction], value_logits: np.ndarray) -> np.ndarray:
    """Compute raw non-root action priors with fixed slot subset ``I_s``."""

    priors = []
    for action in actions:
        prior = 0.0
        for slot_id in action.active_slots:
            prior += float(value_logits[slot_id, action.values[slot_id]])
        priors.append(prior)
    return np.asarray(priors, dtype=np.float64)


def compute_completed_q(visit_counts: np.ndarray, value_sums: np.ndarray, default_value: float) -> np.ndarray:
    """Return completed Q values by filling unvisited actions with ``default_value``."""

    visit_counts = np.asarray(visit_counts, dtype=np.int64)
    value_sums = np.asarray(value_sums, dtype=np.float64)
    return np.where(visit_counts > 0, value_sums / np.maximum(visit_counts, 1), float(default_value))


def sigma_q(q_values: np.ndarray, c_Q: float) -> np.ndarray:
    """Map ``Q`` values in ``[0, 1]`` to bounded score bonuses in ``[-c_Q, c_Q]``."""

    return float(c_Q) * (2.0 * np.asarray(q_values, dtype=np.float64) - 1.0)


def compute_improved_policy_logits(
    prior_logits: np.ndarray,
    completed_q: np.ndarray,
    tau_pi: float,
    c_Q: float,
) -> np.ndarray:
    """Return sampled-subset improved policy logits ``prior / tau + sigma_q(Q)``."""

    return np.asarray(prior_logits, dtype=np.float64) / float(tau_pi) + sigma_q(completed_q, c_Q)


def softmax_over_sampled_actions(logits: np.ndarray) -> np.ndarray:
    """Return a probability distribution over the sampled action subset."""

    if logits.size == 0:
        return logits.astype(np.float64)
    stable = logits - np.max(logits)
    probs = np.exp(stable)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return np.full_like(stable, fill_value=1.0 / len(stable), dtype=np.float64)
    return probs / probs_sum


def joint_action_to_proposal_slots(action: JointAction) -> list[ProposalSlot]:
    """Convert one raw joint action into env-compatible proposal slots.

    Each proposal slot contains exactly one candidate point because AlphaZero
    v1 samples exactly one ``y`` value per active slot and fixes
    ``num_search_attempts=1``.
    """

    proposal_slots: list[ProposalSlot] = []
    for slot_id in action.active_slots:
        proposal_slots.append(
            ProposalSlot(
                slot_id=slot_id,
                candidates=[
                    CandidatePoint(
                        x=slot_id // 2,
                        y=action.values[slot_id],
                        score=None,
                        slot_id=slot_id,
                    )
                ],
            )
        )
    return proposal_slots


def simulate_action_from_state(
    state: SearchState,
    action: JointAction,
    n: int,
    config: AlphaZeroConfig,
) -> SimulationResult:
    """Run one pure-function AlphaZero transition from ``state`` with ``action``.

    ``state.slot_state_ids`` is an ``int64`` vector of shape ``[2n]`` and
    ``action`` is a raw proposal action sampled by root or non-root search. The
    function resolves, evaluates, repairs, and applies the action without using
    the mutable environment backend, then returns the child state,
    repair-filtered ``executed_action``, normalized step reward, and terminal
    metadata. The transition always uses ``allow_existing_deletions=False`` and
    ``num_search_attempts=1``.
    """

    existing_points, point_to_slot = slot_state_ids_to_points_and_assignment(state.slot_state_ids)
    search_result = search_repaired_proposals(
        joint_action_to_proposal_slots(action),
        set(existing_points),
        existing_point_to_slot=dict(point_to_slot),
        num_search_attempts=1,
        allow_existing_deletions=False,
    )
    final_resolved = list(search_result["final_resolved_proposals"])
    executed_slots = [int(proposal.slot_id) for proposal in final_resolved]
    executed_values = {int(proposal.slot_id): int(proposal.point[1]) for proposal in final_resolved}
    executed_action = ExecutedAction(active_slots=executed_slots, values=executed_values)
    next_slot_state_ids = apply_executed_action_to_slot_state(state.slot_state_ids, executed_action, n)
    next_state = SearchState(slot_state_ids=next_slot_state_ids, episode_step=state.episode_step + 1)
    raw_reward_points = len(executed_slots)
    step_reward = raw_reward_points / float(2 * n)

    stop_reason: str | None = None
    if filled_count(next_slot_state_ids) == 2 * n:
        stop_reason = "filled_all_slots"
    elif count_feasible_masked_slots(next_slot_state_ids, config) < config.search.J_nonroot:
        stop_reason = "insufficient_feasible_slots"
    elif config.search.terminate_on_no_progress and (
        not executed_slots or np.array_equal(next_slot_state_ids, state.slot_state_ids)
    ):
        stop_reason = "no_progress"
    elif next_state.episode_step >= config.self_play.max_episode_steps:
        stop_reason = "episode_step_limit"

    return SimulationResult(
        next_state=next_state,
        executed_action=executed_action,
        raw_reward_points=raw_reward_points,
        step_reward=step_reward,
        done=stop_reason is not None,
        stop_reason=stop_reason,
        aux_info=search_result,
    )


def get_sequence_of_considered_visits(max_num_considered_actions: int, num_simulations: int) -> tuple[int, ...]:
    """Return the Sequential Halving considered-visit schedule.

    This is a NumPy-free port of the ``mctx`` utility used by Gumbel AlphaZero.
    The returned tuple has length ``num_simulations`` and each entry is the
    visit count that is currently eligible for selection.
    """

    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence: list[int] = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for index in range(num_considered):
                visits[index] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


def get_considered_visit_sequence(num_actions: int, root_budget: int) -> list[int]:
    """Return the considered-visit schedule for one root search call."""

    return list(get_sequence_of_considered_visits(num_actions, root_budget))


def _make_empty_child_node(next_state: SearchState) -> SearchNode:
    """Construct one unevaluated child node for a cached transition."""

    return SearchNode(state=next_state)


def _accumulate_root_mcts_conflict_counts(
    running_counts: dict[int, float],
    conflict_report: dict[str, object],
    root_active_slot_ids: set[int],
    num_actions: int,
) -> None:
    """Accumulate uniform root-level MCTS conflict mass from one sampled action.

    ``running_counts`` is mutated in place. ``conflict_report`` is the one-step
    search payload for a sampled root action, ``root_active_slot_ids`` is the
    set of slots that were already filled in the root state, and ``num_actions``
    is the total number of sampled root actions used for the uniform average.
    """

    if num_actions <= 0 or not root_active_slot_ids:
        return

    action_counts = collect_existing_slot_line_conflict_counts(
        conflict_report,
        allowed_slot_ids=root_active_slot_ids,
    )
    weight = 1.0 / float(num_actions)
    for slot_id, count in action_counts.items():
        running_counts[int(slot_id)] = running_counts.get(int(slot_id), 0.0) + (weight * float(count))


def _ensure_node_evaluated(
    node: SearchNode,
    model,
    config: AlphaZeroConfig,
    device: torch.device,
) -> SearchModelOutput:
    """Ensure one search node has cached model output and scalar value."""

    if node.unified_logits is None or node.value is None:
        model_output = evaluate_search_state(model, node.state, config.n, device)
        node.unified_logits = model_output.unified_logits
        node.value = model_output.value
    return SearchModelOutput(unified_logits=np.asarray(node.unified_logits), value=float(node.value))


def expand_nonroot_node(
    node: SearchNode,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
) -> None:
    """Expand one non-root node once and cache its sampled action subset.

    The node must already have ``unified_logits`` and ``value`` cached. On the
    first expansion the function fixes ``J_nonroot`` slots with Gumbel top-k, samples
    value pools for those slots, samples ``K_nonroot`` value assignments, and
    initializes visit statistics. Subsequent calls are no-ops.
    """

    if node.expanded:
        return
    if node.unified_logits is None or node.value is None:
        raise ValueError("expand_nonroot_node requires a cached model evaluation")

    _ensure_node_legality_cache(node, config)
    mask_logits = np.asarray(node.unified_logits[:, 0], dtype=np.float64)
    value_logits = np.asarray(node.unified_logits[:, 1:], dtype=np.float64)
    feasible_slot_ids = np.asarray(node.feasible_masked_slots or [], dtype=np.int64)
    fixed_count = min(config.search.J_nonroot, feasible_slot_ids.size)
    if fixed_count > 0:
        legal_row_counts = np.asarray(
            [node.legal_row_counts_by_slot[int(slot_id)] for slot_id in feasible_slot_ids],
            dtype=np.float64,
        )
        fixed_local = gumbel_topk(
            compute_slot_select_scores(
                mask_logits=mask_logits[feasible_slot_ids],
                value_logits=value_logits[feasible_slot_ids],
                legal_row_counts=legal_row_counts,
                tau=config.search.tau_slot_nonroot,
                coef_mask_logit=config.search.slot_score_coef_mask_logit,
                coef_scarcity=config.search.slot_score_coef_scarcity,
                coef_entropy=config.search.slot_score_coef_entropy,
            ),
            fixed_count,
            rng,
        )
        node.fixed_slots = [int(feasible_slot_ids[index]) for index in fixed_local]
        node.value_pools = sample_value_pools_for_slots(
            value_logits=value_logits,
            slot_ids=node.fixed_slots,
            L=config.search.L_nonroot,
            tau_val=config.search.tau_val_nonroot,
            legal_rows_by_column=node.legal_rows_by_column,
            rng=rng,
        )
        sampled_actions: list[JointAction] = []
        for _ in range(config.search.K_nonroot):
            values: dict[int, int] = {}
            for slot_id in node.fixed_slots:
                support = np.asarray(node.value_pools[slot_id], dtype=np.int64)
                if support.size == 0:
                    values = {}
                    break
                row_scores = value_logits[slot_id, support] / float(config.search.tau_val_nonroot)
                values[slot_id] = int(support[gumbel_argmax(row_scores, rng)])
            if len(values) == len(node.fixed_slots):
                sampled_actions.append(JointAction(active_slots=list(node.fixed_slots), values=values))
        node.sampled_actions = sampled_actions
        node.prior_logits = compute_nonroot_action_prior_logits(sampled_actions, value_logits)
    else:
        node.fixed_slots = []
        node.value_pools = {}
        node.sampled_actions = []
        node.prior_logits = np.zeros((0,), dtype=np.float64)

    num_actions = len(node.sampled_actions)
    node.visit_counts = np.zeros(num_actions, dtype=np.int64)
    node.value_sums = np.zeros(num_actions, dtype=np.float64)
    node.executed_actions = [ExecutedAction(active_slots=[], values={}) for _ in range(num_actions)]
    node.step_rewards = [0.0 for _ in range(num_actions)]
    node.stop_reasons = [None for _ in range(num_actions)]
    node.expanded = True


def _materialize_child_transition(
    node: SearchNode,
    action_idx: int,
    config: AlphaZeroConfig,
    state_observer=None,
    ply_from_root: int | None = None,
) -> SimulationResult:
    """Ensure one action has a cached child transition and return it."""

    if action_idx in node.child_nodes:
        child_node = node.child_nodes[action_idx]
        return SimulationResult(
            next_state=child_node.state,
            executed_action=node.executed_actions[action_idx],
            raw_reward_points=len(node.executed_actions[action_idx].active_slots),
            step_reward=node.step_rewards[action_idx],
            done=node.stop_reasons[action_idx] is not None,
            stop_reason=node.stop_reasons[action_idx],
            aux_info={},
        )

    simulation = simulate_action_from_state(node.state, node.sampled_actions[action_idx], config.n, config)
    node.child_nodes[action_idx] = _make_empty_child_node(simulation.next_state)
    if len(node.executed_actions) < len(node.sampled_actions):
        node.executed_actions = [ExecutedAction(active_slots=[], values={}) for _ in node.sampled_actions]
        node.step_rewards = [0.0 for _ in node.sampled_actions]
        node.stop_reasons = [None for _ in node.sampled_actions]
    node.executed_actions[action_idx] = simulation.executed_action
    node.step_rewards[action_idx] = simulation.step_reward
    node.stop_reasons[action_idx] = simulation.stop_reason
    if state_observer is not None and ply_from_root is not None:
        state_observer(simulation.next_state.slot_state_ids.copy(), ply_from_root)
    return simulation


def _select_nonroot_action(node: SearchNode, config: AlphaZeroConfig) -> int:
    """Select one non-root action by matching visits to the improved policy."""

    if node.prior_logits is None or node.visit_counts is None or node.value_sums is None or node.value is None:
        raise ValueError("non-root node is missing priors or visit statistics")
    if len(node.sampled_actions) == 0:
        raise ValueError("non-root node has no sampled actions")

    completed_q = compute_completed_q(node.visit_counts, node.value_sums, float(node.value))
    logits = compute_improved_policy_logits(
        prior_logits=node.prior_logits,
        completed_q=completed_q,
        tau_pi=config.search.tau_pi_nonroot,
        c_Q=config.search.c_Q,
    )
    improved_policy = softmax_over_sampled_actions(logits)
    total_visits = int(np.sum(node.visit_counts))
    scores = improved_policy - (node.visit_counts.astype(np.float64) / float(1 + total_visits))
    return int(np.argmax(scores))


def _rollout_from_action(
    node: SearchNode,
    action_idx: int,
    ply_from_root: int,
    model,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
    device: torch.device,
    state_observer=None,
    terminal_reward_baseline_points: float | None = None,
) -> tuple[float, float, float]:
    """Roll out one action and return ``(R_step, leaf_utility, G)``.

    ``R_step`` is the discounted sum of normalized step rewards from the
    current action onward. ``leaf_utility`` is either the configured terminal
    reward or the model value at the search horizon. ``G`` is the combined
    return ``alpha_step * R_step + alpha_term * leaf_utility`` in ``[0, 1]``.
    """

    alpha_step, alpha_term = reward_mixture_weights(config)
    simulation = _materialize_child_transition(
        node,
        action_idx,
        config,
        state_observer=state_observer,
        ply_from_root=ply_from_root,
    )
    child_node = node.child_nodes[action_idx]

    if simulation.stop_reason is not None:
        leaf_utility = terminal_reward_from_points(
            filled_count(simulation.next_state.slot_state_ids),
            config,
            terminal_reward_baseline_points,
        )
        step_return = simulation.step_reward
        total_return = alpha_step * step_return + alpha_term * leaf_utility
        return step_return, leaf_utility, total_return

    if ply_from_root >= config.search.max_search_plies:
        child_output = _ensure_node_evaluated(child_node, model, config, device)
        leaf_utility = float(child_output.value)
        step_return = simulation.step_reward
        total_return = alpha_step * step_return + alpha_term * leaf_utility
        return step_return, leaf_utility, total_return

    child_output = _ensure_node_evaluated(child_node, model, config, device)
    expand_nonroot_node(child_node, rng, config)
    if len(child_node.sampled_actions) == 0:
        leaf_utility = terminal_reward_from_points(
            filled_count(child_node.state.slot_state_ids),
            config,
            terminal_reward_baseline_points,
        )
        step_return = simulation.step_reward
        total_return = alpha_step * step_return + alpha_term * leaf_utility
        return step_return, leaf_utility, total_return

    next_action_idx = _select_nonroot_action(child_node, config)
    child_step_return, leaf_utility, _ = _rollout_from_action(
        child_node,
        next_action_idx,
        ply_from_root + 1,
        model,
        rng,
        config,
        device,
        state_observer,
        terminal_reward_baseline_points,
    )
    step_return = simulation.step_reward + config.search.gamma_step * child_step_return
    total_return = alpha_step * step_return + alpha_term * leaf_utility
    child_node.visit_counts[next_action_idx] += 1
    child_node.value_sums[next_action_idx] += total_return
    return step_return, leaf_utility, total_return


def run_simulation_from_action(
    node: SearchNode,
    action_idx: int,
    ply_from_root: int,
    model,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
    state_observer=None,
    terminal_reward_baseline_points: float | None = None,
) -> float:
    """Run one simulation from a chosen action and return the combined return."""

    device = next(model.parameters()).device
    _, _, total_return = _rollout_from_action(
        node,
        action_idx,
        ply_from_root,
        model,
        rng,
        config,
        device,
        state_observer=state_observer,
        terminal_reward_baseline_points=terminal_reward_baseline_points,
    )
    return float(total_return)


def _initialize_root_node(
    root_state: SearchState,
    model,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
    device: torch.device,
    state_observer=None,
) -> tuple[SearchNode, dict[int, float]]:
    """Evaluate and initialize the sampled root action subset."""

    root_output = evaluate_search_state(model, root_state, config.n, device)
    root_node = SearchNode(
        state=root_state,
        expanded=True,
        unified_logits=root_output.unified_logits,
        value=root_output.value,
    )
    _ensure_node_legality_cache(root_node, config)
    mask_logits = root_output.unified_logits[:, 0]
    value_logits = root_output.unified_logits[:, 1:]
    feasible_slot_ids = np.asarray(root_node.feasible_masked_slots or [], dtype=np.int64)
    if feasible_slot_ids.size < config.search.J_root:
        root_node.candidate_slots = []
        root_node.value_pools = {}
        root_node.sampled_actions = []
        root_node.prior_logits = np.zeros((0,), dtype=np.float64)
        root_node.visit_counts = np.zeros((0,), dtype=np.int64)
        root_node.value_sums = np.zeros((0,), dtype=np.float64)
        root_node.executed_actions = []
        root_node.step_rewards = []
        root_node.stop_reasons = []
        root_node.root_action_gumbels = np.zeros((0,), dtype=np.float64)
        return root_node, {}
    root_node.candidate_slots = sample_root_candidate_slots(
        mask_logits=mask_logits,
        value_logits=value_logits,
        feasible_slot_ids=feasible_slot_ids,
        legal_row_counts_by_slot=root_node.legal_row_counts_by_slot or {},
        J_root_prime=config.search.J_root_prime,
        tau_slot_root=config.search.tau_slot_root,
        config=config,
        rng=rng,
    )
    root_node.value_pools = sample_value_pools_for_slots(
        value_logits=value_logits,
        slot_ids=root_node.candidate_slots,
        L=config.search.L_root,
        tau_val=config.search.tau_val_root,
        legal_rows_by_column=root_node.legal_rows_by_column,
        rng=rng,
    )
    root_node.sampled_actions = sample_root_joint_actions(
        candidate_slots=root_node.candidate_slots,
        value_pools=root_node.value_pools,
        mask_logits=mask_logits,
        value_logits=value_logits,
        legal_row_counts_by_slot=root_node.legal_row_counts_by_slot or {},
        J_root=config.search.J_root,
        K_root=config.search.K_root,
        tau_slot_root=config.search.tau_slot_root,
        tau_val_root=config.search.tau_val_root,
        config=config,
        rng=rng,
    )
    root_node.prior_logits = compute_root_action_prior_logits(root_node.sampled_actions, mask_logits, value_logits)
    num_actions = len(root_node.sampled_actions)
    root_node.visit_counts = np.zeros(num_actions, dtype=np.int64)
    root_node.value_sums = np.zeros(num_actions, dtype=np.float64)
    root_node.executed_actions = [ExecutedAction(active_slots=[], values={}) for _ in range(num_actions)]
    root_node.step_rewards = [0.0 for _ in range(num_actions)]
    root_node.stop_reasons = [None for _ in range(num_actions)]
    root_node.root_action_gumbels = sample_gumbel((num_actions,), rng) if num_actions > 0 else np.zeros((0,), dtype=np.float64)
    root_active_slot_ids = {
        int(slot_id)
        for slot_id in np.flatnonzero(np.asarray(root_state.slot_state_ids, dtype=np.int64) != 0)
    }
    root_mcts_conflict_counts: dict[int, float] = {}

    for action_idx, action in enumerate(root_node.sampled_actions):
        simulation = simulate_action_from_state(root_state, action, config.n, config)
        root_node.child_nodes[action_idx] = _make_empty_child_node(simulation.next_state)
        root_node.executed_actions[action_idx] = simulation.executed_action
        root_node.step_rewards[action_idx] = simulation.step_reward
        root_node.stop_reasons[action_idx] = simulation.stop_reason
        _accumulate_root_mcts_conflict_counts(
            root_mcts_conflict_counts,
            simulation.aux_info.get("conflict_report", {}),
            root_active_slot_ids,
            num_actions,
        )
        if state_observer is not None:
            state_observer(simulation.next_state.slot_state_ids.copy(), 1)
    return root_node, root_mcts_conflict_counts


def run_root_search(
    root_state: SearchState,
    model,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
    state_observer=None,
    best_overall_filled_count_fn=None,
    terminal_reward_baseline_points: float | None = None,
) -> RootSearchResult:
    """Run sampled-action root search with Gumbel Sequential Halving.

    The root evaluation samples a restricted joint-action subset, caches one
    child transition for every sampled root action, then allocates
    ``root_budget`` simulations using the Sequential Halving considered-visit
    schedule. The returned result contains the root node, completed-Q improved
    policy over sampled actions, and the greedy best action index.
    """

    device = next(model.parameters()).device

    def maybe_observe(slot_state_ids: np.ndarray, ply_from_root: int) -> None:
        """Forward only best-beating candidate states to the caller observer."""

        if state_observer is None:
            return
        current_best = -1 if best_overall_filled_count_fn is None else int(best_overall_filled_count_fn())
        if filled_count(slot_state_ids) > current_best:
            state_observer(slot_state_ids, ply_from_root)

    root_node, root_mcts_conflict_counts = _initialize_root_node(
        root_state,
        model,
        rng,
        config,
        device,
        state_observer=maybe_observe,
    )
    num_actions = len(root_node.sampled_actions)

    if num_actions == 0:
        return RootSearchResult(
            root_node=root_node,
            improved_policy=np.zeros((0,), dtype=np.float64),
            best_action_index=-1,
            completed_q=np.zeros((0,), dtype=np.float64),
            root_mcts_conflict_counts=root_mcts_conflict_counts,
        )

    for considered_visit in get_considered_visit_sequence(num_actions, config.search.root_budget):
        completed_q = compute_completed_q(root_node.visit_counts, root_node.value_sums, float(root_node.value))
        ranking_scores = (
            root_node.root_action_gumbels
            + (root_node.prior_logits / float(config.search.tau_pi_root))
            + sigma_q(completed_q, config.search.c_Q)
        )
        eligible = root_node.visit_counts == considered_visit
        if not np.any(eligible):
            eligible = np.ones_like(root_node.visit_counts, dtype=bool)
        masked_scores = np.where(eligible, ranking_scores, -np.inf)
        action_idx = int(np.argmax(masked_scores))
        rollout_return = run_simulation_from_action(
            root_node,
            action_idx,
            ply_from_root=1,
            model=model,
            rng=rng,
            config=config,
            state_observer=maybe_observe,
            terminal_reward_baseline_points=terminal_reward_baseline_points,
        )
        root_node.visit_counts[action_idx] += 1
        root_node.value_sums[action_idx] += rollout_return

    completed_q = compute_completed_q(root_node.visit_counts, root_node.value_sums, float(root_node.value))
    improved_policy = softmax_over_sampled_actions(
        compute_improved_policy_logits(
            prior_logits=root_node.prior_logits,
            completed_q=completed_q,
            tau_pi=config.search.tau_target,
            c_Q=config.search.c_Q,
        )
    )
    best_action_index = int(np.argmax(improved_policy))
    return RootSearchResult(
        root_node=root_node,
        improved_policy=improved_policy,
        best_action_index=best_action_index,
        completed_q=completed_q,
        root_mcts_conflict_counts=root_mcts_conflict_counts,
    )
