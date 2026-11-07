from __future__ import annotations

import numpy as np

from .config import AlphaZeroConfig, reward_mixture_weights
from .state import filled_count
from .types import AlphaZeroSample, EpisodeTransition, RootSearchResult

_MIN_EXP2_EXPONENT = -60.0


def build_root_improved_policy(root_result: RootSearchResult, config: AlphaZeroConfig) -> np.ndarray:
    """Build the root improved policy from completed-Q target logits.

    The returned vector has shape ``[K_root_sampled]``, sums to ``1``, and is
    defined only over the sampled root actions. The fixed root Gumbel used for
    Sequential Halving is intentionally excluded from the training target.
    """

    node = root_result.root_node
    if node.prior_logits is None or node.visit_counts is None or node.value_sums is None or node.value is None:
        raise ValueError("root node is missing prior logits or statistics")

    completed_q = np.where(
        node.visit_counts > 0,
        node.value_sums / np.maximum(node.visit_counts, 1),
        float(node.value),
    )
    logits = node.prior_logits / config.search.tau_target + (config.search.c_Q * (2.0 * completed_q - 1.0))
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return np.full_like(logits, fill_value=1.0 / len(logits), dtype=np.float64)
    return probs / probs_sum


def build_root_marginal_targets(root_result: RootSearchResult, config: AlphaZeroConfig) -> dict[str, object]:
    """Marginalize the root improved policy into fill and value targets.

    The marginals are constructed from root ``executed_action`` objects rather
    than raw proposal actions. The returned dictionary contains candidate slots,
    fill targets, value supports, support-restricted value targets, and value
    weights aligned by candidate-slot position.
    """

    node = root_result.root_node
    improved_policy = root_result.improved_policy
    candidate_slot_ids = list(node.candidate_slots)
    fill_targets: list[float] = []
    value_pool_ids: list[list[int]] = []
    value_targets_on_support: list[list[float]] = []
    value_weights: list[float] = []

    for slot_id in candidate_slot_ids:
        fill_probability = 0.0
        for action_prob, executed_action in zip(improved_policy, node.executed_actions):
            if slot_id in executed_action.values:
                fill_probability += float(action_prob)

        fill_targets.append(fill_probability)
        support = list(node.value_pools.get(slot_id, []))
        value_pool_ids.append(support)
        if not support or fill_probability <= config.search.eps:
            value_targets_on_support.append([0.0 for _ in support])
            value_weights.append(0.0)
            continue

        conditional_probs = []
        norm = 0.0
        for y_value in support:
            marginal = 0.0
            for action_prob, executed_action in zip(improved_policy, node.executed_actions):
                if executed_action.values.get(slot_id) == y_value:
                    marginal += float(action_prob)
            conditional = marginal / max(fill_probability, config.search.eps)
            conditional_probs.append(conditional)
            norm += conditional

        if norm <= config.search.eps:
            value_targets_on_support.append([0.0 for _ in support])
            value_weights.append(0.0)
            continue

        value_targets_on_support.append([value / norm for value in conditional_probs])
        value_weights.append(fill_probability)

    return {
        "candidate_slot_ids": candidate_slot_ids,
        "fill_targets": fill_targets,
        "value_pool_ids": value_pool_ids,
        "value_targets_on_support": value_targets_on_support,
        "value_weights": value_weights,
    }


def attach_value_targets(transitions: list[EpisodeTransition], config: AlphaZeroConfig, n: int) -> list[AlphaZeroSample]:
    """Attach scalar AlphaZero value targets to a self-play episode prefix.

    ``transitions`` is the ordered episode prefix, each item already contains
    root policy marginals plus the realized step reward. The function returns a
    list of ``AlphaZeroSample`` objects with ``value_target`` in ``[0, 1]`` and
    does not mutate the input transition list.
    """

    alpha_step, alpha_term = reward_mixture_weights(config)
    discounted_suffix = 0.0
    terminal_reward = 0.0 if not transitions else float(transitions[0].terminal_reward)
    samples: list[AlphaZeroSample] = []

    for transition in reversed(transitions):
        discounted_suffix = float(transition.step_reward) + config.search.gamma_step * discounted_suffix
        value_target = alpha_step * discounted_suffix + alpha_term * terminal_reward
        samples.append(
            AlphaZeroSample(
                slot_state_ids=transition.slot_state_ids.copy(),
                episode_step=transition.episode_step,
                candidate_slot_ids=list(transition.candidate_slot_ids),
                fill_targets=list(transition.fill_targets),
                value_pool_ids=[list(pool) for pool in transition.value_pool_ids],
                value_targets_on_support=[list(targets) for targets in transition.value_targets_on_support],
                value_weights=list(transition.value_weights),
                value_target=float(value_target),
            )
        )
    samples.reverse()
    return samples


def final_terminal_score(slot_state_ids: np.ndarray, n: int) -> float:
    """Return the normalized terminal score ``filled_count / (2n)``."""

    return filled_count(slot_state_ids) / float(2 * n)


def initial_terminal_baseline_points(config: AlphaZeroConfig) -> float:
    """Return the configured bootstrap baseline for terminal reward shaping."""

    if config.reward.initial_terminal_baseline_points is not None:
        return float(config.reward.initial_terminal_baseline_points)
    return float(config.reward.initial_terminal_baseline_fraction) * float(2 * config.n)


def terminal_reward_from_points(
    terminal_points: float,
    config: AlphaZeroConfig,
    terminal_reward_baseline_points: float | None = None,
) -> float:
    """Return the configured terminal reward for one terminal point count."""

    points = float(terminal_points)
    if config.reward.terminal_reward_mode == "normalized_points":
        return points / float(2 * config.n)
    if config.reward.terminal_reward_mode != "capped_exp2_advantage":
        raise ValueError(f"unsupported terminal_reward_mode: {config.reward.terminal_reward_mode}")

    baseline = (
        initial_terminal_baseline_points(config)
        if terminal_reward_baseline_points is None
        else float(terminal_reward_baseline_points)
    )
    exponent = float(config.reward.terminal_reward_beta) * (points - baseline - float(config.reward.terminal_reward_cap))
    exponent = max(exponent, _MIN_EXP2_EXPONENT)
    return min(2.0 ** exponent, 1.0)
