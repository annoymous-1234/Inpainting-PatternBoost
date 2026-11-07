from __future__ import annotations

import numpy as np

from env import NoThreeInLineEnv

from .config import AlphaZeroConfig
from .search import count_feasible_masked_slots, gumbel_topk, joint_action_to_proposal_slots, run_root_search
from .state import filled_count, search_state_from_env, slot_state_ids_from_env
from .targets import (
    attach_value_targets,
    build_root_marginal_targets,
    final_terminal_score,
    initial_terminal_baseline_points,
    terminal_reward_from_points,
)
from .types import AlphaZeroSample, EpisodeTransition, SearchState


def choose_root_action(
    root_result,
    state: SearchState,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
) -> int:
    """Choose the real root action from the improved policy ``μ_r``.

    For the first ``sample_prefix_steps`` of the real episode the action is
    sampled from a temperature-adjusted ``μ_r``. After that point the action is
    chosen greedily by ``μ_r``.
    """

    if root_result.improved_policy.size == 0:
        return -1
    if state.episode_step >= config.self_play.sample_prefix_steps:
        return int(np.argmax(root_result.improved_policy))

    probs = np.asarray(root_result.improved_policy, dtype=np.float64)
    if config.self_play.root_temperature != 1.0:
        probs = np.power(probs + config.search.eps, 1.0 / config.self_play.root_temperature)
        probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def _build_self_play_env(config: AlphaZeroConfig) -> NoThreeInLineEnv:
    """Construct the real rollout environment used by AlphaZero self-play."""

    env = NoThreeInLineEnv(
        n=config.n,
        num_slots=2 * config.n,
        max_steps=config.self_play.max_episode_steps,
        proposal_search_config={
            "num_search_attempts": 1,
            "allow_existing_deletions": False,
        },
    )
    env.reset()
    return env


def _select_slots_to_remask(
    slot_state_ids: np.ndarray,
    slot_conflict_mass: np.ndarray,
    slot_active_age: np.ndarray,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
) -> list[int]:
    """Sample rollout slots to remask from anneal scores on active env slots.

    ``slot_state_ids`` is the post-step unified slot state of shape ``[2n]``.
    ``slot_conflict_mass`` and ``slot_active_age`` are rollout-side arrays with
    the same shape. The returned list contains up to
    ``local_anneal.num_slots_to_remask`` slot ids chosen without replacement by
    Gumbel top-k over ``score / sampling_temperature``.
    """

    active_slot_ids = np.flatnonzero(np.asarray(slot_state_ids, dtype=np.int64) != 0)
    if active_slot_ids.size == 0:
        return []

    eligible_slot_ids = [
        int(slot_id)
        for slot_id in active_slot_ids
        if int(slot_active_age[int(slot_id)]) >= config.local_anneal.min_active_age_for_remask
    ]
    if not eligible_slot_ids:
        return []

    eligible_array = np.asarray(eligible_slot_ids, dtype=np.int64)
    scores = slot_conflict_mass[eligible_array] / np.maximum(slot_active_age[eligible_array], 1.0)
    positive_mask = scores > 0.0
    if not np.any(positive_mask):
        return []

    positive_slots = eligible_array[positive_mask]
    positive_scores = scores[positive_mask] / float(config.local_anneal.sampling_temperature)
    num_to_sample = min(config.local_anneal.num_slots_to_remask, positive_slots.size)
    selected_local = gumbel_topk(np.asarray(positive_scores, dtype=np.float64), num_to_sample, rng)
    return [int(positive_slots[index]) for index in selected_local]


def _compute_episode_stop_reason(
    next_slot_state_ids: np.ndarray,
    *,
    env_done: bool,
    accepted_any: bool,
    config: AlphaZeroConfig,
) -> str | None:
    """Return the real-rollout terminal reason after env step and remask."""

    if filled_count(next_slot_state_ids) == 2 * config.n:
        return "filled_all_slots"
    if count_feasible_masked_slots(next_slot_state_ids, config) < config.search.J_root:
        return "insufficient_feasible_slots"
    if env_done:
        return "episode_step_limit"
    if config.search.terminate_on_no_progress and not accepted_any:
        return "no_progress"
    return None


def run_self_play_episode_with_trace(
    model,
    rng: np.random.Generator,
    config: AlphaZeroConfig,
    mcts_state_observer=None,
    best_overall_filled_count_fn=None,
    terminal_reward_baseline_points: float | None = None,
) -> tuple[list[AlphaZeroSample], list[dict[str, object]], dict[str, object]]:
    """Run one complete self-play episode and return samples, trace, and summary."""

    reward_baseline = (
        initial_terminal_baseline_points(config)
        if terminal_reward_baseline_points is None
        else float(terminal_reward_baseline_points)
    )
    env = _build_self_play_env(config)
    state = search_state_from_env(env)
    transitions: list[EpisodeTransition] = []
    trace: list[dict[str, object]] = []
    slot_conflict_mass = np.zeros(2 * config.n, dtype=np.float64)
    slot_active_age = np.zeros(2 * config.n, dtype=np.float64)
    num_remask_events = 0
    total_remasked_slots = 0
    final_stop_reason: str | None = None

    while True:
        root_search_observer = None
        if mcts_state_observer is not None:
            root_episode_step = state.episode_step

            def root_search_observer(slot_state_ids, ply_from_root: int) -> None:
                """Attach the current root episode step to MCTS observer events."""

                mcts_state_observer(
                    slot_state_ids=slot_state_ids,
                    root_episode_step=root_episode_step,
                    ply_from_root=ply_from_root,
                )

        root_result = run_root_search(
            state,
            model,
            rng,
            config,
            state_observer=root_search_observer,
            best_overall_filled_count_fn=best_overall_filled_count_fn,
            terminal_reward_baseline_points=reward_baseline,
        )
        if root_result.best_action_index < 0:
            final_stop_reason = "insufficient_feasible_slots"
            break

        root_targets = build_root_marginal_targets(root_result, config)
        chosen_action_index = choose_root_action(root_result, state, rng, config)
        if chosen_action_index < 0:
            final_stop_reason = "insufficient_feasible_slots"
            break

        chosen_action = root_result.root_node.sampled_actions[chosen_action_index]
        chosen_executed = root_result.root_node.executed_actions[chosen_action_index]
        chosen_step_reward = root_result.root_node.step_rewards[chosen_action_index]
        _, _, env_done, info = env.step(joint_action_to_proposal_slots(chosen_action))
        env_conflict_counts = {
            int(slot_id): int(count)
            for slot_id, count in info["existing_slot_line_conflict_counts"].items()
        }
        mcts_conflict_counts = {
            int(slot_id): float(count)
            for slot_id, count in root_result.root_mcts_conflict_counts.items()
        }

        for slot_id, count in env_conflict_counts.items():
            slot_conflict_mass[slot_id] += float(config.local_anneal.alpha_env) * float(count)
        for slot_id, count in mcts_conflict_counts.items():
            slot_conflict_mass[slot_id] += float(config.local_anneal.alpha_mcts) * float(count)

        post_step_slot_state_ids = slot_state_ids_from_env(env)
        for slot_id in np.flatnonzero(post_step_slot_state_ids != 0):
            slot_active_age[int(slot_id)] += 1.0

        remasked_slots: list[int] = []
        if config.local_anneal.enabled and env.timestep in set(config.local_anneal.trigger_steps):
            remask_candidates = _select_slots_to_remask(
                post_step_slot_state_ids,
                slot_conflict_mass,
                slot_active_age,
                rng,
                config,
            )
            if remask_candidates:
                remask_result = env.remask_slots(remask_candidates)
                remasked_slots = [int(slot_id) for slot_id in remask_result["removed_slot_ids"]]
                if remasked_slots:
                    slot_conflict_mass[remasked_slots] = 0.0
                    slot_active_age[remasked_slots] = 0.0
                    num_remask_events += 1
                    total_remasked_slots += len(remasked_slots)

        next_state = search_state_from_env(env)
        transitions.append(
            EpisodeTransition(
                slot_state_ids=state.slot_state_ids.copy(),
                episode_step=state.episode_step,
                candidate_slot_ids=list(root_targets["candidate_slot_ids"]),
                fill_targets=list(root_targets["fill_targets"]),
                value_pool_ids=[list(pool) for pool in root_targets["value_pool_ids"]],
                value_targets_on_support=[list(targets) for targets in root_targets["value_targets_on_support"]],
                value_weights=list(root_targets["value_weights"]),
                step_reward=float(chosen_step_reward),
            )
        )

        real_stop_reason = _compute_episode_stop_reason(
            next_state.slot_state_ids,
            env_done=env_done,
            accepted_any=bool(chosen_executed.active_slots),
            config=config,
        )
        trace.append(
            {
                "episode_step": state.episode_step,
                "chosen_action_index": chosen_action_index,
                "chosen_action": chosen_action,
                "executed_action": chosen_executed,
                "root_value": float(root_result.root_node.value),
                "env_conflict_counts": dict(env_conflict_counts),
                "mcts_conflict_counts": dict(mcts_conflict_counts),
                "remasked_slots": list(remasked_slots),
                "points_after_env_step": int(np.count_nonzero(post_step_slot_state_ids)),
                "points_after_remask": int(filled_count(next_state.slot_state_ids)),
                "stop_reason": real_stop_reason,
            }
        )
        state = next_state
        if real_stop_reason is not None:
            final_stop_reason = real_stop_reason
            break

    final_points = int(filled_count(state.slot_state_ids))
    terminal_score = final_terminal_score(state.slot_state_ids, config.n)
    terminal_reward = terminal_reward_from_points(final_points, config, reward_baseline)
    for transition in transitions:
        transition.terminal_score = terminal_score
        transition.terminal_reward = terminal_reward
    samples = attach_value_targets(transitions, config, config.n)
    episode_summary = {
        "terminal_slot_state_ids": state.slot_state_ids.copy(),
        "terminal_score": float(terminal_score),
        "terminal_reward": float(terminal_reward),
        "terminal_reward_baseline_points": float(reward_baseline),
        "episode_length": len(trace),
        "stop_reason": final_stop_reason,
        "final_points": final_points,
        "total_step_reward": float(sum(transition.step_reward for transition in transitions)),
        "num_remask_events": int(num_remask_events),
        "total_remasked_slots": int(total_remasked_slots),
        "final_slot_conflict_mass": slot_conflict_mass.copy(),
        "final_slot_active_age": slot_active_age.copy(),
    }
    return samples, trace, episode_summary


def run_self_play_episode(model, rng: np.random.Generator, config: AlphaZeroConfig) -> list[AlphaZeroSample]:
    """Run one self-play episode and return the final learner samples only."""

    samples, _, _ = run_self_play_episode_with_trace(model, rng, config)
    return samples
