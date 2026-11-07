"""Rollout smoke tests for the baseline policy and collector."""

from __future__ import annotations

import pytest

from circle_pack.collect import collect_episode
from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import active_packing_is_feasible
from circle_pack.policies.random_legal import RandomLegalPolicy


def test_collect_episode_random_legal_smoke(make_config) -> None:
    config = make_config(
        K=6,
        max_steps=4,
        commit_policy="shrink_to_fit",
        activation_budget=2,
        r_min_commit=0.01,
        policy_name="random_legal",
    )
    env = CirclePackingEnv(config)
    policy = RandomLegalPolicy(config)

    transitions, summary = collect_episode(env, policy, seed=7, record_snapshots=False)
    final_observation = env.observe()

    assert transitions
    assert summary["done"]
    assert summary["num_steps"] <= config.problem.max_steps
    assert summary["total_reward"] == pytest.approx(summary["final_sum_active_r"])
    assert summary["final_sum_active_r"] == pytest.approx(final_observation["sum_active_r"])
    assert active_packing_is_feasible(final_observation["circles"], final_observation["active_mask"])
