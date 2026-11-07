"""Tests for center-only environment step behavior."""

from __future__ import annotations

import numpy as np

from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import is_valid_packing


def test_reset_returns_centers_without_terminal_cache(make_config) -> None:
    env = CirclePackingEnv(make_config(K=4, max_steps=3, activation_budget=2))
    observation = env.reset(seed=0)

    assert observation["centers"].shape == (4, 2)
    assert observation["radii"] is None
    assert observation["circles"] is None
    assert observation["layout_score"] is None
    assert not np.any(observation["active_mask"])


def test_step_commits_by_slot_index_without_partial_reward(make_config) -> None:
    env = CirclePackingEnv(make_config(K=4, max_steps=4, activation_budget=2, commit_order="slot_index"))
    env.reset(seed=1)
    proposal = np.array(
        [[0.80, 0.20], [0.20, 0.20], [0.50, 0.80], [0.70, 0.70]],
        dtype=env.dtype,
    )

    next_obs, reward, done, info = env.step(proposal)

    assert reward == 0.0
    assert not done
    assert info["committed_ids"] == [0, 1]
    assert info["num_committed"] == 2
    np.testing.assert_array_equal(next_obs["active_mask"], np.array([True, True, False, False]))
    assert next_obs["layout_score"] is None


def test_terminal_step_runs_lp_and_returns_layout_score(make_config) -> None:
    env = CirclePackingEnv(make_config(K=3, max_steps=3, activation_budget=2, commit_order="slot_index"))
    env.reset(seed=2)
    first = np.array([[0.25, 0.25], [0.75, 0.25], [0.50, 0.75]], dtype=env.dtype)
    env.step(first)

    next_obs, reward, done, info = env.step(first)

    assert done
    assert np.all(next_obs["active_mask"])
    assert next_obs["radii"].shape == (3,)
    assert next_obs["circles"].shape == (3, 3)
    assert reward == next_obs["layout_score"] == info["layout_score"]
    assert is_valid_packing(next_obs["circles"][:, :2], next_obs["circles"][:, 2], tol=1.0e-9)


def test_random_commit_order_is_seed_deterministic(make_config) -> None:
    config = make_config(K=5, max_steps=2, activation_budget=2, commit_order="random")
    proposal = np.linspace(0.1, 0.9, 10, dtype=np.float64).reshape(5, 2)

    env_a = CirclePackingEnv(config)
    env_b = CirclePackingEnv(config)
    env_a.reset(seed=10)
    env_b.reset(seed=10)

    _, _, _, info_a = env_a.step(proposal)
    _, _, _, info_b = env_b.step(proposal)

    assert info_a["committed_ids"] == info_b["committed_ids"]
    assert info_a["committed_ids"] != [0, 1]


def test_conflict_policy_none_allows_duplicate_centers(make_config) -> None:
    env = CirclePackingEnv(
        make_config(K=2, max_steps=2, activation_budget=2, center_conflict_policy="none", done_on_all_active=True)
    )
    env.reset(seed=3)
    proposal = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=env.dtype)

    observation, reward, done, info = env.step(proposal)

    assert done
    assert info["num_committed"] == 2
    assert observation["layout_score"] == reward
    assert np.count_nonzero(observation["radii"] > 1.0e-12) == 0


def test_conflict_policy_reject_skips_duplicate_candidate(make_config) -> None:
    env = CirclePackingEnv(
        make_config(
            K=2,
            max_steps=2,
            activation_budget=2,
            center_conflict_policy="reject",
            center_duplicate_eps=1.0e-6,
            done_on_all_active=False,
        )
    )
    env.reset(seed=4)
    proposal = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=env.dtype)

    observation, reward, done, info = env.step(proposal)

    assert reward == 0.0
    assert not done
    assert info["committed_ids"] == [0]
    np.testing.assert_array_equal(observation["active_mask"], np.array([True, False]))


def test_greedy_fill_masked_slots_randomly_completes_and_scores(make_config) -> None:
    env = CirclePackingEnv(make_config(K=4, max_steps=2, activation_budget=1))
    env.reset(seed=5)
    env.step(np.full((4, 2), 0.25, dtype=env.dtype), candidate_mask=np.array([True, False, False, False]))

    info = env.greedy_fill_masked_slots()
    observation = env.observe()

    assert info["success"]
    assert info["remaining_masked_ids"] == []
    assert np.all(observation["active_mask"])
    assert observation["layout_score"] is not None
    assert is_valid_packing(observation["circles"][:, :2], observation["circles"][:, 2], tol=1.0e-9)
