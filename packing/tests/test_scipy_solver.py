"""Tests for fixed-center LP radius scoring."""

from __future__ import annotations

import numpy as np

from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import is_valid_packing
from circle_pack.scipy_solver import complete_circles_from_centers, solve_packing_env, solve_radii_for_centers


def test_known_two_circle_lp_solution() -> None:
    centers = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)

    result = solve_radii_for_centers(centers)

    assert result.success
    np.testing.assert_allclose(result.circles[:, 2], [0.25, 0.25], atol=1.0e-10)
    assert result.sum_r == np.float64(0.5)
    assert is_valid_packing(result.circles[:, :2], result.circles[:, 2], tol=1.0e-9)


def test_random_complete_centers_solve_for_multiple_sizes() -> None:
    rng = np.random.default_rng(7)
    for num_slots in (8, 15, 20, 25):
        result = solve_radii_for_centers(rng.uniform(0.0, 1.0, size=(num_slots, 2)))
        assert result.success
        assert result.circles.shape == (num_slots, 3)
        assert is_valid_packing(result.circles[:, :2], result.circles[:, 2], tol=1.0e-9)


def test_duplicate_centers_are_feasible_with_zero_radii() -> None:
    centers = np.array([[0.5, 0.5], [0.5, 0.5], [0.25, 0.25]], dtype=np.float64)

    result = solve_radii_for_centers(centers)

    assert result.success
    assert result.circles[0, 2] == 0.0
    assert result.circles[1, 2] == 0.0
    assert is_valid_packing(result.circles[:, :2], result.circles[:, 2], tol=1.0e-9)


def test_complete_circles_from_centers_returns_legal_layout() -> None:
    centers = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.75]], dtype=np.float64)

    circles = complete_circles_from_centers(centers)

    assert circles.shape == (3, 3)
    np.testing.assert_allclose(circles[:, :2], centers)
    assert is_valid_packing(circles[:, :2], circles[:, 2], tol=1.0e-9)


def test_solve_packing_env_scores_current_centers(make_config) -> None:
    env = CirclePackingEnv(make_config(K=3, max_steps=2, activation_budget=3))
    env.reset(seed=0)
    proposal = np.array([[0.25, 0.25], [0.75, 0.25], [0.50, 0.75]], dtype=env.dtype)
    observation, reward, done, _info = env.step(proposal)

    result = solve_packing_env(env)

    assert done
    assert reward == observation["layout_score"]
    np.testing.assert_allclose(result.circles, observation["circles"])
    assert result.sum_r == observation["layout_score"]
