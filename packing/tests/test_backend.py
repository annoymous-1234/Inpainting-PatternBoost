"""Tests for the append-only backend and environment snapshot support."""

from __future__ import annotations

import numpy as np

from circle_pack.backend import ActiveBackend
from circle_pack.env import CirclePackingEnv


def test_active_backend_append_updates_compact_state() -> None:
    backend = ActiveBackend(dtype=np.float64)
    backend.append(2, np.array([0.25, 0.30]), 0.10)
    backend.append(4, np.array([0.70, 0.80]), 0.15)

    np.testing.assert_array_equal(backend.active_ids, np.array([2, 4], dtype=np.int64))
    np.testing.assert_allclose(
        backend.active_centers,
        np.array([[0.25, 0.30], [0.70, 0.80]], dtype=np.float64),
    )
    np.testing.assert_allclose(backend.active_radii, np.array([0.10, 0.15], dtype=np.float64))
    assert backend.n_active == 2
    assert backend.sum_active_r == 0.25


def test_env_snapshot_restore_round_trip(make_config) -> None:
    env = CirclePackingEnv(make_config(K=4, max_steps=3, activation_budget=2))
    observation = env.reset(seed=11)
    snapshot = env.snapshot()

    proposal = observation["centers"].copy()
    proposal[:] = np.array([[0.1, 0.1], [0.3, 0.3], [0.7, 0.7], [0.9, 0.9]])
    env.step(proposal)

    restored = env.restore(snapshot)
    np.testing.assert_array_equal(restored["active_mask"], observation["active_mask"])
    np.testing.assert_allclose(restored["centers"], observation["centers"])
    assert restored["step_idx"] == observation["step_idx"]
    assert restored["sum_active_r"] == observation["sum_active_r"]
    assert restored["layout_score"] is None
