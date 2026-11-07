"""Tests for PatternBoost partial environment initialization."""

from __future__ import annotations

import numpy as np

from circle_pack.env import CirclePackingEnv
from patternboost.env_init import initialize_partial_env_from_seed


def test_partial_env_initialization_keeps_fixed_core_active(make_config) -> None:
    config = make_config(K=3)
    env = CirclePackingEnv(config)
    seed = np.array(
        [
            [0.20, 0.20, 0.06],
            [0.80, 0.20, 0.06],
            [0.50, 0.75, 0.08],
        ],
        dtype=np.float64,
    )
    edit_mask = np.array([False, True, False], dtype=bool)
    draft = seed.copy()
    draft[1] = np.array([0.50, 0.50, 0.10])

    obs = initialize_partial_env_from_seed(env, seed, edit_mask, draft_centers=draft, seed=11)

    assert obs["active_mask"].tolist() == [True, False, True]
    np.testing.assert_allclose(obs["centers"][0], seed[0, :2])
    np.testing.assert_allclose(obs["centers"][2], seed[2, :2])
    np.testing.assert_allclose(obs["centers"][1], draft[1, :2])
    assert obs["layout_score"] is None
