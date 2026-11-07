"""Tests for PatternBoost edit-set sampling."""

from __future__ import annotations

import numpy as np

from patternboost.edit import sample_random_edit, sample_random_point_knn_edit


def test_random_edit_ratio_and_full_edit() -> None:
    circles = np.zeros((5, 3), dtype=np.float64)
    active_mask = np.ones(5, dtype=bool)
    rng = np.random.default_rng(1)

    edit = sample_random_edit(circles, active_mask, 0.4, rng)
    full = sample_random_edit(circles, active_mask, 1.0, rng)

    assert edit.policy == "random"
    assert int(edit.edit_mask.sum()) == 2
    assert np.all(full.edit_mask)


def test_random_point_knn_uses_nearest_centers() -> None:
    circles = np.array(
        [
            [0.0, 0.0, 0.01],
            [0.1, 0.0, 0.01],
            [0.9, 0.9, 0.01],
        ],
        dtype=np.float64,
    )
    active_mask = np.ones(3, dtype=bool)

    class FixedRng:
        def uniform(self, low, high, size):
            return np.array([0.05, 0.0], dtype=np.float64)

    edit = sample_random_point_knn_edit(circles, active_mask, 1.0, FixedRng(), knn_k=2)

    assert edit.policy == "random_point_knn"
    assert edit.edit_mask.tolist() == [True, True, False]

