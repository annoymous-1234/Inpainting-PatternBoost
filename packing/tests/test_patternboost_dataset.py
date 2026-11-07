"""Tests for PatternBoost supervised dataset construction."""

from __future__ import annotations

import numpy as np

from patternboost.dataset import PatternBoostSupervisedDataset
from patternboost.top_pool import TopPool


def _layout() -> np.ndarray:
    return np.array(
        [
            [0.20, 0.20, 0.07],
            [0.80, 0.20, 0.07],
            [0.20, 0.80, 0.07],
            [0.80, 0.80, 0.07],
        ],
        dtype=np.float64,
    )


def test_patternboost_dataset_uses_explicit_edit_mask(make_config) -> None:
    config = make_config(K=4)
    pool = TopPool(max_size=4)
    assert pool.insert(_layout(), np.ones(4, dtype=bool))
    dataset = PatternBoostSupervisedDataset(pool, config, mode="fm", num_samples=2, seed=3)

    example = dataset[0]
    edit_mask = example["edit_mask"].numpy().astype(bool)
    fixed_mask = example["fixed_mask"].numpy().astype(bool)

    assert np.array_equal(fixed_mask, ~edit_mask)
    np.testing.assert_allclose(
        example["draft_centers"][fixed_mask].numpy(),
        example["target_centers"][fixed_mask].numpy(),
    )
    assert example["vel_target"][fixed_mask].abs().sum().item() == 0.0
