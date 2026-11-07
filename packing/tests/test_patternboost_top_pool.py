"""Tests for PatternBoost TopPool."""

from __future__ import annotations

import numpy as np

from patternboost.top_pool import TopPool


def _layout(scale: float = 1.0) -> np.ndarray:
    return np.array(
        [
            [0.20, 0.20],
            [0.80, 0.20],
            [0.50, 0.75 * scale],
        ],
        dtype=np.float64,
    )


def test_top_pool_insert_prune_dedup_and_save_load(tmp_path) -> None:
    pool = TopPool(max_size=1, dedup_tol=1.0e-5)

    assert pool.insert(_layout(), np.ones(3, dtype=bool), metadata={"source": "a"})
    assert not pool.insert(_layout()[::-1], np.ones(3, dtype=bool), metadata={"source": "dup"})
    assert pool.insert(_layout(0.9), np.ones(3, dtype=bool), metadata={"source": "b"})

    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].metadata["source"] == "b"

    path = tmp_path / "top_pool.pt"
    pool.save(path)
    loaded = TopPool.load(path)
    assert len(loaded) == 1
    assert loaded.best().score == entries[0].score


def test_top_pool_rank_biased_sampling_prefers_high_scores() -> None:
    pool = TopPool(max_size=4, dedup_tol=1.0e-7)
    for idx, scale in enumerate([1.0, 0.9, 0.8]):
        layout = _layout(scale)
        layout[:, 0] += idx * 0.01
        pool.insert(layout, np.ones(3, dtype=bool), metadata={"idx": idx})

    rng = np.random.default_rng(5)
    sampling = {"mode": "score_bucket_rank_mixture", "uniform_mix": 0.0, "top_fraction": 0.3, "alpha": 1.0, "offset": 1.0}
    sampled = [pool.sample_seed(rng, sampling).metadata["idx"] for _ in range(20)]

    assert sampled == [pool.best().metadata["idx"]] * 20


def test_top_pool_uniform_mix_respects_top_fraction() -> None:
    pool = TopPool(max_size=8, dedup_tol=1.0e-7)
    scores = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
    for idx, score in enumerate(scores):
        layout = _layout()
        layout[:, 0] += idx * 0.01
        assert pool.insert(layout, np.ones(3, dtype=bool), score=score, metadata={"idx": idx})

    rng = np.random.default_rng(11)
    sampling = {
        "mode": "score_bucket_rank_mixture",
        "uniform_mix": 1.0,
        "top_fraction": 0.5,
        "alpha": 1.0,
        "offset": 1.0,
    }
    sampled = [pool.sample_seed(rng, sampling).metadata["idx"] for _ in range(100)]

    assert set(sampled) <= {0, 1, 2}
