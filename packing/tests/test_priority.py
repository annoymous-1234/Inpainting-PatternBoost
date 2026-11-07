"""Tests for controller-side raw-signal and demo-priority helpers."""

from __future__ import annotations

import torch

from pipelines.priority import (
    compute_cross_round_drift,
    compute_cross_round_drift_batched,
    compute_slot_dispersion,
    compute_slot_dispersion_batched,
    demo_priority_score,
    reduce_fm_aux_signals,
)


def test_dispersion_and_drift_helpers(make_config) -> None:
    samples = torch.tensor(
        [
            [[0.10, 0.10], [0.50, 0.50]],
            [[0.20, 0.10], [0.45, 0.55]],
            [[0.15, 0.15], [0.48, 0.52]],
        ],
        dtype=torch.float32,
    )
    dispersion = compute_slot_dispersion(samples)
    drift = compute_cross_round_drift(samples[0], samples)

    assert dispersion.shape == (2,)
    assert drift.shape == (2,)
    assert torch.all(dispersion >= 0.0)
    assert torch.all(drift >= 0.0)


def test_batched_dispersion_and_drift_match_per_item_helpers() -> None:
    samples = torch.tensor(
        [
            [
                [[0.10, 0.10], [0.50, 0.50]],
                [[0.20, 0.10], [0.45, 0.55]],
                [[0.15, 0.15], [0.48, 0.52]],
            ],
            [
                [[0.30, 0.30], [0.70, 0.70]],
                [[0.35, 0.25], [0.65, 0.75]],
                [[0.32, 0.28], [0.68, 0.72]],
            ],
        ],
        dtype=torch.float32,
    )
    prev = samples + 0.05

    dispersion = compute_slot_dispersion_batched(samples)
    drift = compute_cross_round_drift_batched(prev, samples)

    assert dispersion.shape == (2, 2)
    assert drift.shape == (2, 2)
    assert torch.allclose(dispersion[0], compute_slot_dispersion(samples[0]))
    assert torch.allclose(dispersion[1], compute_slot_dispersion(samples[1]))
    assert torch.allclose(drift[0], compute_cross_round_drift(prev[0], samples[0]))
    assert torch.allclose(drift[1], compute_cross_round_drift(prev[1], samples[1]))


def test_fm_aux_reduction_and_demo_priority(make_config) -> None:
    config = make_config(K=4)
    aux = {
        "terminal_velocity_norm": torch.tensor([[1.0, 2.0, 0.5, 0.1], [1.5, 1.0, 0.2, 0.4]]),
        "short_displacement": torch.tensor([[0.5, 0.3, 0.8, 0.1], [0.4, 0.2, 0.6, 0.2]]),
    }
    reduced = reduce_fm_aux_signals(aux)
    dispersion = torch.tensor([0.2, 0.4, 0.1, 0.3])
    drift = torch.tensor([0.05, 0.02, 0.01, 0.03])
    free_mask = torch.tensor([True, False, True, True])

    score = demo_priority_score(
        dispersion,
        drift=drift,
        fm_signal=reduced["short_displacement"],
        free_mask=free_mask,
        cfg=config,
    )

    assert reduced["terminal_velocity_norm"].shape == (4,)
    assert reduced["short_displacement"].shape == (4,)
    assert score.shape == (4,)
    assert torch.isneginf(score[1])
    assert score[2] > score[0]
