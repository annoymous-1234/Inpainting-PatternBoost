"""Tests for geometry helpers and cap computations."""

from __future__ import annotations

import numpy as np

from circle_pack.geometry import (
    box_capacity,
    compute_initial_caps,
    full_recompute_caps,
    lazy_update_caps_from_commit,
    latent_to_physical_circle,
)


def test_latent_to_physical_circle_is_single_circle_legal() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(256, 3))
    physical = latent_to_physical_circle(latent)

    assert physical.shape == (256, 3)
    assert np.all(physical[:, 0] >= 0.0)
    assert np.all(physical[:, 0] <= 1.0)
    assert np.all(physical[:, 1] >= 0.0)
    assert np.all(physical[:, 1] <= 1.0)
    assert np.all(physical[:, 2] >= 0.0)
    assert np.all(physical[:, 2] <= 0.5)
    assert np.all(physical[:, 2] <= box_capacity(physical[:, :2]) + 1.0e-12)


def test_initial_caps_match_empty_and_nonempty_active_core() -> None:
    cand_centers = np.array(
        [
            [0.10, 0.10],
            [0.85, 0.50],
            [0.65, 0.50],
        ],
        dtype=np.float64,
    )

    empty_caps = compute_initial_caps(
        cand_centers,
        active_centers=np.empty((0, 2), dtype=np.float64),
        active_radii=np.empty(0, dtype=np.float64),
    )
    np.testing.assert_allclose(empty_caps, np.array([0.10, 0.15, 0.35], dtype=np.float64))

    active_centers = np.array([[0.50, 0.50]], dtype=np.float64)
    active_radii = np.array([0.20], dtype=np.float64)
    caps = compute_initial_caps(cand_centers, active_centers, active_radii)

    expected = np.array([0.10, 0.15, -0.05], dtype=np.float64)
    np.testing.assert_allclose(caps, expected, atol=1.0e-12)


def test_lazy_update_matches_full_recompute_reference() -> None:
    active_centers = np.array([[0.20, 0.20]], dtype=np.float64)
    active_radii = np.array([0.08], dtype=np.float64)
    cand_centers = np.array(
        [
            [0.75, 0.20],
            [0.20, 0.78],
            [0.78, 0.78],
        ],
        dtype=np.float64,
    )
    cap = compute_initial_caps(cand_centers, active_centers, active_radii)
    alive = np.ones(3, dtype=bool)

    accepted_centers = []
    accepted_radii = []
    commit_plan = [(0, 0.10), (1, 0.12)]

    for local_idx, commit_radius in commit_plan:
        alive[local_idx] = False
        accepted_centers.append(cand_centers[local_idx].copy())
        accepted_radii.append(commit_radius)
        lazy_update_caps_from_commit(
            cand_centers,
            cap,
            alive,
            cand_centers[local_idx],
            commit_radius,
        )

        reference = full_recompute_caps(
            cand_centers[alive],
            active_centers,
            active_radii,
            extra_centers=np.asarray(accepted_centers, dtype=np.float64),
            extra_radii=np.asarray(accepted_radii, dtype=np.float64),
        )
        np.testing.assert_allclose(cap[alive], reference, atol=1.0e-12)
