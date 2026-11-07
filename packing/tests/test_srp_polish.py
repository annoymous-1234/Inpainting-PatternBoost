"""Tests for FlowBoost-compatible SRP terminal polishing."""

from __future__ import annotations

import numpy as np

from circle_pack.geometry import is_valid_packing
from circle_pack.srp_polish import SRPPolishConfig, polish_terminal_layout_srp, strict_shrink_radii


def _tiny_config() -> SRPPolishConfig:
    return SRPPolishConfig(srp_Imax=2, srp_m=1, srp_lbfgs_maxiter=3)


def test_srp_polish_returns_legal_circles_for_random_layout() -> None:
    rng = np.random.default_rng(7)
    centers = rng.uniform(0.0, 1.0, size=(8, 2))
    radii = np.zeros(8, dtype=np.float64)

    result = polish_terminal_layout_srp(centers, radii, config=_tiny_config(), seed=11)

    assert result.circles.shape == (8, 3)
    assert result.radii.shape == (8,)
    assert result.score > 0.0
    assert result.legal
    assert is_valid_packing(result.circles[:, :2], result.circles[:, 2], tol=1.0e-9)


def test_srp_polish_demo_smoke() -> None:
    from scripts.demo_srp_polish_random import run_demo

    metrics = run_demo(
        num_samples=5,
        num_points=6,
        num_workers=1,
        seed=123,
        config=_tiny_config(),
    )

    assert metrics["num_samples"] == 5
    assert metrics["num_points"] == 6
    assert metrics["num_success"] == 5
    assert metrics["sum_r_max"] >= metrics["sum_r_mean"] > 0.0


def test_strict_shrink_repairs_wall_and_pair_violations() -> None:
    centers = np.array([[0.05, 0.05], [0.10, 0.05], [0.90, 0.90]], dtype=np.float64)
    radii = np.array([0.08, 0.08, 0.20], dtype=np.float64)

    repaired, info = strict_shrink_radii(centers, radii, margin=1.0e-8)

    assert info["shrink_total"] > 0.0
    assert is_valid_packing(centers, repaired, tol=1.0e-9)
    assert info["min_wall_clearance"] >= 0.0
    assert info["min_pair_clearance"] >= 0.0
