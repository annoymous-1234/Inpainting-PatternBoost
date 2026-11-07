"""LP-cache utilities for center-only PatternBoost candidates."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from circle_pack.geometry import is_valid_packing
from circle_pack.scipy_solver import ScipySolveResult, solve_radii_for_centers
from circle_pack.srp_polish import SRPPolishConfig, SRPPolishResult, polish_terminal_layout_srp
from patternboost.types import PatternBoostCandidate


def cap_expand_layout(
    centers: Any,
    *,
    radius_max: float = 0.5,
    feasibility_tol: float = 1.0e-12,
) -> np.ndarray:
    """Return LP-derived complete circles for fixed centers."""

    return solve_radii_for_centers(
        np.asarray(centers, dtype=np.float64)[:, :2],
        radius_max=radius_max,
        feasibility_tol=feasibility_tol,
    ).circles


def full_solver_polish_layout(
    centers: Any,
    *,
    radius_max: float = 0.5,
    feasibility_tol: float = 1.0e-12,
    num_restarts: int = 16,
    maxiter: int = 300,
    seed: int | None = None,
) -> ScipySolveResult:
    """Legacy-compatible wrapper that scores fixed centers with LP."""

    del num_restarts, maxiter, seed
    return solve_radii_for_centers(
        np.asarray(centers, dtype=np.float64)[:, :2],
        radius_max=radius_max,
        feasibility_tol=feasibility_tol,
    )


def cap_expand_candidate(
    candidate: PatternBoostCandidate,
    *,
    feasibility_tol: float,
    radius_max: float = 0.5,
) -> PatternBoostCandidate:
    """Refresh the LP-only score/radius cache for a full candidate."""

    if not bool(np.all(candidate.active_mask)):
        return candidate
    result = solve_radii_for_centers(candidate.centers, radius_max=radius_max, feasibility_tol=feasibility_tol)
    candidate.circles = result.circles
    candidate.radii = result.circles[:, 2].copy()
    candidate.legal = bool(result.success) and is_valid_packing(
        result.circles[:, :2],
        result.circles[:, 2],
        tol=max(feasibility_tol, 1.0e-9),
    )
    candidate.raw_score = float(result.sum_r) if candidate.legal else None
    candidate.full_active = True
    if result.success:
        candidate.lp_status = result.message if candidate.legal else f"invalid:{result.message}"
    else:
        candidate.lp_status = f"failed:{result.status}:{result.message}"
    return candidate


def select_solver_candidate_indices(
    candidates: list[PatternBoostCandidate],
    *,
    top_fraction: float,
    top_min: int,
    rank_key: str,
) -> list[int]:
    """Return no candidates because full free SLSQP polish is disabled by default."""

    del candidates, top_fraction, top_min, rank_key
    return []


def normalize_candidate_circles(circles: Any, safety_cfg: Any) -> np.ndarray:
    """Legacy helper that returns zero-radius circles clipped to the unit square."""

    centers = np.clip(np.asarray(circles, dtype=np.float64)[:, :2], 0.0, 1.0)
    return np.column_stack((centers, np.zeros(centers.shape[0], dtype=np.float64)))


@dataclass(frozen=True)
class _SRPWorkItem:
    candidate_index: int
    centers: np.ndarray
    radii: np.ndarray | None
    baseline_score: float
    seed: int
    config: SRPPolishConfig


@dataclass(frozen=True)
class _SRPWorkResult:
    candidate_index: int
    baseline_score: float
    result: SRPPolishResult


def candidate_final_score(candidate: PatternBoostCandidate) -> float | None:
    """Return the score that should be used for TopPool insertion."""

    if candidate.polish_score is not None:
        return float(candidate.polish_score)
    if candidate.raw_score is not None:
        return float(candidate.raw_score)
    return None


def srp_config_from_polish_config(polish_cfg: Any) -> SRPPolishConfig:
    """Build the SRP core config from the PatternBoost polish config section."""

    return SRPPolishConfig(
        srp_Imax=int(polish_cfg.srp_Imax),
        srp_m=int(polish_cfg.srp_m),
        srp_beta=float(polish_cfg.srp_beta),
        srp_backtrack=int(polish_cfg.srp_backtrack),
        srp_step_center=float(polish_cfg.srp_step_center),
        srp_step_radius=float(polish_cfg.srp_step_radius),
        srp_w_overlap=float(polish_cfg.srp_w_overlap),
        srp_w_wall=float(polish_cfg.srp_w_wall),
        srp_alpha_sum_r=float(polish_cfg.srp_alpha_sum_r),
        srp_lbfgs_gtol=float(polish_cfg.srp_lbfgs_gtol),
        srp_lbfgs_ftol=float(polish_cfg.srp_lbfgs_ftol),
        srp_lbfgs_maxiter=int(polish_cfg.srp_lbfgs_maxiter),
        srp_lbfgs_maxcor=int(polish_cfg.srp_lbfgs_maxcor),
        srp_use_small_radius_branch=bool(polish_cfg.srp_use_small_radius_branch),
        srp_init_r0_small=float(polish_cfg.srp_init_r0_small),
        srp_init_rjitter_small=float(polish_cfg.srp_init_rjitter_small),
        srp_small_branch_step_scale=float(polish_cfg.srp_small_branch_step_scale),
        srp_strict_shrink=bool(polish_cfg.srp_strict_shrink),
        srp_strict_shrink_margin=float(polish_cfg.srp_strict_shrink_margin),
    )


def select_srp_candidate_indices(
    candidates: list[PatternBoostCandidate],
    *,
    top_k: int,
) -> list[int]:
    """Return full-active candidate indices selected for SRP polishing."""

    if top_k <= 0:
        return []
    eligible = [
        idx
        for idx, candidate in enumerate(candidates)
        if candidate.full_active and candidate.raw_score is not None
    ]
    eligible.sort(key=lambda idx: (float(candidates[idx].raw_score), -idx), reverse=True)
    return eligible[: int(top_k)]


def _run_srp_work_item(item: _SRPWorkItem) -> _SRPWorkResult:
    result = polish_terminal_layout_srp(
        item.centers,
        item.radii,
        config=item.config,
        seed=item.seed,
    )
    return _SRPWorkResult(candidate_index=item.candidate_index, baseline_score=item.baseline_score, result=result)


def _apply_srp_result(
    candidate: PatternBoostCandidate,
    work_result: _SRPWorkResult,
    *,
    min_improvement: float,
) -> bool:
    result = work_result.result
    if not result.legal:
        candidate.polish_score = None
        candidate.metadata["srp_polish_status"] = "rejected_illegal"
        return False
    if result.score <= work_result.baseline_score + float(min_improvement):
        candidate.polish_score = None
        candidate.metadata["srp_polish_status"] = "rejected_no_improvement"
        candidate.metadata["srp_score"] = float(result.score)
        candidate.metadata["srp_branch"] = result.branch
        return False

    candidate.centers = result.centers.copy()
    candidate.radii = result.radii.copy()
    candidate.circles = result.circles.copy()
    candidate.polish_score = float(result.score)
    candidate.legal = True
    candidate.full_active = True
    candidate.lp_status = "srp_flowboost_hard_project"
    candidate.metadata.update(
        {
            "srp_polish_status": "accepted",
            "srp_score": float(result.score),
            "srp_baseline_score": float(work_result.baseline_score),
            "srp_branch": result.branch,
            "srp_loss_after": float(result.loss_after),
            "srp_min_wall_clearance": float(result.min_wall_clearance),
            "srp_min_pair_clearance": float(result.min_pair_clearance),
        }
    )
    return True


def srp_polish_candidate(
    candidate: PatternBoostCandidate,
    *,
    polish_cfg: Any,
    seed: int,
) -> bool:
    """Run SRP polish on one candidate and mutate it only if the result improves."""

    if not candidate.full_active or candidate.raw_score is None:
        return False
    config = srp_config_from_polish_config(polish_cfg)
    work_item = _SRPWorkItem(
        candidate_index=0,
        centers=np.asarray(candidate.centers, dtype=np.float64).copy(),
        radii=None if candidate.radii is None else np.asarray(candidate.radii, dtype=np.float64).copy(),
        baseline_score=float(candidate.raw_score),
        seed=int(seed),
        config=config,
    )
    return _apply_srp_result(
        candidate,
        _run_srp_work_item(work_item),
        min_improvement=float(polish_cfg.srp_min_improvement),
    )


def srp_polish_candidates(
    candidates: list[PatternBoostCandidate],
    *,
    polish_cfg: Any,
    generation_idx: int,
    runtime_seed: int,
) -> list[int]:
    """Polish selected terminal candidates with optional process-level parallelism."""

    if not bool(polish_cfg.srp_enabled):
        return []
    selected = select_srp_candidate_indices(candidates, top_k=int(polish_cfg.srp_top_k))
    if not selected:
        return []

    core_config = srp_config_from_polish_config(polish_cfg)
    work_items = [
        _SRPWorkItem(
            candidate_index=idx,
            centers=np.asarray(candidates[idx].centers, dtype=np.float64).copy(),
            radii=None if candidates[idx].radii is None else np.asarray(candidates[idx].radii, dtype=np.float64).copy(),
            baseline_score=float(candidates[idx].raw_score),
            seed=int(runtime_seed) + 1_000_000 * int(generation_idx) + idx,
            config=core_config,
        )
        for idx in selected
    ]

    workers = int(polish_cfg.srp_num_workers)
    if workers <= 1:
        results = [_run_srp_work_item(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_srp_work_item, work_items))

    results.sort(key=lambda item: item.candidate_index)
    accepted: list[int] = []
    for work_result in results:
        candidate = candidates[work_result.candidate_index]
        if _apply_srp_result(
            candidate,
            work_result,
            min_improvement=float(polish_cfg.srp_min_improvement),
        ):
            accepted.append(work_result.candidate_index)
    return accepted
