"""Tests for PatternBoost polishing utilities."""

from __future__ import annotations

import numpy as np

from circle_pack.config import PatternBoostPolishSection
from circle_pack.geometry import is_valid_packing
from circle_pack.srp_polish import SRPPolishResult
from patternboost.polish import (
    _SRPWorkResult,
    candidate_final_score,
    cap_expand_layout,
    full_solver_polish_layout,
    select_solver_candidate_indices,
    select_srp_candidate_indices,
    srp_polish_candidate,
    srp_polish_candidates,
)
from patternboost.types import PatternBoostCandidate


def test_cap_expansion_keeps_layout_legal_and_improves_score() -> None:
    circles = np.array(
        [
            [0.25, 0.25, 0.05],
            [0.75, 0.25, 0.05],
            [0.50, 0.75, 0.05],
        ],
        dtype=np.float64,
    )
    expanded = cap_expand_layout(circles[:, :2])

    assert expanded[:, 2].sum() >= circles[:, 2].sum()
    assert is_valid_packing(expanded[:, :2], expanded[:, 2])


def test_solver_selection_and_full_solver_polish() -> None:
    assert select_solver_candidate_indices([], top_fraction=0.5, top_min=1, rank_key="polish_score") == []

    circles = np.array([[0.25, 0.25, 0.05], [0.75, 0.75, 0.05]], dtype=np.float64)
    result = full_solver_polish_layout(circles[:, :2], num_restarts=2, maxiter=50, seed=7)

    assert result.sum_r >= circles[:, 2].sum()
    assert is_valid_packing(result.circles[:, :2], result.circles[:, 2])


def _candidate(score: float, *, offset: float = 0.0) -> PatternBoostCandidate:
    centers = np.array(
        [
            [0.25 + offset, 0.25],
            [0.75 + offset, 0.25],
            [0.50 + offset, 0.75],
        ],
        dtype=np.float64,
    )
    radii = np.full(3, 0.05, dtype=np.float64)
    circles = np.column_stack((centers, radii))
    return PatternBoostCandidate(
        centers=centers,
        radii=radii,
        circles=circles,
        active_mask=np.ones(3, dtype=bool),
        raw_score=float(score),
        legal=True,
        full_active=True,
    )


def _srp_result(score: float) -> SRPPolishResult:
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.50, 0.75]], dtype=np.float64)
    radii = np.full(3, score / 3.0, dtype=np.float64)
    circles = np.column_stack((centers, radii))
    return SRPPolishResult(
        centers=centers,
        radii=radii,
        circles=circles,
        score=float(score),
        legal=True,
        branch="smallr",
        loss_after=0.0,
        min_wall_clearance=0.1,
        min_pair_clearance=0.1,
        branches=(),
    )


def test_srp_polish_candidate_rejects_non_improving_result(monkeypatch) -> None:
    candidate = _candidate(1.0)
    cfg = PatternBoostPolishSection(srp_enabled=True, srp_min_improvement=1.0e-6)

    def fake_run(item):
        return _SRPWorkResult(candidate_index=item.candidate_index, baseline_score=item.baseline_score, result=_srp_result(1.0))

    monkeypatch.setattr("patternboost.polish._run_srp_work_item", fake_run)

    assert not srp_polish_candidate(candidate, polish_cfg=cfg, seed=123)
    assert candidate.raw_score == 1.0
    assert candidate.polish_score is None
    assert candidate_final_score(candidate) == 1.0
    assert candidate.metadata["srp_polish_status"] == "rejected_no_improvement"


def test_srp_polish_candidate_keeps_lp_raw_score_and_records_polish_score(monkeypatch) -> None:
    candidate = _candidate(1.0)
    cfg = PatternBoostPolishSection(srp_enabled=True, srp_min_improvement=1.0e-6)

    def fake_run(item):
        return _SRPWorkResult(candidate_index=item.candidate_index, baseline_score=item.baseline_score, result=_srp_result(1.2))

    monkeypatch.setattr("patternboost.polish._run_srp_work_item", fake_run)

    assert srp_polish_candidate(candidate, polish_cfg=cfg, seed=123)
    assert candidate.raw_score == 1.0
    assert candidate.polish_score == 1.2
    assert candidate_final_score(candidate) == 1.2


def test_srp_selection_uses_only_full_candidates_with_lp_raw_score() -> None:
    candidates = [_candidate(0.2), _candidate(0.4), _candidate(0.3)]
    candidates[1].raw_score = None
    candidates[2].full_active = False

    assert select_srp_candidate_indices(candidates, top_k=3) == [0]


def test_srp_polish_candidates_parallel_branch_preserves_candidate_order(monkeypatch) -> None:
    candidates = [_candidate(0.1), _candidate(0.3, offset=-0.02), _candidate(0.2, offset=0.02)]
    cfg = PatternBoostPolishSection(srp_enabled=True, srp_top_k=2, srp_num_workers=2, srp_min_improvement=0.0)

    def fake_run(item):
        return _SRPWorkResult(
            candidate_index=item.candidate_index,
            baseline_score=item.baseline_score,
            result=_srp_result(item.baseline_score + 1.0),
        )

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            del fn
            return [fake_run(item) for item in reversed(list(items))]

    monkeypatch.setattr("patternboost.polish.ProcessPoolExecutor", FakeExecutor)

    accepted = srp_polish_candidates(candidates, polish_cfg=cfg, generation_idx=0, runtime_seed=5)

    assert accepted == [1, 2]
    assert candidates[0].polish_score is None
    assert candidates[1].raw_score == 0.3
    assert candidates[1].polish_score == 1.3
    assert candidates[2].raw_score == 0.2
    assert candidates[2].polish_score == 1.2
