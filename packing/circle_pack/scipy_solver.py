"""SciPy-based non-convex solver for completing a circle packing state."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, linprog, minimize

from circle_pack.geometry import (
    active_packing_is_feasible,
    box_capacity,
    compute_initial_caps,
    is_valid_packing,
    lazy_update_caps_from_commit,
    normalize_physical_circles,
)


@dataclass(frozen=True)
class ScipySolveResult:
    """Result bundle for one completed circle packing solve."""

    circles: np.ndarray
    sum_r: float
    free_sum_r: float
    success: bool
    status: int
    message: str
    n_starts: int
    best_start_idx: int
    min_constraint_margin: float


@dataclass(frozen=True)
class _CandidateResult:
    """Internal candidate bundle for one start or one solver run."""

    circles: np.ndarray
    sum_r: float
    free_sum_r: float
    min_constraint_margin: float
    start_idx: int
    success: bool
    status: int
    message: str


@dataclass(frozen=True)
class _ConstraintLayout:
    """Precomputed indexing for constraint evaluation and Jacobians."""

    num_free: int
    free_pairs: np.ndarray
    fixed_pairs: np.ndarray


def _as_bool_mask(active_mask: Any, num_slots: int) -> np.ndarray:
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (num_slots,):
        raise ValueError(f"active_mask must have shape ({num_slots},), got {mask.shape}.")
    return mask


def _as_circle_array(circles: Any) -> np.ndarray:
    array = np.asarray(circles, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"circles must have shape [K, 3], got {array.shape}.")
    return array


def _as_center_array(centers: Any) -> np.ndarray:
    array = np.asarray(centers, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"centers must have shape [K, 2], got {array.shape}.")
    return array


def solve_radii_for_centers(
    centers: Any,
    *,
    radius_max: float = 0.5,
    feasibility_tol: float = 1.0e-12,
) -> ScipySolveResult:
    """Solve optimal radii for a complete fixed-center layout using linear programming."""

    center_array = np.clip(_as_center_array(centers), 0.0, 1.0)
    num_slots = int(center_array.shape[0])
    if num_slots == 0:
        circles = np.empty((0, 3), dtype=np.float64)
        return ScipySolveResult(
            circles=circles,
            sum_r=0.0,
            free_sum_r=0.0,
            success=True,
            status=0,
            message="No centers to score.",
            n_starts=1,
            best_start_idx=0,
            min_constraint_margin=float("inf"),
        )

    upper = np.minimum(box_capacity(center_array), float(radius_max))
    upper = np.maximum(upper, 0.0)
    bounds = [(0.0, float(value)) for value in upper]

    pair_rows: list[np.ndarray] = []
    pair_caps: list[float] = []
    for i in range(num_slots):
        for j in range(i + 1, num_slots):
            row = np.zeros(num_slots, dtype=np.float64)
            row[i] = 1.0
            row[j] = 1.0
            pair_rows.append(row)
            pair_caps.append(float(np.linalg.norm(center_array[i] - center_array[j])))
    a_ub = np.vstack(pair_rows) if pair_rows else None
    b_ub = np.asarray(pair_caps, dtype=np.float64) if pair_caps else None

    highs_tol = max(float(feasibility_tol), 1.0e-10)
    result = linprog(
        c=-np.ones(num_slots, dtype=np.float64),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": highs_tol, "dual_feasibility_tolerance": highs_tol},
    )
    radii = np.zeros(num_slots, dtype=np.float64)
    if result.x is not None and np.all(np.isfinite(result.x)):
        radii = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        radii = np.minimum(radii, upper)
        radii[radii <= float(feasibility_tol)] = 0.0

    bound_margin = float(np.min(upper - radii)) if num_slots else float("inf")
    if pair_rows:
        pair_margin = float(np.min(b_ub - a_ub @ radii))
        min_margin = min(bound_margin, pair_margin, float(np.min(radii)))
    else:
        min_margin = min(bound_margin, float(np.min(radii)))
    circles = np.column_stack((center_array, radii))
    success = bool(result.success) and is_valid_packing(center_array, radii, tol=max(float(feasibility_tol), 1.0e-9))
    return ScipySolveResult(
        circles=circles,
        sum_r=float(np.sum(radii)),
        free_sum_r=float(np.sum(radii)),
        success=success,
        status=int(result.status),
        message=str(result.message),
        n_starts=1,
        best_start_idx=0,
        min_constraint_margin=float(min_margin),
    )


def complete_circles_from_centers(
    centers: Any,
    *,
    radius_max: float = 0.5,
    feasibility_tol: float = 1.0e-12,
) -> np.ndarray:
    """Return LP-derived `[K, 3]` circles for complete fixed centers."""

    return solve_radii_for_centers(
        centers,
        radius_max=radius_max,
        feasibility_tol=feasibility_tol,
    ).circles


def _build_constraint_layout(num_free: int, num_fixed: int) -> _ConstraintLayout:
    free_pairs = np.array([(i, j) for i in range(num_free) for j in range(i + 1, num_free)], dtype=np.int64)
    if free_pairs.size == 0:
        free_pairs = np.empty((0, 2), dtype=np.int64)
    fixed_pairs = np.array([(i, j) for i in range(num_free) for j in range(num_fixed)], dtype=np.int64)
    if fixed_pairs.size == 0:
        fixed_pairs = np.empty((0, 2), dtype=np.int64)
    return _ConstraintLayout(num_free=num_free, free_pairs=free_pairs, fixed_pairs=fixed_pairs)


def _objective(z: np.ndarray) -> float:
    return -float(np.sum(z[2::3]))


def _objective_grad(z: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(z, dtype=np.float64)
    grad[2::3] = -1.0
    return grad


def _pair_radius_cap(distance_sq: float, other_radius: float, delta: float) -> float:
    slack_sq = max(distance_sq - delta, 0.0)
    return float(np.sqrt(slack_sq) - other_radius)


def _constraint_values(
    z: np.ndarray,
    layout: _ConstraintLayout,
    fixed_centers: np.ndarray,
    fixed_radii: np.ndarray,
    delta: float,
) -> np.ndarray:
    free = np.asarray(z, dtype=np.float64).reshape(layout.num_free, 3)
    centers = free[:, :2]
    radii = free[:, 2]

    values = np.empty(
        4 * layout.num_free + layout.free_pairs.shape[0] + layout.fixed_pairs.shape[0],
        dtype=np.float64,
    )
    values[0 : layout.num_free] = centers[:, 0] - radii - delta
    values[layout.num_free : 2 * layout.num_free] = 1.0 - centers[:, 0] - radii - delta
    values[2 * layout.num_free : 3 * layout.num_free] = centers[:, 1] - radii - delta
    values[3 * layout.num_free : 4 * layout.num_free] = 1.0 - centers[:, 1] - radii - delta

    offset = 4 * layout.num_free
    for pair_idx, (i, j) in enumerate(layout.free_pairs):
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        rr = radii[i] + radii[j]
        values[offset + pair_idx] = dx * dx + dy * dy - rr * rr - delta

    offset += layout.free_pairs.shape[0]
    for pair_idx, (i, j) in enumerate(layout.fixed_pairs):
        dx = centers[i, 0] - fixed_centers[j, 0]
        dy = centers[i, 1] - fixed_centers[j, 1]
        rr = radii[i] + fixed_radii[j]
        values[offset + pair_idx] = dx * dx + dy * dy - rr * rr - delta

    return values


def _constraint_jacobian(
    z: np.ndarray,
    layout: _ConstraintLayout,
    fixed_centers: np.ndarray,
    fixed_radii: np.ndarray,
    delta: float,
) -> np.ndarray:
    del delta
    free = np.asarray(z, dtype=np.float64).reshape(layout.num_free, 3)
    centers = free[:, :2]
    radii = free[:, 2]

    jac = np.zeros(
        (
            4 * layout.num_free + layout.free_pairs.shape[0] + layout.fixed_pairs.shape[0],
            3 * layout.num_free,
        ),
        dtype=np.float64,
    )

    for i in range(layout.num_free):
        base = 3 * i
        jac[i, base] = 1.0
        jac[i, base + 2] = -1.0

        jac[layout.num_free + i, base] = -1.0
        jac[layout.num_free + i, base + 2] = -1.0

        jac[2 * layout.num_free + i, base + 1] = 1.0
        jac[2 * layout.num_free + i, base + 2] = -1.0

        jac[3 * layout.num_free + i, base + 1] = -1.0
        jac[3 * layout.num_free + i, base + 2] = -1.0

    offset = 4 * layout.num_free
    for pair_idx, (i, j) in enumerate(layout.free_pairs):
        row = offset + pair_idx
        base_i = 3 * i
        base_j = 3 * j
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        rr = radii[i] + radii[j]

        jac[row, base_i] = 2.0 * dx
        jac[row, base_i + 1] = 2.0 * dy
        jac[row, base_i + 2] = -2.0 * rr

        jac[row, base_j] = -2.0 * dx
        jac[row, base_j + 1] = -2.0 * dy
        jac[row, base_j + 2] = -2.0 * rr

    offset += layout.free_pairs.shape[0]
    for pair_idx, (i, j) in enumerate(layout.fixed_pairs):
        row = offset + pair_idx
        del j
        base_i = 3 * i
        dx = centers[i, 0] - fixed_centers[layout.fixed_pairs[pair_idx, 1], 0]
        dy = centers[i, 1] - fixed_centers[layout.fixed_pairs[pair_idx, 1], 1]
        rr = radii[i] + fixed_radii[layout.fixed_pairs[pair_idx, 1]]

        jac[row, base_i] = 2.0 * dx
        jac[row, base_i + 1] = 2.0 * dy
        jac[row, base_i + 2] = -2.0 * rr

    return jac


def _normalize_free_circles(
    free_circles: np.ndarray,
    radius_max: float,
) -> np.ndarray:
    safety_cfg = SimpleNamespace(
        clip_positions=True,
        clip_radii=True,
        radius_max=float(radius_max),
    )
    return normalize_physical_circles(free_circles, safety_cfg).astype(np.float64, copy=False)


def _build_feasible_start(
    free_circles: np.ndarray,
    fixed_centers: np.ndarray,
    fixed_radii: np.ndarray,
    radius_max: float,
    delta: float,
) -> np.ndarray:
    normalized = _normalize_free_circles(free_circles, radius_max)
    num_free = normalized.shape[0]
    if num_free == 0:
        return normalized

    centers = normalized[:, :2].copy()
    radii_prop = normalized[:, 2].copy()
    linear_cap = compute_initial_caps(centers, fixed_centers, fixed_radii).astype(np.float64, copy=True)
    alive = np.ones(num_free, dtype=bool)
    radii = np.zeros(num_free, dtype=np.float64)

    accepted_centers = np.empty((0, 2), dtype=np.float64)
    accepted_radii = np.empty(0, dtype=np.float64)

    for i in range(num_free):
        box_cap = min(
            centers[i, 0] - delta,
            1.0 - centers[i, 0] - delta,
            centers[i, 1] - delta,
            1.0 - centers[i, 1] - delta,
        )
        exact_cap = box_cap

        for fixed_idx in range(fixed_radii.shape[0]):
            delta_xy = centers[i] - fixed_centers[fixed_idx]
            distance_sq = float(np.dot(delta_xy, delta_xy))
            exact_cap = min(exact_cap, _pair_radius_cap(distance_sq, float(fixed_radii[fixed_idx]), delta))

        for prev_idx in range(accepted_radii.shape[0]):
            delta_xy = centers[i] - accepted_centers[prev_idx]
            distance_sq = float(np.dot(delta_xy, delta_xy))
            exact_cap = min(exact_cap, _pair_radius_cap(distance_sq, float(accepted_radii[prev_idx]), delta))

        radii[i] = max(0.0, min(float(radii_prop[i]), float(linear_cap[i]), float(exact_cap)))
        alive[i] = False
        lazy_update_caps_from_commit(centers, linear_cap, alive, centers[i], float(radii[i]))
        accepted_centers = np.vstack((accepted_centers, centers[i].reshape(1, 2)))
        accepted_radii = np.append(accepted_radii, radii[i])

    return np.column_stack((centers, radii))


def _complete_with_free_circles(
    base_circles: np.ndarray,
    free_ids: np.ndarray,
    free_solution: np.ndarray,
    feasibility_tol: float,
) -> np.ndarray:
    completed = np.asarray(base_circles, dtype=np.float64).copy()
    completed[free_ids] = np.asarray(free_solution, dtype=np.float64)
    small = completed[:, 2] <= feasibility_tol
    completed[small, 2] = 0.0
    return completed


def _evaluate_full_candidate(
    base_circles: np.ndarray,
    free_ids: np.ndarray,
    free_solution: np.ndarray,
    fixed_sum_r: float,
    start_idx: int,
    success: bool,
    status: int,
    message: str,
    feasibility_tol: float,
    layout: _ConstraintLayout,
    fixed_centers: np.ndarray,
    fixed_radii: np.ndarray,
    delta: float,
) -> _CandidateResult:
    free_solution = np.asarray(free_solution, dtype=np.float64).reshape(layout.num_free, 3)
    circles = _complete_with_free_circles(base_circles, free_ids, free_solution, feasibility_tol)
    margin = float(np.min(_constraint_values(free_solution.reshape(-1), layout, fixed_centers, fixed_radii, delta)))
    free_sum_r = float(np.sum(circles[free_ids, 2]))
    return _CandidateResult(
        circles=circles,
        sum_r=float(fixed_sum_r + free_sum_r),
        free_sum_r=free_sum_r,
        min_constraint_margin=margin,
        start_idx=int(start_idx),
        success=bool(success),
        status=int(status),
        message=str(message),
    )


def _make_zero_radius_completion(
    base_circles: np.ndarray,
    free_ids: np.ndarray,
    radius_max: float,
    feasibility_tol: float,
) -> np.ndarray:
    completed = np.asarray(base_circles, dtype=np.float64).copy()
    if free_ids.size == 0:
        return completed
    normalized = _normalize_free_circles(completed[free_ids], radius_max)
    normalized[:, 2] = 0.0
    completed[free_ids] = normalized
    completed[completed[:, 2] <= feasibility_tol, 2] = 0.0
    return completed


def solve_packing_observation(
    observation: dict[str, Any],
    *,
    radius_max: float = 0.5,
    feasibility_tol: float = 1.0e-12,
    num_restarts: int = 16,
    maxiter: int = 300,
    seed: int | None = None,
) -> ScipySolveResult:
    """Solve the free slots of one environment observation with SciPy."""

    circles = _as_circle_array(observation["circles"])
    active_mask = _as_bool_mask(observation["active_mask"], circles.shape[0])

    if not active_packing_is_feasible(circles, active_mask, tol=feasibility_tol):
        raise ValueError("The active core is infeasible; refusing to solve from an invalid environment state.")

    free_ids = np.flatnonzero(~active_mask).astype(np.int64, copy=False)
    fixed_ids = np.flatnonzero(active_mask).astype(np.int64, copy=False)
    fixed_centers = circles[fixed_ids, :2].astype(np.float64, copy=False)
    fixed_radii = circles[fixed_ids, 2].astype(np.float64, copy=False)
    fixed_sum_r = float(np.sum(fixed_radii))

    if free_ids.size == 0:
        final_circles = circles.copy()
        return ScipySolveResult(
            circles=final_circles,
            sum_r=float(np.sum(final_circles[:, 2])),
            free_sum_r=0.0,
            success=True,
            status=0,
            message="No free slots to optimize.",
            n_starts=0,
            best_start_idx=-1,
            min_constraint_margin=float("inf"),
        )

    num_starts = max(1, int(num_restarts))
    rng = np.random.default_rng(seed)
    delta = max(1.0e-9, 100.0 * float(feasibility_tol))
    layout = _build_constraint_layout(int(free_ids.size), int(fixed_ids.size))

    bounds_upper = np.tile(np.array([1.0, 1.0, float(radius_max)], dtype=np.float64), layout.num_free)
    bounds = Bounds(np.zeros(3 * layout.num_free, dtype=np.float64), bounds_upper)
    constraint = NonlinearConstraint(
        lambda z: _constraint_values(z, layout, fixed_centers, fixed_radii, delta),
        0.0,
        np.inf,
        jac=lambda z: _constraint_jacobian(z, layout, fixed_centers, fixed_radii, delta),
    )

    initial_candidates: list[_CandidateResult] = []
    solve_candidates: list[_CandidateResult] = []

    warm_start = circles[free_ids].astype(np.float64, copy=True)
    for start_idx in range(num_starts):
        if start_idx == 0:
            proposal = warm_start
        else:
            proposal = np.empty((layout.num_free, 3), dtype=np.float64)
            proposal[:, :2] = rng.uniform(0.0, 1.0, size=(layout.num_free, 2))
            proposal[:, 2] = rng.uniform(0.0, float(radius_max), size=layout.num_free)

        initial_free = _build_feasible_start(proposal, fixed_centers, fixed_radii, radius_max, delta)
        initial_candidate = _evaluate_full_candidate(
            circles,
            free_ids,
            initial_free,
            fixed_sum_r,
            start_idx,
            True,
            0,
            "Feasible initial start.",
            feasibility_tol,
            layout,
            fixed_centers,
            fixed_radii,
            delta,
        )
        initial_candidates.append(initial_candidate)

        result = minimize(
            _objective,
            initial_free.reshape(-1),
            method="SLSQP",
            jac=_objective_grad,
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": int(maxiter), "ftol": float(feasibility_tol), "disp": False},
        )
        if not np.all(np.isfinite(result.x)):
            continue
        solve_candidate = _evaluate_full_candidate(
            circles,
            free_ids,
            result.x.reshape(layout.num_free, 3),
            fixed_sum_r,
            start_idx,
            bool(result.success),
            int(result.status),
            str(result.message),
            feasibility_tol,
            layout,
            fixed_centers,
            fixed_radii,
            delta,
        )
        if solve_candidate.min_constraint_margin >= -float(feasibility_tol):
            solve_candidates.append(solve_candidate)

    if solve_candidates:
        best = max(solve_candidates, key=lambda candidate: (candidate.sum_r, candidate.free_sum_r, -candidate.start_idx))
    else:
        best_initial = max(
            initial_candidates,
            key=lambda candidate: (candidate.sum_r, candidate.free_sum_r, -candidate.start_idx),
        )
        if best_initial.free_sum_r > float(feasibility_tol):
            best = _CandidateResult(
                circles=best_initial.circles,
                sum_r=best_initial.sum_r,
                free_sum_r=best_initial.free_sum_r,
                min_constraint_margin=best_initial.min_constraint_margin,
                start_idx=best_initial.start_idx,
                success=False,
                status=-1,
                message="Falling back to the best feasible initial solution.",
            )
        else:
            zero_completion = _make_zero_radius_completion(circles, free_ids, radius_max, feasibility_tol)
            best = _CandidateResult(
                circles=zero_completion,
                sum_r=float(np.sum(zero_completion[:, 2])),
                free_sum_r=0.0,
                min_constraint_margin=float("inf"),
                start_idx=best_initial.start_idx,
                success=False,
                status=-2,
                message="No feasible positive-radius free-circle solution found; returning fixed circles with zero-radius free slots.",
            )

    if not is_valid_packing(best.circles[:, :2], best.circles[:, 2], tol=feasibility_tol):
        raise AssertionError("SciPy solver produced an invalid full packing.")

    return ScipySolveResult(
        circles=best.circles,
        sum_r=best.sum_r,
        free_sum_r=best.free_sum_r,
        success=best.success,
        status=best.status,
        message=best.message,
        n_starts=num_starts,
        best_start_idx=best.start_idx,
        min_constraint_margin=best.min_constraint_margin,
    )


def solve_packing_env(
    env: Any,
    *,
    num_restarts: int = 16,
    maxiter: int = 300,
    seed: int | None = None,
) -> ScipySolveResult:
    """Solve a packing completion directly from a live environment object."""

    observation = env.observe()
    if "centers" in observation and observation.get("circles") is None:
        return solve_radii_for_centers(
            observation["centers"],
            radius_max=float(env.config.safety.radius_max),
            feasibility_tol=float(env.config.safety.feasibility_tol),
        )
    return solve_packing_observation(
        observation,
        radius_max=float(env.config.safety.radius_max),
        feasibility_tol=float(env.config.safety.feasibility_tol),
        num_restarts=num_restarts,
        maxiter=maxiter,
        seed=seed,
    )
