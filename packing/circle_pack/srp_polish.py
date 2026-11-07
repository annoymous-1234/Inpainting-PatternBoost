"""FlowBoost-compatible SRP polishing for complete circle layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numba import njit
from scipy.optimize import linprog, minimize

from circle_pack.geometry import is_valid_packing


EPS = 1e-12


@dataclass(frozen=True)
class SRPPolishConfig:
    """Hyperparameters for FlowBoost-style SRP terminal polishing."""

    srp_Imax: int = 400
    srp_m: int = 30
    srp_beta: float = 0.99
    srp_backtrack: int = 5
    srp_step_center: float = 0.05
    srp_step_radius: float = 0.001
    srp_w_overlap: float = 1000.0
    srp_w_wall: float = 1000.0
    srp_alpha_sum_r: float = 0.2
    srp_lbfgs_gtol: float = 1.0e-8
    srp_lbfgs_ftol: float = 1.0e-12
    srp_lbfgs_maxiter: int = 1000
    srp_lbfgs_maxcor: int = 20
    srp_use_small_radius_branch: bool = True
    srp_init_r0_small: float = 0.01
    srp_init_rjitter_small: float = 0.005
    srp_small_branch_step_scale: float = 0.2
    srp_strict_shrink: bool = True
    srp_strict_shrink_margin: float = 1.0e-8


@dataclass(frozen=True)
class SRPBranchResult:
    """Result for one SRP initialization branch."""

    centers: np.ndarray
    radii: np.ndarray
    circles: np.ndarray
    score: float
    legal: bool
    branch: str
    loss_after: float
    min_wall_clearance: float
    min_pair_clearance: float
    lp_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SRPPolishResult:
    """Best result from all SRP initialization branches."""

    centers: np.ndarray
    radii: np.ndarray
    circles: np.ndarray
    score: float
    legal: bool
    branch: str
    loss_after: float
    min_wall_clearance: float
    min_pair_clearance: float
    branches: tuple[SRPBranchResult, ...]


def init_radii(N, r0=0.01, jitter=0.005):
    r = r0 + jitter * (np.random.rand(N) - 0.5) * 2.0
    r = np.maximum(r, 1e-4)
    return r.astype(np.float64)


@njit(cache=True, fastmath=True)
def _unit(dx, dy):
    d = math.sqrt(dx * dx + dy * dy) + EPS
    return dx / d, dy / d, d


@njit(cache=True, fastmath=True)
def loss_and_grad(X, N, w_overlap, w_wall, alpha):
    centers = X[: 2 * N].reshape((N, 2))
    radii = X[2 * N : 2 * N + N]

    L = 0.0
    g = np.zeros_like(X)

    # Walls
    for i in range(N):
        ri = radii[i]
        xi = centers[i, 0]
        yi = centers[i, 1]
        # x-left
        v = ri - xi
        if v > 0.0:
            L += w_wall * (v * v)
            g[2 * i + 0] += -2.0 * w_wall * v
            g[2 * N + i] += 2.0 * w_wall * v
        # x-right
        v = ri - (1.0 - xi)
        if v > 0.0:
            L += w_wall * (v * v)
            g[2 * i + 0] += 2.0 * w_wall * v
            g[2 * N + i] += 2.0 * w_wall * v
        # y-bottom
        v = ri - yi
        if v > 0.0:
            L += w_wall * (v * v)
            g[2 * i + 1] += -2.0 * w_wall * v
            g[2 * N + i] += 2.0 * w_wall * v
        # y-top
        v = ri - (1.0 - yi)
        if v > 0.0:
            L += w_wall * (v * v)
            g[2 * i + 1] += 2.0 * w_wall * v
            g[2 * N + i] += 2.0 * w_wall * v

    # Pair overlaps
    for i in range(N):
        xi0, yi0, ri = centers[i, 0], centers[i, 1], radii[i]
        for j in range(i + 1, N):
            dx = xi0 - centers[j, 0]
            dy = yi0 - centers[j, 1]
            ux, uy, d = _unit(dx, dy)
            over = (ri + radii[j]) - d
            if over > 0.0:
                L += w_overlap * (over * over)
                c = 2.0 * w_overlap * over
                # i center  (NOTE the minus signs)
                g[2 * i + 0] += -c * ux
                g[2 * i + 1] += -c * uy
                # j center  (opposite)
                g[2 * j + 0] += c * ux
                g[2 * j + 1] += c * uy
                # radii
                g[2 * N + i] += c
                g[2 * N + j] += c

    # Maximize sum radii (minimize -sum radii)
    L += -alpha * np.sum(radii)
    for i in range(N):
        g[2 * N + i] += -alpha

    return L, g


@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k] * v[k]
    return math.sqrt(s) + 1e-18


@njit(cache=True, fastmath=True)
def _seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True, fastmath=True)
def srp_adaptive(
    X,
    N,
    Imax,
    m,
    step_center,
    step_radius,
    beta,
    backtrack,
    w_overlap,
    w_wall,
    alpha,
):
    Xc = X.copy()
    eta = 1.0
    for _ in range(Imax):
        # jitter
        noise = np.zeros_like(Xc)
        for i in range(N):
            noise[2 * i + 0] = (np.random.rand() * 2.0 - 1.0) * (eta * step_center)
            noise[2 * i + 1] = (np.random.rand() * 2.0 - 1.0) * (eta * step_center)
        for i in range(N):
            noise[2 * N + i] = (np.random.rand() * 2.0 - 1.0) * (eta * step_radius)
        Xtrial = Xc + noise

        # m gradient steps with simple backtracking
        for __ in range(m):
            L0, g = loss_and_grad(Xtrial, N, w_overlap, w_wall, alpha)

            step = np.empty_like(g)
            for i in range(N):
                step[2 * i + 0] = eta * step_center
                step[2 * i + 1] = eta * step_center
            for i in range(N):
                step[2 * N + i] = eta * step_radius

            gn = _l2_norm(g)
            if gn > 0.0:
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = loss_and_grad(Xprop, N, w_overlap, w_wall, alpha)

                bt = 0
                while Lprop > L0 and bt < backtrack:
                    for i in range(N):
                        step[2 * i + 0] *= 0.5
                        step[2 * i + 1] *= 0.5
                    for i in range(N):
                        step[2 * N + i] *= 0.5
                    Xprop = Xtrial - step * (g / gn)
                    Lprop, _ = loss_and_grad(Xprop, N, w_overlap, w_wall, alpha)
                    bt += 1

                Xtrial = Xprop

        # clip to domain for stability
        for i in range(N):
            if Xtrial[2 * i + 0] < 0.0:
                Xtrial[2 * i + 0] = 0.0
            if Xtrial[2 * i + 0] > 1.0:
                Xtrial[2 * i + 0] = 1.0
            if Xtrial[2 * i + 1] < 0.0:
                Xtrial[2 * i + 1] = 0.0
            if Xtrial[2 * i + 1] > 1.0:
                Xtrial[2 * i + 1] = 1.0
        for i in range(N):
            if Xtrial[2 * N + i] < 0.0:
                Xtrial[2 * N + i] = 0.0
            if Xtrial[2 * N + i] > 0.5:
                Xtrial[2 * N + i] = 0.5

        Xc = Xtrial
        eta *= beta
    return Xc


def local_optimize(
    X0,
    N,
    w_overlap,
    w_wall,
    alpha,
    gtol=1e-8,
    ftol=1e-12,
    maxiter=1000,
    maxcor=20,
):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

    def fun(x):
        L, _ = loss_and_grad(x, N, w_overlap, w_wall, alpha)
        return L

    def jac(x):
        _, g = loss_and_grad(x, N, w_overlap, w_wall, alpha)
        return g

    res = minimize(
        fun=fun,
        x0=X0,
        method="L-BFGS-B",
        jac=jac,
        bounds=bounds,
        options={"gtol": gtol, "ftol": ftol, "maxiter": maxiter, "maxcor": maxcor},
    )
    return res.x, res.fun


@njit(cache=True, fastmath=True)
def min_wall_clearance(centers, radii):
    N = centers.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = centers[i, 0], centers[i, 1]
        ri = radii[i]
        c1 = xi - ri
        if c1 < best:
            best = c1
        c2 = 1.0 - xi - ri
        if c2 < best:
            best = c2
        c3 = yi - ri
        if c3 < best:
            best = c3
        c4 = 1.0 - yi - ri
        if c4 < best:
            best = c4
    return best


@njit(cache=True, fastmath=True)
def min_pair_clearance(centers, radii):
    N = centers.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi, ri = centers[i, 0], centers[i, 1], radii[i]
        for j in range(i + 1, N):
            dx = xi - centers[j, 0]
            dy = yi - centers[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            clr = d - (ri + radii[j])
            if clr < best:
                best = clr
    return best


def hard_project_max_sum_radii(centers, radii, safety=1e-9, pair_safety_mul=1.0):
    """
    Given centers (N,2) in [0,1]^2 and current radii (N,),
    compute radii' that:
      maximize sum(r_i)
      subject to:
        0 <= r_i <= wall_i
        r_i + r_j <= d_ij
    and then shrink by a tiny `safety` to avoid visual tangency.
    Returns: radii_proj (N,), info dict with diagnostics.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    # --- diagnostics helpers ---
    def _min_wall_clear(cent, rr):
        xi, yi = cent[:, 0], cent[:, 1]
        return float(np.min([xi - rr, 1.0 - xi - rr, yi - rr, 1.0 - yi - rr]))

    def _min_pair_clear(cent, rr):
        dx = cent[:, None, 0] - cent[None, :, 0]
        dy = cent[:, None, 1] - cent[None, :, 1]
        D = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(D, np.inf)
        S = rr[:, None] + rr[None, :]
        return float(np.min(D - S))

    N = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]

    # Wall upper bounds for each circle
    wall_ub = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    wall_ub = np.clip(wall_ub - safety, 0.0, None)  # tiny safety

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx * dx + dy * dy)

    # Build LP:
    # maximize sum r  <=>  minimize -sum r
    c = -np.ones(N, dtype=float)

    # Bounds: 0 <= r_i <= wall_ub[i]
    bounds = [(0.0, float(wall_ub[i])) for i in range(N)]

    # Inequalities A_ub @ r <= b_ub
    rows = []
    rhs = []
    for i in range(N):
        for j in range(i + 1, N):
            dij = D[i, j]
            rhs_ij = max(dij * pair_safety_mul - safety, 0.0)
            row = np.zeros(N, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(rhs_ij)

    if rows:
        A_ub = np.vstack(rows)
        b_ub = np.asarray(rhs, dtype=float)
    else:
        A_ub = None
        b_ub = None

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "sum_r": None,
        "violations": None,
    }

    if not res.success:
        # Fallback: clamp to wall_ub (feasible w.r.t. walls; pairs may violate)
        r_proj = np.minimum(radii, wall_ub).copy()
        info["sum_r"] = float(np.sum(r_proj))
        info["violations"] = {
            "min_wall_clear": _min_wall_clear(centers, r_proj),
            "min_pair_clear": _min_pair_clear(centers, r_proj),
        }
        info["message"] = f"LP failed; returned clamped radii. HiGHS status {res.status}: {res.message}"
        return r_proj, info

    r_star = res.x

    # Final tiny shrink to guarantee strict feasibility in plotting
    r_proj = np.maximum(0.0, r_star - safety)

    mwc = _min_wall_clear(centers, r_proj)
    mpc = _min_pair_clear(centers, r_proj)

    info["sum_r"] = float(np.sum(r_proj))
    info["violations"] = {"min_wall_clear": mwc, "min_pair_clear": mpc}
    return r_proj, info


def strict_shrink_radii(centers: Any, radii: Any, margin: float = 1.0e-8) -> tuple[np.ndarray, dict[str, Any]]:
    """Shrink radii until wall and pair constraints have at least `margin` clearance."""

    centers = _as_centers(centers)
    r = np.asarray(radii, dtype=np.float64).copy()
    if r.shape != (centers.shape[0],):
        raise ValueError(f"radii must have shape ({centers.shape[0]},), got {r.shape}.")
    margin = max(float(margin), 0.0)
    before_sum = float(np.sum(r))

    x = centers[:, 0]
    y = centers[:, 1]
    wall_ub = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    r = np.minimum(np.maximum(r, 0.0), np.maximum(0.0, wall_ub - margin))

    num_circles = int(r.shape[0])
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            cap = max(0.0, distance - margin)
            excess = float(r[i] + r[j] - cap)
            if excess <= 0.0:
                continue
            first, second = (i, j) if r[i] >= r[j] else (j, i)
            delta = min(float(r[first]), excess)
            r[first] -= delta
            excess -= delta
            if excess > 0.0:
                r[second] = max(0.0, r[second] - excess)

    mwc = float(min_wall_clearance(centers, r))
    mpc = float(min_pair_clearance(centers, r))
    info = {
        "margin": margin,
        "sum_before": before_sum,
        "sum_after": float(np.sum(r)),
        "shrink_total": float(before_sum - np.sum(r)),
        "min_wall_clearance": mwc,
        "min_pair_clearance": mpc,
    }
    return r, info


def _as_centers(centers: Any) -> np.ndarray:
    array = np.asarray(centers, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in {2, 3}:
        raise ValueError(f"centers must have shape [N,2] or legacy [N,3], got {array.shape}.")
    return array[:, :2].copy()


def _as_radii(radii: Any | None, num_circles: int) -> np.ndarray:
    if radii is None:
        return np.zeros(num_circles, dtype=np.float64)
    array = np.asarray(radii, dtype=np.float64)
    if array.shape != (num_circles,):
        raise ValueError(f"radii must have shape ({num_circles},), got {array.shape}.")
    return array.copy()


def _run_branch(
    centers0: np.ndarray,
    radii0: np.ndarray,
    config: SRPPolishConfig,
    *,
    branch: str,
    step_scale: float = 1.0,
    seed: int | None = None,
) -> SRPBranchResult:
    if seed is not None:
        _seed_numba(int(seed))
    N = int(centers0.shape[0])
    X0 = np.concatenate([centers0.ravel(), radii0], axis=0).astype(np.float64, copy=False)
    X_srp = srp_adaptive(
        X0,
        N,
        Imax=int(config.srp_Imax),
        m=int(config.srp_m),
        step_center=float(config.srp_step_center) * float(step_scale),
        step_radius=float(config.srp_step_radius) * float(step_scale),
        beta=float(config.srp_beta),
        backtrack=int(config.srp_backtrack),
        w_overlap=float(config.srp_w_overlap),
        w_wall=float(config.srp_w_wall),
        alpha=float(config.srp_alpha_sum_r),
    )
    X_fin, L_fin = local_optimize(
        X_srp,
        N,
        w_overlap=float(config.srp_w_overlap),
        w_wall=float(config.srp_w_wall),
        alpha=float(config.srp_alpha_sum_r),
        gtol=float(config.srp_lbfgs_gtol),
        ftol=float(config.srp_lbfgs_ftol),
        maxiter=int(config.srp_lbfgs_maxiter),
        maxcor=int(config.srp_lbfgs_maxcor),
    )

    centers = X_fin[: 2 * N].reshape((N, 2))
    radii = X_fin[2 * N : 2 * N + N]
    r_proj, info = hard_project_max_sum_radii(centers, radii, safety=1e-9)
    if config.srp_strict_shrink:
        r_proj, shrink_info = strict_shrink_radii(
            centers,
            r_proj,
            margin=float(config.srp_strict_shrink_margin),
        )
        info = {
            **info,
            "strict_shrink": shrink_info,
        }
    circles = np.column_stack((centers, r_proj))
    score = float(np.sum(r_proj))
    mwc = float(min_wall_clearance(centers, r_proj))
    mpc = float(min_pair_clearance(centers, r_proj))
    legal = is_valid_packing(centers, r_proj, tol=1.0e-9)
    return SRPBranchResult(
        centers=centers.astype(np.float64, copy=True),
        radii=r_proj.astype(np.float64, copy=True),
        circles=circles.astype(np.float64, copy=True),
        score=score,
        legal=legal,
        branch=branch,
        loss_after=float(L_fin),
        min_wall_clearance=mwc,
        min_pair_clearance=mpc,
        lp_info=info,
    )


def polish_terminal_layout_srp(
    centers: Any,
    radii: Any | None = None,
    *,
    config: SRPPolishConfig | None = None,
    seed: int | None = None,
) -> SRPPolishResult:
    """Run FlowBoost-style SRP + L-BFGS-B + LP on one complete layout."""

    cfg = SRPPolishConfig() if config is None else config
    centers0 = _as_centers(centers)
    radii_flow = _as_radii(radii, centers0.shape[0])

    if seed is not None:
        np.random.seed(int(seed))

    branches: list[SRPBranchResult] = [
        _run_branch(centers0, radii_flow, cfg, branch="flowr", seed=seed),
    ]
    if cfg.srp_use_small_radius_branch:
        radii_small = init_radii(
            centers0.shape[0],
            r0=float(cfg.srp_init_r0_small),
            jitter=float(cfg.srp_init_rjitter_small),
        )
        small_seed = None if seed is None else int(seed) + 1
        branches.append(
            _run_branch(
                centers0,
                radii_small,
                cfg,
                branch="smallr",
                step_scale=float(cfg.srp_small_branch_step_scale),
                seed=small_seed,
            )
        )

    best = max(branches, key=lambda item: (item.score, item.legal))
    return SRPPolishResult(
        centers=best.centers,
        radii=best.radii,
        circles=best.circles,
        score=best.score,
        legal=best.legal,
        branch=best.branch,
        loss_after=best.loss_after,
        min_wall_clearance=best.min_wall_clearance,
        min_pair_clearance=best.min_pair_clearance,
        branches=tuple(branches),
    )
