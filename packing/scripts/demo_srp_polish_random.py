"""Demo FlowBoost-style SRP polishing on random terminal center layouts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from circle_pack.srp_polish import SRPPolishConfig, polish_terminal_layout_srp


def _run_one(payload: tuple[int, int, int, SRPPolishConfig]) -> dict[str, Any]:
    sample_idx, num_points, seed, config = payload
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.0, 1.0, size=(num_points, 2)).astype(np.float64)
    radii = np.zeros(num_points, dtype=np.float64)
    result = polish_terminal_layout_srp(centers, radii, config=config, seed=seed)
    return {
        "sample": int(sample_idx),
        "success": bool(result.legal),
        "sum_r": float(result.score),
        "branch": result.branch,
        "min_wall_clearance": float(result.min_wall_clearance),
        "min_pair_clearance": float(result.min_pair_clearance),
    }


def run_demo(
    *,
    num_samples: int,
    num_points: int,
    num_workers: int,
    seed: int,
    config: SRPPolishConfig,
) -> dict[str, Any]:
    """Run the random-center SRP polishing demo and return aggregate metrics."""

    work = [(idx, int(num_points), int(seed) + idx, config) for idx in range(int(num_samples))]
    start = perf_counter()
    if int(num_workers) <= 1:
        rows = [_run_one(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            rows = list(executor.map(_run_one, work))
    elapsed = perf_counter() - start

    rows.sort(key=lambda item: item["sample"])
    sum_r = np.asarray([row["sum_r"] for row in rows], dtype=np.float64)
    min_wall = np.asarray([row["min_wall_clearance"] for row in rows], dtype=np.float64)
    min_pair = np.asarray([row["min_pair_clearance"] for row in rows], dtype=np.float64)
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    return {
        "num_samples": int(num_samples),
        "num_points": int(num_points),
        "num_workers": int(num_workers),
        "num_success": int(np.count_nonzero(success)),
        "sum_r_max": float(np.max(sum_r)) if sum_r.size else 0.0,
        "sum_r_mean": float(np.mean(sum_r)) if sum_r.size else 0.0,
        "min_wall_clearance_min": float(np.min(min_wall)) if min_wall.size else 0.0,
        "min_wall_clearance_mean": float(np.mean(min_wall)) if min_wall.size else 0.0,
        "min_pair_clearance_min": float(np.min(min_pair)) if min_pair.size else 0.0,
        "min_pair_clearance_mean": float(np.mean(min_pair)) if min_pair.size else 0.0,
        "elapsed_sec": float(elapsed),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--srp-Imax", type=int, default=400)
    parser.add_argument("--srp-m", type=int, default=30)
    parser.add_argument("--srp-beta", type=float, default=0.99)
    parser.add_argument("--srp-backtrack", type=int, default=5)
    parser.add_argument("--srp-step-center", type=float, default=0.05)
    parser.add_argument("--srp-step-radius", type=float, default=0.001)
    parser.add_argument("--lbfgs-maxiter", type=int, default=1000)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = SRPPolishConfig(
        srp_Imax=args.srp_Imax,
        srp_m=args.srp_m,
        srp_beta=args.srp_beta,
        srp_backtrack=args.srp_backtrack,
        srp_step_center=args.srp_step_center,
        srp_step_radius=args.srp_step_radius,
        srp_lbfgs_maxiter=args.lbfgs_maxiter,
        srp_use_small_radius_branch=True,
    )
    metrics = run_demo(
        num_samples=args.num_samples,
        num_points=args.num_points,
        num_workers=args.num_workers,
        seed=args.seed,
        config=config,
    )
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
        return
    print(f"Samples: {metrics['num_samples']} x {metrics['num_points']}")
    print(f"Workers: {metrics['num_workers']}")
    print(f"Successful/legal layouts: {metrics['num_success']}/{metrics['num_samples']}")
    print(f"sum_r max: {metrics['sum_r_max']:.8f}")
    print(f"sum_r mean: {metrics['sum_r_mean']:.8f}")
    print(f"min wall clearance: min={metrics['min_wall_clearance_min']:.3e} mean={metrics['min_wall_clearance_mean']:.3e}")
    print(f"min pair clearance: min={metrics['min_pair_clearance_min']:.3e} mean={metrics['min_pair_clearance_mean']:.3e}")
    print(f"elapsed sec: {metrics['elapsed_sec']:.3f}")


if __name__ == "__main__":
    main()
