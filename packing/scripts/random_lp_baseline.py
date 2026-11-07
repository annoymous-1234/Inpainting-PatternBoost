#!/usr/bin/env python3
"""Random + LP baseline for circle packing (the ``Random + LP`` row of Table 4).

Each round samples ``--samples`` layouts of ``K`` independent uniform centers in
the unit square, solves the exact radius LP for every layout, and keeps the best
``--pool`` scores seen so far. There is no model and no active core: this is the
from-scratch floor the learned methods are measured against.

The defaults follow the protocol of the runs archived under
``exp-result/packing/main/ddpm/random_baseline/``: the directories are named
``..._K<K>_r40_s2048_pool2048_seed<S>``, i.e. 40 rounds, 2048 samples per round,
pool size 2048.

Only the fields the table aggregation reads are written, so the output can be
consumed exactly like a PatternBoost run directory:

    <out>/<run-name>/metrics.jsonl   one record per round, plus run_start/run_complete
    <out>/<run-name>/results.json    {"summary": {"best_score": ...}, "status": "ok"}

CPU only; no GPU is touched. Importing the repo pulls in torch, so the ``ec``
environment is still required.

Example
-------
    python scripts/random_lp_baseline.py --K 30 --seed 23 \
        --out-dir checkpoints/random_lp/rebuttal --run-name K30_seed23
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.geometry import sample_random_centers  # noqa: E402
from circle_pack.scipy_solver import solve_radii_for_centers  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random + LP baseline for circle packing.")
    p.add_argument("--K", type=int, required=True, help="number of circles")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--rounds", type=int, default=40, help="generations (default 40)")
    p.add_argument("--samples", type=int, default=2048, help="layouts per round (default 2048)")
    p.add_argument("--pool", type=int, default=2048, help="retained pool size (default 2048)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/random_lp"))
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def pool_stats(scores: list[float]) -> dict[str, float | int | None]:
    """Return the pool summary fields used by the aggregation."""
    if not scores:
        return {"pool_size": 0, "pool_score_best": None, "pool_score_mean": None,
                "pool_score_p50": None, "pool_score_p75": None, "pool_score_p90": None}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "pool_size": int(arr.size),
        "pool_score_best": float(arr.max()),
        "pool_score_mean": float(arr.mean()),
        "pool_score_p50": float(np.percentile(arr, 50)),
        "pool_score_p75": float(np.percentile(arr, 75)),
        "pool_score_p90": float(np.percentile(arr, 90)),
    }


def main() -> int:
    args = parse_args()
    run_name = args.run_name or f"K{args.K}_r{args.rounds}_s{args.samples}_pool{args.pool}_seed{args.seed}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def log(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    rng = np.random.default_rng(args.seed)
    started = time.time()
    log({"event": "run_start", "baseline": "random_lp", "K": args.K, "seed": args.seed,
         "rounds": args.rounds, "samples_per_round": args.samples, "pool_size_max": args.pool,
         **pool_stats([])})

    pool: list[float] = []
    for generation in range(1, args.rounds + 1):
        before = max(pool) if pool else None
        round_scores: list[float] = []
        for _ in range(args.samples):
            centers = sample_random_centers(rng, args.K)
            result = solve_radii_for_centers(centers)
            # An infeasible solve would be a solver failure, not a legal layout;
            # skipping keeps one bad LP from poisoning the round statistics.
            if getattr(result, "success", True):
                round_scores.append(float(result.sum_r))
        pool = sorted(pool + round_scores, reverse=True)[: args.pool]
        after = max(pool) if pool else None
        cand = np.asarray(round_scores, dtype=np.float64) if round_scores else np.empty(0)
        log({
            "event": "generation", "generation": generation,
            "best_before": before, "best_after": after,
            "num_candidates": int(cand.size),
            "candidate_final_score_mean": float(cand.mean()) if cand.size else None,
            "candidate_final_score_max": float(cand.max()) if cand.size else None,
            **pool_stats(pool),
        })
        print(f"generation={generation} best_after={after:.6f} "
              f"candidate_mean={cand.mean():.6f} pool_size={len(pool)}", flush=True)

    best = max(pool) if pool else 0.0
    elapsed = time.time() - started
    log({"event": "run_complete", "best_score": best, "elapsed_sec": elapsed, **pool_stats(pool)})
    (run_dir / "results.json").write_text(
        json.dumps({"summary": {"best_score": best, **pool_stats(pool)},
                    "status": "ok", "elapsed_sec": elapsed,
                    "config": {"K": args.K, "seed": args.seed, "rounds": args.rounds,
                               "samples_per_round": args.samples, "pool": args.pool}},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status=ok best={best:.6f} run_dir={run_dir} elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
