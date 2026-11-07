#!/usr/bin/env python3
"""Circle packing -- matched-magnitude random-perturbation control.

With a continuous search space, masked centers can move by a negligible amount
and still score like the parent, so a displacement measurement alone cannot rule
out inheritance. This script holds the active core and the edit set fixed and
swaps only how the masked centers are placed.

    learned arm : read from candidates.jsonl, no model is loaded
                  delta = raw_score - metadata.seed_score   (final_score if SRP is on)
    random arm  : recomputed here, model-free
                  masked centers <- parent center + N(0, sigma^2), clipped to the
                  unit square, then scored with the same LP

sigma defaults to 0.04, matching the run's own measured inference displacement
(``inference_center_distance_mean`` in metrics.jsonl, ~0.03-0.04). The script
prints the run's actual displacement so the choice can be checked, and
``--sigma-sweep`` traces the whole curve so the learned point can be placed
against random perturbations of *every* magnitude, not just one.

Pairing.  Each learned record carries ``metadata.seed_hash`` (its parent) and
``metadata.edit_ratio``/``edit_policy`` (its edit set). Records whose parent is
found in any pool snapshot (``top_pool.pt`` plus any
``top_pool_generation_XXX.pt``) are replayed with the same parent, same policy
and same ratio. The edit mask itself is not logged and both policies are
rng-driven, so the mask is resampled using the repo's own ``sample_edit_set``;
the mask *distribution* is therefore matched exactly even though individual
masks are not.

Usage:
    cd packing
    python analysis/batch41_random_control.py checkpoints/<run> --num-samples 1000
    python analysis/batch41_random_control.py checkpoints/<run> --sigma-sweep 0.01 0.02 0.04 0.08 0.16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circle_pack.geometry import is_valid_packing  # noqa: E402
from circle_pack.scipy_solver import solve_radii_for_centers  # noqa: E402
from patternboost.edit import sample_edit_set  # noqa: E402
from patternboost.top_pool import TopPool  # noqa: E402


# --------------------------------------------------------------------------
# run directory contract
# --------------------------------------------------------------------------


class RunDir:
    """Locate and validate the files this analysis needs inside one run dir."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.candidates = path / "candidates.jsonl"
        self.top_pool = path / "top_pool.pt"
        self.config = path / "config_snapshot.yaml"
        self.metrics = path / "metrics.jsonl"

    def require(self) -> None:
        missing = [p.name for p in (self.candidates, self.top_pool, self.config) if not p.exists()]
        if missing:
            raise SystemExit(
                f"{self.path} is missing required file(s): {', '.join(missing)}\n"
                "This script needs candidates.jsonl (learned deltas), "
                "top_pool.pt (parent layouts) and config_snapshot.yaml (LP tolerances)."
            )


def load_pool_entries(run: RunDir) -> tuple[dict[str, Any], dict[str, int]]:
    """Merge every pool snapshot in the run into one hash -> entry map.

    ``top_pool.pt`` is rewritten in place each generation, so on its own it only
    exposes parents that survived to the end -- candidates seeded from entries
    that were later evicted get dropped. Runs with
    ``top_pool_snapshot_every_generations`` also write
    ``top_pool_generation_XXX.pt``; merging them recovers those parents.

    A canonical hash determines the layout, and the score is a function of the
    layout, so an entry is identical wherever it appears and the union is safe.
    """

    entries: dict[str, Any] = {}
    provenance: dict[str, int] = {}
    snapshots = sorted(run.path.glob("top_pool_generation_*.pt")) + [run.top_pool]
    for path in snapshots:
        pool = TopPool.load(path)
        added = 0
        for entry in pool.entries():
            if entry.canonical_hash not in entries:
                entries[entry.canonical_hash] = entry
                added += 1
        provenance[path.name] = added
    return entries, provenance


class RunConfig:
    """The subset of config_snapshot.yaml that defines the scoring protocol."""

    def __init__(self, config_path: Path) -> None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        safety = raw.get("safety", {})
        self.radius_max = float(safety.get("radius_max", 0.5))
        self.feasibility_tol = float(safety.get("feasibility_tol", 1.0e-12))
        pb = raw.get("patternboost", {})
        polish = pb.get("polish", {})
        self.srp_enabled = bool(polish.get("srp_enabled", polish.get("enabled", False)))
        edit = pb.get("edit", {})
        self.edit_config = SimpleNamespace(
            policies=list(edit.get("policies", ["random", "random_point_knn"])),
            policy_weights=list(edit.get("policy_weights", [1.0] * len(edit.get("policies", ["random", "random_point_knn"])))),
            edit_ratios=[float(value) for value in edit.get("edit_ratios", [0.2])],
            ratio_weights=list(edit.get("ratio_weights", [1.0] * len(edit.get("edit_ratios", [0.2])))),
            knn_k=edit.get("knn_k"),
        )
        self.num_circles = raw.get("problem", {}).get("K", raw.get("num_circles"))

    def describe(self) -> dict[str, Any]:
        return {
            "num_circles": self.num_circles,
            "radius_max": self.radius_max,
            "feasibility_tol": self.feasibility_tol,
            "srp_enabled": self.srp_enabled,
            "edit_policies": self.edit_config.policies,
            "edit_ratios": self.edit_config.edit_ratios,
        }


def measured_displacement(metrics_path: Path) -> dict[str, Any] | None:
    """Return the run's own learned displacement, used to calibrate sigma."""

    if not metrics_path.exists():
        return None
    values = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get("inference_center_distance_mean")
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                values.append(float(value))
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "num_generations": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "last": float(array[-1]),
    }


# --------------------------------------------------------------------------
# candidate sampling
# --------------------------------------------------------------------------


def stream_candidates(path: Path) -> Iterator[dict[str, Any]]:
    """Yield candidate records one at a time; candidates.jsonl can be gigabytes."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def learned_delta(record: dict[str, Any], use_final_score: bool) -> float | None:
    """Return child - parent for one learned candidate, or None if unusable."""

    metadata = record.get("metadata") or {}
    seed_score = metadata.get("seed_score")
    child = record.get("final_score") if use_final_score else record.get("raw_score")
    if seed_score is None or child is None:
        return None
    return float(child) - float(seed_score)


def reservoir_sample(
    path: Path,
    pool_hashes: set[str],
    num_samples: int,
    rng: np.random.Generator,
    *,
    use_final_score: bool,
    min_generation: int | None,
    max_generation: int | None,
) -> tuple[list[dict[str, Any]], Counter]:
    """Uniformly sample eligible learned records in a single streaming pass."""

    chosen: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen = 0
    for record in stream_candidates(path):
        stats["total"] += 1
        metadata = record.get("metadata") or {}
        generation = record.get("generation")
        if min_generation is not None and generation is not None and generation < min_generation:
            stats["out_of_generation_range"] += 1
            continue
        if max_generation is not None and generation is not None and generation > max_generation:
            stats["out_of_generation_range"] += 1
            continue
        if not record.get("legal") or not record.get("full_active"):
            stats["illegal_or_incomplete"] += 1
            continue
        delta = learned_delta(record, use_final_score)
        if delta is None:
            stats["missing_scores"] += 1
            continue
        seed_hash = metadata.get("seed_hash")
        if seed_hash not in pool_hashes:
            stats["parent_not_in_final_pool"] += 1
            continue
        stats["ok"] += 1
        seen += 1
        slim = {
            "generation": generation,
            "candidate_index": record.get("candidate_index"),
            "seed_hash": seed_hash,
            "seed_score": float(metadata["seed_score"]),
            "edit_ratio": float(metadata.get("edit_ratio", 0.2)),
            "edit_policy": str(metadata.get("edit_policy", "random")),
            "delta_learned": delta,
        }
        if len(chosen) < num_samples:
            chosen.append(slim)
        else:
            index = int(rng.integers(0, seen))
            if index < num_samples:
                chosen[index] = slim
    stats["eligible"] = seen
    return chosen, stats


# --------------------------------------------------------------------------
# random arm
# --------------------------------------------------------------------------


def score_centers(centers: np.ndarray, config: RunConfig) -> float | None:
    """Score a complete layout with the same LP the run used; None if infeasible."""

    result = solve_radii_for_centers(
        centers,
        radius_max=config.radius_max,
        feasibility_tol=config.feasibility_tol,
    )
    if not result.success:
        return None
    circles = result.circles
    if not is_valid_packing(circles[:, :2], circles[:, 2], tol=max(config.feasibility_tol, 1.0e-9)):
        return None
    return float(result.sum_r)


def perturb_in_place(
    parent_centers: np.ndarray,
    edit_mask: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Displace the edited centers by N(0, sigma^2), clipped to the unit square.

    Unedited centers are left untouched, so the preserved active core is bitwise
    identical to the one the model was conditioned on.
    """

    centers = parent_centers.copy()
    noise = rng.normal(0.0, sigma, size=(int(edit_mask.sum()), 2))
    centers[edit_mask] = np.clip(centers[edit_mask] + noise, 0.0, 1.0)
    return centers


def run_control_arm(
    samples: list[dict[str, Any]],
    entries: dict[str, Any],
    config: RunConfig,
    sigmas: list[float],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Compute the random-arm delta at every sigma for every sampled candidate."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        entry = entries[sample["seed_hash"]]
        parent_centers = np.asarray(entry.centers, dtype=np.float64)
        parent_score = float(entry.score)
        # Resample an edit set from the run's own sampler at this candidate's
        # ratio and policy, so the mask distribution matches the learned arm.
        edit_config = SimpleNamespace(
            policies=[sample["edit_policy"]],
            policy_weights=[1.0],
            edit_ratios=[sample["edit_ratio"]],
            ratio_weights=[1.0],
            knn_k=config.edit_config.knn_k,
        )
        circles = np.column_stack((parent_centers, np.asarray(entry.radii, dtype=np.float64)))
        active_mask = np.ones(parent_centers.shape[0], dtype=bool)
        edit_set = sample_edit_set(circles, active_mask, edit_config, rng)
        row = {
            **sample,
            "pool_parent_score": parent_score,
            "num_edited": int(edit_set.edit_mask.sum()),
        }
        for sigma in sigmas:
            centers = perturb_in_place(parent_centers, edit_set.edit_mask, sigma, rng)
            score = score_centers(centers, config)
            row[f"delta_random_sigma{sigma:g}"] = None if score is None else score - parent_score
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f"  ... {index + 1}/{len(samples)} control layouts", flush=True)
    return rows


# --------------------------------------------------------------------------
# statistics and reporting
# --------------------------------------------------------------------------


def summarize_pairs(rows: list[dict[str, Any]], sigma: float) -> dict[str, Any]:
    key = f"delta_random_sigma{sigma:g}"
    usable = [row for row in rows if row[key] is not None]
    learned = np.array([row["delta_learned"] for row in usable], dtype=float)
    random_arm = np.array([row[key] for row in usable], dtype=float)
    diff = learned - random_arm
    count = diff.size
    if count == 0:
        return {"num_pairs": 0, "lp_failures": len(rows)}
    stderr = float(diff.std(ddof=1) / math.sqrt(count)) if count > 1 else float("nan")
    return {
        "sigma": sigma,
        "num_pairs": int(count),
        "lp_failures": len(rows) - count,
        "delta_learned_mean": float(learned.mean()),
        "delta_learned_std": float(learned.std(ddof=1)) if count > 1 else float("nan"),
        "delta_random_mean": float(random_arm.mean()),
        "delta_random_std": float(random_arm.std(ddof=1)) if count > 1 else float("nan"),
        "paired_gap_mean": float(diff.mean()),
        "paired_gap_stderr": stderr,
        "paired_gap_ci95": [float(diff.mean() - 1.96 * stderr), float(diff.mean() + 1.96 * stderr)],
        "learned_wins_frac": float(np.mean(diff > 0)),
    }


def stratify(rows: list[dict[str, Any]], key: str, sigma: float) -> dict[str, dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return {
        str(value): summarize_pairs(group, sigma)
        for value, group in sorted(groups.items(), key=lambda item: str(item[0]))
        if len(group) >= 2
    }


def write_plot(
    rows: list[dict[str, Any]],
    sigmas: list[float],
    primary_sigma: float,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", flush=True)
        return

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    key = f"delta_random_sigma{primary_sigma:g}"
    usable = [row for row in rows if row[key] is not None]
    learned = np.array([row["delta_learned"] for row in usable], dtype=float)
    random_arm = np.array([row[key] for row in usable], dtype=float)
    span = max(abs(learned).max(), abs(random_arm).max())
    bins = np.linspace(-span, span, 61)
    left.hist(random_arm, bins=bins, alpha=0.6, density=True, label=rf"random, $\sigma$={primary_sigma:g}", color="#9aa0a6")
    left.hist(learned, bins=bins, alpha=0.6, density=True, label="learned", color="#1a73e8")
    left.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    left.set_xlabel(r"$\Delta$ = $\sum r$(child) $-$ $\sum r$(parent)")
    left.set_ylabel("density")
    left.set_title("per-candidate score change")
    left.legend()

    means, errs = [], []
    for sigma in sigmas:
        values = np.array(
            [row[f"delta_random_sigma{sigma:g}"] for row in rows if row[f"delta_random_sigma{sigma:g}"] is not None],
            dtype=float,
        )
        means.append(values.mean())
        errs.append(values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else 0.0)
    right.errorbar(sigmas, means, yerr=np.array(errs) * 1.96, marker="o", color="#9aa0a6", label="random perturbation")
    learned_err = learned.std(ddof=1) / math.sqrt(learned.size) if learned.size > 1 else 0.0
    right.axhline(learned.mean(), color="#1a73e8", linewidth=1.8, label="learned")
    right.fill_between(
        sigmas,
        learned.mean() - 1.96 * learned_err,
        learned.mean() + 1.96 * learned_err,
        color="#1a73e8",
        alpha=0.2,
        linewidth=0,
    )
    right.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    right.set_xscale("log")
    right.set_xlabel(r"random perturbation magnitude $\sigma$")
    right.set_ylabel(r"mean $\Delta$")
    right.set_title("learned vs the whole random-magnitude curve")
    right.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run directory holding candidates.jsonl + top_pool.pt + config_snapshot.yaml")
    parser.add_argument("--num-samples", type=int, default=1000, help="learned candidates to pair against (default 1000)")
    parser.add_argument("--sigma", type=float, default=0.04, help="perturbation magnitude for the headline arm (default 0.04)")
    parser.add_argument("--sigma-sweep", type=float, nargs="*", default=None, help="extra sigmas to trace the full curve")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-generation", type=int, default=None)
    parser.add_argument("--max-generation", type=int, default=None)
    parser.add_argument("--use-final-score", action="store_true", help="use final_score instead of raw_score (set this if the run had SRP on)")
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <run_dir>/analysis_batch41")
    args = parser.parse_args()

    run = RunDir(args.run_dir)
    run.require()
    config = RunConfig(run.config)
    use_final_score = args.use_final_score or config.srp_enabled
    out_dir = args.out_dir or (args.run_dir / "analysis_batch41")
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = sorted({args.sigma, *(args.sigma_sweep or [])})

    print(f"run     : {args.run_dir}")
    print(f"config  : {json.dumps(config.describe())}")
    print(f"score   : {'final_score (SRP)' if use_final_score else 'raw_score (no SRP)'}")

    displacement = measured_displacement(run.metrics)
    if displacement is not None:
        print(
            f"learned displacement : mean={displacement['mean']:.4f} "
            f"range=[{displacement['min']:.4f}, {displacement['max']:.4f}] over {displacement['num_generations']} generations"
        )
        if not displacement["min"] <= args.sigma <= displacement["max"]:
            print(
                f"NOTE: --sigma {args.sigma:g} is outside the run's own displacement range; "
                "the random arm is no longer magnitude-matched."
            )
    else:
        print("learned displacement : not found in metrics.jsonl (cannot verify sigma is matched)")

    entries, provenance = load_pool_entries(run)
    pool_hashes = set(entries)
    print(
        f"pool    : {len(pool_hashes)} distinct entries from {len(provenance)} snapshot(s), "
        f"best={max(entry.score for entry in entries.values()):.4f}"
    )
    if len(provenance) > 1:
        print(f"          new entries per snapshot: {json.dumps(provenance)}")
    else:
        print(
            "          only the final pool is available; candidates seeded from evicted entries "
            "will be dropped. Enable patternboost.runtime.top_pool_snapshot_every_generations on future runs."
        )

    rng = np.random.default_rng(args.seed)
    samples, scan_stats = reservoir_sample(
        run.candidates,
        pool_hashes,
        args.num_samples,
        rng,
        use_final_score=use_final_score,
        min_generation=args.min_generation,
        max_generation=args.max_generation,
    )
    print(f"scanned : {json.dumps(dict(sorted(scan_stats.items())))}")
    if not samples:
        raise SystemExit(
            "no eligible candidates. Check the 'scanned' counts above: "
            "parent_not_in_final_pool means no sampled parent was in any pool snapshot "
            "(try --min-generation); missing_scores means metadata.seed_score is absent, "
            "i.e. this run predates the provenance logging."
        )
    print(f"paired  : running {len(samples)} x {len(sigmas)} control layouts", flush=True)

    rows = run_control_arm(samples, entries, config, sigmas, rng)

    report = {
        "run_dir": str(args.run_dir),
        "config": config.describe(),
        "args": {
            "num_samples": args.num_samples,
            "sigma": args.sigma,
            "sigma_sweep": sigmas,
            "seed": args.seed,
            "use_final_score": use_final_score,
            "min_generation": args.min_generation,
            "max_generation": args.max_generation,
        },
        "scan": dict(sorted(scan_stats.items())),
        "pool_size": len(pool_hashes),
        "pool_snapshots": provenance,
        "learned_displacement": displacement,
        "overall": summarize_pairs(rows, args.sigma),
        "sigma_sweep": [summarize_pairs(rows, sigma) for sigma in sigmas],
        "by_edit_ratio": stratify(rows, "edit_ratio", args.sigma),
        "by_edit_policy": stratify(rows, "edit_policy", args.sigma),
        "by_generation": stratify(rows, "generation", args.sigma),
    }
    (out_dir / "batch41_random_control.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (out_dir / "batch41_random_control_pairs.csv").open("w", encoding="utf-8") as handle:
        sigma_columns = ",".join(f"delta_random_sigma{sigma:g}" for sigma in sigmas)
        handle.write(f"generation,seed_hash,parent_score,edit_ratio,edit_policy,num_edited,delta_learned,{sigma_columns}\n")
        for row in rows:
            values = ",".join(
                "" if row[f"delta_random_sigma{sigma:g}"] is None else f"{row[f'delta_random_sigma{sigma:g}']:.6f}"
                for sigma in sigmas
            )
            handle.write(
                f"{row['generation']},{row['seed_hash'][:12]},{row['pool_parent_score']:.6f},"
                f"{row['edit_ratio']},{row['edit_policy']},{row['num_edited']},{row['delta_learned']:.6f},{values}\n"
            )

    write_plot(
        rows,
        sigmas,
        args.sigma,
        out_dir / "batch41_random_control.png",
        f"Circle packing K={config.num_circles} -- learned vs matched random perturbation",
    )

    overall = report["overall"]
    print()
    print(f"pairs                {overall['num_pairs']}  (LP failures {overall['lp_failures']})")
    print(f"mean delta learned   {overall['delta_learned_mean']:+.4f}")
    print(f"mean delta random    {overall['delta_random_mean']:+.4f}  (sigma={args.sigma:g})")
    print(
        f"paired gap           {overall['paired_gap_mean']:+.4f} "
        f"(95% CI {overall['paired_gap_ci95'][0]:+.4f} .. {overall['paired_gap_ci95'][1]:+.4f})"
    )
    print(f"learned wins         {overall['learned_wins_frac']:.3f}")
    if len(sigmas) > 1:
        print()
        print(f"{'sigma':>8s} {'mean delta random':>20s} {'gap vs learned':>16s}")
        for entry in report["sigma_sweep"]:
            print(f"{entry['sigma']:8.3f} {entry['delta_random_mean']:+20.4f} {entry['paired_gap_mean']:+16.4f}")
    print()
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
