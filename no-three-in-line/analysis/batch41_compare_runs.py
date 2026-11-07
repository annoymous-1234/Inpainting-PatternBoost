#!/usr/bin/env python3
"""No-three-in-line -- learned vs random-operator run comparison.

Companion to ``batch41_random_control.py``. That script rebuilds the random arm
post-hoc on one run's final pool; this one compares two *end-to-end runs* whose
configs differ only in ``completion_operator``:

    learned run : patternboost.generation.completion_operator = learned
    random run  : patternboost.generation.completion_operator = random

Both runs log every candidate's ``delta_score`` (= child - parent), so the
comparison is a pure log read -- no model, no environment, no pool.

What this buys over the post-hoc control: the random arm here ran inside the
real search loop for the whole run, so the per-generation axis is available and
the comparison is not restricted to parents that survived into the final pool.
What it costs: the two runs' pools diverge, so candidates are not paired on the
same parent. Parent-score distributions are reported per generation so the
confound is visible; note that a stronger learned pool makes further gains
*harder*, so a positive gap under drift is conservative.

Usage:
    cd no-three-in-line
    python analysis/batch41_compare_runs.py \
        --learned checkpoints/n25_main_stage1_rt23/exp \
        --random  checkpoints/n25_random_op_mrv/exp
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

def record_delta(record: dict[str, Any]) -> tuple[float, float] | None:
    """Return (delta, parent_score) for one record, or None if unrecoverable.

    Runs launched before the provenance logging landed have no ``delta_score``.
    ``score - seed_score`` reproduces it exactly when both are present. Unlike
    the paired control script there is no pool here, so a log carrying neither
    cannot be rescued.
    """

    if record.get("delta_score") is not None and record.get("seed_score") is not None:
        return float(record["delta_score"]), float(record["seed_score"])
    if record.get("score") is not None and record.get("seed_score") is not None:
        return float(record["score"]) - float(record["seed_score"]), float(record["seed_score"])
    return None


def peek_schema(path: Path) -> list[str]:
    """Return the field names of the first candidate record, for diagnostics."""

    for record in stream_candidates(path):
        return sorted(record.keys())
    return []


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


class ArmAccumulator:
    """Streaming aggregate of one arm's delta distribution."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.deltas: list[float] = []
        self.by_generation: dict[int, list[float]] = defaultdict(list)
        self.by_mask_ratio: dict[float, list[float]] = defaultdict(list)
        self.parent_by_generation: dict[int, list[float]] = defaultdict(list)
        self.child_scores: list[float] = []
        self.skipped = 0

    def add(self, record: dict[str, Any], min_generation: int | None, max_generation: int | None) -> None:
        resolved = record_delta(record)
        if resolved is None or record.get("generation") is None or record.get("mask_ratio") is None:
            self.skipped += 1
            return
        delta, parent_score = resolved
        generation = int(record["generation"])
        if min_generation is not None and generation < min_generation:
            return
        if max_generation is not None and generation > max_generation:
            return
        mask_ratio = float(record["mask_ratio"])
        if mask_ratio <= 0.0:
            return
        self.deltas.append(delta)
        self.by_generation[generation].append(delta)
        self.by_mask_ratio[mask_ratio].append(delta)
        self.parent_by_generation[generation].append(parent_score)
        if record.get("score") is not None:
            self.child_scores.append(float(record["score"]))

    def summary(self) -> dict[str, Any]:
        values = np.asarray(self.deltas, dtype=float)
        count = values.size
        if count == 0:
            return {"num_candidates": 0}
        stderr = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else float("nan")
        return {
            "num_candidates": int(count),
            "delta_mean": float(values.mean()),
            "delta_std": float(values.std(ddof=1)) if count > 1 else float("nan"),
            "delta_stderr": stderr,
            "delta_median": float(np.median(values)),
            "improved_frac": float(np.mean(values > 0)),
            "unchanged_frac": float(np.mean(values == 0)),
            "parent_score_mean": float(
                np.mean([score for scores in self.parent_by_generation.values() for score in scores])
            ),
            "child_score_mean": float(np.mean(self.child_scores)) if self.child_scores else None,
            "skipped_malformed": self.skipped,
        }

    def generation_curve(self) -> list[dict[str, Any]]:
        rows = []
        for generation in sorted(self.by_generation):
            values = np.asarray(self.by_generation[generation], dtype=float)
            parents = np.asarray(self.parent_by_generation[generation], dtype=float)
            rows.append(
                {
                    "generation": generation,
                    "num_candidates": int(values.size),
                    "delta_mean": float(values.mean()),
                    "delta_stderr": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
                    "parent_score_mean": float(parents.mean()),
                }
            )
        return rows

    def mask_ratio_table(self) -> dict[str, dict[str, Any]]:
        table = {}
        for ratio in sorted(self.by_mask_ratio):
            values = np.asarray(self.by_mask_ratio[ratio], dtype=float)
            table[f"{ratio:g}"] = {
                "num_candidates": int(values.size),
                "delta_mean": float(values.mean()),
                "delta_stderr": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
            }
        return table


def load_arm(run_dir: Path, label: str, min_generation: int | None, max_generation: int | None) -> ArmAccumulator:
    candidates = run_dir / "candidates.jsonl"
    if not candidates.exists():
        raise SystemExit(f"{candidates} not found; this run has no candidate log (log_candidates_jsonl must be true)")
    arm = ArmAccumulator(label)
    for record in stream_candidates(candidates):
        arm.add(record, min_generation, max_generation)
    if not arm.deltas:
        raise SystemExit(
            f"{candidates} yielded no usable records ({arm.skipped} skipped).\n"
            f"Record fields present: {peek_schema(candidates)}\n"
            "This script needs seed_score plus either delta_score or score. A log missing "
            "seed_score predates the candidate provenance logging; use "
            "batch41_random_control.py, which can recover the parent score from top_pool.pt."
        )
    return arm


def config_fingerprint(run_dir: Path) -> dict[str, Any]:
    """Pull the fields that must match between the two arms."""

    path = run_dir / "config_snapshot.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    pb = raw.get("patternboost", raw)
    generation = pb.get("generation", {})
    saturation = pb.get("saturation", {})
    return {
        "n": raw.get("n", pb.get("n")),
        "completion_operator": generation.get("completion_operator", "learned"),
        "num_generations": generation.get("num_generations"),
        "candidates_per_generation": generation.get("candidates_per_generation"),
        "remask_modes": generation.get("remask_modes"),
        "remask_mode_weights": generation.get("remask_mode_weights"),
        "mask_ratios": pb.get("mask_ratios"),
        "saturation_slot_order": saturation.get("slot_order"),
        "saturation_y_order": saturation.get("y_order"),
        "seed": pb.get("runtime", {}).get("seed"),
    }


def compare_configs(learned: dict[str, Any], random_arm: dict[str, Any]) -> list[str]:
    """Report every field that differs, so a broken control is visible."""

    expected_differences = {"completion_operator", "seed"}
    warnings = []
    for key in sorted(set(learned) | set(random_arm)):
        left, right = learned.get(key), random_arm.get(key)
        if left != right and key not in expected_differences:
            warnings.append(f"config mismatch on {key!r}: learned={left!r} random={right!r}")
    if random_arm.get("completion_operator") != "random":
        warnings.append(
            f"the --random run has completion_operator={random_arm.get('completion_operator')!r}, expected 'random'"
        )
    if learned.get("completion_operator") not in (None, "learned"):
        warnings.append(
            f"the --learned run has completion_operator={learned.get('completion_operator')!r}, expected 'learned'"
        )
    if learned.get("seed") != random_arm.get("seed"):
        warnings.append(
            f"seeds differ (learned={learned.get('seed')}, random={random_arm.get('seed')}); "
            "prefer matched seeds so pool trajectories start identically"
        )
    return warnings


def gap_stats(learned: ArmAccumulator, random_arm: ArmAccumulator) -> dict[str, Any]:
    """Unpaired difference of means with a Welch standard error."""

    left = np.asarray(learned.deltas, dtype=float)
    right = np.asarray(random_arm.deltas, dtype=float)
    stderr = math.sqrt(left.var(ddof=1) / left.size + right.var(ddof=1) / right.size)
    gap = float(left.mean() - right.mean())
    return {
        "gap_mean": gap,
        "gap_stderr": float(stderr),
        "gap_ci95": [gap - 1.96 * stderr, gap + 1.96 * stderr],
        "welch_t": float(gap / stderr) if stderr > 0 else float("nan"),
    }


def write_plot(learned: ArmAccumulator, random_arm: ArmAccumulator, out_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", flush=True)
        return

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left_values = np.asarray(learned.deltas, dtype=float)
    right_values = np.asarray(random_arm.deltas, dtype=float)
    lo = math.floor(min(left_values.min(), right_values.min())) - 0.5
    hi = math.ceil(max(left_values.max(), right_values.max())) + 0.5
    bins = np.arange(lo, hi + 1, 1.0)
    left.hist(right_values, bins=bins, alpha=0.6, density=True, label="random operator", color="#9aa0a6")
    left.hist(left_values, bins=bins, alpha=0.6, density=True, label="learned operator", color="#1a73e8")
    left.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    left.set_xlabel(r"$\Delta$ = score(child) $-$ score(parent)")
    left.set_ylabel("density")
    left.set_title("per-candidate score change")
    left.legend()

    for arm, color, label in ((random_arm, "#9aa0a6", "random operator"), (learned, "#1a73e8", "learned operator")):
        curve = arm.generation_curve()
        x = np.array([row["generation"] for row in curve], dtype=float)
        y = np.array([row["delta_mean"] for row in curve], dtype=float)
        err = np.array([row["delta_stderr"] for row in curve], dtype=float)
        right.plot(x, y, color=color, label=label, linewidth=1.6)
        right.fill_between(x, y - 1.96 * err, y + 1.96 * err, color=color, alpha=0.2, linewidth=0)
    right.set_xlabel("generation")
    right.set_ylabel(r"mean $\Delta$ per candidate")
    right.set_title("edit quality over training")
    right.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learned", type=Path, required=True, help="run dir of the learned-operator run")
    parser.add_argument("--random", dest="random_run", type=Path, required=True, help="run dir of the random-operator run")
    parser.add_argument("--min-generation", type=int, default=None)
    parser.add_argument("--max-generation", type=int, default=None)
    parser.add_argument("--late-generations", type=int, default=20, help="also report the last N generations separately")
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <learned run>/analysis_batch41")
    args = parser.parse_args()

    learned_config = config_fingerprint(args.learned)
    random_config = config_fingerprint(args.random_run)
    warnings = compare_configs(learned_config, random_config)
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)

    learned = load_arm(args.learned, "learned", args.min_generation, args.max_generation)
    random_arm = load_arm(args.random_run, "random", args.min_generation, args.max_generation)

    max_generation = max(max(learned.by_generation), max(random_arm.by_generation))
    late_start = max(1, max_generation - args.late_generations + 1)
    late_learned = load_arm(args.learned, "learned_late", late_start, args.max_generation)
    late_random = load_arm(args.random_run, "random_late", late_start, args.max_generation)

    out_dir = args.out_dir or (args.learned / "analysis_batch41")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "learned_run": str(args.learned),
        "random_run": str(args.random_run),
        "config_learned": learned_config,
        "config_random": random_config,
        "config_warnings": warnings,
        "overall": {
            "learned": learned.summary(),
            "random": random_arm.summary(),
            "gap": gap_stats(learned, random_arm),
        },
        f"late_{args.late_generations}_generations": {
            "generation_range": [late_start, max_generation],
            "learned": late_learned.summary(),
            "random": late_random.summary(),
            "gap": gap_stats(late_learned, late_random),
        },
        "by_mask_ratio": {
            "learned": learned.mask_ratio_table(),
            "random": random_arm.mask_ratio_table(),
        },
        "generation_curve": {
            "learned": learned.generation_curve(),
            "random": random_arm.generation_curve(),
        },
    }
    (out_dir / "batch41_compare_runs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (out_dir / "batch41_compare_runs_curve.csv").open("w", encoding="utf-8") as handle:
        handle.write("generation,arm,num_candidates,delta_mean,delta_stderr,parent_score_mean\n")
        for label, arm in (("learned", learned), ("random", random_arm)):
            for row in arm.generation_curve():
                handle.write(
                    f"{row['generation']},{label},{row['num_candidates']},"
                    f"{row['delta_mean']},{row['delta_stderr']},{row['parent_score_mean']}\n"
                )

    write_plot(
        learned,
        random_arm,
        out_dir / "batch41_compare_runs.png",
        f"N3L n={learned_config.get('n')} -- learned vs random completion operator",
    )

    overall = report["overall"]
    late = report[f"late_{args.late_generations}_generations"]
    print()
    print(f"{'':22s} {'learned':>12s} {'random':>12s} {'gap':>12s}")
    print(
        f"{'candidates':22s} {overall['learned']['num_candidates']:12d} "
        f"{overall['random']['num_candidates']:12d} {'':>12s}"
    )
    print(
        f"{'mean delta (all gens)':22s} {overall['learned']['delta_mean']:+12.3f} "
        f"{overall['random']['delta_mean']:+12.3f} {overall['gap']['gap_mean']:+12.3f}"
    )
    print(
        f"{f'mean delta (last {args.late_generations})':22s} {late['learned']['delta_mean']:+12.3f} "
        f"{late['random']['delta_mean']:+12.3f} {late['gap']['gap_mean']:+12.3f}"
    )
    print(
        f"{'improved frac':22s} {overall['learned']['improved_frac']:12.3f} "
        f"{overall['random']['improved_frac']:12.3f} {'':>12s}"
    )
    print(
        f"{'mean parent score':22s} {overall['learned']['parent_score_mean']:12.3f} "
        f"{overall['random']['parent_score_mean']:12.3f} {'':>12s}"
    )
    print()
    print(
        f"gap 95% CI (all gens)  {overall['gap']['gap_ci95'][0]:+.3f} .. {overall['gap']['gap_ci95'][1]:+.3f}"
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
