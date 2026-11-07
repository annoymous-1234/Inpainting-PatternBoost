#!/usr/bin/env python3
"""No-three-in-line -- paired random-completion control.

Tests whether the gain comes from the generative model or from local inheritance,
by holding the active core and the mask fixed and swapping only how the masked
slots get filled.

    learned arm : read from candidates.jsonl (delta_score), no model is loaded
    random arm  : recomputed here, model-free, by replaying the run's own
                  remask -> greedy_saturate path with completion_operator=random

The random arm reproduces ``PatternBoostGenerator`` exactly: under
``completion_operator="random"`` the generator skips decoding entirely and lets
``greedy_saturate`` fill the remasked slots with uniformly random legal rows
(see ``patternboost/generator.py::_uses_random_operator``). So the only removed
ingredient is the learned decoder.

Pairing.  Each learned candidate record carries ``seed_hash`` (its parent) and
``remask_mode``/``mask_ratio`` (its mask). Records are kept when their parent
still exists in a pool snapshot, and the control replays that parent.

The mask itself is *not* recoverable, in either remask mode. ``random`` is an
explicit rng draw. ``replaceability`` looks deterministic but is not: it ranks
filled slots by how many legal rows their column would have if the point were
removed, and on a near-saturated construction that count is 1 for essentially
every slot -- so the ranking is all ties and the 1e-6 tie-break jitter picks the
mask. Measured overlap between two draws at the same ratio is ~0.4.

So the control resamples the mask through the run's own ``generation_remask``
using the recorded mode and ratio. The mask *distribution* is therefore matched
exactly and the parent is identical; individual masks are not paired. The
report includes the mean masked-slot count of both arms so the matched-magnitude
claim can be checked rather than assumed.

Usage:
    cd no-three-in-line
    python analysis/batch41_random_control.py checkpoints/<run>/exp --num-samples 4000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import NoThreeInLineEnv  # noqa: E402
from models import slot_states_to_state_ids  # noqa: E402
from patternboost.remask import (  # noqa: E402
    random_generation_remask,
    replaceability_generation_remask,
)
from patternboost.state import (  # noqa: E402
    canonicalize_state_ids,
    filled_count,
    state_ids_to_env_reset_args,
)
from patternboost.top_pool import TopPool  # noqa: E402
from saturation import greedy_saturate  # noqa: E402

REPLACEABILITY_MODE = "replaceability"


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
        missing = [
            str(p.name)
            for p in (self.candidates, self.top_pool, self.config)
            if not p.exists()
        ]
        if missing:
            raise SystemExit(
                f"{self.path} is missing required file(s): {', '.join(missing)}\n"
                "This script needs candidates.jsonl (learned deltas), "
                "top_pool.pt (parent states) and config_snapshot.yaml (env params)."
            )


def load_pool_entries(run: RunDir) -> tuple[dict[str, Any], dict[str, int]]:
    """Merge every pool snapshot in the run into one hash -> entry map.

    ``top_pool.pt`` is rewritten in place each generation, so on its own it only
    exposes parents that survived to the end -- candidates seeded from entries
    that were later evicted get dropped. Runs with
    ``top_pool_snapshot_every_generations`` also write
    ``top_pool_generation_XXX.pt``; merging them recovers those parents.

    A canonical hash determines the state, and the score is a function of the
    state, so an entry is identical wherever it appears and the union is safe.
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
    """The subset of config_snapshot.yaml that defines the completion protocol."""

    def __init__(self, config_path: Path) -> None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pb = raw.get("patternboost", raw)
        grid_size = _find_scalar(raw, "n")
        if grid_size is None:
            raise SystemExit(f"{config_path} does not define the grid size 'n'")
        self.n = int(grid_size)
        saturation = pb.get("saturation", {})
        self.saturation_enabled = bool(saturation.get("enabled", True))
        self.saturation_max_steps = saturation.get("max_steps")
        self.saturation_slot_order = str(saturation.get("slot_order", "mrv"))
        self.saturation_y_order = str(saturation.get("y_order", "random"))
        decode = pb.get("decode", {})
        self.env_search_attempts = int(decode.get("env_num_search_attempts", 1))
        self.env_allow_deletions = bool(decode.get("env_allow_existing_deletions", False))
        self.completion_operator = str(
            pb.get("generation", {}).get("completion_operator", "learned")
        )

    def describe(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "saturation_enabled": self.saturation_enabled,
            "saturation_slot_order": self.saturation_slot_order,
            "saturation_y_order": self.saturation_y_order,
            "env_num_search_attempts": self.env_search_attempts,
            "env_allow_existing_deletions": self.env_allow_deletions,
            "completion_operator": self.completion_operator,
        }


def _find_scalar(mapping: Any, key: str) -> Any:
    """Find ``key`` bound to a scalar anywhere in a nested mapping, or None.

    The snapshot puts ``n`` at the top level, but tolerating a nested layout
    keeps the script working against older or hand-edited snapshots.
    """

    queue = [mapping]
    while queue:
        node = queue.pop(0)
        if not isinstance(node, dict):
            continue
        if key in node and not isinstance(node[key], (dict, list)):
            return node[key]
        queue.extend(value for value in node.values() if isinstance(value, dict))
    return None


# --------------------------------------------------------------------------
# candidate sampling
# --------------------------------------------------------------------------


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


def reservoir_sample(
    records: Iterator[dict[str, Any]],
    keep: "Filter",
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], Counter]:
    """Uniformly sample ``num_samples`` eligible records in a single pass."""

    chosen: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen = 0
    for record in records:
        stats["total"] += 1
        verdict = keep(record)
        stats[verdict] += 1
        if verdict != "ok":
            continue
        seen += 1
        slim = {
            "generation": record.get("generation"),
            "candidate_index": record.get("candidate_index"),
            "seed_hash": record["seed_hash"],
            "seed_score": record.get("seed_score"),
            "delta_score": record.get("delta_score"),
            "score": record.get("score"),
            "mask_ratio": float(record["mask_ratio"]),
            "remask_mode": record["remask_mode"],
            "selected_slot_ids": sorted(
                {
                    int(slot_id)
                    for step in record.get("step_stats", []) or []
                    for slot_id in step.get("selected_slot_ids", [])
                }
            ),
        }
        if len(chosen) < num_samples:
            chosen.append(slim)
        else:
            index = int(rng.integers(0, seen))
            if index < num_samples:
                chosen[index] = slim
    stats["eligible"] = seen
    return chosen, stats


class Filter:
    """Eligibility test for a learned candidate record."""

    def __init__(
        self,
        pool_hashes: set[str],
        *,
        min_generation: int | None,
        max_generation: int | None,
        mask_ratios: list[float] | None,
        remask_modes: list[str] | None,
    ) -> None:
        self.pool_hashes = pool_hashes
        self.min_generation = min_generation
        self.max_generation = max_generation
        self.mask_ratios = mask_ratios
        self.remask_modes = remask_modes

    def __call__(self, record: dict[str, Any]) -> str:
        # seed_hash is the only field with no fallback: without it we cannot
        # tell which pool entry was the parent, so there is nothing to pair.
        if "seed_hash" not in record:
            return "missing_seed_hash"
        if record.get("score") is None and record.get("delta_score") is None:
            return "missing_child_score"
        if record.get("mask_ratio") is None:
            return "missing_mask_ratio"
        generation = record.get("generation")
        if self.min_generation is not None and generation is not None:
            if generation < self.min_generation:
                return "out_of_generation_range"
        if self.max_generation is not None and generation is not None:
            if generation > self.max_generation:
                return "out_of_generation_range"
        mask_ratio = float(record.get("mask_ratio", 0.0))
        if mask_ratio <= 0.0:
            return "mask_ratio_zero"
        if self.mask_ratios is not None and not any(
            math.isclose(mask_ratio, value) for value in self.mask_ratios
        ):
            return "mask_ratio_filtered"
        if record.get("remask_mode") is None:
            return "missing_remask_mode"
        if self.remask_modes is not None and record["remask_mode"] not in self.remask_modes:
            return "remask_mode_filtered"
        if record["seed_hash"] not in self.pool_hashes:
            return "parent_not_in_final_pool"
        return "ok"


# --------------------------------------------------------------------------
# random arm
# --------------------------------------------------------------------------


def resolve_learned_delta(sample: dict[str, Any], parent_score: int) -> tuple[float, str]:
    """Return this candidate's learned delta and where it came from.

    Runs launched before the seed_score/delta_score provenance logging only
    record the child ``score``. Their parent score is still recoverable: the
    generator seeds from a pool entry and ``TopPoolEntry.score`` *is* the
    ``seed_score`` it would have logged, so ``score - parent_score`` is the same
    number -- provided the parent survived into the final pool, which the
    eligibility filter already guarantees.
    """

    if sample["delta_score"] is not None:
        return float(sample["delta_score"]), "delta_score"
    if sample["seed_score"] is not None and sample["score"] is not None:
        return float(sample["score"]) - float(sample["seed_score"]), "score-seed_score"
    return float(sample["score"]) - float(parent_score), "score-pool_parent_score"


def rebuild_mask(
    parent_state: np.ndarray,
    n: int,
    mask_ratio: float,
    remask_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Draw a mask from the same distribution the run used for this candidate.

    Both modes are rng-driven in practice (see the module docstring on
    replaceability ties), so this is a fresh draw at the recorded mode and
    ratio, not a replay.
    """

    if remask_mode == REPLACEABILITY_MODE:
        return replaceability_generation_remask(parent_state, n, mask_ratio, rng)
    return random_generation_remask(parent_state, mask_ratio, rng)


def random_completion_score(
    partial_state: np.ndarray,
    config: RunConfig,
    rng: np.random.Generator,
) -> int:
    """Fill the masked slots with random legal rows and return the final score.

    This is exactly what ``PatternBoostGenerator`` does under
    ``completion_operator="random"``: build the remasked env, then let
    ``greedy_saturate`` pick a legal row uniformly at random for each slot it
    selects. No model is involved.
    """

    init_points, init_assignment = state_ids_to_env_reset_args(partial_state)
    env = NoThreeInLineEnv(
        n=config.n,
        num_slots=2 * config.n,
        proposal_search_config={
            "num_search_attempts": config.env_search_attempts,
            "allow_existing_deletions": config.env_allow_deletions,
        },
    )
    env.reset(init_points=init_points, init_slot_assignment=init_assignment)
    if config.saturation_enabled:
        greedy_saturate(
            env,
            rng,
            max_steps=config.saturation_max_steps,
            slot_order=config.saturation_slot_order,
            y_order=config.saturation_y_order,
        )
    final_state = slot_states_to_state_ids(env.slot_states, env.n).cpu().numpy()
    return filled_count(canonicalize_state_ids(final_state, config.n))


def run_control_arm(
    samples: list[dict[str, Any]],
    entries: dict[str, Any],
    config: RunConfig,
    rng: np.random.Generator,
    repeats: int,
) -> list[dict[str, Any]]:
    """Compute the random-arm delta for every sampled learned candidate."""

    rows: list[dict[str, Any]] = []
    delta_sources: Counter = Counter()
    for index, sample in enumerate(samples):
        entry = entries[sample["seed_hash"]]
        parent_state = np.asarray(entry.slot_state_ids, dtype=np.int64)
        parent_score = int(entry.score)
        delta_learned, delta_source = resolve_learned_delta(sample, parent_score)
        delta_sources[delta_source] += 1
        deltas = []
        masked_counts = []
        for _ in range(repeats):
            partial_state, remasked_slots = rebuild_mask(
                parent_state,
                config.n,
                sample["mask_ratio"],
                sample["remask_mode"],
                rng,
            )
            masked_counts.append(len(remasked_slots))
            deltas.append(random_completion_score(partial_state, config, rng) - parent_score)
        rows.append(
            {
                **sample,
                "pool_parent_score": parent_score,
                "delta_random": float(np.mean(deltas)),
                "delta_random_draws": deltas,
                "delta_learned": delta_learned,
                "delta_source": delta_source,
                "control_masked_slots": float(np.mean(masked_counts)),
                "learned_selected_slots": len(sample["selected_slot_ids"]),
            }
        )
        if (index + 1) % 200 == 0:
            print(f"  ... {index + 1}/{len(samples)} control completions", flush=True)
    if len(delta_sources) > 1 or "delta_score" not in delta_sources:
        print(f"learned delta sources: {json.dumps(dict(delta_sources))}", flush=True)
    return rows


# --------------------------------------------------------------------------
# statistics and reporting
# --------------------------------------------------------------------------


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    learned = np.array([row["delta_learned"] for row in rows], dtype=float)
    random_arm = np.array([row["delta_random"] for row in rows], dtype=float)
    diff = learned - random_arm
    n = len(rows)
    stderr = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "num_pairs": n,
        "delta_learned_mean": float(learned.mean()),
        "delta_learned_std": float(learned.std(ddof=1)) if n > 1 else float("nan"),
        "delta_random_mean": float(random_arm.mean()),
        "delta_random_std": float(random_arm.std(ddof=1)) if n > 1 else float("nan"),
        "paired_gap_mean": float(diff.mean()),
        "paired_gap_stderr": stderr,
        "paired_gap_ci95": [
            float(diff.mean() - 1.96 * stderr),
            float(diff.mean() + 1.96 * stderr),
        ],
        "learned_wins_frac": float(np.mean(diff > 0)),
        "ties_frac": float(np.mean(diff == 0)),
    }


def stratify(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return {
        str(value): summarize(group)
        for value, group in sorted(groups.items(), key=lambda item: str(item[0]))
        if len(group) >= 2
    }


def write_plot(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", flush=True)
        return

    learned = np.array([row["delta_learned"] for row in rows], dtype=float)
    random_arm = np.array([row["delta_random"] for row in rows], dtype=float)
    ratios = sorted({row["mask_ratio"] for row in rows})

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    lo = math.floor(min(learned.min(), random_arm.min())) - 0.5
    hi = math.ceil(max(learned.max(), random_arm.max())) + 0.5
    bins = np.arange(lo, hi + 1, 1.0)
    left.hist(random_arm, bins=bins, alpha=0.6, label="random completion", color="#9aa0a6")
    left.hist(learned, bins=bins, alpha=0.6, label="learned completion", color="#1a73e8")
    left.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    left.set_xlabel(r"$\Delta$ = score(child) $-$ score(parent)")
    left.set_ylabel("candidates")
    left.set_title("per-candidate score change")
    left.legend()

    means_l, means_r, errs_l, errs_r = [], [], [], []
    for ratio in ratios:
        group = [row for row in rows if row["mask_ratio"] == ratio]
        gl = np.array([row["delta_learned"] for row in group])
        gr = np.array([row["delta_random"] for row in group])
        means_l.append(gl.mean())
        means_r.append(gr.mean())
        errs_l.append(gl.std(ddof=1) / math.sqrt(len(gl)) if len(gl) > 1 else 0.0)
        errs_r.append(gr.std(ddof=1) / math.sqrt(len(gr)) if len(gr) > 1 else 0.0)
    x = np.arange(len(ratios), dtype=float)
    right.bar(x - 0.19, means_r, 0.36, yerr=errs_r, label="random", color="#9aa0a6", capsize=3)
    right.bar(x + 0.19, means_l, 0.36, yerr=errs_l, label="learned", color="#1a73e8", capsize=3)
    right.axhline(0.0, color="black", linewidth=0.8)
    right.set_xticks(x)
    right.set_xticklabels([f"{ratio:g}" for ratio in ratios])
    right.set_xlabel("mask ratio")
    right.set_ylabel(r"mean $\Delta$")
    right.set_title("by mask ratio (same parent, same mask)")
    right.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run directory holding candidates.jsonl + top_pool.pt + config_snapshot.yaml")
    parser.add_argument("--num-samples", type=int, default=2000, help="learned candidates to pair against (default 2000)")
    parser.add_argument("--repeats", type=int, default=1, help="random completions per pair, averaged (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-generation", type=int, default=None)
    parser.add_argument("--max-generation", type=int, default=None)
    parser.add_argument("--mask-ratios", type=float, nargs="*", default=None, help="restrict to these mask ratios")
    parser.add_argument(
        "--remask-modes",
        nargs="*",
        default=None,
        choices=["random", "replaceability"],
        help="restrict to these remask modes (default: both). The mask is resampled either way; "
             "see the module docstring on why replaceability is not replayable.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <run_dir>/analysis_batch41")
    args = parser.parse_args()

    run = RunDir(args.run_dir)
    run.require()
    config = RunConfig(run.config)
    if config.completion_operator != "learned":
        print(
            f"WARNING: this run's completion_operator is {config.completion_operator!r}; "
            "its candidates.jsonl is not a learned arm.",
            flush=True,
        )
    out_dir = args.out_dir or (args.run_dir / "analysis_batch41")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run     : {args.run_dir}")
    print(f"config  : {json.dumps(config.describe())}")

    entries, provenance = load_pool_entries(run)
    pool_hashes = set(entries)
    print(
        f"pool    : {len(pool_hashes)} distinct entries from {len(provenance)} snapshot(s), "
        f"best={max(entry.score for entry in entries.values())}"
    )
    if len(provenance) > 1:
        print(f"          new entries per snapshot: {json.dumps(provenance)}")
    else:
        print(
            "          only the final pool is available; candidates seeded from evicted entries "
            "will be dropped. Enable patternboost.runtime.top_pool_snapshot_every_generations on future runs."
        )

    rng = np.random.default_rng(args.seed)
    keep = Filter(
        pool_hashes,
        min_generation=args.min_generation,
        max_generation=args.max_generation,
        mask_ratios=args.mask_ratios,
        remask_modes=args.remask_modes,
    )
    schema = peek_schema(run.candidates)
    print(f"log     : {len(schema)} fields per record -> {schema}")
    samples, scan_stats = reservoir_sample(stream_candidates(run.candidates), keep, args.num_samples, rng)
    print(f"scanned : {json.dumps(dict(sorted(scan_stats.items())))}")
    if not samples:
        dominant = max(
            ((reason, count) for reason, count in scan_stats.items() if reason not in {"total", "eligible", "ok"}),
            key=lambda item: item[1],
            default=("unknown", 0),
        )
        hints = {
            "missing_seed_hash": (
                "records have no seed_hash, so the parent cannot be identified. "
                "This log predates candidate provenance logging and cannot be used for the paired design; "
                "use batch41_compare_runs.py against a learned/random run pair instead."
            ),
            "missing_child_score": "records have neither 'score' nor 'delta_score'.",
            "missing_mask_ratio": "records have no mask_ratio, so the mask cannot be reproduced.",
            "missing_remask_mode": "records have no remask_mode, so the mask cannot be reproduced.",
            "parent_not_in_final_pool": (
                "no sampled parent was found in any available pool snapshot. If only top_pool.pt exists, "
                "the run had no per-generation snapshots -- try --min-generation to restrict to late "
                "generations, where parents are most likely to have survived."
            ),
            "remask_mode_filtered": "no candidate used the remask mode(s) requested by --remask-modes.",
            "mask_ratio_zero": "every candidate had mask_ratio 0 (nothing was masked).",
        }
        raise SystemExit(
            f"no eligible candidates. Dominant reason: {dominant[0]} ({dominant[1]} records).\n"
            f"{hints.get(dominant[0], '')}\n"
            f"Record fields present: {schema}"
        )
    print(f"paired  : running random completion for {len(samples)} candidates", flush=True)

    rows = run_control_arm(samples, entries, config, rng, args.repeats)

    report = {
        "run_dir": str(args.run_dir),
        "config": config.describe(),
        "args": {
            "num_samples": args.num_samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "remask_modes": args.remask_modes,
            "min_generation": args.min_generation,
            "max_generation": args.max_generation,
            "mask_ratios": args.mask_ratios,
        },
        "scan": dict(sorted(scan_stats.items())),
        "pool_size": len(pool_hashes),
        "pool_snapshots": provenance,
        "pairing": "same parent and same mask distribution (mode + ratio); mask resampled, not replayed",
        "control_masked_slots_mean": float(np.mean([row["control_masked_slots"] for row in rows])),
        "learned_selected_slots_mean": float(np.mean([row["learned_selected_slots"] for row in rows])),
        "overall": summarize(rows),
        "by_mask_ratio": stratify(rows, "mask_ratio"),
        "by_parent_score": stratify(rows, "pool_parent_score"),
        "by_generation": stratify(rows, "generation"),
    }

    (out_dir / "batch41_random_control.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "batch41_random_control_pairs.csv").open("w", encoding="utf-8") as handle:
        handle.write("generation,seed_hash,parent_score,mask_ratio,remask_mode,delta_learned,delta_random\n")
        for row in rows:
            handle.write(
                f"{row['generation']},{row['seed_hash'][:12]},{row['pool_parent_score']},"
                f"{row['mask_ratio']},{row['remask_mode']},{row['delta_learned']},{row['delta_random']}\n"
            )
    write_plot(rows, out_dir / "batch41_random_control.png", f"N3L n={config.n} -- learned vs random completion")

    overall = report["overall"]
    print()
    print(f"pairs                {overall['num_pairs']}")
    print(
        f"masked slots         control {report['control_masked_slots_mean']:.2f} "
        f"vs learned-decoded {report['learned_selected_slots_mean']:.2f} (magnitude match)"
    )
    print(f"mean delta learned   {overall['delta_learned_mean']:+.3f}")
    print(f"mean delta random    {overall['delta_random_mean']:+.3f}")
    print(
        f"paired gap           {overall['paired_gap_mean']:+.3f} "
        f"(95% CI {overall['paired_gap_ci95'][0]:+.3f} .. {overall['paired_gap_ci95'][1]:+.3f})"
    )
    print(f"learned wins         {overall['learned_wins_frac']:.3f}  (ties {overall['ties_frac']:.3f})")
    print()
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
