#!/usr/bin/env python3
"""Circle packing -- paired transfer control.

A gain over the random operator alone cannot rule out memorisation: the model is
retrained every generation on the very pool the parents are drawn from, so even
a copier would beat the random arm. This script removes that confound by adding
an arm whose model never saw this pool.

    home     : the model trained by the run the parents come from
    transfer : a model from a different run (different seed -> different pool)
    random   : completion_operator=random_perturb, no model at all

All arms complete the *same* parent under the *same* edit set. Reading:

    transfer - random  > 0   structure a fresh model applies to layouts it never
                             trained on; memorisation cannot produce this
    home - transfer    > 0   the premium from having trained on this pool
    transfer ~ random        the gain over random was memorisation

Method.  Every arm runs the production ``PatternBoostGenerator`` with the *pool
run's own config*; only the completion operator differs. ``_FixedEditGenerator``
overrides ``_initialize_candidate_env`` so the edit set comes from a plan drawn
once up front rather than from a fresh ``sample_edit_set`` per candidate.
Everything after that -- the outer-loop rounds, the DDPM sampler, commitment,
the radius LP -- is untouched library code, and the random arm is the same
``random_perturb`` operator the full-loop ablation used.

Decoding is whatever the pool run configured: with ``modes: [full_episode]``,
``outer_loop.max_rounds: 6`` and ``env.activation_budget: 4`` that is the
paper's iterative "Ours" setting, six rounds each committing at most four of the
K circles. (The one-shot ablation is ``activation_budget >= K`` with a single
round; the script prints which one it is rather than assuming.)

Pairing is tighter than replaying candidates.jsonl, which can match only the edit
*distribution*. Here the edit mask, the parent, and the candidate seed are identical across
arms, so the jittered draft that seeds the sampler is identical too. What is not
shared is the stochasticity downstream: the streams diverge as soon as the
operators do.

Usage:
    cd packing
    python analysis/transfer_control.py \
        --pool-run checkpoints/K26_ddpm_main_batch4 \
        --transfer rt7=checkpoints/K26_ddpm_transfer_rt7/generation_040.pt \
        --parent-generations 20 30 40 \
        --num-parents 4096

``--home`` defaults to the highest-numbered ``generation_*.pt`` inside
``--pool-run``, so the usual invocation only names the transfer donors.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circle_pack.config import ProjectConfig, load_config  # noqa: E402
from circle_pack.env import CirclePackingEnv  # noqa: E402
from models.factory import build_model  # noqa: E402
from patternboost.edit import sample_edit_set  # noqa: E402
from patternboost.env_init import initialize_partial_env_from_seed  # noqa: E402
from patternboost.generator import PatternBoostGenerator  # noqa: E402
from patternboost.top_pool import TopPool  # noqa: E402
from patternboost.types import CircleTopPoolEntry, EditSet  # noqa: E402
from pipelines.data import build_jittered_draft  # noqa: E402

RANDOM_ARM = "random"
HOME_ARM = "home"
RANDOM_OPERATOR = "random_perturb"


# --------------------------------------------------------------------------
# run directory contract
# --------------------------------------------------------------------------


class RunDir:
    """Locate the files this analysis needs inside one PatternBoost run dir."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.top_pool = path / "top_pool.pt"
        # resolved_config.yaml is what the run actually executed; config_snapshot
        # is the file as written. They agree, but prefer the resolved one.
        self.config = path / "resolved_config.yaml"
        if not self.config.exists():
            self.config = path / "config_snapshot.yaml"

    def require(self) -> None:
        missing = [p.name for p in (self.config, self.top_pool) if not p.exists()]
        if missing:
            raise SystemExit(
                f"{self.path} is missing required file(s): {', '.join(missing)}\n"
                "This script needs resolved_config.yaml (the completion protocol) "
                "and top_pool.pt (parent layouts)."
            )

    def snapshots(self) -> list[Path]:
        return sorted(self.path.glob("top_pool_generation_*.pt")) + [self.top_pool]

    def latest_checkpoint(self) -> Path:
        """Return the highest-numbered training checkpoint written by this run.

        ``latest.pt`` holds the same weights but is overwritten in place and
        carries no generation number, so the explicit per-generation file is
        preferred: the report should say which generation the home model is.
        """

        pattern = re.compile(r"generation_(\d+)\.pt$")
        best: tuple[int, Path] | None = None
        for path in self.path.glob("generation_*.pt"):
            match = pattern.search(path.name)
            if match is None:
                continue
            generation = int(match.group(1))
            if best is None or generation > best[0]:
                best = (generation, path)
        if best is None:
            fallback = self.path / "latest.pt"
            if fallback.exists():
                return fallback
            raise SystemExit(
                f"{self.path} holds no model checkpoint. Pass --home explicitly, or check "
                "that the run had patternboost.runtime.save_checkpoints on."
            )
        return best[1]


def snapshot_generation(path: Path) -> int | None:
    match = re.search(r"top_pool_generation_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# parents
# --------------------------------------------------------------------------


def load_pool_entries(
    run: RunDir,
    cfg: ProjectConfig,
    source: str,
    generations: list[int] | None,
) -> tuple[dict[str, CircleTopPoolEntry], dict[str, int | None], dict[str, int]]:
    """Return parent layouts keyed by canonical hash, plus their provenance.

    ``top_pool.pt`` is rewritten in place each generation, so on its own it
    exposes only the layouts that survived to the end. Runs launched with
    ``top_pool_snapshot_every_generations`` also write
    ``top_pool_generation_XXX.pt``; merging them recovers earlier, typically
    weaker parents and lets the report break the result down by how far into the
    run a parent came from.

    ``generations`` restricts the union to named snapshots. The final
    generation's snapshot has the same contents as ``top_pool.pt``, so naming it
    is equivalent to including the final pool. The recorded generation is the
    *earliest* snapshot an entry appears in.
    """

    paths = run.snapshots() if source == "union" else [run.top_pool]
    if source == "union" and generations:
        wanted = set(generations)
        available = {snapshot_generation(path) for path in paths}
        missing = sorted(wanted - available)
        if missing:
            have = sorted(g for g in available if g is not None)
            raise SystemExit(
                f"{run.path} has no pool snapshot for generation(s) {missing}; "
                f"available: {have or 'none'}"
            )
        paths = [path for path in paths if snapshot_generation(path) in wanted]

    entries: dict[str, CircleTopPoolEntry] = {}
    origin: dict[str, int | None] = {}
    provenance: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        pool = TopPool.load(path)
        added = 0
        for entry in pool.entries():
            if entry.canonical_hash in entries:
                continue
            entries[entry.canonical_hash] = entry
            origin[entry.canonical_hash] = snapshot_generation(path)
            added += 1
        provenance[path.name] = added
    if not entries:
        raise SystemExit(f"no pool entries found under {run.path}")
    del cfg
    return entries, origin, provenance


# --------------------------------------------------------------------------
# edit plan
# --------------------------------------------------------------------------


@dataclass
class Trial:
    """One parent plus one edit set, shared by every arm."""

    index: int
    entry: CircleTopPoolEntry
    parent_generation: int | None
    edit_set: EditSet
    mode: str
    candidate_seed: int


def build_plan(
    parents: list[tuple[CircleTopPoolEntry, int | None]],
    cfg: ProjectConfig,
    repeats: int,
    base_seed: int,
    rng: np.random.Generator,
) -> list[Trial]:
    """Draw one edit set per (parent, repeat) and freeze it for every arm.

    Drawing the plan up front is what makes the arms comparable: the edit set is
    a property of the trial, not of the arm, so a difference between arms can
    only come from how the freed circles were placed. ``candidate_seed`` is
    fixed per trial for the same reason -- it seeds the partial-environment
    initialisation, which every arm must share.
    """

    modes = list(cfg.patternboost.generation.modes)
    weights = list(cfg.patternboost.generation.mode_weights or [1.0] * len(modes))
    total = float(sum(weights))
    probabilities = [float(weight) / total for weight in weights]

    trials: list[Trial] = []
    for entry, generation in parents:
        for _ in range(repeats):
            index = len(trials)
            edit_set = sample_edit_set(entry.centers, entry.active_mask, cfg.patternboost.edit, rng)
            mode = str(rng.choice(modes, p=probabilities))
            trials.append(
                Trial(
                    index=index,
                    entry=entry,
                    parent_generation=generation,
                    edit_set=edit_set,
                    mode=mode,
                    candidate_seed=base_seed + index,
                )
            )
    return trials


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


class _FixedEditGenerator(PatternBoostGenerator):
    """A generator that consumes a pre-drawn edit set instead of sampling one.

    Only ``_initialize_candidate_env`` changes, and only in where the edit set
    comes from: the draft construction and environment initialisation below are
    the same two calls the library makes.
    """

    def load_plan(self, trials: list[Trial]) -> None:
        self._trials = trials
        self._cursor = 0

    def _initialize_candidate_env(
        self,
        seed_entry: CircleTopPoolEntry,
        candidate_seed: int | None,
    ) -> tuple[EditSet, CirclePackingEnv]:
        trial = self._trials[self._cursor]
        self._cursor += 1
        if trial.entry.canonical_hash != seed_entry.canonical_hash:
            raise RuntimeError(
                "edit plan and seed batch fell out of step; seed entries must be "
                "passed in plan order"
            )
        edit_set = trial.edit_set
        fixed_mask = ~np.asarray(edit_set.edit_mask, dtype=bool)
        _, draft_centers = build_jittered_draft(seed_entry.centers, fixed_mask, self.rng, cfg=self.cfg)
        env = CirclePackingEnv(self.cfg)
        initialize_partial_env_from_seed(
            env,
            seed_entry.centers,
            edit_set.edit_mask,
            draft_centers=draft_centers,
            seed=candidate_seed,
        )
        return edit_set, env


def with_random_operator(cfg: ProjectConfig, sigma: float) -> ProjectConfig:
    """Return a copy of ``cfg`` whose completion operator is the random one."""

    clone = copy.deepcopy(cfg)
    generation = clone.patternboost.generation
    for field, value in (("completion_operator", RANDOM_OPERATOR), ("random_center_jitter_std", float(sigma))):
        try:
            setattr(generation, field, value)
        except Exception:  # frozen dataclass
            object.__setattr__(generation, field, value)
    return clone


def load_model(checkpoint_path: Path, cfg: ProjectConfig, device: torch.device) -> tuple[Any, dict[str, Any]]:
    """Build the architecture from the *pool run's* config and load one checkpoint.

    Building every arm from one config is deliberate: a donor whose architecture
    differs fails loudly on ``strict=True`` instead of quietly half-loading.
    """

    model = build_model(cfg).to(device=device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise SystemExit(f"{checkpoint_path} has no model_state_dict")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    meta = {
        "path": str(checkpoint_path),
        "generation": payload.get("generation"),
        "mean_loss": payload.get("mean_loss"),
    }
    match = re.search(r"generation_(\d+)\.pt$", checkpoint_path.name)
    if match:
        meta["generation"] = int(match.group(1))
    donor = RunDir(checkpoint_path.parent)
    if donor.config.exists():
        donor_cfg = load_config(donor.config)
        meta["donor"] = {
            "run_dir": str(checkpoint_path.parent),
            "K": int(donor_cfg.problem.K),
            "seed": int(donor_cfg.patternboost.runtime.seed),
            "num_generations": int(donor_cfg.patternboost.generation.num_generations),
            "training_mode": str(donor_cfg.patternboost.training.mode),
        }
    else:
        meta["donor"] = {"run_dir": str(checkpoint_path.parent)}
    return model, meta


def score_of(result: Any, use_final_score: bool) -> float | None:
    """Return one candidate's score, or None when it is unusable.

    A candidate whose generation raised, whose packing is illegal, or whose LP
    did not return leaves nothing to compare; the caller drops that trial from
    *every* arm so the pairing survives.
    """

    candidate = result.candidate
    if not candidate.legal:
        return None
    if use_final_score and candidate.polish_score is not None:
        return float(candidate.polish_score)
    if candidate.raw_score is None:
        return None
    return float(candidate.raw_score)


def run_arm(
    name: str,
    trials: list[Trial],
    cfg: ProjectConfig,
    model: Any,
    seed: int,
    batch_size: int,
    inference_mode: str,
    use_final_score: bool,
) -> np.ndarray:
    """Return the per-trial delta (child score minus parent score) for one arm.

    NaN marks a trial this arm could not score; the caller intersects the NaN
    masks across arms so every reported statistic is computed on trials all arms
    completed.
    """

    generator = _FixedEditGenerator(model, cfg, np.random.default_rng(seed))
    generator.load_plan(trials)

    deltas = np.full(len(trials), np.nan, dtype=float)
    started = time.time()
    for start in range(0, len(trials), batch_size):
        chunk = trials[start : start + batch_size]
        work_items = [
            generator.prepare_batch_work_item(
                trial.entry,
                mode=trial.mode,
                candidate_seed=trial.candidate_seed,
                candidate_index=trial.index,
            )
            for trial in chunk
        ]
        results = generator.generate_batch(work_items, inference_mode=inference_mode)
        by_index = {int(r.candidate.metadata.get("candidate_index", -1)): r for r in results}
        for trial in chunk:
            result = by_index.get(trial.index)
            if result is None:
                continue
            score = score_of(result, use_final_score)
            if score is not None:
                deltas[trial.index] = score - float(trial.entry.score)
        done = start + len(chunk)
        if done % (batch_size * 10) == 0 or done == len(trials):
            rate = done / max(time.time() - started, 1e-9)
            print(f"  [{name}] {done}/{len(trials)}  ({rate:.1f} trials/s)", flush=True)
    return deltas


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def arm_stats(delta: np.ndarray) -> dict[str, Any]:
    count = int(delta.size)
    return {
        "num_trials": count,
        "delta_mean": float(delta.mean()) if count else float("nan"),
        "delta_std": float(delta.std(ddof=1)) if count > 1 else float("nan"),
        "frac_improved": float(np.mean(delta > 0)) if count else float("nan"),
    }


def paired_gap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return the paired difference ``left - right`` with a normal 95% CI."""

    diff = left - right
    count = int(diff.size)
    stderr = float(diff.std(ddof=1) / math.sqrt(count)) if count > 1 else float("nan")
    mean = float(diff.mean()) if count else float("nan")
    return {
        "num_pairs": count,
        "gap_mean": mean,
        "gap_stderr": stderr,
        "gap_ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr],
        "left_wins_frac": float(np.mean(diff > 0)) if count else float("nan"),
    }


def breakdown(trials: list[Trial], deltas: dict[str, np.ndarray], key: Any, arms: list[str]) -> dict[str, Any]:
    """Group trials by ``key(trial)`` and report every arm plus its gap to random."""

    buckets: dict[Any, list[int]] = {}
    for position, trial in enumerate(trials):
        buckets.setdefault(key(trial), []).append(position)
    report: dict[str, Any] = {}
    for bucket, positions in sorted(buckets.items(), key=lambda item: (item[0] is None, item[0])):
        index = np.asarray(positions, dtype=int)
        entry: dict[str, Any] = {"num_trials": int(index.size)}
        for arm in arms:
            entry[f"delta_{arm}_mean"] = float(deltas[arm][index].mean())
        for arm in arms:
            if arm == RANDOM_ARM:
                continue
            entry[f"gap_{arm}_vs_random"] = paired_gap(deltas[arm][index], deltas[RANDOM_ARM][index])["gap_mean"]
        report[str(bucket)] = entry
    return report


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def parse_donor(value: str) -> tuple[str, Path]:
    """Parse ``name=path``; a bare path is named after its run directory."""

    if "=" in value:
        name, _, path = value.partition("=")
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.parent.name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired transfer control for circle packing.")
    parser.add_argument(
        "--pool-run",
        type=Path,
        required=True,
        help="run directory supplying the parents and the completion protocol "
        "(e.g. checkpoints/K26_ddpm_main_batch4)",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="checkpoint of the model that trained on this pool "
        "(default: the highest-numbered generation_*.pt inside --pool-run)",
    )
    parser.add_argument(
        "--transfer",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="checkpoint from a different run; repeatable",
    )
    parser.add_argument("--num-parents", type=int, default=4096)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="independent edit-set draws per parent (default 1); total trials = parents x repeats",
    )
    parser.add_argument("--parent-source", choices=("union", "final"), default="union")
    parser.add_argument(
        "--parent-generations",
        type=int,
        nargs="*",
        default=None,
        metavar="G",
        help="restrict the union to these snapshot generations, e.g. 20 30 40 "
        "(default: every snapshot). Ignored when --parent-source is 'final'",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="random arm center jitter (default 0.05; "
        "the full-loop ablation swept 0.04-0.10)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="default: cuda when available")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="candidates per batched sampler call (default: the run's candidate_batch_size)",
    )
    parser.add_argument(
        "--use-final-score",
        action="store_true",
        help="score with polish_score instead of raw_score; set this if the run had SRP on",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <pool-run>/analysis_transfer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = RunDir(args.pool_run)
    run.require()

    cfg = load_config(run.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_mode = str(cfg.patternboost.training.mode)
    if inference_mode != "ddpm":
        print(f"note: pool run trains in {inference_mode!r} mode; arms will sample with it")

    # ---- parents ----------------------------------------------------------
    entries, origin, provenance = load_pool_entries(run, cfg, args.parent_source, args.parent_generations)
    rng = np.random.default_rng(args.seed)
    hashes = sorted(entries)
    take = min(args.num_parents, len(hashes))
    chosen = rng.choice(len(hashes), size=take, replace=False)
    parents = [(entries[hashes[int(i)]], origin[hashes[int(i)]]) for i in chosen]

    # ---- edit plan --------------------------------------------------------
    trials = build_plan(parents, cfg, args.repeats, base_seed=args.seed * 1_000_003, rng=rng)

    # ---- arms -------------------------------------------------------------
    home_path = args.home or run.latest_checkpoint()
    donors = [parse_donor(value) for value in args.transfer]
    if not donors:
        raise SystemExit("no --transfer donor given; the transfer arm is the point of this script")

    batch_size = args.batch_size or max(1, int(cfg.patternboost.generation.candidate_batch_size))
    arms = [RANDOM_ARM, HOME_ARM] + [name for name, _ in donors]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"duplicate arm name in {arms}")

    rounds = int(cfg.outer_loop.max_rounds)
    budget = int(cfg.env.activation_budget)
    decoding = "one-shot" if budget >= int(cfg.problem.K) else "iterative"
    print(f"pool run       {run.path}")
    print(f"parent pools   {', '.join(f'{name}(+{added})' for name, added in provenance.items())}")
    print(f"parents        {take} of {len(entries)} (source={args.parent_source}) "
          f"x {args.repeats} edit draw(s) = {len(trials)} trials")
    print(f"decoding       {decoding} (activation_budget={budget}, max_rounds={rounds}, K={cfg.problem.K}); "
          f"sampler={inference_mode}")
    print(f"edit           policies={list(cfg.patternboost.edit.policies)} "
          f"ratios={list(cfg.patternboost.edit.edit_ratios)}")
    print(f"random arm     {RANDOM_OPERATOR}, jitter std {args.sigma}")
    print(f"device         {device}   batch_size={batch_size}")
    print(f"arms           {', '.join(arms)}")
    print()

    deltas: dict[str, np.ndarray] = {}
    model_meta: dict[str, Any] = {}

    print(f"[{RANDOM_ARM}] no model")
    deltas[RANDOM_ARM] = run_arm(
        RANDOM_ARM, trials, with_random_operator(cfg, args.sigma), None,
        args.seed, batch_size, inference_mode, args.use_final_score,
    )

    for name, checkpoint_path in [(HOME_ARM, home_path)] + donors:
        if not Path(checkpoint_path).exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        model, meta = load_model(Path(checkpoint_path), cfg, device)
        model_meta[name] = meta
        print(f"[{name}] {checkpoint_path}  (donor seed={meta['donor'].get('seed')})")
        deltas[name] = run_arm(
            name, trials, cfg, model, args.seed, batch_size, inference_mode, args.use_final_score
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- keep only trials every arm scored --------------------------------
    usable = np.ones(len(trials), dtype=bool)
    dropped: Counter = Counter()
    for name in arms:
        bad = np.isnan(deltas[name])
        dropped[name] = int(bad.sum())
        usable &= ~bad
    kept = [trial for position, trial in enumerate(trials) if usable[position]]
    deltas = {name: values[usable] for name, values in deltas.items()}
    if not kept:
        raise SystemExit("no trial was scored by every arm; nothing to compare")
    if any(dropped.values()):
        print(f"\ndropped per arm (illegal or unscored): {dict(dropped)}; "
              f"{len(kept)}/{len(trials)} trials kept")

    # ---- statistics -------------------------------------------------------
    transfer_names = [name for name, _ in donors]
    report: dict[str, Any] = {
        "pool_run": str(run.path),
        "args": {
            "num_parents": take,
            "repeats": args.repeats,
            "parent_source": args.parent_source,
            "parent_generations": args.parent_generations,
            "sigma": args.sigma,
            "seed": args.seed,
            "device": str(device),
            "batch_size": batch_size,
            "use_final_score": args.use_final_score,
        },
        "protocol": {
            "K": int(cfg.problem.K),
            "decoding": decoding,
            "activation_budget": budget,
            "max_rounds": rounds,
            "inference_mode": inference_mode,
            "edit_policies": list(cfg.patternboost.edit.policies),
            "edit_ratios": list(cfg.patternboost.edit.edit_ratios),
            "srp_enabled": bool(cfg.patternboost.polish.srp_enabled),
        },
        "pool_snapshots": provenance,
        "pool_entries_available": len(entries),
        "models": model_meta,
        "num_trials_planned": len(trials),
        "num_trials_kept": len(kept),
        "dropped_per_arm": dict(dropped),
        "arms": {name: arm_stats(deltas[name]) for name in arms},
        "gap_vs_random": {
            name: paired_gap(deltas[name], deltas[RANDOM_ARM]) for name in [HOME_ARM] + transfer_names
        },
        "home_minus_transfer": {
            name: paired_gap(deltas[HOME_ARM], deltas[name]) for name in transfer_names
        },
        "by_edit_ratio": breakdown(kept, deltas, lambda t: t.edit_set.ratio, arms),
        "by_edit_policy": breakdown(kept, deltas, lambda t: t.edit_set.policy, arms),
        "by_parent_generation": breakdown(kept, deltas, lambda t: t.parent_generation, arms),
    }

    out_dir = args.out_dir or (run.path / "analysis_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "transfer_control.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    csv_path = out_dir / "transfer_control_pairs.csv"
    columns = ",".join(f"delta_{name}" for name in arms)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "trial,parent_hash,parent_score,parent_generation,edit_ratio,"
            f"edit_policy,num_edited,{columns}\n"
        )
        for position, trial in enumerate(kept):
            values = ",".join(f"{deltas[name][position]:.6f}" for name in arms)
            handle.write(
                f"{trial.index},{trial.entry.canonical_hash},{float(trial.entry.score):.6f},"
                f"{trial.parent_generation if trial.parent_generation is not None else ''},"
                f"{trial.edit_set.ratio},{trial.edit_set.policy},"
                f"{int(np.sum(trial.edit_set.edit_mask))},{values}\n"
            )

    # ---- console summary --------------------------------------------------
    print()
    print(f"trials               {len(kept)}")
    for name in arms:
        stats = report["arms"][name]
        print(f"  delta[{name:<10}]   {stats['delta_mean']:+.4f}   (>0 {stats['frac_improved']:.2%})")
    print()
    for name in [HOME_ARM] + transfer_names:
        gap = report["gap_vs_random"][name]
        low, high = gap["gap_ci95"]
        print(f"  {name:<10} - random   {gap['gap_mean']:+.4f}"
              f"   CI95 [{low:+.4f}, {high:+.4f}]   wins {gap['left_wins_frac']:.2%}")
    print()
    for name in transfer_names:
        gap = report["home_minus_transfer"][name]
        low, high = gap["gap_ci95"]
        print(f"  home - {name:<10}  {gap['gap_mean']:+.4f}   CI95 [{low:+.4f}, {high:+.4f}]")
    print()
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
