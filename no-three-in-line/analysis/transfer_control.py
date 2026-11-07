#!/usr/bin/env python3
"""No-three-in-line -- paired transfer control.

A gain over the random operator alone cannot rule out memorisation: the model is
retrained every generation on the very pool the parents are drawn from, so even
a copier would beat the random arm. This script removes that confound by adding
a third arm whose model *never saw this pool*.

    home     : the model trained by the run the parents come from
    transfer : a model from a different run (different seed -> different pool)
    random   : no model at all

All arms complete the *same* parent under the *same* mask. Reading:

    transfer - random  > 0   the operator carries structure a fresh model can
                             apply to constructions it never trained on;
                             memorisation cannot produce this
    home - transfer    > 0   the premium from having trained on this pool
                             (adaptation and/or memorisation)
    transfer ~ random        the gain over random was memorisation

Method.  Every arm runs the production ``PatternBoostGenerator``; only the
decoder differs.  ``_FixedMaskGenerator`` overrides ``_initialize_candidate`` so
that the remasked environment comes from a mask plan drawn once up front instead
of from a fresh rng draw per candidate.  Everything after that -- slot
selection, proposal search, the decode schedule, repair, greedy saturation and
scoring -- is the untouched library code, and the random arm is literally
``completion_operator="random"``, the same path the full-loop random-operator
ablation used.

This is a strictly tighter pairing than replaying candidates.jsonl, which could
only match the mask *distribution* (masks are not recoverable from the log: the
``replaceability`` ranking is decided by 1e-6 tie-break jitter on a nearly
saturated board, so two draws at the same ratio overlap only ~0.4). Here every
arm is handed the identical partial state.

What is *not* shared is the stochasticity downstream of the mask. Each arm
starts from an rng seeded identically, but the streams diverge as soon as the
decoders do: different decoders select different slots and consume different
numbers of draws, so slot selection and saturation are matched in distribution
rather than draw for draw. That is unavoidable between different operators, and
it is why the parent and the mask -- the two things a completion is conditioned
on -- are the ones held fixed.

Cost is one forward pass per decode step per model arm; 4096 parents against two
model arms is about three generations' worth of the original run's inference.

Usage:
    cd no-three-in-line
    python analysis/transfer_control.py \
        --pool-run checkpoints/n25_main_batch4/exp \
        --transfer rt23=checkpoints/n25_main_stage1_rt23/exp/patternboost_gen_64_epoch_1.pt \
        --transfer rt44=checkpoints/n25_main_stage1_rt44/exp/patternboost_gen_64_epoch_1.pt \
        --num-parents 4096

``--home`` defaults to the highest-numbered ``patternboost_gen_*_epoch_1.pt``
inside ``--pool-run``, so the usual invocation only names the transfer donors.
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
from typing import Any, Iterable

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import NoThreeInLineEnv  # noqa: E402
from patternboost.config import load_patternboost_config  # noqa: E402
from patternboost.generator import PatternBoostGenerator, _CandidateState  # noqa: E402
from patternboost.remask import generation_remask  # noqa: E402
from patternboost.state import state_ids_to_env_reset_args  # noqa: E402
from patternboost.top_pool import TopPool  # noqa: E402
from patternboost.types import TopPoolEntry  # noqa: E402
from supervised.config import load_supervised_config  # noqa: E402
from supervised.trainer import build_model_from_project_config  # noqa: E402

RANDOM_ARM = "random"
HOME_ARM = "home"


# --------------------------------------------------------------------------
# run directory contract
# --------------------------------------------------------------------------


class RunDir:
    """Locate the files this analysis needs inside one PatternBoost run dir."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.config = path / "config_snapshot.yaml"
        self.top_pool = path / "top_pool.pt"
        self.metrics = path / "metrics.jsonl"

    def require(self) -> None:
        missing = [p.name for p in (self.config, self.top_pool) if not p.exists()]
        if missing:
            raise SystemExit(
                f"{self.path} is missing required file(s): {', '.join(missing)}\n"
                "This script needs config_snapshot.yaml (the completion protocol) "
                "and top_pool.pt (parent states)."
            )

    def snapshots(self) -> list[Path]:
        """Return per-generation pool snapshots, oldest first, then the final pool."""

        return sorted(self.path.glob("top_pool_generation_*.pt")) + [self.top_pool]

    def latest_checkpoint(self) -> Path:
        """Return the highest-numbered training checkpoint written by this run.

        ``latest.pt`` is a copy of the same file, but it is overwritten in place
        and carries no generation number, so the explicit per-generation file is
        preferred: the report should be able to say which generation the home
        model came from.
        """

        pattern = re.compile(r"patternboost_gen_(\d+)_epoch_\d+\.pt$")
        best: tuple[int, Path] | None = None
        for path in self.path.glob("patternboost_gen_*_epoch_*.pt"):
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
                f"{self.path} holds no model checkpoint. Pass --home explicitly, or "
                "check that the run was launched with training.checkpoint_every_generations set."
            )
        return best[1]


def snapshot_generation(path: Path) -> int | None:
    """Return the generation number encoded in a pool snapshot filename."""

    match = re.search(r"top_pool_generation_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# parents
# --------------------------------------------------------------------------


def load_pool_entries(
    run: RunDir,
    source: str,
    generations: list[int] | None = None,
) -> tuple[dict[str, TopPoolEntry], dict[str, int | None], dict[str, int]]:
    """Return parent entries keyed by canonical hash, plus their provenance.

    ``top_pool.pt`` is rewritten in place each generation, so on its own it
    exposes only the constructions that survived to the end. Runs launched with
    ``top_pool_snapshot_every_generations`` also write ``top_pool_generation_XXX.pt``;
    merging them recovers earlier, typically weaker parents and lets the report
    break the result down by how far into the run a parent came from.

    ``generations`` restricts the union to named snapshots. A handful of
    well-separated generations already spans the range of parent quality, and
    every extra snapshot mostly re-contributes constructions the later ones
    already hold, so this trades little coverage for a much smaller parent set.
    The final generation's snapshot has the same contents as ``top_pool.pt``, so
    naming it is equivalent to including the final pool.

    A canonical hash determines the state and the score is a function of the
    state, so an entry is identical wherever it appears and the union is safe.
    The recorded generation is the *earliest* snapshot an entry appears in.
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
    entries: dict[str, TopPoolEntry] = {}
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
    return entries, origin, provenance


# --------------------------------------------------------------------------
# mask plan
# --------------------------------------------------------------------------


@dataclass
class Trial:
    """One parent plus one mask, shared by every arm."""

    index: int
    entry: TopPoolEntry
    parent_generation: int | None
    remask_mode: str
    mask_ratio: float
    partial_state: np.ndarray
    num_masked: int


def active_mask_ratios(config: Any, generation_number: int) -> list[float]:
    """Return the mask ratios the run used at ``generation_number`` (1-based).

    Mirrors ``PatternBoostRunner._active_mask_ratios`` so the control samples
    from the same distribution the run itself was sampling from.
    """

    active = list(config.mask_ratios)
    schedule = config.schedules.mask_ratios_by_generation
    if schedule is not None:
        for entry in schedule:
            if entry.start <= generation_number:
                active = list(entry.mask_ratios)
    return active


def active_remask_weights(config: Any, generation_number: int) -> list[float] | None:
    """Return the remask-mode weights the run used at ``generation_number``."""

    active = config.generation.remask_mode_weights
    schedule = config.schedules.remask_mode_weights_by_generation
    if schedule is not None:
        for entry in schedule:
            if entry.start <= generation_number:
                active = list(entry.weights)
    return None if active is None else list(active)


def build_plan(
    parents: list[tuple[TopPoolEntry, int | None]],
    config: Any,
    ratios: list[float],
    modes: list[str],
    weights: list[float] | None,
    repeats: int,
    rng: np.random.Generator,
) -> list[Trial]:
    """Draw one mask per (parent, repeat) and freeze it for every arm.

    Drawing the whole plan up front is what makes the arms comparable: the mask
    is a property of the trial, not of the arm, so a difference between arms can
    only come from how the masked slots were filled.
    """

    probabilities = None
    if weights is not None:
        total = float(sum(weights))
        probabilities = [float(weight) / total for weight in weights]

    # A ratio of 0.0 masks nothing, so every arm returns the parent unchanged.
    # Keeping those records would dilute every mean with a block of guaranteed
    # zeros.
    ratios = [ratio for ratio in ratios if ratio > 0.0]
    if not ratios:
        raise SystemExit("every configured mask ratio is 0.0; nothing would be masked")

    trials: list[Trial] = []
    for entry, generation in parents:
        parent_state = np.asarray(entry.slot_state_ids, dtype=np.int64)
        for _ in range(repeats):
            mode = str(rng.choice(modes, p=probabilities))
            ratio = float(rng.choice(ratios))
            partial_state, removed = generation_remask(parent_state, config.n, ratio, mode, rng)
            trials.append(
                Trial(
                    index=len(trials),
                    entry=entry,
                    parent_generation=generation,
                    remask_mode=mode,
                    mask_ratio=ratio,
                    partial_state=partial_state,
                    num_masked=len(removed),
                )
            )
    return trials


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


class _FixedMaskGenerator(PatternBoostGenerator):
    """A generator that consumes a pre-drawn mask instead of sampling one.

    Only ``_initialize_candidate`` changes. Decoding, proposal search, repair,
    saturation and scoring stay on the library path, so an arm here behaves
    exactly like the corresponding arm of a real run.
    """

    def load_plan(self, trials: list[Trial]) -> None:
        self._trials = trials
        self._cursor = 0

    def _initialize_candidate(self, seed_entry: TopPoolEntry) -> _CandidateState:
        trial = self._trials[self._cursor]
        self._cursor += 1
        if trial.entry.canonical_hash != seed_entry.canonical_hash:
            raise RuntimeError(
                "mask plan and seed batch fell out of step; seed entries must be "
                "passed in plan order"
            )
        init_points, init_assignment = state_ids_to_env_reset_args(trial.partial_state)
        env = NoThreeInLineEnv(
            n=self.config.n,
            num_slots=2 * self.config.n,
            proposal_search_config={
                "num_search_attempts": self.config.decode.env_num_search_attempts,
                "allow_existing_deletions": self.config.decode.env_allow_existing_deletions,
            },
        )
        env.reset(init_points=init_points, init_slot_assignment=init_assignment)
        return _CandidateState(
            env=env,
            seed_hash=seed_entry.canonical_hash,
            seed_score=int(seed_entry.score),
            remask_mode=trial.remask_mode,
            mask_ratio=trial.mask_ratio,
        )


def with_random_operator(config: Any) -> Any:
    """Return a copy of ``config`` whose completion operator is the random one."""

    clone = copy.deepcopy(config)
    try:
        clone.generation.completion_operator = RANDOM_ARM
    except Exception:  # frozen dataclass
        object.__setattr__(clone.generation, "completion_operator", RANDOM_ARM)
    return clone


def load_model(
    checkpoint_path: Path,
    project_config_path: Path,
    n: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the architecture from the *pool run's* config and load one checkpoint.

    Building from a single project config for every arm is deliberate: it makes
    a donor whose architecture differs fail loudly on ``strict=True`` instead of
    quietly loading a partially matching model.
    """

    supervised_config = load_supervised_config(project_config_path)
    supervised_config.n = n
    model = build_model_from_project_config(supervised_config)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise SystemExit(f"{checkpoint_path} has no model_state_dict")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    meta = {
        "path": str(checkpoint_path),
        "generation": payload.get("epoch"),
        "stage_name": payload.get("stage_name"),
    }
    match = re.search(r"patternboost_gen_(\d+)_epoch_\d+\.pt$", checkpoint_path.name)
    if match:
        meta["generation"] = int(match.group(1))
    return model, meta


def describe_donor(checkpoint_path: Path) -> dict[str, Any]:
    """Return the donor run's identifying config fields, when they are on disk.

    Reported rather than enforced: a donor with a different ``n`` or a different
    model block would already have failed the strict state-dict load, but the
    seed is worth recording because 'different seed' is the whole premise of the
    transfer arm.
    """

    snapshot = checkpoint_path.parent / "config_snapshot.yaml"
    if not snapshot.exists():
        return {"run_dir": str(checkpoint_path.parent)}
    raw = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    patternboost = raw.get("patternboost", {}) or {}
    runtime = patternboost.get("runtime", {}) or {}
    return {
        "run_dir": str(checkpoint_path.parent),
        "n": raw.get("n"),
        "seed": runtime.get("seed"),
        "num_generations": (patternboost.get("generation", {}) or {}).get("num_generations"),
        "completion_operator": (patternboost.get("generation", {}) or {}).get(
            "completion_operator", "learned"
        ),
    }


def run_arm(
    name: str,
    trials: list[Trial],
    config: Any,
    model: torch.nn.Module | None,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Return the per-trial delta (child score minus parent score) for one arm.

    Every arm starts from a fresh rng seeded identically. The streams diverge
    once the decoders do, so slot selection and saturation are matched in
    distribution, not draw for draw; the parent and the mask are what is held
    fixed. ``batch_size`` also participates in that stream (candidates in a
    batch interleave their decode steps), so it is held fixed across arms.
    """

    generator = _FixedMaskGenerator(
        model,
        config,
        device,
        np.random.default_rng(seed),
    )
    generator.load_plan(trials)

    deltas = np.empty(len(trials), dtype=float)
    started = time.time()
    for start in range(0, len(trials), batch_size):
        chunk = trials[start : start + batch_size]
        results = generator.generate_batch([trial.entry for trial in chunk])
        for offset, result in enumerate(results):
            deltas[start + offset] = float(result.score) - float(result.seed_score)
        done = start + len(chunk)
        if done % (batch_size * 10) == 0 or done == len(trials):
            rate = done / max(time.time() - started, 1e-9)
            print(f"  [{name}] {done}/{len(trials)}  ({rate:.0f} trials/s)", flush=True)
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
        "frac_equal": float(np.mean(delta == 0)) if count else float("nan"),
        "frac_worse": float(np.mean(delta < 0)) if count else float("nan"),
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
        "tie_frac": float(np.mean(diff == 0)) if count else float("nan"),
    }


def breakdown(
    trials: list[Trial],
    deltas: dict[str, np.ndarray],
    key: Any,
    arms: list[str],
) -> dict[str, Any]:
    """Group trials by ``key(trial)`` and report every arm plus its gap to random."""

    buckets: dict[Any, list[int]] = {}
    for trial in trials:
        buckets.setdefault(key(trial), []).append(trial.index)
    report: dict[str, Any] = {}
    for bucket, indices in sorted(buckets.items(), key=lambda item: (item[0] is None, item[0])):
        index = np.asarray(indices, dtype=int)
        entry: dict[str, Any] = {"num_trials": int(index.size)}
        for arm in arms:
            entry[f"delta_{arm}_mean"] = float(deltas[arm][index].mean())
        for arm in arms:
            if arm == RANDOM_ARM:
                continue
            entry[f"gap_{arm}_vs_random"] = paired_gap(
                deltas[arm][index], deltas[RANDOM_ARM][index]
            )["gap_mean"]
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
    # checkpoints/<run>/exp/patternboost_gen_64_epoch_1.pt -> <run>
    parent = path.parent
    name = parent.parent.name if parent.name == "exp" else parent.name
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired transfer control for no-three-in-line.",
    )
    parser.add_argument(
        "--pool-run",
        type=Path,
        required=True,
        help="run directory supplying the parents and the completion protocol "
        "(e.g. checkpoints/n25_main_batch4/exp)",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="checkpoint of the model that trained on this pool "
        "(default: the highest-numbered checkpoint inside --pool-run)",
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
        help="independent mask draws per parent (default 1); total trials = parents x repeats",
    )
    parser.add_argument(
        "--parent-source",
        choices=("union", "final"),
        default="union",
        help="'union' merges pool snapshots (recovers parents evicted before "
        "the end); 'final' uses top_pool.pt only (default: union)",
    )
    parser.add_argument(
        "--parent-generations",
        type=int,
        nargs="*",
        default=None,
        metavar="G",
        help="restrict the union to these snapshot generations, e.g. 32 48 64 "
        "(default: every snapshot). Ignored when --parent-source is 'final'",
    )
    parser.add_argument(
        "--mask-generation",
        type=int,
        default=None,
        help="sample masks from the schedule active at this 1-based generation "
        "(default: the run's last generation)",
    )
    parser.add_argument("--mask-ratios", type=float, nargs="*", default=None)
    parser.add_argument("--remask-modes", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="default: cuda when available")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="candidates per model forward (default 64). The runs themselves used "
        "candidate_batch_size=1; batching only changes throughput and the rng "
        "interleaving, and it is held fixed across arms",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to <pool-run>/analysis_transfer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = RunDir(args.pool_run)
    run.require()

    config = load_patternboost_config(run.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- parents ----------------------------------------------------------
    entries, origin, provenance = load_pool_entries(
        run, args.parent_source, args.parent_generations
    )
    rng = np.random.default_rng(args.seed)
    hashes = sorted(entries)
    take = min(args.num_parents, len(hashes))
    chosen = rng.choice(len(hashes), size=take, replace=False)
    parents = [(entries[hashes[int(i)]], origin[hashes[int(i)]]) for i in chosen]

    # ---- mask plan --------------------------------------------------------
    generation_number = args.mask_generation or int(config.generation.num_generations)
    ratios = args.mask_ratios or active_mask_ratios(config, generation_number)
    modes = args.remask_modes or list(config.generation.remask_modes)
    weights = None if args.remask_modes else active_remask_weights(config, generation_number)
    if args.remask_modes is not None and weights is not None and len(weights) != len(modes):
        weights = None
    trials = build_plan(parents, config, ratios, modes, weights, args.repeats, rng)

    # ---- arms -------------------------------------------------------------
    home_path = args.home or run.latest_checkpoint()
    donors = [parse_donor(value) for value in args.transfer]
    if not donors:
        raise SystemExit(
            "no --transfer donor given; the transfer arm is the point of this script"
        )

    batch_size = max(1, int(args.batch_size))
    arms = [RANDOM_ARM, HOME_ARM] + [name for name, _ in donors]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"duplicate arm name in {arms}")

    print(f"pool run       {run.path}")
    print(f"parent pools   {', '.join(f'{name}(+{added})' for name, added in provenance.items())}")
    print(f"parents        {take} of {len(entries)} (source={args.parent_source}) "
          f"x {args.repeats} mask draw(s) = {len(trials)} trials")
    print(f"mask           ratios={ratios} modes={modes} weights={weights} "
          f"(schedule at generation {generation_number})")
    print(f"device         {device}   batch_size={batch_size}")
    print(f"arms           {', '.join(arms)}")
    print()

    deltas: dict[str, np.ndarray] = {}
    model_meta: dict[str, Any] = {}

    print(f"[{RANDOM_ARM}] no model")
    deltas[RANDOM_ARM] = run_arm(
        RANDOM_ARM, trials, with_random_operator(config), None, device, args.seed, batch_size
    )

    for name, checkpoint_path in [(HOME_ARM, home_path)] + donors:
        if not Path(checkpoint_path).exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        model, meta = load_model(Path(checkpoint_path), run.config, config.n, device)
        meta["donor"] = describe_donor(Path(checkpoint_path))
        model_meta[name] = meta
        print(f"[{name}] {checkpoint_path}  (donor seed={meta['donor'].get('seed')})")
        deltas[name] = run_arm(name, trials, config, model, device, args.seed, batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- statistics -------------------------------------------------------
    transfer_names = [name for name, _ in donors]
    report: dict[str, Any] = {
        "pool_run": str(run.path),
        "args": {
            "num_parents": take,
            "repeats": args.repeats,
            "parent_source": args.parent_source,
            "parent_generations": args.parent_generations,
            "mask_generation": generation_number,
            "mask_ratios": ratios,
            "remask_modes": modes,
            "remask_mode_weights": weights,
            "seed": args.seed,
            "device": str(device),
            "batch_size": batch_size,
        },
        "pool_snapshots": provenance,
        "pool_entries_available": len(entries),
        "models": model_meta,
        "num_trials": len(trials),
        "arms": {name: arm_stats(deltas[name]) for name in arms},
        "gap_vs_random": {
            name: paired_gap(deltas[name], deltas[RANDOM_ARM])
            for name in [HOME_ARM] + transfer_names
        },
        "home_minus_transfer": {
            name: paired_gap(deltas[HOME_ARM], deltas[name]) for name in transfer_names
        },
        "by_mask_ratio": breakdown(trials, deltas, lambda t: t.mask_ratio, arms),
        "by_remask_mode": breakdown(trials, deltas, lambda t: t.remask_mode, arms),
        "by_parent_score": breakdown(trials, deltas, lambda t: int(t.entry.score), arms),
        "by_parent_generation": breakdown(trials, deltas, lambda t: t.parent_generation, arms),
    }

    out_dir = args.out_dir or (run.path / "analysis_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "transfer_control.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    csv_path = out_dir / "transfer_control_pairs.csv"
    columns = ",".join(f"delta_{name}" for name in arms)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "trial,parent_hash,parent_score,parent_generation,mask_ratio,"
            f"remask_mode,num_masked,{columns}\n"
        )
        for trial in trials:
            values = ",".join(f"{deltas[name][trial.index]:.1f}" for name in arms)
            handle.write(
                f"{trial.index},{trial.entry.canonical_hash},{int(trial.entry.score)},"
                f"{trial.parent_generation if trial.parent_generation is not None else ''},"
                f"{trial.mask_ratio},{trial.remask_mode},{trial.num_masked},{values}\n"
            )

    # ---- console summary --------------------------------------------------
    print()
    print(f"trials               {len(trials)}")
    for name in arms:
        stats = report["arms"][name]
        print(
            f"  delta[{name:<10}]   {stats['delta_mean']:+.4f}"
            f"   (>0 {stats['frac_improved']:.2%}, =0 {stats['frac_equal']:.2%})"
        )
    print()
    for name in [HOME_ARM] + transfer_names:
        gap = report["gap_vs_random"][name]
        low, high = gap["gap_ci95"]
        print(
            f"  {name:<10} - random   {gap['gap_mean']:+.4f}"
            f"   CI95 [{low:+.4f}, {high:+.4f}]   wins {gap['left_wins_frac']:.2%}"
        )
    print()
    for name in transfer_names:
        gap = report["home_minus_transfer"][name]
        low, high = gap["gap_ci95"]
        print(
            f"  home - {name:<10}  {gap['gap_mean']:+.4f}   CI95 [{low:+.4f}, {high:+.4f}]"
        )
    print()
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
