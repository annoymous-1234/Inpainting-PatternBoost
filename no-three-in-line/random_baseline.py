from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from env import NoThreeInLineEnv
from geometry import Point
from patternboost.state import points_to_canonical_state_ids
from patternboost.top_pool import TopPool
from proposal_eval import CandidatePoint, ProposalSlot, compute_existing_legal_rows_by_column


@dataclass(frozen=True)
class RandomBaselineConfig:
    """Configure a random greedy no-three-in-line baseline experiment.

    ``n`` is the board side length, ``trials`` is the number of independent
    random greedy runs, ``seed`` is the base seed used to derive per-trial
    seeds, and ``output_dir``/``run_name`` choose where artifacts are written.
    ``log_steps`` enables per-move JSONL records, ``save_points`` controls
    whether final point sets and placement orders are stored, and ``max_steps``
    optionally caps each trial for debugging. Instances do not mutate external
    state and assume the caller passes positive ``n`` and ``trials``.
    """

    n: int
    trials: int = 100
    seed: int = 0
    output_dir: Path = Path("checkpoints/random_baseline")
    run_name: str | None = None
    log_steps: bool = False
    save_points: bool = True
    max_steps: int | None = None


@dataclass(frozen=True)
class RandomTopPoolBaselineConfig:
    """Configure a PatternBoost-style random TopPool baseline experiment.

    ``n`` is the board side length, ``top_pool_size`` is the maximum number of
    retained pool entries, ``candidates_per_generation`` is the number of
    independent random greedy boards generated each generation, and
    ``num_generations`` is the number of repeated generate-insert rounds.
    ``seed`` derives deterministic per-candidate seeds, output fields select
    artifact locations, and logging flags control point/step payload size. The
    dataclass does not mutate state and assumes positive counts.
    """

    n: int
    top_pool_size: int
    candidates_per_generation: int
    num_generations: int
    seed: int = 0
    output_dir: Path = Path("checkpoints/random_baseline")
    run_name: str | None = None
    log_steps: bool = False
    save_points: bool = True
    max_steps: int | None = None


@dataclass
class RandomBaselineTrialResult:
    """Store the outcome and diagnostics for one random baseline trial.

    ``trial_index`` and ``seed`` identify the run, ``n`` is the board side
    length, ``final_point_count``/``num_steps`` summarize the final board, and
    ``termination_reason`` records why placement stopped. ``slot_state_ids`` is
    an ``int64`` NumPy array of shape ``[2n]`` on CPU with ``0`` for empty slots
    and ``y + 1`` for filled rows. ``selected_points`` is the sorted final point
    set, ``placement_order`` is the accepted move sequence, and ``step_records``
    contains optional per-move diagnostics. The dataclass carries data only and
    does not mutate external state.
    """

    trial_index: int
    seed: int
    n: int
    final_point_count: int
    num_steps: int
    termination_reason: str
    remaining_legal_count: int
    slot_state_ids: np.ndarray
    selected_points: list[Point]
    placement_order: list[Point]
    elapsed_seconds: float
    step_records: list[dict[str, Any]]


def _validate_config(config: RandomBaselineConfig) -> None:
    """Validate random baseline configuration before running trials.

    ``config`` is the experiment configuration to check. The function returns
    ``None``, mutates no state, and raises ``ValueError`` when counts or bounds
    are invalid.
    """

    if config.n <= 0:
        raise ValueError("n must be positive")
    if config.trials <= 0:
        raise ValueError("trials must be positive")
    if config.max_steps is not None and config.max_steps < 0:
        raise ValueError("max_steps must be non-negative when provided")


def _validate_top_pool_config(config: RandomTopPoolBaselineConfig) -> None:
    """Validate random TopPool baseline configuration before running.

    ``config`` is the PatternBoost-style random baseline configuration. The
    function returns ``None``, mutates no state, and raises ``ValueError`` when
    dimensions, pool size, generation counts, or optional step bounds are
    invalid.
    """

    if config.n <= 0:
        raise ValueError("n must be positive")
    if config.top_pool_size <= 0:
        raise ValueError("top_pool_size must be positive")
    if config.candidates_per_generation <= 0:
        raise ValueError("candidates_per_generation must be positive")
    if config.num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if config.max_steps is not None and config.max_steps < 0:
        raise ValueError("max_steps must be non-negative when provided")


def _trial_seed(base_seed: int, trial_index: int) -> int:
    """Return the deterministic seed assigned to one trial.

    ``base_seed`` is the experiment seed and ``trial_index`` is the zero-based
    trial number. The function returns an integer seed, does not mutate state,
    and assumes adjacent seeds are sufficient for independent baseline runs.
    """

    return int(base_seed) + int(trial_index)


def _candidate_seed(base_seed: int, generation_index: int, candidate_index: int) -> int:
    """Return a deterministic seed for one generated TopPool candidate.

    ``base_seed`` is the experiment seed, ``generation_index`` is zero-based,
    and ``candidate_index`` is zero-based within that generation. The function
    returns an integer seed, does not mutate state, and assumes the multiplier
    leaves enough room for ordinary candidate counts.
    """

    return int(base_seed) + 1_000_003 * int(generation_index) + int(candidate_index)


def _legal_points(env: NoThreeInLineEnv) -> list[Point]:
    """Return all currently legal empty lattice points for the environment.

    ``env`` supplies the board size and active-line backend. The helper builds
    per-column NumPy masks with shape ``[n]`` and dtype ``bool`` on CPU, where
    ``True`` means the row is empty and does not lie on a two-point active
    line. It returns ``(x, y)`` points, does not mutate ``env``, and assumes the
    backend represents a legal persistent point set.
    """

    legal_rows_by_column = compute_existing_legal_rows_by_column(
        existing_points=env.backend.get_selected_points(),
        n=env.n,
        candidate_columns=list(range(env.n)),
        backend=env.backend,
    )
    legal_points: list[Point] = []
    for column in range(env.n):
        legal_rows = np.flatnonzero(legal_rows_by_column[column]).tolist()
        legal_points.extend((column, int(row)) for row in legal_rows)
    return legal_points


def _state_ids_from_points(points: list[Point], n: int) -> np.ndarray:
    """Encode a point set in the canonical PatternBoost slot format.

    ``points`` is a legal board state and ``n`` is the board side length. The
    returned NumPy array has shape ``[2n]`` and dtype ``int64`` on CPU, where
    each column owns two slots, ``0`` means empty, and nonzero values store
    ``y + 1``. The helper does not mutate inputs and assumes no column contains
    more than two selected rows.
    """

    return points_to_canonical_state_ids(points, n)


def _masked_slot_for_column(env: NoThreeInLineEnv, column: int) -> int | None:
    """Return an open column-pair slot for ``column`` when one exists.

    ``env`` is the mutable environment and ``column`` is the x-coordinate of a
    legal candidate point. The function returns a masked slot id or ``None``,
    does not mutate state, and assumes the environment uses the project's
    two-slots-per-column convention.
    """

    for slot_id in (2 * int(column), 2 * int(column) + 1):
        slot_state = env.slot_states.get(slot_id)
        if slot_state is not None and slot_state.status == "MASKED":
            return slot_id
    return None


def run_random_baseline_trial(
    config: RandomBaselineConfig,
    trial_index: int,
) -> RandomBaselineTrialResult:
    """Run one random legal-fill baseline trial from an empty board.

    ``config`` provides board size and optional step logging controls, and
    ``trial_index`` selects the deterministic per-trial seed. The function
    returns a trial result, mutates only its local environment, and assumes
    each sampled point is chosen uniformly from the currently legal cells.
    """

    _validate_config(config)
    seed = _trial_seed(config.seed, trial_index)
    rng = random.Random(seed)
    env = NoThreeInLineEnv(n=config.n, num_slots=2 * config.n, max_steps=config.max_steps)
    env.reset()

    started_at = time.perf_counter()
    placement_order: list[Point] = []
    step_records: list[dict[str, Any]] = []
    termination_reason = "saturated"
    remaining_legal_count = 0

    while True:
        legal_points = _legal_points(env)
        remaining_legal_count = len(legal_points)
        if not legal_points:
            termination_reason = (
                "board_full"
                if int(env.observe()["num_selected"]) == config.n * config.n
                else "saturated"
            )
            break
        if config.max_steps is not None and env.timestep >= config.max_steps:
            termination_reason = "max_steps"
            break

        point = legal_points[rng.randrange(len(legal_points))]
        slot_id = _masked_slot_for_column(env, point[0])
        if slot_id is None:
            termination_reason = "slots_exhausted"
            break

        obs, reward, done, info = env.step(
            [
                ProposalSlot(
                    slot_id=slot_id,
                    candidates=[
                        CandidatePoint(
                            x=point[0],
                            y=point[1],
                            score=1.0,
                            slot_id=slot_id,
                        )
                    ],
                )
            ]
        )

        accepted_points = list(info["accepted_proposal_points"])
        if accepted_points:
            placement_order.extend(accepted_points)
        step_records.append(
            {
                "trial_index": trial_index,
                "seed": seed,
                "step_index": env.timestep,
                "slot_id": slot_id,
                "point": point,
                "legal_count_before": remaining_legal_count,
                "selected_count_after": int(obs["num_selected"]),
                "active_line_count_after": len(env.backend.line_to_points),
                "reward": float(reward),
                "accepted_points": accepted_points,
                "rejected_points": list(info["rejected_proposal_points"]),
                "num_conflict_lines": int(info["num_conflict_lines"]),
            }
        )

        if accepted_points != [point]:
            termination_reason = "conflict"
            remaining_legal_count = len(_legal_points(env))
            break
        if done:
            termination_reason = "max_steps"
            remaining_legal_count = len(_legal_points(env))
            break

    elapsed_seconds = time.perf_counter() - started_at
    selected_points = sorted(env.backend.get_selected_points())
    slot_state_ids = _state_ids_from_points(selected_points, config.n)
    return RandomBaselineTrialResult(
        trial_index=trial_index,
        seed=seed,
        n=config.n,
        final_point_count=len(selected_points),
        num_steps=env.timestep,
        termination_reason=termination_reason,
        remaining_legal_count=remaining_legal_count,
        slot_state_ids=slot_state_ids,
        selected_points=selected_points,
        placement_order=placement_order,
        elapsed_seconds=elapsed_seconds,
        step_records=step_records if config.log_steps else [],
    )


def _json_safe(value: Any) -> Any:
    """Convert experiment diagnostics into JSON-serializable values.

    ``value`` may contain dataclasses, paths, NumPy scalars or arrays, tuples,
    lists, dictionaries, or primitive values. The function returns a converted
    value, does not mutate the input, and assumes arrays are small diagnostics.
    """

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object to disk with stable formatting.

    ``path`` is the destination and ``payload`` is converted to JSON-safe
    values before serialization. The function creates parent directories,
    mutates only the destination file, and assumes the payload is finite.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL artifact.

    ``path`` is the destination and ``record`` is converted to JSON-safe values
    before serialization. The function creates parent directories, mutates only
    that file, and assumes records are independent lines.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True) + "\n")


def _append_jsonl_many(path: Path, records: list[dict[str, Any]]) -> None:
    """Append multiple JSON records to a JSONL artifact in one file write.

    ``path`` is the destination and ``records`` are converted to JSON-safe
    values before serialization. The function creates parent directories,
    mutates only that file, and assumes records are independent JSONL rows.
    Empty record lists are ignored.
    """

    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True) + "\n"
        for record in records
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)


def _run_record(
    result: RandomBaselineTrialResult,
    *,
    save_points: bool,
) -> dict[str, Any]:
    """Return the JSONL record for one completed trial.

    ``result`` is the completed trial payload and ``save_points`` controls
    whether point lists are included. The function returns a new dictionary,
    does not mutate inputs, and assumes step-level records are written
    separately when requested.
    """

    record: dict[str, Any] = {
        "event": "trial",
        "trial_index": result.trial_index,
        "seed": result.seed,
        "n": result.n,
        "final_point_count": result.final_point_count,
        "num_steps": result.num_steps,
        "termination_reason": result.termination_reason,
        "remaining_legal_count": result.remaining_legal_count,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if save_points:
        record["slot_state_ids"] = result.slot_state_ids
        record["selected_points"] = result.selected_points
        record["placement_order"] = result.placement_order
    return record


def _score_stats(scores: list[int]) -> dict[str, float | int]:
    """Return compact descriptive statistics for final point counts.

    ``scores`` contains one final point count per trial. The function returns
    min, max, mean, and population standard deviation, mutates no state, and
    assumes ``scores`` is non-empty.
    """

    return {
        "min": min(scores),
        "max": max(scores),
        "mean": float(fmean(scores)),
        "pstdev": float(pstdev(scores)) if len(scores) > 1 else 0.0,
    }


def _counter_from_scores(scores: list[int]) -> Counter[int]:
    """Return a score histogram for JSONL metrics.

    ``scores`` is a list of integer point counts. The function returns a
    ``Counter`` keyed by score, does not mutate inputs, and assumes score
    histograms should keep integer ordering until JSON serialization.
    """

    return Counter(int(score) for score in scores)


def _top_pool_score_counts(pool: TopPool) -> Counter[int]:
    """Return the retained TopPool score distribution.

    ``pool`` is the mutable high-score pool to inspect. The function returns a
    score ``Counter``, does not mutate the pool, and assumes pool entries have
    integer ``score`` fields.
    """

    return Counter(int(entry.score) for entry in pool.entries())


def _format_counter(counter: Counter[int] | dict[Any, Any]) -> str:
    """Return a compact stable histogram string for console output.

    ``counter`` maps scores or labels to counts. The function returns a string,
    mutates no state, and assumes keys can be sorted after their string
    representation is considered for mixed-key dictionaries.
    """

    if not counter:
        return "{}"
    keys = sorted(counter, key=lambda key: (str(type(key)), key))
    return "{" + ", ".join(f"{key}:{counter[key]}" for key in keys) + "}"


def _mean_int(values: list[int]) -> float:
    """Return the arithmetic mean of integer diagnostics.

    ``values`` is a possibly empty list of integer counts. The function returns
    a float, does not mutate inputs, and uses ``0.0`` when no values exist.
    """

    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _random_top_pool_run_start_record(
    config: RandomTopPoolBaselineConfig,
    *,
    run_id: str,
    run_dir: Path,
    top_pool_path: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the PatternBoost-style run-start metrics record.

    ``config`` describes the run and the path arguments identify output
    artifacts. The function returns a new JSON-friendly dictionary, does not
    mutate state, and assumes the TopPool starts empty for this random
    baseline.
    """

    return {
        "event": "run_start",
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "run_dir": run_dir,
        "top_pool_path": top_pool_path,
        "n": config.n,
        "seed": config.seed,
        "num_generations": config.num_generations,
        "candidates_per_generation": config.candidates_per_generation,
        "pool_max_size": config.top_pool_size,
        "initial_best_score": 0,
        "initial_pool_size": 0,
        "initial_score_counts": Counter(),
        "baseline": "random_greedy_top_pool",
    }


def _random_top_pool_candidate_record(
    result: RandomBaselineTrialResult,
    *,
    run_id: str,
    generation_index: int,
    candidate_index: int,
    inserted: bool,
    save_points: bool,
) -> dict[str, Any]:
    """Build one candidate JSONL record for the random TopPool baseline.

    ``result`` is the generated greedy board, ``run_id`` identifies the
    experiment, generation/candidate indices are zero-based, ``inserted`` says
    whether TopPool retained the candidate, and ``save_points`` controls point
    payloads. The function returns a new dictionary and does not mutate the
    result or TopPool.
    """

    record = {
        "event": "candidate",
        "run_id": run_id,
        "generation": generation_index + 1,
        "candidate_index": candidate_index,
        "score": result.final_point_count,
        "inserted": bool(inserted),
        "seed": result.seed,
        "source": "random_greedy",
        "termination_reason": result.termination_reason,
        "remaining_legal_count": result.remaining_legal_count,
        "num_decode_steps": result.num_steps,
        "selected_total": result.num_steps,
        "accepted_total": result.final_point_count,
        "rejected_total": result.num_steps - result.final_point_count,
        "point_gain_total": result.final_point_count,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if save_points:
        record["selected_points"] = result.selected_points
        record["placement_order"] = result.placement_order
    if result.step_records:
        record["step_records"] = result.step_records
    return record


def _random_top_pool_generation_record(
    *,
    config: RandomTopPoolBaselineConfig,
    run_id: str,
    generation_index: int,
    top_pool: TopPool,
    top_pool_path: Path,
    best_before: int,
    pool_size_before: int,
    pool_score_counts_before: Counter[int],
    candidate_results: list[RandomBaselineTrialResult],
    inserted_results: list[RandomBaselineTrialResult],
) -> dict[str, Any]:
    """Build one PatternBoost-style generation metrics record.

    ``config`` describes the experiment, ``run_id`` identifies the run,
    ``generation_index`` is zero-based, and the pool/before/result arguments
    contain the generation state transition. The function returns a metrics
    dictionary, does not mutate state, and assumes insertions have already been
    applied to ``top_pool``.
    """

    best = top_pool.best()
    best_after = 0 if best is None else int(best.score)
    step_counts = [int(result.num_steps) for result in candidate_results]
    accepted_counts = [int(result.final_point_count) for result in candidate_results]
    rejected_counts = [
        int(result.num_steps - result.final_point_count)
        for result in candidate_results
    ]
    generated_scores = [int(result.final_point_count) for result in candidate_results]
    inserted_scores = [int(result.final_point_count) for result in inserted_results]
    return {
        "event": "generation",
        "run_id": run_id,
        "generation": generation_index + 1,
        "num_generations": config.num_generations,
        "best_before": best_before,
        "best_after": best_after,
        "best_delta": best_after - best_before,
        "pool_size_before": pool_size_before,
        "pool_size_after": len(top_pool),
        "pool_score_counts_before": pool_score_counts_before,
        "pool_score_counts_after": _top_pool_score_counts(top_pool),
        "inserted_count": len(inserted_results),
        "generated_count": len(candidate_results),
        "generated_score_counts": _counter_from_scores(generated_scores),
        "inserted_score_counts": _counter_from_scores(inserted_scores),
        "decode_steps_mean": _mean_int(step_counts),
        "selected_mean": _mean_int(step_counts),
        "accepted_mean": _mean_int(accepted_counts),
        "rejected_mean": _mean_int(rejected_counts),
        "point_gain_mean": _mean_int(accepted_counts),
        "training_metrics": {},
        "checkpoint_path": None,
        "top_pool_path": top_pool_path,
        "baseline": "random_greedy_top_pool",
    }


def _print_top_pool_generation_summary(summary: dict[str, Any]) -> None:
    """Print one PatternBoost-style random baseline generation summary.

    ``summary`` is a generation metrics dictionary. The function writes to
    standard output, mutates no experiment state, and assumes histogram fields
    use ``Counter``-like mappings.
    """

    print(
        "random baseline | "
        f"generation {summary['generation']}/{summary['num_generations']} | "
        f"best {summary['best_before']}->{summary['best_after']} | "
        f"pool_entries {summary['pool_size_before']}->{summary['pool_size_after']} | "
        f"inserted {summary['inserted_count']}/{summary['generated_count']} | "
        f"generated_scores {_format_counter(summary['generated_score_counts'])} | "
        f"inserted_scores {_format_counter(summary['inserted_score_counts'])}",
        flush=True,
    )
    print(
        "random baseline | "
        f"generation {summary['generation']}/{summary['num_generations']} detail | "
        f"pool_scores {_format_counter(summary['pool_score_counts_after'])} | "
        f"decode_steps_mean {summary['decode_steps_mean']:.2f} | "
        f"accepted_mean {summary['accepted_mean']:.2f}",
        flush=True,
    )


def _random_top_pool_run_complete_record(
    *,
    config: RandomTopPoolBaselineConfig,
    run_id: str,
    run_dir: Path,
    top_pool: TopPool,
    top_pool_path: Path,
    generation_results: list[RandomBaselineTrialResult],
    inserted_count: int,
) -> dict[str, Any]:
    """Build the final PatternBoost-style run-complete metrics record.

    ``config`` and path arguments describe the experiment, ``top_pool`` is the
    final retained pool, ``generation_results`` are all generated candidates,
    and ``inserted_count`` counts retained candidate insertions. The function
    returns a metrics dictionary, does not mutate state, and assumes TopPool has
    already been saved.
    """

    best = top_pool.best()
    return {
        "event": "run_complete",
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": run_dir,
        "initial_best_score": 0,
        "best_score": 0 if best is None else int(best.score),
        "num_pool_entries": len(top_pool),
        "num_generation_results": len(generation_results),
        "num_inserted_results": int(inserted_count),
        "final_score_counts": _top_pool_score_counts(top_pool),
        "top_pool_path": top_pool_path,
        "baseline": "random_greedy_top_pool",
    }


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a short human-readable random baseline report.

    ``path`` is the markdown destination and ``summary`` is the JSON-friendly
    experiment summary. The function writes one file, mutates no other state,
    and assumes the summary contains the standard score and path fields.
    """

    score_stats = summary["score_stats"]
    best = summary["best_trial"]
    lines = [
        "# Random Baseline Report",
        "",
        f"- n: {summary['config']['n']}",
        f"- trials: {summary['config']['trials']}",
        f"- base seed: {summary['config']['seed']}",
        f"- mean final points: {score_stats['mean']:.3f}",
        f"- min/max final points: {score_stats['min']} / {score_stats['max']}",
        f"- population std: {score_stats['pstdev']:.3f}",
        f"- best trial: index={best['trial_index']} seed={best['seed']} score={best['final_point_count']}",
        f"- runs jsonl: `{summary['runs_jsonl_path']}`",
    ]
    if summary.get("steps_jsonl_path") is not None:
        lines.append(f"- steps jsonl: `{summary['steps_jsonl_path']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_top_pool_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a short report for the random TopPool baseline.

    ``path`` is the markdown destination and ``summary`` is the aggregate
    experiment summary. The function writes one file, mutates no other state,
    and assumes generation summaries include score histograms and best scores.
    """

    lines = [
        "# Random TopPool Baseline Report",
        "",
        f"- n: {summary['config']['n']}",
        f"- TopPool size: {summary['config']['top_pool_size']}",
        f"- generations: {summary['config']['num_generations']}",
        f"- candidates per generation: {summary['config']['candidates_per_generation']}",
        f"- base seed: {summary['config']['seed']}",
        f"- final best score: {summary['best_score']}",
        f"- final pool entries: {summary['num_pool_entries']}",
        f"- inserted candidates: {summary['num_inserted_results']}/{summary['num_generation_results']}",
        f"- metrics jsonl: `{summary['metrics_log_path']}`",
        f"- candidates jsonl: `{summary['candidates_log_path']}`",
        f"- top pool: `{summary['top_pool_path']}`",
        "",
        "## Generations",
        "",
        "| Generation | Best | Pool Entries | Inserted | Generated Scores | Pool Scores |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for record in summary["generation_records"]:
        lines.append(
            f"| {record['generation']} | {record['best_after']} | "
            f"{record['pool_size_after']} | {record['inserted_count']}/{record['generated_count']} | "
            f"{record['generated_score_counts']} | {record['pool_score_counts_after']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_random_baseline_experiment(config: RandomBaselineConfig) -> dict[str, Any]:
    """Run multiple random baseline trials and write experiment artifacts.

    ``config`` controls board size, trial count, random seed, and output
    options. The function returns the aggregate summary, writes JSON/JSONL and
    markdown artifacts under the run directory, and assumes independent trials
    are represented by deterministic seed offsets.
    """

    _validate_config(config)
    run_name = config.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / run_name
    runs_jsonl_path = run_dir / "runs.jsonl"
    steps_jsonl_path = run_dir / "steps.jsonl" if config.log_steps else None
    summary_json_path = run_dir / "summary.json"
    report_path = run_dir / "random_baseline_report.md"

    run_dir.mkdir(parents=True, exist_ok=True)
    if runs_jsonl_path.exists():
        runs_jsonl_path.unlink()
    if steps_jsonl_path is not None and steps_jsonl_path.exists():
        steps_jsonl_path.unlink()

    started_at = time.perf_counter()
    results: list[RandomBaselineTrialResult] = []
    for trial_index in range(config.trials):
        result = run_random_baseline_trial(config, trial_index)
        results.append(result)
        _append_jsonl(runs_jsonl_path, _run_record(result, save_points=config.save_points))
        if steps_jsonl_path is not None:
            for record in result.step_records:
                _append_jsonl(steps_jsonl_path, {"event": "step", **record})

    elapsed_seconds = time.perf_counter() - started_at
    scores = [result.final_point_count for result in results]
    best_result = max(results, key=lambda item: (item.final_point_count, -item.trial_index))
    worst_result = min(results, key=lambda item: (item.final_point_count, item.trial_index))
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": run_dir,
        "runs_jsonl_path": runs_jsonl_path,
        "steps_jsonl_path": steps_jsonl_path,
        "summary_json_path": summary_json_path,
        "report_path": report_path,
        "elapsed_seconds": elapsed_seconds,
        "config": asdict(config),
        "score_stats": _score_stats(scores),
        "score_histogram": dict(sorted(Counter(scores).items())),
        "termination_counts": dict(sorted(Counter(result.termination_reason for result in results).items())),
        "best_trial": _run_record(best_result, save_points=config.save_points),
        "worst_trial": _run_record(worst_result, save_points=config.save_points),
    }

    _write_json(summary_json_path, summary)
    _write_markdown_report(report_path, _json_safe(summary))
    return _json_safe(summary)


def run_random_top_pool_baseline_experiment(config: RandomTopPoolBaselineConfig) -> dict[str, Any]:
    """Run a PatternBoost-style random greedy TopPool baseline.

    ``config`` controls the board size, TopPool size, number of generations,
    candidates per generation, seed, and output options. The function writes
    PatternBoost-like ``metrics.jsonl``, ``candidates.jsonl``, ``top_pool.pt``,
    ``summary.json``, and markdown artifacts, returns the aggregate summary,
    and mutates only files under the selected run directory.
    """

    _validate_top_pool_config(config)
    run_name = config.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / run_name
    run_id = run_dir.name
    metrics_log_path = run_dir / "metrics.jsonl"
    candidates_log_path = run_dir / "candidates.jsonl"
    top_pool_path = run_dir / "top_pool.pt"
    summary_json_path = run_dir / "summary.json"
    report_path = run_dir / "random_top_pool_baseline_report.md"

    run_dir.mkdir(parents=True, exist_ok=True)
    for path in (metrics_log_path, candidates_log_path):
        if path.exists():
            path.unlink()

    top_pool = TopPool(n=config.n, max_size=config.top_pool_size)
    created_at_utc = datetime.now(timezone.utc).isoformat()
    top_pool.save(top_pool_path)
    _append_jsonl(
        metrics_log_path,
        _random_top_pool_run_start_record(
            config,
            run_id=run_id,
            run_dir=run_dir,
            top_pool_path=top_pool_path,
            created_at_utc=created_at_utc,
        ),
    )

    started_at = time.perf_counter()
    generation_records: list[dict[str, Any]] = []
    generation_results: list[RandomBaselineTrialResult] = []
    total_inserted_count = 0

    for generation_index in range(config.num_generations):
        best = top_pool.best()
        best_before = 0 if best is None else int(best.score)
        pool_size_before = len(top_pool)
        pool_score_counts_before = _top_pool_score_counts(top_pool)
        candidate_results: list[RandomBaselineTrialResult] = []
        inserted_results: list[RandomBaselineTrialResult] = []
        candidate_log_records: list[dict[str, Any]] = []

        for candidate_index in range(config.candidates_per_generation):
            candidate_config = RandomBaselineConfig(
                n=config.n,
                trials=1,
                seed=_candidate_seed(config.seed, generation_index, candidate_index),
                output_dir=config.output_dir,
                run_name=config.run_name,
                log_steps=config.log_steps,
                save_points=config.save_points,
                max_steps=config.max_steps,
            )
            result = run_random_baseline_trial(candidate_config, trial_index=0)
            inserted = top_pool.insert(
                result.slot_state_ids,
                metadata={
                    "source": "random_greedy",
                    "generation": generation_index + 1,
                    "candidate_index": candidate_index,
                    "seed": result.seed,
                    "termination_reason": result.termination_reason,
                },
            )
            if inserted:
                inserted_results.append(result)
                total_inserted_count += 1
            candidate_results.append(result)
            generation_results.append(result)
            candidate_log_records.append(
                _random_top_pool_candidate_record(
                    result,
                    run_id=run_id,
                    generation_index=generation_index,
                    candidate_index=candidate_index,
                    inserted=inserted,
                    save_points=config.save_points,
                ),
            )
            if len(candidate_log_records) >= 1024:
                _append_jsonl_many(candidates_log_path, candidate_log_records)
                candidate_log_records = []

        _append_jsonl_many(candidates_log_path, candidate_log_records)

        top_pool.save(top_pool_path)
        generation_record = _random_top_pool_generation_record(
            config=config,
            run_id=run_id,
            generation_index=generation_index,
            top_pool=top_pool,
            top_pool_path=top_pool_path,
            best_before=best_before,
            pool_size_before=pool_size_before,
            pool_score_counts_before=pool_score_counts_before,
            candidate_results=candidate_results,
            inserted_results=inserted_results,
        )
        generation_records.append(_json_safe(generation_record))
        _append_jsonl(metrics_log_path, generation_record)
        _print_top_pool_generation_summary(generation_record)

    top_pool.save(top_pool_path)
    run_complete_record = _random_top_pool_run_complete_record(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        top_pool=top_pool,
        top_pool_path=top_pool_path,
        generation_results=generation_results,
        inserted_count=total_inserted_count,
    )
    _append_jsonl(metrics_log_path, run_complete_record)

    elapsed_seconds = time.perf_counter() - started_at
    best = top_pool.best()
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": run_dir,
        "metrics_log_path": metrics_log_path,
        "candidates_log_path": candidates_log_path,
        "top_pool_path": top_pool_path,
        "summary_json_path": summary_json_path,
        "report_path": report_path,
        "elapsed_seconds": elapsed_seconds,
        "config": asdict(config),
        "best_score": 0 if best is None else int(best.score),
        "num_pool_entries": len(top_pool),
        "num_generation_results": len(generation_results),
        "num_inserted_results": total_inserted_count,
        "final_score_counts": _top_pool_score_counts(top_pool),
        "generation_records": generation_records,
    }

    _write_json(summary_json_path, summary)
    _write_top_pool_markdown_report(report_path, _json_safe(summary))
    return _json_safe(summary)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the random baseline entrypoint.

    The returned namespace contains mode, board, trial or TopPool generation
    options, seed, and logging controls. The function reads process arguments
    only, does not mutate experiment state, and assumes validation happens when
    the selected config is constructed.
    """

    parser = argparse.ArgumentParser(
        description="Run a random greedy no-three-in-line baseline from empty boards."
    )
    parser.add_argument(
        "--mode",
        choices=("trials", "toppool"),
        default="trials",
        help="Run independent trials or a PatternBoost-style TopPool simulation.",
    )
    parser.add_argument("--n", type=int, default=15, help="Board side length.")
    parser.add_argument("--trials", type=int, default=100, help="Number of random trials.")
    parser.add_argument(
        "--top-pool-size",
        type=int,
        default=256,
        help="Maximum retained entries for --mode toppool.",
    )
    parser.add_argument(
        "--candidates-per-generation",
        type=int,
        default=100,
        help="Random greedy candidates generated per generation for --mode toppool.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="Number of generate-insert rounds for --mode toppool.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/random_baseline"),
        help="Root directory for experiment artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output subdirectory name. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-trial step cap for debugging.",
    )
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Write one steps.jsonl record for each accepted or rejected move.",
    )
    parser.add_argument(
        "--no-save-points",
        dest="save_points",
        action="store_false",
        help="Omit final point sets and placement orders from runs.jsonl and summary.",
    )
    parser.set_defaults(save_points=True)
    return parser.parse_args()


def main() -> None:
    """Run the random baseline CLI and print a compact completion summary.

    The function constructs the selected independent-trial or TopPool config
    from command-line arguments, writes experiment artifacts, and prints key
    aggregate metrics. It mutates only files under the selected output
    directory.
    """

    args = parse_args()
    if args.mode == "toppool":
        top_pool_config = RandomTopPoolBaselineConfig(
            n=args.n,
            top_pool_size=args.top_pool_size,
            candidates_per_generation=args.candidates_per_generation,
            num_generations=args.num_generations,
            seed=args.seed,
            output_dir=args.output_dir,
            run_name=args.run_name,
            log_steps=args.log_steps,
            save_points=args.save_points,
            max_steps=args.max_steps,
        )
        summary = run_random_top_pool_baseline_experiment(top_pool_config)
        print(
            "random toppool baseline complete "
            f"n={top_pool_config.n} "
            f"generations={top_pool_config.num_generations} "
            f"candidates_per_generation={top_pool_config.candidates_per_generation} "
            f"best={summary['best_score']} "
            f"pool_entries={summary['num_pool_entries']} "
            f"run_dir={summary['run_dir']}"
        )
        return

    config = RandomBaselineConfig(
        n=args.n,
        trials=args.trials,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_steps=args.log_steps,
        save_points=args.save_points,
        max_steps=args.max_steps,
    )
    summary = run_random_baseline_experiment(config)
    score_stats = summary["score_stats"]
    print(
        "random baseline complete "
        f"n={config.n} "
        f"trials={config.trials} "
        f"mean={score_stats['mean']:.3f} "
        f"best={score_stats['max']} "
        f"run_dir={summary['run_dir']}"
    )


if __name__ == "__main__":
    main()
