"""Run one formal PatternBoost experiment with reproducible artifacts."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import ProjectConfig, load_config
from patternboost.runner import PatternBoostRunner


GENERATION_FIELDS = [
    "generation",
    "seed",
    "mean_train_loss",
    "pool_score_best",
    "pool_score_p90",
    "pool_score_mean",
    "pool_size",
    "best_before",
    "best_after",
    "num_candidates",
    "num_inserted",
    "num_solver_polished",
    "legal_rate",
    "full_active_rate",
    "generation_error_count",
    "greedy_completion_success_rate",
    "candidate_raw_score_mean",
    "candidate_raw_score_p75",
    "candidate_raw_score_p90",
    "candidate_polish_score_mean",
    "candidate_polish_score_p75",
    "candidate_polish_score_p90",
    "candidate_final_score_mean",
    "candidate_final_score_p75",
    "candidate_final_score_p90",
    "inference_center_distance_count",
    "inference_center_distance_candidate_count",
    "inference_center_distance_mean",
    "inference_center_distance_std",
    "inference_center_distance_min",
    "inference_center_distance_p50",
    "inference_center_distance_p75",
    "inference_center_distance_p90",
    "inference_center_distance_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one formal PatternBoost experiment.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to unified config YAML.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Device override.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing experiment directory.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_config_yaml(path: Path, config: ProjectConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(asdict(config)), handle, sort_keys=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True))
            handle.write("\n")


def _validate_experiment_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("Experiment name must not be empty.")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Experiment name must be a single directory name.")
    return name


def experiment_name_from_config(config: ProjectConfig, *, now: datetime | None = None) -> str:
    configured = config.patternboost.runtime.run_name
    if configured:
        return _validate_experiment_name(configured)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return _validate_experiment_name(f"{timestamp}_K{config.problem.K}")


def resolve_experiment_config(config: ProjectConfig, *, experiment_name: str, device: str | None = None) -> ProjectConfig:
    runtime = config.patternboost.runtime
    runtime = replace(
        runtime,
        checkpoint_dir="checkpoints",
        run_name=experiment_name,
        device=device or runtime.device,
        log_jsonl=True,
    )
    pool = replace(config.patternboost.pool, save_path=None)
    patternboost = replace(config.patternboost, pool=pool, runtime=runtime)
    return replace(config, patternboost=patternboost)


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _git_info() -> dict[str, Any]:
    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def _torch_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    config_path: Path,
    experiment_name: str,
    run_dir: Path,
    started_at: str,
    finished_at: str | None = None,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "argv": list(sys.argv),
        "args": _json_safe(vars(args)),
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "run_dir": str(run_dir.resolve()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "torch": _torch_info(),
        "git": _git_info(),
    }


def generation_records_from_metrics(metrics: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in metrics:
        if record.get("event") != "generation":
            continue
        generation_record = {field: record.get(field) for field in GENERATION_FIELDS if field != "seed"}
        generation_record["seed"] = int(seed)
        records.append(generation_record)
    return records


def write_generation_records(*, metrics_path: Path, output_path: Path, seed: int) -> list[dict[str, Any]]:
    records = generation_records_from_metrics(_read_jsonl(metrics_path), seed=seed)
    _write_jsonl(output_path, records)
    return records


def artifact_checks(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    config_snapshot_path: Path,
    resolved_config_path: Path,
    manifest_path: Path,
    generations_log_path: Path,
) -> dict[str, Any]:
    top_pool_path = Path(summary.get("top_pool_path") or run_dir / "top_pool.pt")
    metrics_log_path = Path(summary.get("metrics_log_path") or run_dir / "metrics.jsonl")
    latest_path = run_dir / "latest.pt"
    candidates_path = run_dir / "candidates.jsonl"
    return {
        "top_pool_exists": top_pool_path.exists(),
        "latest_checkpoint_exists": latest_path.exists(),
        "generation_checkpoint_count": len(list(run_dir.glob("generation_*.pt"))) if run_dir.exists() else 0,
        "candidates_jsonl_exists": candidates_path.exists(),
        "metrics_jsonl_exists": metrics_log_path.exists(),
        "generations_jsonl_exists": generations_log_path.exists(),
        "config_snapshot_exists": config_snapshot_path.exists(),
        "resolved_config_exists": resolved_config_path.exists(),
        "manifest_exists": manifest_path.exists(),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    original_config = load_config(config_path)
    experiment_name = experiment_name_from_config(original_config)
    resolved_config = resolve_experiment_config(original_config, experiment_name=experiment_name, device=args.device)

    run_dir = Path(resolved_config.patternboost.runtime.checkpoint_dir) / experiment_name
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"Experiment directory exists: {run_dir}. Use --force to overwrite.")
    if run_dir.exists():
        shutil.rmtree(run_dir)

    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    finished_at: str | None = None
    status = "ok"
    error: str | None = None
    summary: dict[str, Any] = {}

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    resolved_config_path = run_dir / "resolved_config.yaml"
    manifest_path = run_dir / "run_manifest.json"
    generations_log_path = run_dir / "generations.jsonl"
    results_path = run_dir / "results.json"

    try:
        runner = PatternBoostRunner(resolved_config)
        shutil.copy2(config_path, config_snapshot_path)
        _write_config_yaml(resolved_config_path, resolved_config)
        _write_json(
            manifest_path,
            build_manifest(
                args=args,
                config_path=config_path,
                experiment_name=experiment_name,
                run_dir=run_dir,
                started_at=started_at,
            ),
        )
        summary = runner.run()
        del runner
    except Exception:
        status = "error"
        error = traceback.format_exc()
        run_dir.mkdir(parents=True, exist_ok=True)
        if not config_snapshot_path.exists():
            shutil.copy2(config_path, config_snapshot_path)
        if not resolved_config_path.exists():
            _write_config_yaml(resolved_config_path, resolved_config)
    finally:
        finished_at = datetime.now().isoformat(timespec="seconds")

    metrics_log_path = Path(summary.get("metrics_log_path") or run_dir / "metrics.jsonl")
    generation_records = write_generation_records(
        metrics_path=metrics_log_path,
        output_path=generations_log_path,
        seed=resolved_config.patternboost.runtime.seed,
    )
    elapsed_sec = time.perf_counter() - start
    _write_json(
        manifest_path,
        build_manifest(
            args=args,
            config_path=config_path,
            experiment_name=experiment_name,
            run_dir=run_dir,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
        ),
    )
    checks = artifact_checks(
        run_dir=run_dir,
        summary=summary,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        manifest_path=manifest_path,
        generations_log_path=generations_log_path,
    )
    result = {
        "experiment_name": experiment_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": float(elapsed_sec),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path.resolve()),
        "resolved_config_path": str(resolved_config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "summary": summary,
        "artifact_checks": checks,
        "manifest_path": str(manifest_path.resolve()),
        "generations_log_path": str(generations_log_path.resolve()),
        "metrics_log_path": str(metrics_log_path.resolve()),
        "num_generations_recorded": len(generation_records),
        "error": error,
    }
    _write_json(results_path, result)
    return result


def main() -> None:
    args = parse_args()
    try:
        result = run_experiment(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"results_path={Path(result['run_dir']) / 'results.json'}")
    print(f"status={result['status']}")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
