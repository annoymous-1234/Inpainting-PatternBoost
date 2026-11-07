"""Smoke tests for PatternBoost scripts."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path, config) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    return path


def _run_script(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _script_config(tmp_path, make_config):
    config = make_config(K=4, max_steps=2, activation_budget=1)
    pb = replace(
        config.patternboost,
        pool=replace(config.patternboost.pool, max_size=8, init_sources=["greedy"], greedy_num_seeds=4),
        training=replace(
            config.patternboost.training,
            samples_per_generation=2,
            epochs_per_generation=1,
            batch_size=1,
            lr=1.0e-3,
        ),
        generation=replace(
            config.patternboost.generation,
            num_generations=1,
            candidates_per_generation=1,
            modes=["one_shot"],
            mode_weights=[1.0],
        ),
        polish=replace(config.patternboost.polish, refresh_lp_cache=False),
        runtime=replace(config.patternboost.runtime, checkpoint_dir=str(tmp_path), run_name="script_run", device="cpu"),
    )
    model = replace(
        config.model,
        preset="custom",
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2.0,
        use_global_token=False,
    )
    return replace(
        config,
        project=replace(config.project, dtype="float32"),
        model=model,
        train=replace(config.train, dataset_path=str(tmp_path / "missing_teacher.pt")),
        fm=replace(config.fm, num_samples=1, num_steps=1),
        outer_loop=replace(config.outer_loop, max_rounds=1),
        patternboost=pb,
    )


def _formal_script_config(tmp_path, make_config, *, run_name: str, save_checkpoints: bool = False):
    config = _script_config(tmp_path, make_config)
    runtime = replace(
        config.patternboost.runtime,
        run_name=run_name,
        save_checkpoints=save_checkpoints,
        log_candidates_jsonl=False,
    )
    return replace(config, patternboost=replace(config.patternboost, runtime=runtime))


def test_train_patternboost_script_runs(tmp_path, make_config) -> None:
    config_path = _write_config(tmp_path / "patternboost.yaml", _script_config(tmp_path, make_config))

    result = _run_script("scripts/train_patternboost.py", "--config", str(config_path), "--device", "cpu")

    assert "best_score=" in result.stdout
    assert (tmp_path / "script_run" / "latest.pt").exists()


def test_demo_patternboost_script_runs(tmp_path, make_config) -> None:
    config = _script_config(tmp_path, make_config)
    config = replace(
        config,
        patternboost=replace(
            config.patternboost,
            runtime=replace(config.patternboost.runtime, checkpoint_dir=str(tmp_path), run_name=None, device="cpu"),
        ),
    )
    config_path = _write_config(tmp_path / "patternboost_demo.yaml", config)

    result = _run_script(
        "scripts/demo_patternboost.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--num-generations",
        "1",
        "--candidates",
        "1",
    )

    assert "best_score=" in result.stdout
    assert "top_pool_path=" in result.stdout


def test_run_patternboost_experiment_writes_formal_single_run_artifacts(tmp_path, make_config) -> None:
    config_path = _write_config(
        tmp_path / "formal.yaml",
        _formal_script_config(tmp_path, make_config, run_name="formal_run", save_checkpoints=False),
    )

    result = _run_script(
        str(REPO_ROOT / "scripts" / "run_patternboost_experiment.py"),
        "--config",
        str(config_path),
        "--device",
        "cpu",
        cwd=tmp_path,
    )

    run_dir = tmp_path / "checkpoints" / "formal_run"
    assert "status=ok" in result.stdout
    assert (run_dir / "config_snapshot.yaml").read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "generations.jsonl").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "top_pool.pt").exists()
    assert not (run_dir / "latest.pt").exists()
    assert not list(run_dir.glob("generation_*.pt"))

    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["patternboost"]["runtime"]["checkpoint_dir"] == "checkpoints"
    assert resolved["patternboost"]["runtime"]["run_name"] == "formal_run"
    assert resolved["patternboost"]["runtime"]["log_jsonl"] is True
    assert resolved["patternboost"]["pool"]["save_path"] is None

    generation_lines = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert len(generation_lines) == 1
    generation_record = generation_lines[0]
    for key in [
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
    ]:
        assert key in generation_record
    assert "candidate_cap_score_mean" not in generation_record

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    metrics_generation = next(record for record in metrics if record.get("event") == "generation")
    assert "candidate_raw_score_p75" in metrics_generation
    assert "candidate_polish_score_p90" in metrics_generation
    assert "candidate_final_score_p90" in metrics_generation
    assert "inference_center_distance_p90" in metrics_generation
    assert "candidate_cap_score_p90" not in metrics_generation

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["experiment_name"] == "formal_run"
    assert results["status"] == "ok"
    assert results["num_generations_recorded"] == 1
    assert results["artifact_checks"]["top_pool_exists"] is True
    assert results["artifact_checks"]["latest_checkpoint_exists"] is False
    assert results["artifact_checks"]["generation_checkpoint_count"] == 0


def test_run_patternboost_experiment_force_and_checkpoints(tmp_path, make_config) -> None:
    config_path = _write_config(
        tmp_path / "formal_ckpt.yaml",
        _formal_script_config(tmp_path, make_config, run_name="formal_ckpt", save_checkpoints=True),
    )
    run_dir = tmp_path / "checkpoints" / "formal_ckpt"
    run_dir.mkdir(parents=True)
    marker_path = run_dir / "marker.txt"
    marker_path.write_text("old", encoding="utf-8")

    failed = _run_script(
        str(REPO_ROOT / "scripts" / "run_patternboost_experiment.py"),
        "--config",
        str(config_path),
        "--device",
        "cpu",
        cwd=tmp_path,
        check=False,
    )
    assert failed.returncode != 0
    assert marker_path.exists()

    result = _run_script(
        str(REPO_ROOT / "scripts" / "run_patternboost_experiment.py"),
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--force",
        cwd=tmp_path,
    )

    assert "status=ok" in result.stdout
    assert not marker_path.exists()
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "generation_001.pt").exists()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["artifact_checks"]["latest_checkpoint_exists"] is True
    assert results["artifact_checks"]["generation_checkpoint_count"] == 1

    resume_config = _formal_script_config(tmp_path, make_config, run_name="formal_resume", save_checkpoints=True)
    resume_config = replace(
        resume_config,
        patternboost=replace(
            resume_config.patternboost,
            resume=replace(
                resume_config.patternboost.resume,
                enabled=True,
                checkpoint_path=str(run_dir / "latest.pt"),
                top_pool_path=str(run_dir / "top_pool.pt"),
                load_optimizer=True,
            ),
        ),
    )
    resume_config_path = _write_config(tmp_path / "formal_resume.yaml", resume_config)
    resume_result = _run_script(
        str(REPO_ROOT / "scripts" / "run_patternboost_experiment.py"),
        "--config",
        str(resume_config_path),
        "--device",
        "cpu",
        cwd=tmp_path,
    )
    resume_run_dir = tmp_path / "checkpoints" / "formal_resume"
    assert "status=ok" in resume_result.stdout
    assert (resume_run_dir / "generation_002.pt").exists()
    assert (run_dir / "latest.pt").exists()
    metrics = [json.loads(line) for line in (resume_run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    run_start = next(record for record in metrics if record.get("event") == "run_start")
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    assert run_start["resume_enabled"] is True
    assert run_start["resume_from_generation"] == 1
    assert run_start["resume_loaded_optimizer"] is True
    assert generation_record["generation"] == 2
    resume_results = json.loads((resume_run_dir / "results.json").read_text(encoding="utf-8"))
    assert resume_results["summary"]["resume_enabled"] is True
    assert resume_results["summary"]["resume_from_generation"] == 1


def test_run_patternboost_experiment_default_name_uses_timestamp_and_k(tmp_path, make_config) -> None:
    module_path = REPO_ROOT / "scripts" / "run_patternboost_experiment.py"
    spec = importlib.util.spec_from_file_location("run_patternboost_experiment_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = _formal_script_config(tmp_path, make_config, run_name="unused")
    config = replace(
        config,
        patternboost=replace(config.patternboost, runtime=replace(config.patternboost.runtime, run_name=None)),
    )

    name = module.experiment_name_from_config(config, now=datetime(2026, 4, 30, 15, 30, 12))

    assert name == "20260430_153012_K4"


def test_noise_long_v1_specs_follow_factored_design() -> None:
    module_path = REPO_ROOT / "tmp_experiments" / "patternboost_ddpm_noise_long_v1" / "common.py"
    spec = importlib.util.spec_from_file_location("noise_long_v1_common_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    specs = module.build_noise_long_specs()
    assert len(specs) == 12
    assert {item.overrides["uniform_mix"] for item in specs} == {module.FIXED_UNIFORM_MIX}

    joint_specs = [item for item in specs if item.group == "joint_binary"]
    assert len(joint_specs) == 8
    observed_joint = {
        (
            item.overrides["ddpm_noising_center"],
            item.overrides["ddpm_use_draft_warmstart"],
            item.overrides["use_draft_match"],
        )
        for item in joint_specs
    }
    assert observed_joint == {
        (noising_center, use_draft_warmstart, use_draft_match)
        for noising_center in ["target", "draft"]
        for use_draft_warmstart in [True, False]
        for use_draft_match in [False, True]
    }
    for item in joint_specs:
        assert item.overrides["ddpm_draft_strength"] == module.DEFAULT_DDPM_DRAFT_STRENGTH
        assert item.overrides["draft_center_jitter_std"] == module.DEFAULT_DRAFT_CENTER_JITTER_STD

    strength_specs = [item for item in specs if item.group == "draft_strength"]
    assert [item.name for item in strength_specs] == ["strength_s0.5"]
    for item in strength_specs:
        assert item.overrides["ddpm_noising_center"] == "target"
        assert item.overrides["ddpm_use_draft_warmstart"] is True
        assert item.overrides["use_draft_match"] is False
        assert item.overrides["draft_center_jitter_std"] == module.DEFAULT_DRAFT_CENTER_JITTER_STD

    jitter_specs = [item for item in specs if item.group == "draft_jitter"]
    assert [item.name for item in jitter_specs] == ["jitter_j0.0", "jitter_j0.06", "jitter_j0.10"]
    for item in jitter_specs:
        assert item.overrides["ddpm_noising_center"] == "target"
        assert item.overrides["ddpm_use_draft_warmstart"] is True
        assert item.overrides["use_draft_match"] is False
        assert item.overrides["ddpm_draft_strength"] == module.DEFAULT_DDPM_DRAFT_STRENGTH
