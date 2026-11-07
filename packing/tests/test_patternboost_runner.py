"""Smoke tests for the PatternBoost runner."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from patternboost.runner import PatternBoostRunner
from patternboost.top_pool import TopPool
from patternboost.types import PatternBoostCandidate


def _runner_config(tmp_path, make_config):
    config = make_config(K=4, max_steps=2, activation_budget=1)
    model = replace(
        config.model,
        preset="custom",
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2.0,
        use_global_token=False,
    )
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
            candidates_per_generation=2,
            modes=["one_shot"],
            mode_weights=[1.0],
        ),
        polish=replace(config.patternboost.polish, refresh_lp_cache=False),
        runtime=replace(
            config.patternboost.runtime,
            checkpoint_dir=str(tmp_path),
            run_name="pb_test",
            device="cpu",
            top_pool_snapshot_every_generations=1,
        ),
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


def test_patternboost_runner_one_generation_smoke(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    runner = PatternBoostRunner(config)
    summary = runner.run()

    assert summary["best_score"] > 0.0
    assert summary["num_pool_entries"] > 0
    assert (tmp_path / "pb_test" / "top_pool.pt").exists()
    assert (tmp_path / "pb_test" / "top_pool_generation_001.pt").exists()
    assert (tmp_path / "pb_test" / "latest.pt").exists()
    metrics_path = tmp_path / "pb_test" / "metrics.jsonl"
    assert metrics_path.exists()
    assert (tmp_path / "pb_test" / "candidates.jsonl").exists()
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    for key in [
        "num_candidates",
        "num_inserted",
        "full_active_rate",
        "legal_rate",
        "generation_error_count",
        "greedy_completion_success_rate",
        "raw_score_mean",
        "raw_score_max",
        "polish_score_mean",
        "polish_score_max",
        "final_score_mean",
        "final_score_max",
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
        "edit_policy_counts",
        "generation_mode_counts",
        "pool_size",
        "pool_score_best",
        "pool_score_mean",
        "pool_score_p50",
        "pool_score_p75",
        "pool_score_p90",
    ]:
        assert key in generation_record
    assert "cap_score_mean" not in generation_record
    assert "candidate_cap_score_mean" not in generation_record
    assert generation_record["pool_size"] == summary["pool_size"]
    assert generation_record["pool_score_best"] == summary["pool_score_best"]
    candidate_records = [
        json.loads(line)
        for line in (tmp_path / "pb_test" / "candidates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert candidate_records
    assert {"raw_score", "polish_score", "final_score"}.issubset(candidate_records[0])
    assert "cap_score" not in candidate_records[0]
    assert "inference_center_distances" not in candidate_records[0]
    assert "inference_center_distances" not in candidate_records[0]["metadata"]


def test_patternboost_runner_rejects_fm_mode_without_fm_head(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    model = replace(config.model, heads=replace(config.model.heads, enable_fm_head=False))
    pb = replace(config.patternboost, training=replace(config.patternboost.training, mode="fm"))
    config = replace(config, model=model, patternboost=pb)

    with pytest.raises(ValueError, match="enable_fm_head"):
        PatternBoostRunner(config)


def test_patternboost_runner_can_skip_checkpoints_and_candidate_log(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    runtime = replace(
        config.patternboost.runtime,
        run_name="pb_no_artifacts",
        save_checkpoints=False,
        log_candidates_jsonl=False,
        top_pool_snapshot_every_generations=0,
    )
    config = replace(config, patternboost=replace(config.patternboost, runtime=runtime))

    runner = PatternBoostRunner(config)
    summary = runner.run()

    run_dir = tmp_path / "pb_no_artifacts"
    assert (run_dir / "top_pool.pt").exists()
    assert not (run_dir / "latest.pt").exists()
    assert not list(run_dir.glob("generation_*.pt"))
    assert not (run_dir / "candidates.jsonl").exists()
    assert summary["latest_checkpoint_path"] is None
    assert summary["candidates_log_path"] is None

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    assert generation_record["num_candidates"] == 2
    assert generation_record["num_inserted"] >= 0
    assert generation_record["full_active_rate"] == 1.0
    assert generation_record["legal_rate"] == 1.0
    assert generation_record["generation_error_count"] == 0


def _candidate_with_scores(
    raw_score,
    polish_score=None,
    *,
    offset: float = 0.0,
    inference_distances=None,
) -> PatternBoostCandidate:
    centers = np.array(
        [
            [0.20 + offset, 0.20],
            [0.80 + offset, 0.20],
            [0.50 + offset, 0.80],
        ],
        dtype=np.float64,
    )
    radii = np.full(3, 0.05, dtype=np.float64)
    distances = None if inference_distances is None else np.asarray(inference_distances, dtype=np.float64)
    return PatternBoostCandidate(
        centers=centers,
        radii=radii,
        circles=np.column_stack((centers, radii)),
        active_mask=np.ones(3, dtype=bool),
        raw_score=raw_score,
        polish_score=polish_score,
        legal=raw_score is not None or polish_score is not None,
        full_active=True,
        inference_center_distances=distances,
        inference_center_distance_count=0 if distances is None else int(distances.size),
        inference_center_distance_mean=None if distances is None or distances.size == 0 else float(np.mean(distances)),
        inference_center_distance_max=None if distances is None or distances.size == 0 else float(np.max(distances)),
    )


def test_candidate_summary_uses_non_null_raw_polish_and_final_scores() -> None:
    runner = object.__new__(PatternBoostRunner)
    candidates = [
        _candidate_with_scores(1.0, inference_distances=[0.1, 0.2]),
        _candidate_with_scores(None, offset=0.02),
        _candidate_with_scores(2.0, polish_score=3.0, offset=-0.02, inference_distances=[0.4]),
    ]

    summary = runner._candidate_summary(candidates)

    assert summary["candidate_raw_score_mean"] == 1.5
    assert summary["candidate_polish_score_mean"] == 3.0
    assert summary["candidate_final_score_mean"] == 2.0
    assert summary["raw_score_mean"] == 1.5
    assert summary["polish_score_mean"] == 3.0
    assert summary["final_score_mean"] == 2.0
    assert "candidate_cap_score_mean" not in summary
    assert "cap_score_mean" not in summary
    assert summary["inference_center_distance_count"] == 3
    assert summary["inference_center_distance_candidate_count"] == 2
    assert np.isclose(summary["inference_center_distance_mean"], np.mean([0.1, 0.2, 0.4]))
    assert summary["inference_center_distance_p50"] == 0.2
    assert summary["inference_center_distance_max"] == 0.4


def test_insert_candidates_uses_polish_score_then_raw_and_skips_missing_score() -> None:
    runner = object.__new__(PatternBoostRunner)
    runner.top_pool = TopPool(8, dedup_tol=1.0e-12, feasibility_tol=1.0e-12, radius_max=0.5)
    runner._log_candidate = lambda *args, **kwargs: None
    candidates = [
        _candidate_with_scores(1.0, offset=0.0),
        _candidate_with_scores(1.5, polish_score=2.5, offset=0.05),
        _candidate_with_scores(None, offset=-0.05),
    ]

    assert runner._insert_candidates(candidates, generation_idx=0) == 2

    scores = sorted(entry.score for entry in runner.top_pool.entries())
    assert scores == [1.0, 2.5]
    assert candidates[0].inserted is True
    assert candidates[1].inserted is True
    assert candidates[2].inserted is False


def test_patternboost_runner_batched_ddpm_generation_smoke(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    pb = replace(
        config.patternboost,
        training=replace(config.patternboost.training, mode="ddpm", samples_per_generation=2, batch_size=1),
        generation=replace(
            config.patternboost.generation,
            candidates_per_generation=4,
            candidate_batch_size=2,
            modes=["full_episode"],
            mode_weights=[1.0],
        ),
        runtime=replace(
            config.patternboost.runtime,
            run_name="pb_batched_ddpm",
            save_checkpoints=False,
            log_candidates_jsonl=False,
            top_pool_snapshot_every_generations=0,
        ),
    )
    config = replace(
        config,
        ddpm=replace(config.ddpm, num_samples=1, num_steps=2),
        outer_loop=replace(config.outer_loop, max_rounds=1, candidate_mode="all_free"),
        patternboost=pb,
    )

    runner = PatternBoostRunner(config)
    summary = runner.run()

    run_dir = tmp_path / "pb_batched_ddpm"
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    assert summary["best_score"] > 0.0
    assert generation_record["num_candidates"] == 4
    assert generation_record["full_active_rate"] == 1.0
    assert generation_record["legal_rate"] == 1.0
    assert generation_record["generation_error_count"] == 0
    assert not (run_dir / "candidates.jsonl").exists()


def test_patternboost_runner_batched_ddpm_variance_demo_smoke(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    pb = replace(
        config.patternboost,
        training=replace(config.patternboost.training, mode="ddpm", samples_per_generation=2, batch_size=1),
        generation=replace(
            config.patternboost.generation,
            candidates_per_generation=4,
            candidate_batch_size=2,
            modes=["full_episode"],
            mode_weights=[1.0],
        ),
        runtime=replace(
            config.patternboost.runtime,
            run_name="pb_batched_variance",
            save_checkpoints=False,
            log_candidates_jsonl=False,
            top_pool_snapshot_every_generations=0,
        ),
    )
    config = replace(
        config,
        ddpm=replace(config.ddpm, num_samples=2, num_steps=2),
        outer_loop=replace(config.outer_loop, max_rounds=1, candidate_mode="variance_demo"),
        patternboost=pb,
    )

    runner = PatternBoostRunner(config)
    summary = runner.run()

    run_dir = tmp_path / "pb_batched_variance"
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    assert summary["best_score"] > 0.0
    assert generation_record["num_candidates"] == 4
    assert generation_record["full_active_rate"] == 1.0
    assert generation_record["legal_rate"] == 1.0
    assert generation_record["generation_error_count"] == 0


def test_patternboost_runner_batched_fm_generation_smoke(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    pb = replace(
        config.patternboost,
        training=replace(config.patternboost.training, mode="fm", samples_per_generation=2, batch_size=1),
        generation=replace(
            config.patternboost.generation,
            candidates_per_generation=4,
            candidate_batch_size=2,
            modes=["full_episode"],
            mode_weights=[1.0],
        ),
        runtime=replace(
            config.patternboost.runtime,
            run_name="pb_batched_fm",
            save_checkpoints=False,
            log_candidates_jsonl=False,
            top_pool_snapshot_every_generations=0,
        ),
    )
    config = replace(
        config,
        fm=replace(config.fm, num_samples=1, num_steps=2),
        outer_loop=replace(config.outer_loop, max_rounds=1, candidate_mode="all_free"),
        patternboost=pb,
    )

    runner = PatternBoostRunner(config)
    summary = runner.run()

    run_dir = tmp_path / "pb_batched_fm"
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    assert summary["best_score"] > 0.0
    assert generation_record["num_candidates"] == 4
    assert generation_record["full_active_rate"] == 1.0
    assert generation_record["legal_rate"] == 1.0
    assert generation_record["generation_error_count"] == 0


def test_patternboost_runner_batched_generation_falls_back_to_serial(tmp_path, make_config, monkeypatch) -> None:
    config = _runner_config(tmp_path, make_config)
    pb = replace(
        config.patternboost,
        training=replace(config.patternboost.training, mode="ddpm", samples_per_generation=2, batch_size=1),
        generation=replace(
            config.patternboost.generation,
            candidates_per_generation=2,
            candidate_batch_size=2,
            modes=["one_shot"],
            mode_weights=[1.0],
        ),
        runtime=replace(
            config.patternboost.runtime,
            run_name="pb_batched_fallback",
            save_checkpoints=False,
            log_candidates_jsonl=True,
            top_pool_snapshot_every_generations=0,
        ),
    )
    config = replace(config, ddpm=replace(config.ddpm, num_samples=1, num_steps=2), patternboost=pb)

    def fail_batch(*args, **kwargs):
        raise RuntimeError("forced batch failure")

    monkeypatch.setattr("patternboost.generator.PatternBoostGenerator.generate_batch", fail_batch)

    runner = PatternBoostRunner(config)
    runner.run()

    run_dir = tmp_path / "pb_batched_fallback"
    candidates = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines() if line.strip()]
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    generation_record = next(record for record in metrics if record.get("event") == "generation")
    assert generation_record["num_candidates"] == 2
    assert generation_record["full_active_rate"] == 1.0
    assert all("forced batch failure" in (record["metadata"].get("batch_fallback_error") or "") for record in candidates)


def test_patternboost_runner_delegates_srp_polishing(tmp_path, make_config, monkeypatch) -> None:
    config = _runner_config(tmp_path, make_config)
    pb = replace(
        config.patternboost,
        polish=replace(config.patternboost.polish, srp_enabled=True, srp_top_k=1, srp_num_workers=1),
        runtime=replace(config.patternboost.runtime, run_name="pb_srp_delegate", save_checkpoints=False),
    )
    config = replace(config, patternboost=pb)
    calls = []

    def fake_srp_polish(candidates, *, polish_cfg, generation_idx, runtime_seed):
        calls.append((len(candidates), polish_cfg.srp_top_k, generation_idx, runtime_seed))
        return []

    monkeypatch.setattr("patternboost.runner.srp_polish_candidates", fake_srp_polish)

    runner = PatternBoostRunner(config)
    candidates = []
    runner._polish_candidates(candidates, generation_idx=3)

    assert calls == [(0, 1, 3, config.patternboost.runtime.seed)]


def test_patternboost_runner_resumes_checkpoint_top_pool_and_optimizer(tmp_path, make_config) -> None:
    first_config = _runner_config(tmp_path, make_config)
    first_runner = PatternBoostRunner(first_config)
    first_summary = first_runner.run()
    first_run_dir = tmp_path / "pb_test"
    checkpoint_path = first_run_dir / "latest.pt"
    top_pool_path = first_run_dir / "top_pool.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["generation"] == 1
    assert "optimizer_state_dict" in checkpoint

    resume = replace(
        first_config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(checkpoint_path),
        top_pool_path=str(top_pool_path),
        load_optimizer=True,
    )
    pb = replace(
        first_config.patternboost,
        resume=resume,
        runtime=replace(first_config.patternboost.runtime, run_name="pb_resume"),
    )
    resume_config = replace(first_config, patternboost=pb)

    resume_runner = PatternBoostRunner(resume_config)
    resume_summary = resume_runner.run()
    resume_run_dir = tmp_path / "pb_resume"
    resume_checkpoint = torch.load(resume_run_dir / "latest.pt", map_location="cpu", weights_only=False)
    metrics = [json.loads(line) for line in (resume_run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    run_start = next(record for record in metrics if record.get("event") == "run_start")
    generation_record = next(record for record in metrics if record.get("event") == "generation")

    assert run_start["initial_best_score"] == first_summary["best_score"]
    assert run_start["resume_enabled"] is True
    assert run_start["resume_from_generation"] == 1
    assert run_start["resume_loaded_optimizer"] is True
    assert generation_record["generation"] == 2
    assert (resume_run_dir / "generation_002.pt").exists()
    assert resume_checkpoint["generation"] == 2
    assert resume_summary["resume_from_generation"] == 1
    assert resume_summary["resume_loaded_optimizer"] is True
    assert checkpoint_path.exists()
    assert top_pool_path.exists()


def test_patternboost_resume_can_rebuild_optimizer_when_checkpoint_lacks_optimizer(tmp_path, make_config) -> None:
    first_config = _runner_config(tmp_path, make_config)
    first_runner = PatternBoostRunner(first_config)
    first_runner.run()
    first_run_dir = tmp_path / "pb_test"
    checkpoint_path = first_run_dir / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer_state_dict")
    no_optimizer_path = tmp_path / "no_optimizer.pt"
    torch.save(checkpoint, no_optimizer_path)

    resume = replace(
        first_config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(no_optimizer_path),
        top_pool_path=str(first_run_dir / "top_pool.pt"),
        load_optimizer=False,
    )
    pb = replace(
        first_config.patternboost,
        resume=resume,
        runtime=replace(first_config.patternboost.runtime, run_name="pb_resume_rebuild_optimizer"),
    )
    runner = PatternBoostRunner(replace(first_config, patternboost=pb))

    assert runner.resume_loaded_optimizer is False
    assert runner.resume_from_generation == 1


def test_patternboost_resume_rejects_checkpoint_without_optimizer_when_requested(tmp_path, make_config) -> None:
    first_config = _runner_config(tmp_path, make_config)
    first_runner = PatternBoostRunner(first_config)
    first_runner.run()
    first_run_dir = tmp_path / "pb_test"
    checkpoint = torch.load(first_run_dir / "latest.pt", map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer_state_dict")
    no_optimizer_path = tmp_path / "no_optimizer.pt"
    torch.save(checkpoint, no_optimizer_path)
    resume = replace(
        first_config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(no_optimizer_path),
        top_pool_path=str(first_run_dir / "top_pool.pt"),
        load_optimizer=True,
    )
    pb = replace(
        first_config.patternboost,
        resume=resume,
        runtime=replace(first_config.patternboost.runtime, run_name="pb_resume_missing_optimizer"),
    )

    with pytest.raises(ValueError, match="optimizer_state_dict"):
        PatternBoostRunner(replace(first_config, patternboost=pb))


def test_patternboost_resume_rejects_invalid_paths_and_top_pool(tmp_path, make_config) -> None:
    config = _runner_config(tmp_path, make_config)
    runner = PatternBoostRunner(config)
    runner.run()
    run_dir = tmp_path / "pb_test"

    missing_resume = replace(
        config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(tmp_path / "missing.pt"),
        top_pool_path=str(run_dir / "top_pool.pt"),
    )
    missing_pb = replace(
        config.patternboost,
        resume=missing_resume,
        runtime=replace(config.patternboost.runtime, run_name="pb_missing_checkpoint"),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        PatternBoostRunner(replace(config, patternboost=missing_pb))

    same_top_pool_resume = replace(
        config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(run_dir / "latest.pt"),
        top_pool_path=str(run_dir / "top_pool.pt"),
    )
    same_top_pool_pb = replace(
        config.patternboost,
        pool=replace(config.patternboost.pool, save_path=str(run_dir / "top_pool.pt")),
        resume=same_top_pool_resume,
        runtime=replace(config.patternboost.runtime, run_name="pb_same_top_pool"),
    )
    with pytest.raises(ValueError, match="must differ"):
        PatternBoostRunner(replace(config, patternboost=same_top_pool_pb))

    empty_pool_path = tmp_path / "empty_top_pool.pt"
    TopPool(8).save(empty_pool_path)
    empty_resume = replace(
        config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(run_dir / "latest.pt"),
        top_pool_path=str(empty_pool_path),
    )
    empty_pb = replace(
        config.patternboost,
        resume=empty_resume,
        runtime=replace(config.patternboost.runtime, run_name="pb_empty_top_pool"),
    )
    with pytest.raises(ValueError, match="empty"):
        PatternBoostRunner(replace(config, patternboost=empty_pb))

    mismatch_pool = TopPool(8)
    mismatch_pool.insert(np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]], dtype=np.float64))
    mismatch_pool_path = tmp_path / "mismatch_top_pool.pt"
    mismatch_pool.save(mismatch_pool_path)
    mismatch_resume = replace(
        config.patternboost.resume,
        enabled=True,
        checkpoint_path=str(run_dir / "latest.pt"),
        top_pool_path=str(mismatch_pool_path),
    )
    mismatch_pb = replace(
        config.patternboost,
        resume=mismatch_resume,
        runtime=replace(config.patternboost.runtime, run_name="pb_mismatch_top_pool"),
    )
    with pytest.raises(ValueError, match="K mismatch"):
        PatternBoostRunner(replace(config, patternboost=mismatch_pb))


@pytest.mark.parametrize("operator", ["random_perturb", "random_scratch"])
def test_patternboost_runner_random_operator_builds_no_model(tmp_path, make_config, operator) -> None:
    config = _runner_config(tmp_path, make_config)
    generation = replace(
        config.patternboost.generation,
        completion_operator=operator,
        candidate_batch_size=2,
    )
    runtime = replace(config.patternboost.runtime, run_name=f"pb_{operator}", save_checkpoints=True)
    config = replace(
        config,
        patternboost=replace(config.patternboost, generation=generation, runtime=runtime),
    )

    runner = PatternBoostRunner(config)
    assert runner.model is None
    assert runner.optimizer is None
    assert not runner.uses_learned_operator

    summary = runner.run()

    # No training happened, so no *model* checkpoint may be advertised or
    # written, even though save_checkpoints is on. TopPool snapshots are a
    # separate artifact and must still be produced.
    run_dir = tmp_path / f"pb_{operator}"
    assert summary["latest_checkpoint_path"] is None
    assert not (run_dir / "latest.pt").exists()
    assert not list(run_dir.glob("generation_*.pt"))
    assert (run_dir / "top_pool.pt").exists()

    records = [
        json.loads(line)
        for line in (tmp_path / f"pb_{operator}" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generations = [record for record in records if record.get("event") == "generation"]
    assert generations
    assert all(record["mean_train_loss"] == 0.0 for record in generations)
