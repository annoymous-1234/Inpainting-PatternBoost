"""Tests for PatternBoost candidate generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import numpy as np
import torch

from models.factory import build_model
from patternboost.generator import PatternBoostGenerator, compute_inference_center_distances
from patternboost.types import CircleTopPoolEntry
from circle_pack.scipy_solver import solve_radii_for_centers
from pipelines.types import BatchedProposalBundle

REPO_ROOT = Path(__file__).resolve().parents[1]


def _entry() -> CircleTopPoolEntry:
    centers = np.array(
        [
            [0.20, 0.20],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.80, 0.80],
        ],
        dtype=np.float64,
    )
    result = solve_radii_for_centers(centers)
    return CircleTopPoolEntry(
        centers=centers,
        radii=result.circles[:, 2],
        circles=result.circles,
        active_mask=np.ones(4, dtype=bool),
        score=float(result.sum_r),
        lp_status=result.message,
        canonical_hash="seed",
    )


def _small_config(make_config):
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
    return replace(
        config,
        project=replace(config.project, dtype="float32"),
        model=model,
        fm=replace(config.fm, num_samples=1, num_steps=1),
        outer_loop=replace(config.outer_loop, max_rounds=1),
        patternboost=replace(config.patternboost, polish=replace(config.patternboost.polish, refresh_lp_cache=False)),
    )


def test_compute_inference_center_distances_uses_edit_mask() -> None:
    start = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    end = np.array([[0.3, 0.4], [0.7, 0.5], [1.0, 0.0]], dtype=np.float64)
    distances = compute_inference_center_distances(start, end, np.array([True, False, True]))

    assert np.allclose(distances, [0.5, 1.0])
    assert compute_inference_center_distances(start, end, np.zeros(3, dtype=bool)).size == 0


def test_one_shot_generation_calls_one_outer_round_and_greedy_completes(make_config, monkeypatch) -> None:
    config = _small_config(make_config)
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(4))
    calls = {"count": 0}

    def fake_run_outer_round(*args, **kwargs):
        calls["count"] += 1
        return "record", None

    monkeypatch.setattr("patternboost.generator.run_outer_round", fake_run_outer_round)

    result = generator.generate_one_shot(_entry(), inference_mode="fm", candidate_seed=5)

    assert calls["count"] == 1
    assert result.candidate.full_active
    assert result.records == ["record"]
    assert result.candidate.inference_center_distance_count == int(result.edit_set.edit_mask.sum())
    assert result.candidate.inference_center_distance_mean == 0.0
    assert result.candidate.inference_center_distance_max == 0.0


def test_one_shot_generation_records_model_output_distance(make_config, monkeypatch) -> None:
    config = _small_config(make_config)
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(5))

    def fake_run_outer_round(env, *args, **kwargs):
        del args, kwargs
        free = ~np.asarray(env.active_mask, dtype=bool)
        direction = np.where(env.centers[free, 0] <= 0.5, 1.0, -1.0)
        env.centers[free, 0] += 0.1 * direction
        return "record", None

    monkeypatch.setattr("patternboost.generator.run_outer_round", fake_run_outer_round)

    result = generator.generate_one_shot(_entry(), inference_mode="fm", candidate_seed=6)

    assert result.candidate.inference_center_distance_count == int(result.edit_set.edit_mask.sum())
    assert np.isclose(result.candidate.inference_center_distance_mean, 0.1)
    assert np.isclose(result.candidate.inference_center_distance_max, 0.1)


def test_full_episode_generation_runs(make_config) -> None:
    config = _small_config(make_config)
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(6))

    result = generator.generate_full_episode(_entry(), inference_mode="fm", candidate_seed=7)

    assert result.candidate.circles.shape == (4, 3)
    assert result.generation_mode == "full_episode"


def test_fm_batched_generation_preserves_candidate_order(make_config, monkeypatch) -> None:
    config = _small_config(make_config)
    config = replace(
        config,
        fm=replace(config.fm, num_samples=2, num_steps=1),
        patternboost=replace(
            config.patternboost,
            generation=replace(config.patternboost.generation, candidate_batch_size=2),
        ),
    )
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(8))
    calls: list[int] = []

    def fake_sample_fm_proposals_batched(model, slot_latent, slot_centers, fixed_mask, cfg, latent_to_center_fn):
        del model, fixed_mask, latent_to_center_fn
        batch_size, num_slots, _ = slot_centers.shape
        sample_count = int(cfg.fm.num_samples)
        calls.append(batch_size)
        center_samples = slot_centers.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        latent_samples = slot_latent.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        aux = {"short_displacement": torch.ones(batch_size, sample_count, num_slots, device=slot_centers.device)}
        return BatchedProposalBundle(latent_samples=latent_samples, center_samples=center_samples, aux=aux)

    monkeypatch.setattr("patternboost.generator.sample_fm_proposals_batched", fake_sample_fm_proposals_batched)
    entry = _entry()
    work_items = [
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=11, candidate_index=1),
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=10, candidate_index=0),
    ]

    results = generator.generate_batch(work_items, inference_mode="fm")

    assert calls == [2]
    assert [result.candidate.metadata["candidate_index"] for result in results] == [0, 1]
    assert all(result.candidate.full_active for result in results)


def test_fm_batched_generation_records_output_distance(make_config, monkeypatch) -> None:
    config = _small_config(make_config)
    config = replace(config, fm=replace(config.fm, num_samples=2, num_steps=1))
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(11))

    def fake_sample_fm_proposals_batched(model, slot_latent, slot_centers, fixed_mask, cfg, latent_to_center_fn):
        del model, fixed_mask, latent_to_center_fn
        batch_size, num_slots, _ = slot_centers.shape
        sample_count = int(cfg.fm.num_samples)
        offset = torch.zeros_like(slot_centers)
        offset[..., 0] = torch.where(slot_centers[..., 0] <= 0.5, 0.1, -0.1)
        center_samples = slot_centers.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        center_samples = center_samples + offset.unsqueeze(1)
        latent_samples = slot_latent.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        aux = {"short_displacement": torch.ones(batch_size, sample_count, num_slots, device=slot_centers.device)}
        return BatchedProposalBundle(latent_samples=latent_samples, center_samples=center_samples, aux=aux)

    monkeypatch.setattr("patternboost.generator.sample_fm_proposals_batched", fake_sample_fm_proposals_batched)
    entry = _entry()
    work_items = [
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=16, candidate_index=0),
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=17, candidate_index=1),
    ]

    results = generator.generate_batch(work_items, inference_mode="fm")

    assert [result.candidate.metadata["candidate_index"] for result in results] == [0, 1]
    for result in results:
        assert result.candidate.inference_center_distance_count == int(result.edit_set.edit_mask.sum())
        assert np.isclose(result.candidate.inference_center_distance_mean, 0.1)
        assert np.isclose(result.candidate.inference_center_distance_max, 0.1)


def test_fm_batched_generation_runs_mixed_modes(make_config, monkeypatch) -> None:
    config = _small_config(make_config)
    config = replace(config, fm=replace(config.fm, num_samples=2, num_steps=1))
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(9))
    calls: list[int] = []

    def fake_sample_fm_proposals_batched(model, slot_latent, slot_centers, fixed_mask, cfg, latent_to_center_fn):
        del model, fixed_mask, latent_to_center_fn
        batch_size, num_slots, _ = slot_centers.shape
        sample_count = int(cfg.fm.num_samples)
        calls.append(batch_size)
        center_samples = slot_centers.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        latent_samples = slot_latent.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        aux = {"terminal_velocity_norm": torch.ones(batch_size, sample_count, num_slots, device=slot_centers.device)}
        return BatchedProposalBundle(latent_samples=latent_samples, center_samples=center_samples, aux=aux)

    monkeypatch.setattr("patternboost.generator.sample_fm_proposals_batched", fake_sample_fm_proposals_batched)
    entry = _entry()
    work_items = [
        generator.prepare_batch_work_item(entry, mode="full_episode", candidate_seed=12, candidate_index=0),
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=13, candidate_index=1),
    ]

    results = generator.generate_batch(work_items, inference_mode="fm")

    assert calls == [1, 1]
    assert [result.generation_mode for result in results] == ["full_episode", "one_shot"]
    assert all(result.candidate.full_active for result in results)


def test_fm_batched_variance_demo_uses_aux_signal(make_config, monkeypatch) -> None:
    config = _small_config(make_config)
    config = replace(
        config,
        fm=replace(config.fm, num_samples=2, num_steps=1),
        outer_loop=replace(config.outer_loop, candidate_mode="variance_demo"),
    )
    model = build_model(config)
    generator = PatternBoostGenerator(model, config, np.random.default_rng(10))
    saw_fm_signal: list[bool] = []

    def fake_sample_fm_proposals_batched(model, slot_latent, slot_centers, fixed_mask, cfg, latent_to_center_fn):
        del model, fixed_mask, latent_to_center_fn
        batch_size, num_slots, _ = slot_centers.shape
        sample_count = int(cfg.fm.num_samples)
        center_samples = slot_centers.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        latent_samples = slot_latent.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).clone()
        aux = {"short_displacement": torch.ones(batch_size, sample_count, num_slots, device=slot_centers.device)}
        return BatchedProposalBundle(latent_samples=latent_samples, center_samples=center_samples, aux=aux)

    def fake_demo_priority_score(dispersion, *, drift=None, fm_signal=None, free_mask=None, cfg=None):
        del dispersion, drift, cfg
        saw_fm_signal.append(fm_signal is not None)
        return torch.ones_like(torch.as_tensor(free_mask, dtype=torch.float32))

    monkeypatch.setattr("patternboost.generator.sample_fm_proposals_batched", fake_sample_fm_proposals_batched)
    monkeypatch.setattr("patternboost.generator.demo_priority_score", fake_demo_priority_score)
    entry = _entry()
    work_items = [
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=14, candidate_index=0),
        generator.prepare_batch_work_item(entry, mode="one_shot", candidate_seed=15, candidate_index=1),
    ]

    results = generator.generate_batch(work_items, inference_mode="fm")

    assert saw_fm_signal == [True, True]
    assert all(result.candidate.full_active for result in results)


def _random_operator_config(make_config, operator: str, sigma: float = 0.04):
    config = _small_config(make_config)
    generation = replace(
        config.patternboost.generation,
        completion_operator=operator,
        random_center_jitter_std=sigma,
    )
    return replace(config, patternboost=replace(config.patternboost, generation=generation))


def test_random_proposals_preserve_fixed_slots_and_match_sigma() -> None:
    from circle_pack.config import load_config
    from models.sampling import sample_random_proposals, sample_random_proposals_batched

    config = load_config(REPO_ROOT / "config.yaml")
    sigma = 0.04
    generation = replace(
        config.patternboost.generation,
        completion_operator="random_perturb",
        random_center_jitter_std=sigma,
    )
    config = replace(config, patternboost=replace(config.patternboost, generation=generation))

    num_slots = int(config.problem.K)
    num_samples = int(config.ddpm.num_samples)
    batch_size = 64
    rng = np.random.default_rng(0)
    centers = rng.uniform(0.3, 0.7, size=(batch_size, num_slots, 2))
    fixed = np.zeros((batch_size, num_slots), dtype=bool)
    fixed[:, : num_slots // 2] = True

    bundle = sample_random_proposals_batched(centers, fixed, config)
    samples = bundle.center_samples

    assert tuple(samples.shape) == (batch_size, num_samples, num_slots, 2)
    assert tuple(bundle.latent_samples.shape) == (batch_size, num_samples, num_slots, 2)

    reference = torch.as_tensor(centers, dtype=config.torch_dtype)
    reference = reference.unsqueeze(1).expand(batch_size, num_samples, num_slots, 2)
    assert torch.equal(samples[:, :, : num_slots // 2], reference[:, :, : num_slots // 2])
    assert bool((samples >= 0.0).all() and (samples <= 1.0).all())

    # Free-slot displacement is the 2D Gaussian norm, so its mean is sigma*sqrt(pi/2).
    moved = (samples[:, :, num_slots // 2 :] - reference[:, :, num_slots // 2 :]).norm(dim=-1)
    assert float(moved.mean()) == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.05)

    single = sample_random_proposals(centers[0], fixed[0], config)
    assert tuple(single.center_samples.shape) == (num_samples, num_slots, 2)


def test_random_scratch_proposals_ignore_parent_centers() -> None:
    from circle_pack.config import load_config
    from models.sampling import sample_random_proposals_batched

    config = load_config(REPO_ROOT / "config.yaml")
    generation = replace(config.patternboost.generation, completion_operator="random_scratch")
    config = replace(config, patternboost=replace(config.patternboost, generation=generation))

    num_slots = int(config.problem.K)
    centers = np.full((8, num_slots, 2), 0.5, dtype=np.float64)
    fixed = np.zeros((8, num_slots), dtype=bool)
    fixed[:, 0] = True

    samples = sample_random_proposals_batched(centers, fixed, config).center_samples
    assert torch.equal(samples[:, :, 0], torch.full_like(samples[:, :, 0], 0.5))
    assert float(samples[:, :, 1:].std()) > 0.1


@pytest.mark.parametrize("operator", ["random_perturb", "random_scratch"])
@pytest.mark.parametrize("inference_mode", ["ddpm", "fm"])
def test_random_operator_generation_never_touches_the_model(make_config, operator, inference_mode) -> None:
    config = _random_operator_config(make_config, operator)
    generator = PatternBoostGenerator(None, config, np.random.default_rng(11))

    result = generator.generate_full_episode(_entry(), inference_mode=inference_mode, candidate_seed=3)

    assert result.candidate.circles.shape == (4, 3)
    assert result.candidate.full_active


@pytest.mark.parametrize("inference_mode", ["ddpm", "fm"])
def test_random_operator_batched_generation_never_touches_the_model(make_config, inference_mode) -> None:
    config = _random_operator_config(make_config, "random_perturb")
    generator = PatternBoostGenerator(None, config, np.random.default_rng(12))

    work_items = [
        generator.prepare_batch_work_item(_entry(), mode="one_shot", candidate_seed=idx, candidate_index=idx)
        for idx in range(3)
    ]
    results = generator.generate_batch(work_items, inference_mode=inference_mode)

    assert [int(r.candidate.metadata["candidate_index"]) for r in results] == [0, 1, 2]
    assert all(r.candidate.full_active for r in results)


@pytest.mark.parametrize(("training_mode", "expected_key"), [("ddpm", "ddpm"), ("fm", "fm")])
def test_random_sample_count_follows_the_configured_head(make_config, training_mode, expected_key) -> None:
    from models.sampling import sample_random_proposals_batched

    config = _random_operator_config(make_config, "random_perturb")
    config = replace(
        config,
        ddpm=replace(config.ddpm, num_samples=2),
        fm=replace(config.fm, num_samples=5),
        patternboost=replace(
            config.patternboost,
            training=replace(config.patternboost.training, mode=training_mode),
        ),
    )
    expected = {"ddpm": 2, "fm": 5}[expected_key]

    num_slots = int(config.problem.K)
    bundle = sample_random_proposals_batched(
        np.full((1, num_slots, 2), 0.5),
        np.zeros((1, num_slots), dtype=bool),
        config,
    )
    assert bundle.center_samples.shape[1] == expected
