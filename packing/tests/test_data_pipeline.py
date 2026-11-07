"""Tests for center-only supervised data construction and dataset helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from circle_pack.config import DDPMSection, FMSection, ModelSection, ModelSlotFeaturesSection
from circle_pack.collect import collect_episode
from circle_pack.env import CirclePackingEnv
from circle_pack.policies import make_policy
from pipelines.data import (
    CirclePackingSupervisedDataset,
    assign_slots_random,
    assign_slots_xy,
    build_jittered_draft,
    build_synthetic_draft,
    generate_synthetic_teacher_layouts,
    load_teacher_layouts,
    make_patternboost_supervised_example,
    make_supervised_example,
    match_target_to_draft_greedy,
    save_teacher_layouts,
)


def _teacher_layout() -> np.ndarray:
    return np.array(
        [
            [0.70, 0.80, 0.10],
            [0.20, 0.60, 0.08],
            [0.40, 0.30, 0.05],
            [0.10, 0.10, 0.00],
        ],
        dtype=np.float64,
    )


def _teacher_centers() -> np.ndarray:
    return _teacher_layout()[:, :2]


def _alpha_bar_from_example(example: dict[str, torch.Tensor], config) -> float:
    step_idx = int(round(float(example["inner_time"].item()) * max(config.ddpm.num_steps - 1, 1)))
    betas = np.linspace(config.ddpm.beta_start, config.ddpm.beta_end, config.ddpm.num_steps, dtype=np.float64)
    return float(np.cumprod(1.0 - betas, axis=0)[step_idx])


def test_assignment_helpers_have_expected_behavior() -> None:
    teacher = _teacher_layout()
    rng = np.random.default_rng(7)

    random_layout, random_order = assign_slots_random(teacher, rng)
    xy_layout, xy_order = assign_slots_xy(teacher)

    assert random_layout.shape == (4, 2)
    assert sorted(random_order.tolist()) == list(range(teacher.shape[0]))
    assert xy_order.tolist() == [3, 1, 2, 0]
    assert np.allclose(xy_layout[:, 0], np.array([0.10, 0.20, 0.40, 0.70]))


def test_draft_match_greedy_follows_close_slots() -> None:
    teacher = _teacher_layout()
    draft = teacher[[2, 0, 3, 1]].copy()
    draft[:, :2] += np.array([[0.01, -0.01], [0.0, 0.01], [-0.01, 0.0], [0.0, -0.01]])

    matched, assignment = match_target_to_draft_greedy(draft, teacher)

    assert matched.shape == (4, 2)
    assert sorted(assignment.tolist()) == list(range(teacher.shape[0]))
    assert np.allclose(matched[0], teacher[2, :2], atol=1.0e-6)
    assert np.allclose(matched[1], teacher[0, :2], atol=1.0e-6)


def test_jittered_draft_only_changes_free_slots(make_config) -> None:
    config = make_config(K=4)
    teacher = _teacher_layout()
    fixed_mask = np.array([True, False, True, False], dtype=bool)

    draft_latent, draft_centers = build_jittered_draft(teacher, fixed_mask, np.random.default_rng(19), cfg=config)

    assert draft_latent.shape == (4, 2)
    assert draft_centers.shape == (4, 2)
    np.testing.assert_allclose(draft_centers[fixed_mask], teacher[fixed_mask, :2])
    assert np.all((0.0 <= draft_centers) & (draft_centers <= 1.0))


def test_synthetic_draft_and_supervised_examples(make_config) -> None:
    config = make_config(K=4)
    teacher = _teacher_layout()
    target_latent = np.array([[0.1, -0.2], [-0.4, 0.5], [0.0, 0.2], [0.7, -0.6]], dtype=np.float32)
    rng = np.random.default_rng(11)

    draft_latent, draft_centers = build_synthetic_draft(
        target_latent,
        teacher,
        mode="mixed",
        corruption_scale=0.15,
        rng=rng,
        cfg=config,
    )
    assert draft_latent.shape == (4, 2)
    assert draft_centers.shape == (4, 2)
    assert np.all((0.0 <= draft_centers) & (draft_centers <= 1.0))

    ddpm_cfg = make_config(
        K=4,
        model=ModelSection(
            preset="tiny",
            d_model=192,
            n_heads=6,
            n_layers=4,
            mlp_ratio=4.0,
            dropout=0.0,
            slot_features=ModelSlotFeaturesSection(
                use_fixed_flag=True,
                use_latent=True,
                use_physical=True,
                use_wall_slack=True,
                use_extra_slot_features=True,
                extra_slot_feature_dim=4,
            ),
        ),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=8,
            beta_start=1.0e-4,
            beta_end=0.02,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=True,
            fixed_mode="frozen_full_state",
        ),
    )
    ddpm_example = make_supervised_example(teacher, mode="ddpm", cfg=ddpm_cfg, rng=np.random.default_rng(3))
    fm_example = make_supervised_example(teacher, mode="fm", cfg=config, rng=np.random.default_rng(5))

    assert ddpm_example["slot_latent"].shape == (4, 2)
    assert ddpm_example["slot_centers"].shape == (4, 2)
    assert ddpm_example["noise_target"].shape == (4, 2)
    assert ddpm_example["target_latent"].shape == (4, 2)
    assert ddpm_example["target_centers"].shape == (4, 2)
    assert ddpm_example["fixed_mask"].shape == (4,)
    assert ddpm_example["extra_slot_features"].shape == (4, 4)
    assert ddpm_example["assignment"].dtype == torch.int64

    assert fm_example["slot_latent"].shape == (4, 2)
    assert fm_example["slot_centers"].shape == (4, 2)
    assert fm_example["draft_latent"].shape == (4, 2)
    assert fm_example["draft_centers"].shape == (4, 2)
    assert fm_example["vel_target"].shape == (4, 2)
    assert "extra_slot_features" not in fm_example


def test_fm_draft_feature_supervised_examples(make_config) -> None:
    model = ModelSection(
        preset="tiny",
        d_model=192,
        n_heads=6,
        n_layers=4,
        mlp_ratio=4.0,
        dropout=0.0,
        slot_features=ModelSlotFeaturesSection(
            use_fixed_flag=False,
            use_latent=True,
            use_physical=False,
            use_wall_slack=False,
            use_extra_slot_features=True,
            extra_slot_feature_dim=4,
        ),
    )
    config = make_config(
        K=4,
        model=model,
        fm=FMSection(
            num_samples=2,
            num_steps=4,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
            use_draft_feature=True,
        ),
    )
    teacher = _teacher_layout()
    edit_mask = np.array([False, True, False, True], dtype=bool)

    standard = make_supervised_example(teacher, mode="fm", cfg=config, rng=np.random.default_rng(31))
    patternboost = make_patternboost_supervised_example(
        teacher,
        edit_mask,
        mode="fm",
        cfg=config,
        rng=np.random.default_rng(32),
    )

    assert standard["extra_slot_features"].shape == (4, 4)
    assert patternboost["extra_slot_features"].shape == (4, 4)
    torch.testing.assert_close(standard["extra_slot_features"][:, :2], standard["draft_latent"])
    torch.testing.assert_close(standard["extra_slot_features"][:, 2:], standard["draft_centers"])
    torch.testing.assert_close(patternboost["extra_slot_features"][:, :2], patternboost["draft_latent"])
    torch.testing.assert_close(patternboost["extra_slot_features"][:, 2:], patternboost["draft_centers"])


def test_ddpm_target_centered_example_uses_standard_noise_target(make_config) -> None:
    config = make_config(
        K=4,
        train=replace(make_config(K=4).train, fixed_ratio_min=0.5, fixed_ratio_max=0.5),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=8,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
            noising_center="target",
            draft_strength=1.0,
            use_timestep_subsampling=False,
            infer_num_steps=4,
        ),
    )
    example = make_supervised_example(_teacher_layout(), mode="ddpm", cfg=config, rng=np.random.default_rng(21))
    alpha_bar = _alpha_bar_from_example(example, config)
    fixed_mask = example["fixed_mask"].numpy().astype(bool)
    free_mask = ~fixed_mask
    expected_free = (
        np.sqrt(alpha_bar) * example["target_latent"][free_mask].numpy()
        + np.sqrt(1.0 - alpha_bar) * example["noise_target"][free_mask].numpy()
    )

    np.testing.assert_allclose(example["slot_latent"][free_mask].numpy(), expected_free, atol=1.0e-6)
    np.testing.assert_allclose(example["slot_latent"][fixed_mask].numpy(), example["target_latent"][fixed_mask].numpy())
    assert "extra_slot_features" not in example


def test_ddpm_draft_centered_example_uses_effective_noise_target(make_config) -> None:
    config = make_config(
        K=4,
        train=replace(make_config(K=4).train, fixed_ratio_min=0.5, fixed_ratio_max=0.5),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=8,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
            noising_center="draft",
            draft_strength=0.5,
            use_timestep_subsampling=False,
            infer_num_steps=4,
        ),
    )
    example = make_supervised_example(_teacher_layout(), mode="ddpm", cfg=config, rng=np.random.default_rng(22))
    alpha_bar = _alpha_bar_from_example(example, config)
    fixed_mask = example["fixed_mask"].numpy().astype(bool)
    free_mask = ~fixed_mask
    expected_noise_target = (
        example["slot_latent"][free_mask].numpy()
        - np.sqrt(alpha_bar) * example["target_latent"][free_mask].numpy()
    ) / np.sqrt(1.0 - alpha_bar)

    np.testing.assert_allclose(example["noise_target"][free_mask].numpy(), expected_noise_target, atol=1.0e-6)
    np.testing.assert_allclose(example["draft_centers"][fixed_mask].numpy(), example["target_centers"][fixed_mask].numpy())
    np.testing.assert_allclose(example["slot_latent"][fixed_mask].numpy(), example["target_latent"][fixed_mask].numpy())


def test_ddpm_draft_strength_zero_uses_pure_noise_input(make_config) -> None:
    config = make_config(
        K=4,
        train=replace(make_config(K=4).train, fixed_ratio_min=0.5, fixed_ratio_max=0.5),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=8,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
            noising_center="draft",
            draft_strength=0.0,
            use_timestep_subsampling=False,
            infer_num_steps=4,
        ),
    )
    example = make_supervised_example(_teacher_layout(), mode="ddpm", cfg=config, rng=np.random.default_rng(23))
    alpha_bar = _alpha_bar_from_example(example, config)
    fixed_mask = example["fixed_mask"].numpy().astype(bool)
    free_mask = ~fixed_mask
    recovered_pure_noise = (
        example["noise_target"][free_mask].numpy() * np.sqrt(1.0 - alpha_bar)
        + np.sqrt(alpha_bar) * example["target_latent"][free_mask].numpy()
    )

    np.testing.assert_allclose(example["slot_latent"][free_mask].numpy(), recovered_pure_noise, atol=2.0e-6)


def test_supervised_dataset_returns_on_the_fly_examples(make_config) -> None:
    config = make_config(K=4)
    teacher_bank = np.stack([_teacher_layout(), _teacher_layout()[::-1]], axis=0)
    dataset = CirclePackingSupervisedDataset(teacher_bank, mode="fm", cfg=config, seed=17)

    example = dataset[0]
    assert len(dataset) == 2
    assert example["slot_latent"].shape == (4, 2)
    assert example["slot_centers"].shape == (4, 2)
    assert example["fixed_mask"].shape == (4,)


def test_teacher_layout_roundtrip_pt_and_jsonl(tmp_path) -> None:
    teacher_bank = np.stack([_teacher_centers(), _teacher_centers()[::-1]], axis=0)

    pt_path = tmp_path / "teacher_layouts.pt"
    jsonl_path = tmp_path / "teacher_layouts.jsonl"
    save_teacher_layouts(teacher_bank, pt_path, metadata={"seed": 7})
    save_teacher_layouts(teacher_bank, jsonl_path, metadata={"seed": 7})

    pt_loaded = load_teacher_layouts(pt_path)
    jsonl_loaded = load_teacher_layouts(jsonl_path)

    assert pt_loaded.shape == teacher_bank.shape
    assert jsonl_loaded.shape == teacher_bank.shape
    assert np.allclose(pt_loaded, teacher_bank)
    assert np.allclose(jsonl_loaded, teacher_bank)


def test_teacher_layout_loading_rejects_bad_files(tmp_path) -> None:
    unsupported_path = tmp_path / "teacher_layouts.txt"
    unsupported_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.pt or \.jsonl"):
        load_teacher_layouts(unsupported_path)

    missing_centers_path = tmp_path / "missing_centers.jsonl"
    missing_centers_path.write_text('{"layout_index": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing centers/circles"):
        load_teacher_layouts(missing_centers_path)

    invalid_shape_path = tmp_path / "invalid_shape.pt"
    torch.save({"teacher_layouts": torch.zeros(2, 4, 1)}, invalid_shape_path)
    with pytest.raises(ValueError, match=r"shape \[num_layouts, K, 2\]"):
        load_teacher_layouts(invalid_shape_path)


def test_generate_synthetic_teacher_layouts_matches_final_rollout_state(make_config) -> None:
    config = make_config(K=4, max_steps=4, activation_budget=1, policy_name="random_legal")
    generated = generate_synthetic_teacher_layouts(config, count=1, seed=9)

    rollout_config = replace(config, project=replace(config.project, dtype="float64"))
    env = CirclePackingEnv(rollout_config)
    policy = make_policy(rollout_config.rollout.policy_name, rollout_config)
    collect_episode(env, policy, seed=9, record_snapshots=False)
    if not bool(np.all(env.active_mask)):
        env.greedy_fill_masked_slots()
    final_obs = env.observe()

    assert generated.shape == (1, config.problem.K, 2)
    assert np.allclose(generated[0], final_obs["centers"].astype(config.numpy_dtype))
