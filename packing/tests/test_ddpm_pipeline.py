"""Tests for the supervised DDPM loss wrapper and DDPM proposal sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from circle_pack.config import DDPMSection, ModelSection, ModelSlotFeaturesSection
from circle_pack.geometry import center_to_latent_approx, latent_to_center_simple
from models.factory import build_model
from models.objectives import ddpm_supervised_loss
from models.sampling import sample_ddpm_proposals, sample_ddpm_proposals_batched
from pipelines.data import make_supervised_example


def _tiny_model() -> ModelSection:
    return ModelSection(
        preset="tiny",
        d_model=192,
        n_heads=6,
        n_layers=4,
        mlp_ratio=4.0,
        dropout=0.0,
    )


def _teacher_layout() -> np.ndarray:
    return np.array(
        [
            [0.18, 0.18, 0.05],
            [0.35, 0.60, 0.08],
            [0.70, 0.40, 0.10],
            [0.82, 0.80, 0.04],
        ],
        dtype=np.float64,
    )


def _stack_examples(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in examples[0]:
        batch[key] = torch.stack([example[key] for example in examples], dim=0)
    return batch


def test_ddpm_supervised_loss_runs(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=8,
            beta_start=1.0e-4,
            beta_end=0.02,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
        ),
    )
    model = build_model(config)
    teacher = _teacher_layout()
    batch = _stack_examples(
        [
            make_supervised_example(teacher, mode="ddpm", cfg=config, rng=np.random.default_rng(1)),
            make_supervised_example(teacher[::-1], mode="ddpm", cfg=config, rng=np.random.default_rng(2)),
        ]
    )

    loss, stats = ddpm_supervised_loss(model, batch, config)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert stats["free_slot_count"] >= 0.0


def test_ddpm_supervised_loss_runs_with_outer_step_embedding(make_config) -> None:
    config = make_config(
        K=4,
        model=replace(
            _tiny_model(),
            use_outer_step_embedding=True,
            outer_step_embedding_type="learned",
        ),
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
            draft_strength=1.0,
            use_timestep_subsampling=False,
            infer_num_steps=4,
        ),
    )
    model = build_model(config)
    teacher = _teacher_layout()
    batch = _stack_examples(
        [
            make_supervised_example(teacher, mode="ddpm", cfg=config, rng=np.random.default_rng(11)),
            make_supervised_example(teacher[::-1], mode="ddpm", cfg=config, rng=np.random.default_rng(12)),
        ]
    )

    loss, stats = ddpm_supervised_loss(model, batch, config)

    assert torch.isfinite(loss)
    assert stats["free_slot_count"] >= 0.0


@pytest.mark.parametrize("fixed_mode", ["conditional", "frozen_full_state"])
def test_ddpm_sampler_returns_proposal_bundle(make_config, fixed_mode: str) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        ddpm=DDPMSection(
            num_samples=3,
            num_steps=6,
            beta_start=1.0e-4,
            beta_end=0.02,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode=fixed_mode,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_ddpm_proposals(
        model,
        latent,
        centers,
        fixed_mask,
        config,
        latent_to_center_simple,
    )

    assert bundle.latent_samples.shape == (config.ddpm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (config.ddpm.num_samples, 4, 2)
    assert bundle.aux["step_trace"].ndim == 1
    assert torch.allclose(bundle.latent_samples[:, 0], latent.expand(config.ddpm.num_samples, -1, -1)[:, 0])
    assert torch.allclose(bundle.center_samples[:, 2], centers.expand(config.ddpm.num_samples, -1, -1)[:, 2])


def test_batched_ddpm_sampler_returns_batched_bundle(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=5,
            beta_start=1.0e-4,
            beta_end=0.02,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
        ),
    )
    model = build_model(config)
    centers_one = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype)
    centers = torch.stack((centers_one, torch.flip(centers_one, dims=(0,)), centers_one.roll(1, dims=0)), dim=0)
    latent = torch.as_tensor(center_to_latent_approx(centers.numpy()), dtype=config.torch_dtype)
    fixed_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    bundle = sample_ddpm_proposals_batched(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (3, config.ddpm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (3, config.ddpm.num_samples, 4, 2)
    assert bundle.aux["step_trace"].ndim == 1
    for batch_idx in range(3):
        for slot_idx in fixed_mask[batch_idx].nonzero(as_tuple=False).flatten().tolist():
            expected_latent = latent[batch_idx, slot_idx].expand(config.ddpm.num_samples, -1)
            expected_center = centers[batch_idx, slot_idx].expand(config.ddpm.num_samples, -1)
            assert torch.allclose(bundle.latent_samples[batch_idx, :, slot_idx], expected_latent)
            assert torch.allclose(bundle.center_samples[batch_idx, :, slot_idx], expected_center)


def test_batched_ddpm_sampler_runs_with_draft_features(make_config) -> None:
    slot_cfg = ModelSlotFeaturesSection(
        use_fixed_flag=False,
        use_latent=True,
        use_physical=False,
        use_wall_slack=False,
        use_extra_slot_features=True,
        extra_slot_feature_dim=4,
    )
    config = make_config(
        K=4,
        model=replace(_tiny_model(), slot_features=slot_cfg),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=4,
            beta_start=1.0e-4,
            beta_end=0.02,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=True,
            fixed_mode="frozen_full_state",
        ),
    )
    model = build_model(config)
    centers_one = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype)
    centers = torch.stack((centers_one, torch.flip(centers_one, dims=(0,))), dim=0)
    latent = torch.as_tensor(center_to_latent_approx(centers.numpy()), dtype=config.torch_dtype)
    fixed_mask = torch.tensor([[True, False, False, True], [False, True, True, False]], dtype=torch.bool)

    bundle = sample_ddpm_proposals_batched(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (2, config.ddpm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (2, config.ddpm.num_samples, 4, 2)


def test_ddpm_sampler_runs_with_outer_step_embedding(make_config) -> None:
    config = make_config(
        K=4,
        model=replace(
            _tiny_model(),
            use_outer_step_embedding=True,
            outer_step_embedding_type="sin",
        ),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=6,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
            noising_center="draft",
            draft_strength=1.0,
            use_timestep_subsampling=False,
            infer_num_steps=3,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_ddpm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (config.ddpm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (config.ddpm.num_samples, 4, 2)


@pytest.mark.parametrize("draft_strength", [0.0, 0.5, 1.0])
def test_ddpm_sampler_draft_strength_modes_run(make_config, draft_strength: float) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=6,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
            noising_center="draft",
            draft_strength=draft_strength,
            use_timestep_subsampling=False,
            infer_num_steps=3,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[False, True, False, True]], dtype=torch.bool)

    bundle = sample_ddpm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (config.ddpm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (config.ddpm.num_samples, 4, 2)
    assert torch.allclose(bundle.latent_samples[:, 1], latent.expand(config.ddpm.num_samples, -1, -1)[:, 1])
    assert torch.allclose(bundle.center_samples[:, 3], centers.expand(config.ddpm.num_samples, -1, -1)[:, 3])


def test_ddpm_sampler_timestep_subsampling_trace(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
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
            draft_strength=1.0,
            use_timestep_subsampling=True,
            infer_num_steps=4,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_ddpm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)
    trace = bundle.aux["step_trace"].detach().cpu().tolist()

    assert len(trace) == config.ddpm.infer_num_steps
    assert trace[0] == config.ddpm.num_steps - 1
    assert trace[-1] == 0
    assert all(left >= right for left, right in zip(trace, trace[1:]))


def test_ddpm_sampler_tau_start_controls_start_step(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=8,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=0.5,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
            noising_center="draft",
            draft_strength=1.0,
            use_timestep_subsampling=False,
            infer_num_steps=4,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_ddpm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert int(bundle.aux["start_step"].item()) == round(config.ddpm.tau_start_infer * (config.ddpm.num_steps - 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_ddpm_sampler_cuda_smoke(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        ddpm=DDPMSection(
            num_samples=2,
            num_steps=4,
            beta_start=1.0e-4,
            beta_end=0.02,
            tau_start_infer=1.0,
            use_draft_warmstart=True,
            use_draft_feature=False,
            fixed_mode="frozen_full_state",
        ),
    )
    model = build_model(config).to(device=torch.device("cuda"))
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype, device=model.device).unsqueeze(0)
    latent = torch.as_tensor(
        center_to_latent_approx(centers.squeeze(0).detach().cpu().numpy()),
        dtype=config.torch_dtype,
        device=model.device,
    ).unsqueeze(0)
    fixed_mask = torch.tensor([[False, True, False, False]], dtype=torch.bool, device=model.device)

    bundle = sample_ddpm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.is_cuda
    assert bundle.center_samples.is_cuda
