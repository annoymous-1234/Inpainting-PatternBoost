"""Tests for the supervised FM loss wrapper and FM proposal sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from circle_pack.config import FMSection, ModelSection, ModelSlotFeaturesSection
from circle_pack.geometry import center_to_latent_approx, latent_to_center_simple
from models.factory import build_model
from models.objectives import fm_supervised_loss
from models.sampling import sample_fm_proposals, sample_fm_proposals_batched
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


def _tiny_model_with_draft_features() -> ModelSection:
    return replace(
        _tiny_model(),
        slot_features=ModelSlotFeaturesSection(
            use_fixed_flag=False,
            use_latent=True,
            use_physical=False,
            use_wall_slack=False,
            use_extra_slot_features=True,
            extra_slot_feature_dim=4,
        ),
    )


def _teacher_layout() -> np.ndarray:
    return np.array(
        [
            [0.16, 0.22, 0.06],
            [0.42, 0.58, 0.07],
            [0.67, 0.36, 0.09],
            [0.85, 0.81, 0.03],
        ],
        dtype=np.float64,
    )


def _stack_examples(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in examples[0]:
        batch[key] = torch.stack([example[key] for example in examples], dim=0)
    return batch


def test_fm_supervised_loss_runs(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        fm=FMSection(
            num_samples=2,
            num_steps=8,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
        ),
    )
    model = build_model(config)
    teacher = _teacher_layout()
    batch = _stack_examples(
        [
            make_supervised_example(teacher, mode="fm", cfg=config, rng=np.random.default_rng(4)),
            make_supervised_example(teacher[::-1], mode="fm", cfg=config, rng=np.random.default_rng(5)),
        ]
    )

    loss, stats = fm_supervised_loss(model, batch, config)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert stats["free_slot_count"] >= 0.0


def test_fm_supervised_loss_runs_with_outer_step_embedding(make_config) -> None:
    config = make_config(
        K=4,
        model=replace(
            _tiny_model(),
            use_outer_step_embedding=True,
            outer_step_embedding_type="learned",
        ),
        fm=FMSection(
            num_samples=2,
            num_steps=8,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
        ),
    )
    model = build_model(config)
    teacher = _teacher_layout()
    batch = _stack_examples(
        [
            make_supervised_example(teacher, mode="fm", cfg=config, rng=np.random.default_rng(14)),
            make_supervised_example(teacher[::-1], mode="fm", cfg=config, rng=np.random.default_rng(15)),
        ]
    )

    loss, stats = fm_supervised_loss(model, batch, config)

    assert torch.isfinite(loss)
    assert stats["free_slot_count"] >= 0.0


def test_fm_supervised_loss_runs_with_draft_features(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_with_draft_features(),
        fm=FMSection(
            num_samples=2,
            num_steps=6,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
            use_draft_feature=True,
        ),
    )
    model = build_model(config)
    teacher = _teacher_layout()
    batch = _stack_examples(
        [
            make_supervised_example(teacher, mode="fm", cfg=config, rng=np.random.default_rng(24)),
            make_supervised_example(teacher[::-1], mode="fm", cfg=config, rng=np.random.default_rng(25)),
        ]
    )

    loss, stats = fm_supervised_loss(model, batch, config)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["free_slot_count"] >= 0.0
    assert batch["extra_slot_features"].shape == (2, 4, 4)


def test_fm_supervised_loss_rejects_unserved_extra_slot_features(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_with_draft_features(),
        fm=FMSection(
            num_samples=2,
            num_steps=6,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
            use_draft_feature=False,
        ),
    )
    model = build_model(config)
    teacher = _teacher_layout()
    data_config = replace(config, model=_tiny_model())
    batch = _stack_examples(
        [
            make_supervised_example(teacher, mode="fm", cfg=data_config, rng=np.random.default_rng(26)),
            make_supervised_example(teacher[::-1], mode="fm", cfg=data_config, rng=np.random.default_rng(27)),
        ]
    )

    with pytest.raises(ValueError, match="fm.use_draft_feature"):
        fm_supervised_loss(model, batch, config)


@pytest.mark.parametrize("fixed_mode", ["conditional", "frozen_full_state"])
def test_fm_sampler_returns_proposal_bundle(make_config, fixed_mode: str) -> None:
    config = make_config(
        K=4,
        model=_tiny_model(),
        fm=FMSection(
            num_samples=3,
            num_steps=6,
            source_noise_std=0.05,
            fixed_mode=fixed_mode,
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_fm_proposals(
        model,
        latent,
        centers,
        fixed_mask,
        config,
        latent_to_center_simple,
    )

    assert bundle.latent_samples.shape == (config.fm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (config.fm.num_samples, 4, 2)
    assert bundle.aux["terminal_velocity_norm"].shape == (config.fm.num_samples, 4)
    assert bundle.aux["short_displacement"].shape == (config.fm.num_samples, 4)
    assert torch.allclose(bundle.latent_samples[:, 0], latent.expand(config.fm.num_samples, -1, -1)[:, 0])
    assert torch.allclose(bundle.center_samples[:, 2], centers.expand(config.fm.num_samples, -1, -1)[:, 2])


def test_fm_sampler_runs_with_outer_step_embedding(make_config) -> None:
    config = make_config(
        K=4,
        model=replace(
            _tiny_model(),
            use_outer_step_embedding=True,
            outer_step_embedding_type="sin",
        ),
        fm=FMSection(
            num_samples=2,
            num_steps=6,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_fm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (config.fm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (config.fm.num_samples, 4, 2)


def test_fm_sampler_runs_with_draft_features(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_with_draft_features(),
        fm=FMSection(
            num_samples=2,
            num_steps=5,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
            use_draft_feature=True,
        ),
    )
    model = build_model(config)
    centers = torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype).unsqueeze(0)
    latent = torch.as_tensor(center_to_latent_approx(centers.squeeze(0).numpy()), dtype=config.torch_dtype).unsqueeze(0)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    bundle = sample_fm_proposals(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (config.fm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (config.fm.num_samples, 4, 2)
    assert torch.allclose(bundle.latent_samples[:, 0], latent.expand(config.fm.num_samples, -1, -1)[:, 0])
    assert torch.allclose(bundle.center_samples[:, 2], centers.expand(config.fm.num_samples, -1, -1)[:, 2])


def test_batched_fm_sampler_returns_batched_bundle(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_with_draft_features(),
        fm=FMSection(
            num_samples=3,
            num_steps=5,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
            use_draft_feature=True,
        ),
    )
    model = build_model(config)
    centers = torch.stack(
        [
            torch.as_tensor(_teacher_layout()[:, :2], dtype=config.torch_dtype),
            torch.as_tensor(_teacher_layout()[::-1, :2].copy(), dtype=config.torch_dtype),
        ],
        dim=0,
    )
    latent = torch.as_tensor(center_to_latent_approx(centers.numpy()), dtype=config.torch_dtype)
    fixed_mask = torch.tensor([[True, False, True, False], [False, True, False, True]], dtype=torch.bool)

    bundle = sample_fm_proposals_batched(model, latent, centers, fixed_mask, config, latent_to_center_simple)

    assert bundle.latent_samples.shape == (2, config.fm.num_samples, 4, 2)
    assert bundle.center_samples.shape == (2, config.fm.num_samples, 4, 2)
    assert bundle.aux["terminal_velocity_norm"].shape == (2, config.fm.num_samples, 4)
    assert bundle.aux["short_displacement"].shape == (2, config.fm.num_samples, 4)
    for batch_idx in range(2):
        for slot_idx in torch.where(fixed_mask[batch_idx])[0].tolist():
            expected_latent = latent[batch_idx, slot_idx].expand(config.fm.num_samples, -1)
            expected_center = centers[batch_idx, slot_idx].expand(config.fm.num_samples, -1)
            assert torch.allclose(bundle.latent_samples[batch_idx, :, slot_idx], expected_latent)
            assert torch.allclose(bundle.center_samples[batch_idx, :, slot_idx], expected_center)
