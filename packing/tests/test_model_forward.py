"""Tests for model/backbone forward shapes and hidden-state behavior."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circle_pack.config import ModelSection
from models.model import CirclePackingModel


def _tiny_model_section(**overrides) -> ModelSection:
    """Return the validated tiny preset with optional overrides."""

    return ModelSection(
        preset="tiny",
        d_model=192,
        n_heads=6,
        n_layers=4,
        mlp_ratio=4.0,
        dropout=0.0,
        **overrides,
    )


def _make_model_inputs(config, batch_size: int = 2):
    """Create random center-only model inputs for forward tests."""

    dtype = config.torch_dtype
    generator = torch.Generator().manual_seed(2)
    slot_latent = torch.randn(batch_size, config.problem.K, 2, dtype=dtype, generator=generator)
    slot_centers = torch.rand(batch_size, config.problem.K, 2, dtype=dtype, generator=generator)
    fixed_mask = torch.rand(batch_size, config.problem.K, generator=generator) > 0.5
    return slot_latent, slot_centers, fixed_mask


def test_model_forward_with_global_token_and_hidden_return(make_config) -> None:
    config = make_config(K=5, model=_tiny_model_section())
    model = CirclePackingModel(config)
    slot_latent, slot_centers, fixed_mask = _make_model_inputs(config, batch_size=2)

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=torch.tensor([0.1, 0.7], dtype=config.torch_dtype),
        predict_ddpm=True,
        predict_fm=True,
        predict_proposal=True,
        predict_value=True,
        return_hidden=True,
    )

    assert outputs["ddpm_pred"] is not None and outputs["ddpm_pred"].shape == (2, 5, 2)
    assert outputs["fm_pred"] is not None and outputs["fm_pred"].shape == (2, 5, 2)
    assert outputs["proposal_latent"] is not None and outputs["proposal_latent"].shape == (2, 5, 2)
    assert outputs["value"] is not None and outputs["value"].shape == (2, 1)
    assert outputs["slot_hidden"] is not None and outputs["slot_hidden"].shape == (2, 5, 192)
    assert outputs["global_hidden"] is not None and outputs["global_hidden"].shape == (2, 192)


def test_model_forward_without_global_token(make_config) -> None:
    config = make_config(K=4, model=_tiny_model_section(use_global_token=False))
    model = CirclePackingModel(config)
    slot_latent, slot_centers, fixed_mask = _make_model_inputs(config, batch_size=2)

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=torch.tensor([0.2, 0.4], dtype=config.torch_dtype),
        predict_value=True,
        return_hidden=True,
    )

    assert outputs["value"] is not None and outputs["value"].shape == (2, 1)
    assert outputs["slot_hidden"] is not None and outputs["slot_hidden"].shape == (2, 4, 192)
    assert outputs["global_hidden"] is None


def test_time_embedding_inputs_accept_both_vector_and_column_shapes(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_section(use_outer_step_embedding=True, use_inner_time_embedding=True),
    )
    model = CirclePackingModel(config)
    slot_latent, slot_centers, fixed_mask = _make_model_inputs(config, batch_size=2)

    outputs_a = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        outer_step=torch.tensor([1.0, 2.0], dtype=config.torch_dtype),
        inner_time=torch.tensor([[0.1], [0.9]], dtype=config.torch_dtype),
        predict_proposal=True,
    )
    outputs_b = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        outer_step=torch.tensor([[1.0], [2.0]], dtype=config.torch_dtype),
        inner_time=torch.tensor([0.1, 0.9], dtype=config.torch_dtype),
        predict_proposal=True,
    )

    assert outputs_a["proposal_latent"] is not None and outputs_a["proposal_latent"].shape == (2, 4, 2)
    assert outputs_b["proposal_latent"] is not None and outputs_b["proposal_latent"].shape == (2, 4, 2)


@pytest.mark.parametrize("embedding_type", ["learned", "sin"])
def test_outer_step_embedding_can_infer_from_fixed_mask(make_config, embedding_type: str) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_section(
            use_global_token=False,
            use_outer_step_embedding=True,
            outer_step_embedding_type=embedding_type,
            use_inner_time_embedding=True,
        ),
    )
    model = CirclePackingModel(config)
    slot_latent, slot_centers, fixed_mask = _make_model_inputs(config, batch_size=2)
    inner_time = torch.tensor([0.1, 0.9], dtype=config.torch_dtype)

    outputs_auto = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        return_hidden=True,
    )
    outputs_explicit = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        outer_step=fixed_mask.to(dtype=config.torch_dtype).mean(dim=1),
        inner_time=inner_time,
        return_hidden=True,
    )

    assert outputs_auto["slot_hidden"] is not None
    assert outputs_explicit["slot_hidden"] is not None
    torch.testing.assert_close(outputs_auto["slot_hidden"], outputs_explicit["slot_hidden"])


def test_outer_step_embedding_changes_with_fixed_slot_fraction(make_config) -> None:
    config = make_config(
        K=4,
        model=_tiny_model_section(
            use_global_token=False,
            use_outer_step_embedding=True,
            outer_step_embedding_type="learned",
            use_inner_time_embedding=True,
        ),
    )
    model = CirclePackingModel(config)
    slot_latent, slot_centers, _ = _make_model_inputs(config, batch_size=1)
    fixed_none = torch.zeros(1, config.problem.K, dtype=torch.bool)
    fixed_all = torch.ones(1, config.problem.K, dtype=torch.bool)
    inner_time = torch.tensor([0.5], dtype=config.torch_dtype)

    outputs_none = model(slot_latent, slot_centers, fixed_none, inner_time=inner_time, return_hidden=True)
    outputs_all = model(slot_latent, slot_centers, fixed_all, inner_time=inner_time, return_hidden=True)

    assert outputs_none["slot_hidden"] is not None
    assert outputs_all["slot_hidden"] is not None
    assert not torch.allclose(outputs_none["slot_hidden"], outputs_all["slot_hidden"])
