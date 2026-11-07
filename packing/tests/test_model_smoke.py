"""End-to-end smoke tests for the model-side wrapper and decode helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circle_pack.config import ModelSection, load_config
from circle_pack.geometry import latent_to_centers
from models.factory import build_model


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


def _make_inputs(config, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic model inputs on the requested device."""

    dtype = config.torch_dtype
    generator = torch.Generator(device=device).manual_seed(5)
    slot_latent = torch.randn(batch_size, config.problem.K, 2, dtype=dtype, generator=generator, device=device)
    slot_centers = torch.rand(batch_size, config.problem.K, 2, dtype=dtype, generator=generator, device=device)
    fixed_mask = (torch.rand(batch_size, config.problem.K, generator=generator, device=device) > 0.5).to(torch.bool)
    inner_time = torch.rand(batch_size, dtype=dtype, generator=generator, device=device)
    return slot_latent, slot_centers, fixed_mask, inner_time


def test_model_smoke_cpu(make_config) -> None:
    config = make_config(K=5)
    model = build_model(config)
    slot_latent, slot_centers, fixed_mask, inner_time = _make_inputs(config, batch_size=2, device=torch.device("cpu"))

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        predict_ddpm=True,
        predict_fm=True,
        predict_proposal=True,
        predict_value=True,
        return_hidden=True,
    )
    proposal_latent = model.predict_proposal_latent(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
    )
    proposal_centers = model.predict_proposal_centers_for_test(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        latent_to_center_fn=latent_to_centers,
    )

    assert outputs["ddpm_pred"] is not None and outputs["ddpm_pred"].shape == (2, 5, 2)
    assert outputs["fm_pred"] is not None and outputs["fm_pred"].shape == (2, 5, 2)
    assert outputs["proposal_latent"] is not None and outputs["proposal_latent"].shape == (2, 5, 2)
    assert outputs["value"] is not None and outputs["value"].shape == (2, 1)
    assert outputs["slot_hidden"] is not None and outputs["slot_hidden"].shape == (2, 5, config.model.d_model)
    assert proposal_latent.shape == (2, 5, 2)
    assert proposal_centers.shape == (2, 5, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_model_smoke_cuda(make_config) -> None:
    config = make_config(K=4, model=_tiny_model_section())
    model = build_model(config).to(device=torch.device("cuda"))
    slot_latent, slot_centers, fixed_mask, inner_time = _make_inputs(
        config,
        batch_size=1,
        device=torch.device("cuda"),
    )

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        predict_proposal=True,
        return_hidden=True,
    )
    proposal_latent = model.predict_proposal_latent(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
    )
    proposal_centers = model.predict_proposal_centers_for_test(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
    )

    assert outputs["proposal_latent"] is not None and outputs["proposal_latent"].is_cuda
    assert outputs["slot_hidden"] is not None and outputs["slot_hidden"].is_cuda
    assert proposal_latent.is_cuda
    assert proposal_centers.is_cuda
