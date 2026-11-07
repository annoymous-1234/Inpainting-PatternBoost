"""Tests for model-side output heads and fixed-slot zeroing behavior."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from circle_pack.config import ModelSection
from models.heads import DDPMHead, FMHead, SimpleProposalHead, ValueHead
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


def test_head_output_shapes(make_config) -> None:
    config = make_config(K=5, model=_tiny_model_section())
    slot_hidden = torch.randn(2, 5, config.model.d_model, dtype=config.torch_dtype)
    fixed_mask = torch.tensor([[True, False, True, False, False], [False, True, False, True, False]])

    ddpm_head = DDPMHead(config.model.d_model, target="eps", zero_fixed_outputs=False).to(dtype=config.torch_dtype)
    fm_head = FMHead(config.model.d_model, zero_fixed_outputs=False).to(dtype=config.torch_dtype)
    proposal_head = SimpleProposalHead(
        config.model.d_model,
        target="delta_z",
        zero_fixed_outputs=False,
    ).to(dtype=config.torch_dtype)
    value_head = ValueHead(config.model.d_model, config.model.activation).to(dtype=config.torch_dtype)

    assert ddpm_head(slot_hidden, fixed_mask=fixed_mask).shape == (2, 5, 2)
    assert fm_head(slot_hidden, fixed_mask=fixed_mask).shape == (2, 5, 2)
    assert proposal_head(slot_hidden, fixed_mask=fixed_mask).shape == (2, 5, 2)
    assert value_head(torch.randn(2, config.model.d_model, dtype=config.torch_dtype)).shape == (2, 1)


@pytest.mark.parametrize(
    ("head_ctor", "ctor_kwargs"),
    [
        (DDPMHead, {"target": "eps"}),
        (FMHead, {}),
        (SimpleProposalHead, {"target": "delta_z"}),
    ],
)
def test_zero_fixed_outputs_for_slot_heads(make_config, head_ctor, ctor_kwargs) -> None:
    config = make_config(K=5, model=_tiny_model_section())
    slot_hidden = torch.randn(2, 5, config.model.d_model, dtype=config.torch_dtype)
    fixed_mask = torch.tensor([[True, False, True, False, False], [False, True, False, True, False]])

    raw_head = head_ctor(config.model.d_model, zero_fixed_outputs=False, **ctor_kwargs).to(dtype=config.torch_dtype)
    zeroed_head = head_ctor(config.model.d_model, zero_fixed_outputs=True, **ctor_kwargs).to(dtype=config.torch_dtype)
    zeroed_head.load_state_dict(raw_head.state_dict())

    raw_output = raw_head(slot_hidden, fixed_mask=fixed_mask)
    zeroed_output = zeroed_head(slot_hidden, fixed_mask=fixed_mask)

    assert torch.all(zeroed_output[fixed_mask] == 0)
    torch.testing.assert_close(zeroed_output[~fixed_mask], raw_output[~fixed_mask])


class _SpyValueHead(nn.Module):
    """Small test double that captures the value-head input tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        self.last_input = hidden_state.detach().clone()
        return hidden_state.sum(dim=-1, keepdim=True)


def test_value_readout_uses_global_token_or_mean_pool(make_config) -> None:
    config_with_global = make_config(K=4, model=_tiny_model_section(use_global_token=True))
    model_with_global = CirclePackingModel(config_with_global)
    spy_global = _SpyValueHead()
    model_with_global.value_head = spy_global

    inputs = torch.randn(2, 4, 2, dtype=config_with_global.torch_dtype)
    centers = torch.sigmoid(torch.randn(2, 4, 2, dtype=config_with_global.torch_dtype))
    fixed_mask = torch.tensor([[True, False, True, False], [False, False, True, True]])
    outputs_global = model_with_global(
        inputs,
        centers,
        fixed_mask,
        inner_time=torch.tensor([0.1, 0.9], dtype=config_with_global.torch_dtype),
        predict_value=True,
        return_hidden=True,
    )

    assert spy_global.last_input is not None
    torch.testing.assert_close(spy_global.last_input, outputs_global["global_hidden"])

    config_no_global = make_config(K=4, model=_tiny_model_section(use_global_token=False))
    model_no_global = CirclePackingModel(config_no_global)
    spy_no_global = _SpyValueHead()
    model_no_global.value_head = spy_no_global
    outputs_no_global = model_no_global(
        inputs,
        centers,
        fixed_mask,
        inner_time=torch.tensor([0.1, 0.9], dtype=config_no_global.torch_dtype),
        predict_value=True,
        return_hidden=True,
    )

    assert spy_no_global.last_input is not None
    assert outputs_no_global["slot_hidden"] is not None
    torch.testing.assert_close(spy_no_global.last_input, outputs_no_global["slot_hidden"].mean(dim=1))
