"""Tests for geometry-aware attention pair bias."""

from __future__ import annotations

from dataclasses import replace

import torch

from circle_pack.config import ModelPairBiasSection
from models.pair_bias import PairBiasBuilder


def _make_pair_inputs(batch_size: int, num_slots: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic physical center inputs for pair-bias tests."""

    generator = torch.Generator().manual_seed(1)
    centers = torch.rand(batch_size, num_slots, 2, dtype=dtype, generator=generator)
    fixed_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool).expand(batch_size, -1)
    return centers, fixed_mask


def test_pair_bias_scalar_shape(make_config) -> None:
    pair_cfg = replace(make_config(K=4).model.pair_bias, enabled=True)
    config = make_config(K=4, model=replace(make_config(K=4).model, preset="custom", pair_bias=pair_cfg))
    builder = PairBiasBuilder(
        config.model.pair_bias,
        hidden_dim=config.model.d_model,
        num_heads=config.model.n_heads,
        use_global_token=False,
        activation=config.model.activation,
    ).to(dtype=config.torch_dtype)
    slot_centers, fixed_mask = _make_pair_inputs(2, 4, config.torch_dtype)

    pair_bias = builder(slot_centers, fixed_mask)

    assert pair_bias is not None
    assert pair_bias.shape == (2, 4, 4, 1)


def test_pair_bias_per_head_shape(make_config) -> None:
    pair_cfg = replace(make_config(K=4).model.pair_bias, enabled=True, pair_bias_mode="per_head")
    model_cfg = replace(make_config(K=4).model, preset="custom", pair_bias=pair_cfg)
    config = make_config(K=4, model=model_cfg)
    builder = PairBiasBuilder(
        config.model.pair_bias,
        hidden_dim=config.model.d_model,
        num_heads=config.model.n_heads,
        use_global_token=False,
        activation=config.model.activation,
    ).to(dtype=config.torch_dtype)
    slot_centers, fixed_mask = _make_pair_inputs(2, 4, config.torch_dtype)

    pair_bias = builder(slot_centers, fixed_mask)

    assert pair_bias is not None
    assert pair_bias.shape == (2, 4, 4, config.model.n_heads)


def test_pair_bias_global_token_padding_is_zero(make_config) -> None:
    pair_cfg = replace(make_config(K=4).model.pair_bias, enabled=True)
    config = make_config(K=4, model=replace(make_config(K=4).model, preset="custom", pair_bias=pair_cfg))
    builder = PairBiasBuilder(
        config.model.pair_bias,
        hidden_dim=config.model.d_model,
        num_heads=config.model.n_heads,
        use_global_token=True,
        activation=config.model.activation,
    ).to(dtype=config.torch_dtype)
    slot_centers, fixed_mask = _make_pair_inputs(1, 4, config.torch_dtype)

    pair_bias = builder(slot_centers, fixed_mask)

    assert pair_bias is not None
    assert pair_bias.shape == (1, 5, 5, 1)
    assert torch.count_nonzero(pair_bias[:, 0, :, :]) == 0
    assert torch.count_nonzero(pair_bias[:, :, 0, :]) == 0
