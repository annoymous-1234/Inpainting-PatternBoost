"""Tests for model-side slot feature assembly."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from circle_pack.config import ModelSection, ModelSlotFeaturesSection
from models.features import SlotFeatureBuilder


def _make_slot_inputs(batch_size: int, num_slots: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create lightweight center-only slot inputs."""

    generator = torch.Generator().manual_seed(0)
    slot_latent = torch.randn(batch_size, num_slots, 2, dtype=dtype, generator=generator)
    slot_centers = torch.rand(batch_size, num_slots, 2, dtype=dtype, generator=generator)
    fixed_mask = torch.tensor([[True, False, True, False], [False, False, True, True]], dtype=torch.bool)
    return slot_latent, slot_centers, fixed_mask


def test_default_slot_feature_builder_shape_and_order(make_config) -> None:
    config = make_config(K=4)
    builder = SlotFeatureBuilder(config.model.slot_features)
    slot_latent, slot_centers, fixed_mask = _make_slot_inputs(2, 4, config.torch_dtype)

    raw_features = builder(slot_latent, slot_centers, fixed_mask)

    assert raw_features.shape == (2, 4, 2)
    assert builder.feature_names == (
        "latent_u",
        "latent_v",
    )
    assert builder.feature_dim == 2


def test_slot_feature_builder_ablation_shape(make_config) -> None:
    slot_cfg = ModelSlotFeaturesSection(
        use_fixed_flag=True,
        use_latent=False,
        use_physical=False,
        use_wall_slack=True,
        use_extra_slot_features=False,
        extra_slot_feature_dim=0,
    )
    model_cfg = replace(
        config := make_config(K=4).model,
        preset="custom",
        slot_features=slot_cfg,
    )
    config = make_config(K=4, model=model_cfg)
    builder = SlotFeatureBuilder(config.model.slot_features)
    slot_latent, slot_centers, fixed_mask = _make_slot_inputs(2, 4, config.torch_dtype)

    raw_features = builder(slot_latent, slot_centers, fixed_mask)

    assert raw_features.shape == (2, 4, 5)
    assert builder.feature_names == (
        "fixed_flag",
        "left_wall_distance",
        "right_wall_distance",
        "bottom_wall_distance",
        "top_wall_distance",
    )


def test_slot_feature_builder_extra_feature_dimension_enforced(make_config) -> None:
    slot_cfg = ModelSlotFeaturesSection(
        use_fixed_flag=True,
        use_latent=False,
        use_physical=False,
        use_wall_slack=False,
        use_extra_slot_features=True,
        extra_slot_feature_dim=5,
    )
    model_cfg = replace(make_config(K=4).model, preset="custom", slot_features=slot_cfg)
    config = make_config(K=4, model=model_cfg)
    builder = SlotFeatureBuilder(config.model.slot_features)
    slot_latent, slot_centers, fixed_mask = _make_slot_inputs(2, 4, config.torch_dtype)

    with pytest.raises(ValueError):
        builder(
            slot_latent,
            slot_centers,
            fixed_mask,
            extra_slot_features=torch.randn(2, 4, 4, dtype=config.torch_dtype),
        )

    raw_features = builder(
        slot_latent,
        slot_centers,
        fixed_mask,
        extra_slot_features=torch.randn(2, 4, 5, dtype=config.torch_dtype),
    )
    assert raw_features.shape == (2, 4, 6)
