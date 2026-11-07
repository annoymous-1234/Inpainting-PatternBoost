"""Feature builders and simple input projections for the model stack."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from circle_pack.config import ModelSlotFeaturesSection


def build_activation(name: str) -> nn.Module:
    """Return a lightweight activation module by name."""

    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation name: {name!r}.")


def _torch_center_wall_distances(slot_centers: torch.Tensor) -> torch.Tensor:
    """Return distances from each center to unit-square walls.

    Parameters
    ----------
    slot_centers:
        Tensor of shape `[B, K, 2]` storing `(x, y)`.

    Returns
    -------
    torch.Tensor
        Tensor of shape `[B, K, 4]` with `(x, 1-x, y, 1-y)` for each slot.
    """

    x = slot_centers[..., 0:1]
    y = slot_centers[..., 1:2]
    return torch.cat((x, 1.0 - x, y, 1.0 - y), dim=-1)


class SlotFeatureBuilder(nn.Module):
    """Assemble raw per-slot features from latent, physical, and role inputs.

    Feature order is stable and documented through the `feature_names` property:

    1. fixed/free role flag
    2. latent `(u, v)`
    3. physical center `(x, y)`
    4. center wall distances `(left, right, bottom, top)`
    5. extra slot features
    """

    def __init__(self, config: ModelSlotFeaturesSection) -> None:
        super().__init__()
        self.config = config
        self.feature_names = self._build_feature_names()

    @property
    def feature_dim(self) -> int:
        """Return the total raw slot feature dimension."""

        return len(self.feature_names)

    def _build_feature_names(self) -> tuple[str, ...]:
        """Return the stable feature ordering implied by the config toggles."""

        names: list[str] = []
        if self.config.use_fixed_flag:
            names.append("fixed_flag")
        if self.config.use_latent:
            names.extend(("latent_u", "latent_v"))
        if self.config.use_physical:
            names.extend(("physical_x", "physical_y"))
        if self.config.use_wall_slack:
            names.extend(("left_wall_distance", "right_wall_distance", "bottom_wall_distance", "top_wall_distance"))
        if self.config.use_extra_slot_features:
            names.extend(
                f"extra_slot_feature_{idx}" for idx in range(int(self.config.extra_slot_feature_dim))
            )
        return tuple(names)

    def _append_feature_block(
        self,
        blocks: list[torch.Tensor],
        names: list[str],
        tensor: torch.Tensor,
        block_names: Iterable[str],
    ) -> None:
        """Append one feature block and extend the corresponding name list."""

        blocks.append(tensor)
        names.extend(block_names)

    def forward(
        self,
        slot_latent: torch.Tensor,
        slot_centers: torch.Tensor,
        fixed_mask: torch.Tensor,
        extra_slot_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build raw slot features of shape `[B, K, D_in]`.

        Parameters
        ----------
        slot_latent:
            Tensor of shape `[B, K, 2]` storing latent slot vectors `(u, v)`.
        slot_centers:
            Tensor of shape `[B, K, 2]` storing physical slot centers `(x, y)`.
        fixed_mask:
            Boolean tensor of shape `[B, K]` where `True` marks fixed slots.
        extra_slot_features:
            Optional tensor of shape `[B, K, D_extra]` that is concatenated only
            when enabled by config.
        """

        if slot_latent.shape != slot_centers.shape:
            raise ValueError("slot_latent and slot_centers must have identical shapes.")
        if slot_latent.ndim != 3 or slot_latent.shape[-1] != 2:
            raise ValueError("slot_latent and slot_centers must both have shape [B, K, 2].")
        if fixed_mask.shape != slot_latent.shape[:2]:
            raise ValueError("fixed_mask must have shape [B, K] matching the slot tensors.")

        dtype = slot_latent.dtype
        blocks: list[torch.Tensor] = []
        names: list[str] = []

        if self.config.use_fixed_flag:
            self._append_feature_block(
                blocks,
                names,
                fixed_mask.to(dtype=dtype).unsqueeze(-1),
                ("fixed_flag",),
            )
        if self.config.use_latent:
            self._append_feature_block(
                blocks,
                names,
                slot_latent,
                ("latent_u", "latent_v"),
            )
        if self.config.use_physical:
            self._append_feature_block(
                blocks,
                names,
                slot_centers,
                ("physical_x", "physical_y"),
            )
        if self.config.use_wall_slack:
            self._append_feature_block(
                blocks,
                names,
                _torch_center_wall_distances(slot_centers),
                ("left_wall_distance", "right_wall_distance", "bottom_wall_distance", "top_wall_distance"),
            )
        if self.config.use_extra_slot_features:
            if extra_slot_features is None:
                raise ValueError("extra_slot_features must be provided when enabled in config.")
            if extra_slot_features.shape[:2] != slot_latent.shape[:2]:
                raise ValueError("extra_slot_features must have leading shape [B, K].")
            expected_dim = int(self.config.extra_slot_feature_dim)
            if extra_slot_features.shape[-1] != expected_dim:
                raise ValueError(
                    f"extra_slot_features must have trailing dimension {expected_dim}, "
                    f"got {extra_slot_features.shape[-1]}."
                )
            self._append_feature_block(
                blocks,
                names,
                extra_slot_features.to(dtype=dtype),
                tuple(f"extra_slot_feature_{idx}" for idx in range(expected_dim)),
            )

        if tuple(names) != self.feature_names:
            raise AssertionError("SlotFeatureBuilder feature ordering drifted from the documented config.")
        return torch.cat(blocks, dim=-1)


class SlotInputProjector(nn.Module):
    """Project raw slot features into the Transformer model width."""

    def __init__(self, input_dim: int, d_model: int, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            build_activation(activation),
            nn.Linear(d_model, d_model),
        )

    def forward(self, raw_slot_features: torch.Tensor) -> torch.Tensor:
        """Project raw features from `[B, K, D_in]` to `[B, K, d_model]`."""

        return self.net(raw_slot_features)


class ConditionEmbedder(nn.Module):
    """Embed scalar per-batch conditioning inputs into `[B, d_model]` vectors."""

    def __init__(self, d_model: int, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            build_activation(activation),
            nn.Linear(d_model, d_model),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Embed `value` with shape `[B]` or `[B, 1]` into `[B, d_model]`."""

        if value.ndim == 1:
            value = value.unsqueeze(-1)
        elif value.ndim != 2 or value.shape[-1] != 1:
            raise ValueError("ConditionEmbedder expects input of shape [B] or [B, 1].")
        weight = self.net[0].weight
        return self.net(value.to(device=weight.device, dtype=weight.dtype))


class SinusoidalConditionEmbedder(nn.Module):
    """Embed scalar per-batch conditioning inputs with fixed sinusoidal features."""

    def __init__(self, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        half_dim = max(self.d_model // 2, 1)
        exponent = -math.log(float(max_period)) * torch.arange(half_dim, dtype=torch.float32) / float(half_dim)
        self.register_buffer("frequencies", torch.exp(exponent), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Embed `value` with shape `[B]` or `[B, 1]` into `[B, d_model]`."""

        if value.ndim == 1:
            value = value.unsqueeze(-1)
        elif value.ndim != 2 or value.shape[-1] != 1:
            raise ValueError("SinusoidalConditionEmbedder expects input of shape [B] or [B, 1].")
        target_dtype = value.dtype if torch.is_floating_point(value) else self.frequencies.dtype
        frequencies = self.frequencies.to(device=value.device, dtype=target_dtype)
        angles = value.to(dtype=target_dtype) * frequencies.unsqueeze(0) * (2.0 * math.pi)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedding.shape[-1] < self.d_model:
            embedding = torch.nn.functional.pad(embedding, (0, self.d_model - embedding.shape[-1]))
        return embedding[..., : self.d_model]
