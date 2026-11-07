"""Geometry-aware pair-bias builder for full self-attention."""

from __future__ import annotations

import torch
from torch import nn

from circle_pack.config import ModelPairBiasSection
from models.features import build_activation


class PairBiasBuilder(nn.Module):
    """Build optional additive pair bias from physical center geometry.

    Pair features are always derived from `slot_centers` and `fixed_mask`, never
    from latent variables directly.
    """

    def __init__(
        self,
        config: ModelPairBiasSection,
        *,
        hidden_dim: int,
        num_heads: int,
        use_global_token: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = int(num_heads)
        self.use_global_token = bool(use_global_token)
        self.feature_names = self._build_feature_names()
        if self.config.enabled and not self.feature_names:
            raise ValueError("At least one pair feature must be enabled when pair bias is enabled.")

        output_dim = 1 if self.config.pair_bias_mode == "scalar" else self.num_heads
        self.output_dim = output_dim
        self.mlp = None
        if self.config.enabled:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                build_activation(activation),
                nn.Linear(hidden_dim, output_dim),
            )

    @property
    def feature_dim(self) -> int:
        """Return the raw pair feature dimension."""

        return len(self.feature_names)

    def _build_feature_names(self) -> tuple[str, ...]:
        """Return the stable ordering of enabled pair features."""

        names: list[str] = []
        if self.config.use_pair_dxdy:
            names.extend(("dx", "dy"))
        if self.config.use_pair_distance:
            names.append("dist")
        if self.config.use_pair_gap:
            raise ValueError("pair gap is radius-dependent and disabled in center-only mode.")
        if self.config.use_pair_radii:
            raise ValueError("pair radii are disabled in center-only mode.")
        if self.config.use_pair_role_bits:
            names.extend(("both_fixed", "both_free", "mixed"))
        if self.config.use_pair_abs_delta:
            names.extend(("abs_dx", "abs_dy"))
        if self.config.use_pair_sqdist:
            names.append("dist_sq")
        return tuple(names)

    def _pad_global_token(self, pair_bias: torch.Tensor) -> torch.Tensor:
        """Pad pair bias with a zero global-token row and column when enabled."""

        if not self.use_global_token:
            return pair_bias
        batch_size, num_slots, _, out_dim = pair_bias.shape
        padded = pair_bias.new_zeros(batch_size, num_slots + 1, num_slots + 1, out_dim)
        padded[:, 1:, 1:, :] = pair_bias
        return padded

    def forward(self, slot_centers: torch.Tensor, fixed_mask: torch.Tensor) -> torch.Tensor | None:
        """Return pair bias with shape `[B, T, T, C]` or `None`.

        Parameters
        ----------
        slot_centers:
            Tensor of shape `[B, K, 2]`.
        fixed_mask:
            Boolean tensor of shape `[B, K]`.
        """

        if not self.config.enabled:
            return None
        if slot_centers.ndim != 3 or slot_centers.shape[-1] != 2:
            raise ValueError("slot_centers must have shape [B, K, 2].")
        if fixed_mask.shape != slot_centers.shape[:2]:
            raise ValueError("fixed_mask must have shape [B, K] matching slot_centers.")
        if self.mlp is None:
            raise AssertionError("PairBiasBuilder MLP must exist when pair bias is enabled.")

        x = slot_centers[..., 0]
        y = slot_centers[..., 1]

        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        y_i = y.unsqueeze(2)
        y_j = y.unsqueeze(1)
        dx = x_j - x_i
        dy = y_j - y_i
        dist_sq = dx.square() + dy.square()
        eps = torch.finfo(slot_centers.dtype).eps
        dist = torch.sqrt(dist_sq.clamp_min(eps))

        features: list[torch.Tensor] = []
        if self.config.use_pair_dxdy:
            features.extend((dx.unsqueeze(-1), dy.unsqueeze(-1)))
        if self.config.use_pair_distance:
            features.append(dist.unsqueeze(-1))
        if self.config.use_pair_gap:
            raise ValueError("pair gap is radius-dependent and disabled in center-only mode.")
        if self.config.use_pair_radii:
            raise ValueError("pair radii are disabled in center-only mode.")
        if self.config.use_pair_role_bits:
            fixed_i = fixed_mask.unsqueeze(2)
            fixed_j = fixed_mask.unsqueeze(1)
            both_fixed = (fixed_i & fixed_j).to(dtype=slot_centers.dtype)
            both_free = ((~fixed_i) & (~fixed_j)).to(dtype=slot_centers.dtype)
            mixed = (fixed_i ^ fixed_j).to(dtype=slot_centers.dtype)
            features.extend((both_fixed.unsqueeze(-1), both_free.unsqueeze(-1), mixed.unsqueeze(-1)))
        if self.config.use_pair_abs_delta:
            features.extend((dx.abs().unsqueeze(-1), dy.abs().unsqueeze(-1)))
        if self.config.use_pair_sqdist:
            features.append(dist_sq.unsqueeze(-1))

        pair_features = torch.cat(features, dim=-1)
        pair_bias = self.mlp(pair_features)
        return self._pad_global_token(pair_bias)
