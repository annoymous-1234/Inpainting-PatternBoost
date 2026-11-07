"""Shared-slot Transformer backbone for center-only circle packing."""

from __future__ import annotations

import torch
from torch import nn

from circle_pack.config import ModelSection
from models.blocks import TransformerBlock
from models.features import ConditionEmbedder, SinusoidalConditionEmbedder, SlotInputProjector
from models.model_types import BackboneOutput
from models.pair_bias import PairBiasBuilder


def _build_outer_step_embedder(config: ModelSection) -> nn.Module | None:
    """Build the configured outer-progress embedder."""

    if not config.use_outer_step_embedding:
        return None
    if config.outer_step_embedding_type == "learned":
        return ConditionEmbedder(config.d_model, config.activation)
    if config.outer_step_embedding_type == "sin":
        return SinusoidalConditionEmbedder(config.d_model)
    raise ValueError("model.outer_step_embedding_type must be 'learned' or 'sin'.")


class CircleTransformerBackbone(nn.Module):
    """Shared full-attention backbone over all center slots.

    The backbone always processes all `K` slots together. Fixed/free status is
    represented only through input features and optional head-level masking.
    """

    def __init__(self, input_dim: int, config: ModelSection) -> None:
        super().__init__()
        self.config = config
        self.input_projector = SlotInputProjector(input_dim, config.d_model, config.activation)
        self.outer_step_embedder = _build_outer_step_embedder(config)
        self.inner_time_embedder = (
            ConditionEmbedder(config.d_model, config.activation) if config.use_inner_time_embedding else None
        )
        self.pair_bias_builder = PairBiasBuilder(
            config.pair_bias,
            hidden_dim=config.d_model,
            num_heads=config.n_heads,
            use_global_token=config.use_global_token,
            activation=config.activation,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        if config.use_global_token:
            self.global_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        else:
            self.register_parameter("global_token", None)

    def forward(
        self,
        raw_slot_features: torch.Tensor,
        slot_centers: torch.Tensor,
        fixed_mask: torch.Tensor,
        outer_step: torch.Tensor | None = None,
        inner_time: torch.Tensor | None = None,
    ) -> BackboneOutput:
        """Encode slots into hidden states.

        Parameters
        ----------
        raw_slot_features:
            Tensor of shape `[B, K, D_in]`.
        slot_centers:
            Tensor of shape `[B, K, 2]`.
        fixed_mask:
            Boolean tensor of shape `[B, K]`.
        outer_step:
            Optional tensor of shape `[B]` or `[B, 1]`. When omitted and outer-step
            embedding is enabled, this is inferred as the active/fixed slot fraction.
        inner_time:
            Optional tensor of shape `[B]` or `[B, 1]`.
        """

        slot_tokens = self.input_projector(raw_slot_features)
        if self.outer_step_embedder is not None:
            if outer_step is None:
                outer_step = fixed_mask.to(dtype=slot_tokens.dtype).mean(dim=1)
            slot_tokens = slot_tokens + self.outer_step_embedder(outer_step).unsqueeze(1)
        if self.inner_time_embedder is not None:
            if inner_time is None:
                raise ValueError("inner_time must be provided when use_inner_time_embedding is enabled.")
            slot_tokens = slot_tokens + self.inner_time_embedder(inner_time).unsqueeze(1)

        pair_bias = self.pair_bias_builder(slot_centers, fixed_mask)
        if self.global_token is not None:
            global_token = self.global_token.expand(slot_tokens.shape[0], -1, -1)
            tokens = torch.cat((global_token, slot_tokens), dim=1)
        else:
            tokens = slot_tokens

        for block in self.blocks:
            tokens = block(tokens, pair_bias=pair_bias)
        tokens = self.final_norm(tokens)

        if self.global_token is not None:
            return BackboneOutput(slot_hidden=tokens[:, 1:, :], global_hidden=tokens[:, 0, :])
        return BackboneOutput(slot_hidden=tokens, global_hidden=None)
