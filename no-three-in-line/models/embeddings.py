from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .point_features import (
    PointFeatureConfig,
    build_column_anchor_table,
    build_point_feature_table,
    get_point_raw_feature_dim,
)


_TIME_EMBEDDING_TYPES = {"learned", "sinusoidal"}


def _build_sinusoidal_embedding_table(num_steps: int, d_model: int) -> Tensor:
    """Return a fixed sinusoidal table indexed by integer progress ids."""

    positions = torch.arange(num_steps, dtype=torch.float32).unsqueeze(1)
    div_terms = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(num_steps, d_model, dtype=torch.float32)
    table[:, 0::2] = torch.sin(positions * div_terms)
    table[:, 1::2] = torch.cos(positions * div_terms[: table[:, 1::2].shape[1]])
    return table


class PointFeatureProjector(nn.Module):
    """Project raw point features into the shared model width.

    The module applies a small MLP to the last feature dimension. It mutates
    only its own parameters during training and assumes the final input
    dimension equals the configured raw feature width.
    """

    def __init__(self, d_raw: int, d_model: int) -> None:
        """Initialize the raw-feature projector.

        ``d_raw`` is the raw point-feature width and ``d_model`` is the model
        width after projection. The constructor returns ``None``, mutates only
        the new module, and assumes both dimensions are positive.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_raw, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, raw_features: Tensor) -> Tensor:
        """Project raw point features into ``d_model``.

        ``raw_features`` is a floating tensor of shape ``[..., d_raw]`` on any
        device. The method returns a tensor of shape ``[..., d_model]`` on the
        same device, does not mutate the input tensor, and mutates only module
        parameters if gradients are later applied.
        """
        return self.net(raw_features)


class SlotTokenEmbedding(nn.Module):
    """Construct initial slot and optional global token embeddings.

    The module combines state embeddings, projected point or column-anchor
    features, pair embeddings, and step embeddings. Its forward pass does not
    mutate persistent buffers, aside from standard autograd interactions with
    parameters during training.
    """

    def __init__(
        self,
        n: int,
        d_model: int,
        num_steps: int,
        point_feature_config: PointFeatureConfig,
        use_global_token: bool = True,
        time_embedding_type: str = "learned",
    ) -> None:
        """Initialize lookup tables and projection layers for slot tokens.

        ``n`` is the grid size, ``d_model`` is the shared hidden width,
        ``num_steps`` bounds the step embedding table, ``point_feature_config``
        controls the raw point-feature tables, and ``time_embedding_type``
        selects learned or fixed sinusoidal progress embeddings.
        ``use_global_token`` decides whether a learned global token is
        prepended at runtime. The constructor returns ``None``, mutates only
        the new module, and assumes at least one point-feature block is enabled.
        """
        super().__init__()
        if time_embedding_type not in _TIME_EMBEDDING_TYPES:
            raise ValueError("time_embedding_type must be 'learned' or 'sinusoidal'")
        self.n = n
        self.d_model = d_model
        self.num_steps = num_steps
        self.use_global_token = use_global_token
        self.time_embedding_type = time_embedding_type

        d_raw = get_point_raw_feature_dim(n, point_feature_config)
        if d_raw <= 0:
            raise ValueError("point feature configuration must enable at least one feature block")

        self.state_embedding = nn.Embedding(n + 1, d_model)
        self.pair_embedding = nn.Embedding(2, d_model)
        if time_embedding_type == "learned":
            self.step_embedding = nn.Embedding(num_steps, d_model)
            self.register_buffer("step_embedding_table", None)
        else:
            self.step_embedding = None
            self.register_buffer(
                "step_embedding_table",
                _build_sinusoidal_embedding_table(num_steps, d_model),
            )
        self.point_projector = PointFeatureProjector(d_raw=d_raw, d_model=d_model)

        if use_global_token:
            self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.global_count_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.register_parameter("global_token", None)
            self.global_count_proj = None

        token_ids = torch.arange(2 * n, dtype=torch.long)
        self.register_buffer("token_to_column", token_ids // 2)
        self.register_buffer("token_to_pair", token_ids % 2)
        self.register_buffer(
            "point_raw_table",
            build_point_feature_table(n, point_feature_config),
        )
        self.register_buffer(
            "column_anchor_raw_table",
            build_column_anchor_table(self.point_raw_table),
        )

    def forward(
        self,
        slot_state_ids: Tensor,
        step_ids: Tensor,
        current_point_counts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Build initial token embeddings for a batch of slot states.

        ``slot_state_ids`` is a ``LongTensor`` of shape ``[B, 2n]`` whose values
        are ``0`` for ``[MASK]`` and ``y + 1`` for filled slots. ``step_ids`` is
        a ``LongTensor`` of filled-slot progress ids with shape ``[B]``.
        ``current_point_counts`` is optional and may have shape ``[B]`` or
        ``[B, 1]``. The method returns a dictionary with ``token_embeds`` of
        shape ``[B, 2n, d_model]`` or ``[B, 2n + 1, d_model]`` when the global
        token is enabled, ``candidate_point_embeds`` of shape ``[n, n,
        d_model]``, and ``column_anchor_embeds`` of shape ``[n, d_model]``. All
        tensors live on the module device with floating dtype except the
        integer inputs. The method does not mutate input tensors and assumes
        slot ids match the column-pair representation.
        """
        if slot_state_ids.ndim != 2 or slot_state_ids.shape[1] != 2 * self.n:
            raise ValueError(f"slot_state_ids must have shape [B, {2 * self.n}]")
        if step_ids.ndim != 1 or step_ids.shape[0] != slot_state_ids.shape[0]:
            raise ValueError("step_ids must have shape [B]")

        batch_size = slot_state_ids.shape[0]
        state_embeds = self.state_embedding(slot_state_ids)
        pair_embeds = self.pair_embedding(self.token_to_pair).unsqueeze(0)
        step_embeds = self._step_embeds(step_ids).unsqueeze(1)

        candidate_point_embeds = self.point_projector(self.point_raw_table)
        column_anchor_embeds = self.point_projector(self.column_anchor_raw_table)

        column_anchor_by_token = column_anchor_embeds[self.token_to_column]
        column_anchor_by_token = column_anchor_by_token.unsqueeze(0).expand(batch_size, -1, -1)

        point_embeds_by_token = candidate_point_embeds[self.token_to_column]
        point_embeds_by_token = point_embeds_by_token.unsqueeze(0).expand(batch_size, -1, -1, -1)

        filled_y = torch.clamp(slot_state_ids - 1, min=0)
        gather_index = filled_y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.d_model)
        filled_point_embeds = torch.gather(point_embeds_by_token, 2, gather_index).squeeze(2)

        masked = slot_state_ids.eq(0).unsqueeze(-1)
        point_or_anchor = torch.where(masked, column_anchor_by_token, filled_point_embeds)
        token_embeds = state_embeds + point_or_anchor + pair_embeds + step_embeds

        if self.use_global_token:
            if current_point_counts is None:
                count_values = token_embeds.new_zeros((batch_size, 1))
            else:
                if current_point_counts.ndim == 1:
                    count_values = current_point_counts.unsqueeze(-1)
                elif current_point_counts.ndim == 2 and current_point_counts.shape[1] == 1:
                    count_values = current_point_counts
                else:
                    raise ValueError("current_point_counts must have shape [B] or [B, 1]")
                count_values = count_values.to(device=token_embeds.device, dtype=token_embeds.dtype)

            global_token = self.global_token.expand(batch_size, -1, -1)
            global_token = global_token + step_embeds + self.global_count_proj(count_values).unsqueeze(1)
            token_embeds = torch.cat((global_token, token_embeds), dim=1)

        return {
            "token_embeds": token_embeds,
            "candidate_point_embeds": candidate_point_embeds,
            "column_anchor_embeds": column_anchor_embeds,
        }

    def _step_embeds(self, step_ids: Tensor) -> Tensor:
        """Look up learned or sinusoidal embeddings for integer progress ids."""

        if self.time_embedding_type == "learned":
            if self.step_embedding is None:
                raise RuntimeError("learned time embedding module is not initialized")
            return self.step_embedding(step_ids)
        if self.step_embedding_table is None:
            raise RuntimeError("sinusoidal time embedding table is not initialized")
        return self.step_embedding_table[step_ids]
