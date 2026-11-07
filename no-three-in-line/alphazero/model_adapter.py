from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from models import ColumnPairModel, point_feature_config_from_mapping
from project_config import load_project_config

from .config import AlphaZeroConfig
from .state import filled_count
from .types import SearchModelOutput, SearchState


def build_alphazero_model_from_project_config(config: AlphaZeroConfig) -> ColumnPairModel:
    """Build the project model with AlphaZero-specific runtime overrides.

    ``config`` supplies the board size and optionally the source config path.
    The function reads the top-level ``model`` and ``point_features`` mappings,
    forces ``use_global_token=True``, ``use_value_head=True``, and
    ``num_steps=2n + 1``, returns a ``ColumnPairModel``, and does not mutate
    the source project config.
    """

    loaded = load_project_config(config.config_path)
    model_mapping = dict(loaded.get("model", {}))
    point_feature_mapping = dict(loaded.get("point_features", {}))
    point_feature_config = point_feature_config_from_mapping(point_feature_mapping)

    return ColumnPairModel(
        n=config.n,
        num_steps=2 * config.n + 1,
        d_model=int(model_mapping.get("d_model", 256)),
        num_layers=int(model_mapping.get("num_layers", 8)),
        num_heads=int(model_mapping.get("num_heads", 8)),
        d_ff=int(model_mapping.get("d_ff", 1024)),
        dropout=float(model_mapping.get("dropout", 0.0)),
        point_feature_config=point_feature_config,
        use_global_token=True,
        use_value_head=True,
        score_dim=model_mapping.get("score_dim"),
        state_head_type=str(model_mapping.get("state_head_type", "classifier")),
        time_embedding_type=str(model_mapping.get("time_embedding_type", "learned")),
    )


def split_unified_logits(unified_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split unified logits into mask and conditional-value parts.

    ``unified_logits`` is a floating tensor of shape ``[2n, n + 1]`` or
    ``[B, 2n, n + 1]`` on any device. The function returns ``mask_logits`` with
    shape ``[2n]`` or ``[B, 2n]`` and ``value_logits`` with shape ``[2n, n]`` or
    ``[B, 2n, n]``. The logits remain raw and unnormalized.
    """

    return unified_logits[..., 0], unified_logits[..., 1:]


def evaluate_search_state(
    model: nn.Module,
    state: SearchState,
    n: int,
    device: torch.device,
) -> SearchModelOutput:
    """Run one AlphaZero model evaluation on a canonical search state.

    ``state.slot_state_ids`` is an ``int64`` array of shape ``[2n]`` and is
    moved to ``device`` as a batch of size ``1``. The function returns
    ``SearchModelOutput`` whose ``unified_logits`` has shape ``[2n, n + 1]`` in
    raw logit space and whose scalar ``value`` is in ``[0, 1]`` after
    ``sigmoid``. The model is evaluated under ``torch.no_grad()`` and the
    function does not mutate the search state.
    """

    progress_id = filled_count(state.slot_state_ids)
    slot_state_ids = torch.as_tensor(state.slot_state_ids, dtype=torch.long, device=device).unsqueeze(0)
    step_ids = torch.tensor([progress_id], dtype=torch.long, device=device)
    current_point_counts = torch.tensor([progress_id], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(
            slot_state_ids=slot_state_ids,
            step_ids=step_ids,
            current_point_counts=current_point_counts,
        )
        unified_logits = outputs["state_logits"].squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
        raw_value = outputs["value"]
        if raw_value is None:
            raise ValueError("AlphaZero model requires use_value_head=True")
        value = torch.sigmoid(raw_value.squeeze(0)).item()

    if unified_logits.shape != (2 * n, n + 1):
        raise ValueError(f"model returned unexpected state_logits shape: {unified_logits.shape}")
    return SearchModelOutput(unified_logits=unified_logits.astype(np.float32, copy=False), value=float(value))


def resolve_runtime_device(requested_device: str) -> torch.device:
    """Resolve one runtime device string with CUDA fallback handling."""

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)
