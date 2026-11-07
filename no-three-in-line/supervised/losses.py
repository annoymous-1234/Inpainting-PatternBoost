from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig


def compute_token_weights(target_state_ids: Tensor, config: LossConfig) -> Tensor:
    """Build static per-slot loss weights from target state ids.

    ``target_state_ids`` is a ``LongTensor`` of shape ``[S]`` or ``[B, S]`` on
    any device, where ``0`` denotes a masked or empty slot and nonzero ids
    denote filled slots. ``config`` provides the scalar weights for masked and
    filled targets. The function returns a ``FloatTensor`` with the same shape
    and device, does not mutate its inputs, and assumes the training vocabulary
    reserves ``0`` for the masked-state token.
    """

    mask_weights = torch.full_like(target_state_ids, fill_value=config.mask_weight, dtype=torch.float32)
    filled_weights = torch.full_like(target_state_ids, fill_value=config.filled_weight, dtype=torch.float32)
    return torch.where(target_state_ids.eq(0), mask_weights, filled_weights)


def compute_supervised_loss(
    state_logits: Tensor,
    target_state_ids: Tensor,
    token_weights: Tensor,
    supervision_mask: Tensor,
    task_types: list[str],
    config: LossConfig,
) -> dict[str, Tensor]:
    """Compute the task-aware supervised denoising loss for one batch.

    ``state_logits`` is a ``FloatTensor[B, S, V]`` on the training device,
    ``target_state_ids`` is a ``LongTensor[B, S]`` with target class ids,
    ``token_weights`` is a ``FloatTensor[B, S]`` with static mask-vs-filled
    weights, and ``supervision_mask`` is a ``FloatTensor[B, S]`` with ``1.0``
    on supervised slots and ``0.0`` elsewhere. ``task_types`` has length
    ``B`` and marks each sample as ``"full"`` or ``"partial"``. The function
    returns scalar tensors for the batch loss and task-specific summaries, does
    not mutate its inputs, and assumes all batch-aligned tensors share the same
    ``[B, S]`` layout and that ``task_types`` matches the batch order.
    """

    batch_size = state_logits.shape[0]
    per_token_loss = F.cross_entropy(
        state_logits.reshape(-1, state_logits.shape[-1]),
        target_state_ids.reshape(-1),
        reduction="none",
    ).reshape(batch_size, -1)
    # Combine class-based token weighting with task-specific supervision scope
    # so full samples only learn from currently masked filled slots.
    effective_weights = token_weights * supervision_mask
    normalized_sample_loss = (
        (per_token_loss * effective_weights).sum(dim=1) / effective_weights.sum(dim=1).clamp_min(1e-8)
    )

    task_scale = normalized_sample_loss.new_tensor(
        [
            config.partial_loss_weight if task_type == "partial" else config.full_loss_weight
            for task_type in task_types
        ]
    )
    scaled_sample_loss = normalized_sample_loss * task_scale

    full_mask = normalized_sample_loss.new_tensor([task_type == "full" for task_type in task_types], dtype=torch.bool)
    partial_mask = normalized_sample_loss.new_tensor(
        [task_type == "partial" for task_type in task_types],
        dtype=torch.bool,
    )

    zero = normalized_sample_loss.new_zeros(())
    mean_full_loss = scaled_sample_loss[full_mask].mean() if full_mask.any() else zero
    mean_partial_loss = scaled_sample_loss[partial_mask].mean() if partial_mask.any() else zero
    return {
        "loss": scaled_sample_loss.mean(),
        "mean_full_loss": mean_full_loss,
        "mean_partial_loss": mean_partial_loss,
        "mean_token_loss": normalized_sample_loss.mean(),
    }
