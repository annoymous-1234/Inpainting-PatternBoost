from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import AlphaZeroConfig
from .types import AlphaZeroSample


def collate_alphazero_samples(samples: list[AlphaZeroSample], device: torch.device) -> dict[str, torch.Tensor]:
    """Collate variable-support AlphaZero samples into padded batch tensors.

    ``samples`` is a list of final learner samples. The returned dictionary
    contains padded tensors on ``device``: ``slot_state_ids`` with shape
    ``[B, 2n]``, ``episode_steps`` with shape ``[B]``, ``candidate_slot_ids``
    with shape ``[B, C_max]``, ``value_pool_ids`` with shape ``[B, C_max,
    V_max]``, masks aligned with those padded dimensions, and scalar
    ``value_target`` of shape ``[B]``. Candidate-slot padding uses ``-1`` and
    value-support padding also uses ``-1``.
    """

    if not samples:
        raise ValueError("samples must be non-empty")

    batch_size = len(samples)
    num_slots = int(samples[0].slot_state_ids.shape[0])
    max_candidates = max(len(sample.candidate_slot_ids) for sample in samples)
    max_support = max((len(pool) for sample in samples for pool in sample.value_pool_ids), default=0)

    slot_state_ids = torch.zeros((batch_size, num_slots), dtype=torch.long, device=device)
    episode_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    candidate_slot_ids = torch.full((batch_size, max_candidates), -1, dtype=torch.long, device=device)
    candidate_slot_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)
    fill_targets = torch.zeros((batch_size, max_candidates), dtype=torch.float32, device=device)
    value_pool_ids = torch.full((batch_size, max_candidates, max_support), -1, dtype=torch.long, device=device)
    value_pool_mask = torch.zeros((batch_size, max_candidates, max_support), dtype=torch.bool, device=device)
    value_targets = torch.zeros((batch_size, max_candidates, max_support), dtype=torch.float32, device=device)
    value_weights = torch.zeros((batch_size, max_candidates), dtype=torch.float32, device=device)
    value_target = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for batch_index, sample in enumerate(samples):
        slot_state_ids[batch_index] = torch.as_tensor(sample.slot_state_ids, dtype=torch.long, device=device)
        episode_steps[batch_index] = int(sample.episode_step)
        value_target[batch_index] = float(sample.value_target)
        for candidate_index, slot_id in enumerate(sample.candidate_slot_ids):
            candidate_slot_ids[batch_index, candidate_index] = int(slot_id)
            candidate_slot_mask[batch_index, candidate_index] = True
            fill_targets[batch_index, candidate_index] = float(sample.fill_targets[candidate_index])
            value_weights[batch_index, candidate_index] = float(sample.value_weights[candidate_index])
            support = sample.value_pool_ids[candidate_index]
            support_targets = sample.value_targets_on_support[candidate_index]
            for support_index, y_value in enumerate(support):
                value_pool_ids[batch_index, candidate_index, support_index] = int(y_value)
                value_pool_mask[batch_index, candidate_index, support_index] = True
                value_targets[batch_index, candidate_index, support_index] = float(support_targets[support_index])

    return {
        "slot_state_ids": slot_state_ids,
        "episode_steps": episode_steps,
        "candidate_slot_ids": candidate_slot_ids,
        "candidate_slot_mask": candidate_slot_mask,
        "fill_targets": fill_targets,
        "value_pool_ids": value_pool_ids,
        "value_pool_mask": value_pool_mask,
        "value_targets": value_targets,
        "value_weights": value_weights,
        "value_target": value_target,
    }


def compute_policy_losses(
    unified_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: AlphaZeroConfig,
) -> dict[str, torch.Tensor]:
    """Compute AlphaZero fill and value policy losses on sampled supports.

    ``unified_logits`` is a float tensor of shape ``[B, 2n, n + 1]`` on the
    same device as ``batch``. The function returns scalar tensors ``fill_loss``,
    ``val_loss``, and ``policy_loss``. Fill supervision is applied only to the
    sampled root candidate slots, and value supervision is applied only to the
    sampled value supports inside each candidate slot.
    """

    mask_logits = unified_logits[..., 0]
    value_logits = unified_logits[..., 1:]
    log_probs_full = F.log_softmax(unified_logits, dim=-1)
    log_mask_prob = log_probs_full[..., 0]
    log_fill_prob = torch.log1p(-torch.exp(log_mask_prob).clamp(max=1.0 - config.search.eps))
    value_log_probs = F.log_softmax(value_logits, dim=-1)

    candidate_slot_ids = batch["candidate_slot_ids"]
    candidate_slot_mask = batch["candidate_slot_mask"]
    fill_targets = batch["fill_targets"]
    batch_index = torch.arange(unified_logits.shape[0], device=unified_logits.device).unsqueeze(1)
    safe_slot_ids = candidate_slot_ids.clamp(min=0)

    gathered_log_fill = log_fill_prob[batch_index, safe_slot_ids]
    gathered_log_mask = log_mask_prob[batch_index, safe_slot_ids]
    fill_loss_matrix = -(
        fill_targets * gathered_log_fill + (1.0 - fill_targets) * gathered_log_mask
    )
    fill_loss = fill_loss_matrix[candidate_slot_mask].mean() if candidate_slot_mask.any() else unified_logits.sum() * 0.0

    value_pool_ids = batch["value_pool_ids"]
    value_pool_mask = batch["value_pool_mask"]
    value_targets = batch["value_targets"]
    value_weights = batch["value_weights"]
    safe_value_pool_ids = value_pool_ids.clamp(min=0)

    gathered_value_log_probs = value_log_probs[batch_index.unsqueeze(-1), safe_slot_ids.unsqueeze(-1), safe_value_pool_ids]
    weighted_value_ce = -(value_targets * gathered_value_log_probs)
    weighted_value_ce = weighted_value_ce * value_pool_mask
    slot_value_loss = weighted_value_ce.sum(dim=-1)
    slot_has_support = value_pool_mask.any(dim=-1)
    weighted_slot_value_loss = slot_value_loss * value_weights
    if slot_has_support.any():
        val_loss = weighted_slot_value_loss[slot_has_support].sum() / torch.clamp(
            value_weights[slot_has_support].sum(),
            min=config.search.eps,
        )
    else:
        val_loss = unified_logits.sum() * 0.0

    policy_loss = config.training.lambda_fill * fill_loss + config.training.lambda_val * val_loss
    return {
        "fill_loss": fill_loss,
        "val_loss": val_loss,
        "policy_loss": policy_loss,
    }


def compute_value_loss(pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    """Return the mean-squared scalar value loss for AlphaZero training."""

    return F.mse_loss(pred_value, target_value)
