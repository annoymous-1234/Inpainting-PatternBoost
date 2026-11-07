"""Version-1 supervised loss wrappers for DDPM and flow matching.

These losses belong to the training/pipeline layer. They call the existing
shared-slot model and compute supervision only on free slots, leaving any future
optimization loop or trainer abstraction outside this module.
"""

from __future__ import annotations

from typing import Any

import torch

from circle_pack.config import ProjectConfig
from models.sampling import _validate_draft_features, build_ddpm_schedule, ddpm_time_to_step_index, ddpm_velocity_target


def _masked_slot_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    free_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the free-slot MSE and the number of free slot-vectors.

    Parameters
    ----------
    prediction, target:
        Tensors of shape `[B, K, 2]`.
    free_mask:
        Boolean tensor of shape `[B, K]` where `True` marks slots that should
        contribute to the supervised objective.
    """

    expanded_mask = free_mask.unsqueeze(-1)
    squared_error = (prediction - target) ** 2
    masked_error = squared_error * expanded_mask.to(dtype=squared_error.dtype)
    free_vectors = free_mask.sum()
    denom = (free_vectors * prediction.shape[-1]).clamp_min(1)
    loss = masked_error.sum() / denom.to(dtype=squared_error.dtype)
    return loss, free_vectors


def _prepare_batch_value(
    value: torch.Tensor | Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    is_mask: bool = False,
) -> torch.Tensor:
    """Move one batch field onto the requested device and dtype."""

    tensor = torch.as_tensor(value, device=device)
    if is_mask:
        return tensor.to(dtype=torch.bool)
    if dtype is not None:
        return tensor.to(dtype=dtype)
    return tensor


def ddpm_supervised_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    cfg: ProjectConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the version-1 supervised DDPM loss on free slots only.

    Expected batch fields
    ---------------------
    - `slot_latent`: `[B, K, 2]`
    - `slot_centers`: `[B, K, 2]`
    - `fixed_mask`: `[B, K]`
    - `inner_time`: `[B]` or `[B, 1]`
    - `noise_target`: `[B, K, 2]`
    - `target_latent`: `[B, K, 2]`
    - optional `extra_slot_features`: `[B, K, 4]` when DDPM draft-feature
      conditioning is enabled
    """

    device = model.device
    dtype = model.dtype
    slot_latent = _prepare_batch_value(batch["slot_latent"], device=device, dtype=dtype)
    slot_centers = _prepare_batch_value(batch["slot_centers"], device=device, dtype=dtype)
    fixed_mask = _prepare_batch_value(batch["fixed_mask"], device=device, is_mask=True)
    inner_time = _prepare_batch_value(batch["inner_time"], device=device, dtype=dtype)
    noise_target = _prepare_batch_value(batch["noise_target"], device=device, dtype=dtype)
    target_latent = _prepare_batch_value(batch["target_latent"], device=device, dtype=dtype)
    extra_slot_features = None
    if cfg.ddpm.use_draft_feature:
        _validate_draft_features(model, cfg, owner="ddpm")
        extra_slot_features = _prepare_batch_value(batch["extra_slot_features"], device=device, dtype=dtype)

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        extra_slot_features=extra_slot_features,
        predict_ddpm=True,
    )
    prediction = outputs["ddpm_pred"]
    if prediction is None:
        raise AssertionError("ddpm_supervised_loss requires an enabled DDPM head.")

    if cfg.model.heads.ddpm_target == "eps":
        target = noise_target
    elif cfg.model.heads.ddpm_target == "x0":
        target = target_latent
    else:
        schedule = build_ddpm_schedule(
            cfg.ddpm.num_steps,
            cfg.ddpm.beta_start,
            cfg.ddpm.beta_end,
            device=device,
            dtype=dtype,
        )
        step_indices = ddpm_time_to_step_index(inner_time, cfg.ddpm.num_steps)
        alpha_bars_t = schedule["alpha_bars"][step_indices]
        target = ddpm_velocity_target(target_latent, noise_target, alpha_bars_t)

    free_mask = ~fixed_mask
    loss, free_vectors = _masked_slot_mse(prediction, target, free_mask)
    free_mask_expanded = free_mask.unsqueeze(-1)
    pred_norm = torch.linalg.norm(prediction * free_mask_expanded.to(dtype=dtype), dim=-1)
    target_norm = torch.linalg.norm(target * free_mask_expanded.to(dtype=dtype), dim=-1)

    stats = {
        "loss": float(loss.detach().cpu().item()),
        "free_slot_count": float(free_vectors.detach().cpu().item()),
        "pred_norm_mean": float(pred_norm[free_mask].mean().detach().cpu().item()) if free_mask.any() else 0.0,
        "target_norm_mean": (
            float(target_norm[free_mask].mean().detach().cpu().item()) if free_mask.any() else 0.0
        ),
    }
    return loss, stats


def fm_supervised_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    cfg: ProjectConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the version-1 supervised FM loss on free slots only.

    Expected batch fields
    ---------------------
    - `slot_latent`: `[B, K, 2]`
    - `slot_centers`: `[B, K, 2]`
    - `fixed_mask`: `[B, K]`
    - `inner_time`: `[B]` or `[B, 1]`
    - `vel_target`: `[B, K, 2]`
    - optional `extra_slot_features`: `[B, K, 4]` when FM draft-feature
      conditioning is enabled
    """

    device = model.device
    dtype = model.dtype
    slot_latent = _prepare_batch_value(batch["slot_latent"], device=device, dtype=dtype)
    slot_centers = _prepare_batch_value(batch["slot_centers"], device=device, dtype=dtype)
    fixed_mask = _prepare_batch_value(batch["fixed_mask"], device=device, is_mask=True)
    inner_time = _prepare_batch_value(batch["inner_time"], device=device, dtype=dtype)
    vel_target = _prepare_batch_value(batch["vel_target"], device=device, dtype=dtype)
    extra_slot_features = None
    if cfg.fm.use_draft_feature:
        _validate_draft_features(model, cfg, owner="fm")
        extra_slot_features = _prepare_batch_value(batch["extra_slot_features"], device=device, dtype=dtype)
    elif cfg.model.slot_features.use_extra_slot_features:
        raise ValueError(
            "FM supervised loss with model.slot_features.use_extra_slot_features=True requires "
            "fm.use_draft_feature=True."
        )

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        extra_slot_features=extra_slot_features,
        predict_fm=True,
    )
    prediction = outputs["fm_pred"]
    if prediction is None:
        raise AssertionError("fm_supervised_loss requires an enabled FM head.")

    free_mask = ~fixed_mask
    loss, free_vectors = _masked_slot_mse(prediction, vel_target, free_mask)
    free_mask_expanded = free_mask.unsqueeze(-1)
    pred_norm = torch.linalg.norm(prediction * free_mask_expanded.to(dtype=dtype), dim=-1)
    target_norm = torch.linalg.norm(vel_target * free_mask_expanded.to(dtype=dtype), dim=-1)

    stats = {
        "loss": float(loss.detach().cpu().item()),
        "free_slot_count": float(free_vectors.detach().cpu().item()),
        "pred_norm_mean": float(pred_norm[free_mask].mean().detach().cpu().item()) if free_mask.any() else 0.0,
        "target_norm_mean": (
            float(target_norm[free_mask].mean().detach().cpu().item()) if free_mask.any() else 0.0
        ),
    }
    return loss, stats
