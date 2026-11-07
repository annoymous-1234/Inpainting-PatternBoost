"""Raw-signal extraction and tiny demo priority helpers.

This module belongs to the controller-side pipeline. It intentionally avoids
making final environment-side selection decisions beyond a small demo heuristic.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_center_sample_tensor(center_samples: torch.Tensor | Any) -> torch.Tensor:
    """Normalize sample inputs to a float tensor of shape `[N, K, 2]`."""

    tensor = torch.as_tensor(center_samples, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] != 2:
        raise ValueError("center sample tensors must have shape [N, K, 2] or [K, 2].")
    return tensor


def _as_batched_center_sample_tensor(center_samples: torch.Tensor | Any) -> torch.Tensor:
    """Normalize sample inputs to a float tensor of shape `[B, N, K, 2]`."""

    tensor = torch.as_tensor(center_samples, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != 2:
        raise ValueError("batched center sample tensors must have shape [B, N, K, 2] or [N, K, 2].")
    return tensor


def compute_slot_dispersion(
    center_samples: torch.Tensor | Any,
    mode: str = "mean_l2_to_mean",
) -> torch.Tensor:
    """Compute current-round proposal dispersion per slot.

    Parameters
    ----------
    center_samples:
        Tensor or array with shape `[N, K, 2]`.
    mode:
        Dispersion reduction mode. Version 1 supports only `"mean_l2_to_mean"`.

    Returns
    -------
    torch.Tensor
        Tensor of shape `[K]` containing one dispersion value per slot.
    """

    if mode != "mean_l2_to_mean":
        raise ValueError("compute_slot_dispersion only supports mode='mean_l2_to_mean' in version 1.")
    samples = _as_center_sample_tensor(center_samples)
    mean_sample = samples.mean(dim=0, keepdim=True)
    return torch.linalg.norm(samples - mean_sample, dim=-1).mean(dim=0)


def compute_slot_dispersion_batched(
    center_samples: torch.Tensor | Any,
    mode: str = "mean_l2_to_mean",
) -> torch.Tensor:
    """Compute proposal dispersion per candidate and slot.

    Input shape is `[B, N, K, 2]`; output shape is `[B, K]`.
    """

    if mode != "mean_l2_to_mean":
        raise ValueError("compute_slot_dispersion_batched only supports mode='mean_l2_to_mean' in version 1.")
    samples = _as_batched_center_sample_tensor(center_samples)
    mean_sample = samples.mean(dim=1, keepdim=True)
    return torch.linalg.norm(samples - mean_sample, dim=-1).mean(dim=1)


def compute_cross_round_drift(
    prev_center_samples: torch.Tensor | Any,
    curr_center_samples: torch.Tensor | Any,
    mode: str = "mean_to_mean_l2",
) -> torch.Tensor:
    """Compute per-slot drift between two outer-loop rounds."""

    if mode != "mean_to_mean_l2":
        raise ValueError("compute_cross_round_drift only supports mode='mean_to_mean_l2' in version 1.")
    prev_samples = _as_center_sample_tensor(prev_center_samples)
    curr_samples = _as_center_sample_tensor(curr_center_samples)
    prev_mean = prev_samples.mean(dim=0)
    curr_mean = curr_samples.mean(dim=0)
    return torch.linalg.norm(curr_mean - prev_mean, dim=-1)


def compute_cross_round_drift_batched(
    prev_center_samples: torch.Tensor | Any,
    curr_center_samples: torch.Tensor | Any,
    mode: str = "mean_to_mean_l2",
) -> torch.Tensor:
    """Compute per-candidate per-slot drift between two batched outer rounds."""

    if mode != "mean_to_mean_l2":
        raise ValueError("compute_cross_round_drift_batched only supports mode='mean_to_mean_l2' in version 1.")
    prev_samples = _as_batched_center_sample_tensor(prev_center_samples)
    curr_samples = _as_batched_center_sample_tensor(curr_center_samples)
    if prev_samples.shape[0] != curr_samples.shape[0] or prev_samples.shape[2:] != curr_samples.shape[2:]:
        raise ValueError("prev and curr batched center samples must agree in B, K, and coordinate dimensions.")
    prev_mean = prev_samples.mean(dim=1)
    curr_mean = curr_samples.mean(dim=1)
    return torch.linalg.norm(curr_mean - prev_mean, dim=-1)


def reduce_fm_aux_signals(
    aux: dict[str, torch.Tensor],
    reduction: str = "mean_over_samples",
) -> dict[str, torch.Tensor]:
    """Reduce FM sampler auxiliary tensors from `[N, K]` to `[K]`."""

    if reduction != "mean_over_samples":
        raise ValueError("reduce_fm_aux_signals only supports reduction='mean_over_samples' in version 1.")
    reduced: dict[str, torch.Tensor] = {}
    for key, value in aux.items():
        tensor = torch.as_tensor(value)
        if tensor.ndim == 2:
            reduced[key] = tensor.mean(dim=0)
    return reduced


def _reduce_motion_signal(fm_signal: torch.Tensor | dict[str, torch.Tensor] | None) -> torch.Tensor | None:
    """Reduce optional FM signals to one per-slot motion penalty tensor."""

    if fm_signal is None:
        return None
    if isinstance(fm_signal, dict):
        if not fm_signal:
            return None
        stacked = torch.stack([torch.as_tensor(value) for value in fm_signal.values()], dim=0)
        return stacked.mean(dim=0)
    return torch.as_tensor(fm_signal)


def demo_priority_score(
    dispersion: torch.Tensor | Any,
    drift: torch.Tensor | Any = None,
    fm_signal: torch.Tensor | dict[str, torch.Tensor] | Any = None,
    free_mask: torch.Tensor | Any = None,
    cfg: Any = None,
) -> torch.Tensor:
    """Compute a tiny controller-side demo priority score.

    Higher score means higher demo priority.
    """

    dispersion_t = torch.as_tensor(dispersion, dtype=torch.float32)
    score = -dispersion_t
    lambda_var = 1.0
    lambda_drift = 0.0
    lambda_motion = 0.0
    if cfg is not None:
        lambda_var = float(cfg.demo_priority.lambda_var)
        lambda_drift = float(cfg.demo_priority.lambda_drift)
        lambda_motion = float(cfg.demo_priority.lambda_motion)
    score = -lambda_var * dispersion_t

    if drift is not None:
        score = score - lambda_drift * torch.as_tensor(drift, dtype=dispersion_t.dtype, device=dispersion_t.device)
    motion = _reduce_motion_signal(fm_signal)
    if motion is not None:
        score = score - lambda_motion * motion.to(dtype=dispersion_t.dtype, device=dispersion_t.device)
    if free_mask is not None:
        free_mask_t = torch.as_tensor(free_mask, dtype=torch.bool, device=dispersion_t.device)
        score = torch.where(free_mask_t, score, torch.full_like(score, float("-inf")))
    return score
