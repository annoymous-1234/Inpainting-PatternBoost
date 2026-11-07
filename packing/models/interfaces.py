"""Interface helpers shared by model wrappers, tests, and smoke scripts."""

from __future__ import annotations

from typing import Any

import torch

from models.model_types import LatentToCircleFn


def validate_model_inputs(
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    fixed_mask: torch.Tensor,
    extra_slot_features: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Validate common slot-wise model inputs and return `(batch_size, num_slots)`."""

    if slot_latent.ndim != 3 or slot_latent.shape[-1] != 2:
        raise ValueError("slot_latent must have shape [B, K, 2].")
    if slot_centers.shape != slot_latent.shape:
        raise ValueError("slot_centers must have the same shape as slot_latent.")
    if fixed_mask.shape != slot_latent.shape[:2]:
        raise ValueError("fixed_mask must have shape [B, K] matching the slot tensors.")
    if extra_slot_features is not None and extra_slot_features.shape[:2] != slot_latent.shape[:2]:
        raise ValueError("extra_slot_features must have leading shape [B, K].")
    batch_size, num_slots, _ = slot_latent.shape
    return int(batch_size), int(num_slots)


def decode_latent_proposal_for_test(
    proposal_latent: torch.Tensor,
    latent_to_circle_fn: LatentToCircleFn | None = None,
) -> torch.Tensor:
    """Decode latent proposals into physical centers for tests and smoke demos.

    This helper intentionally treats decoding as a convenience layer only. It is
    not the authoritative future decode/fix rule for environment interaction.
    """

    if latent_to_circle_fn is None:
        from circle_pack.geometry import latent_to_centers

        latent_to_circle_fn = latent_to_centers

    proposal_numpy = proposal_latent.detach().cpu().numpy()
    decoded = latent_to_circle_fn(proposal_numpy)
    if isinstance(decoded, torch.Tensor):
        output = decoded.to(device=proposal_latent.device, dtype=proposal_latent.dtype)
    else:
        output = torch.as_tensor(decoded, device=proposal_latent.device, dtype=proposal_latent.dtype)
    if output.shape != proposal_latent.shape:
        raise ValueError("Decoded proposal centers must have the same shape as proposal_latent.")
    return output
