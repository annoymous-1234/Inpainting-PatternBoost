"""Typed containers and protocols for the model-side codebase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

import torch


@dataclass
class BackboneOutput:
    """Structured output from the Transformer backbone.

    Attributes
    ----------
    slot_hidden:
        Tensor of shape `[B, K, d_model]` containing the final hidden state for
        each circle slot token.
    global_hidden:
        Tensor of shape `[B, d_model]` when a global token is enabled, else
        `None`.
    """

    slot_hidden: torch.Tensor
    global_hidden: torch.Tensor | None


class ModelOutputs(TypedDict):
    """Dictionary-shaped model output with optional prediction tensors."""

    ddpm_pred: torch.Tensor | None
    fm_pred: torch.Tensor | None
    proposal_latent: torch.Tensor | None
    value: torch.Tensor | None
    slot_hidden: torch.Tensor | None
    global_hidden: torch.Tensor | None


@runtime_checkable
class LatentToCircleFn(Protocol):
    """Protocol for test-time latent-to-center decoding callbacks.

    The callback may accept and return numpy arrays or torch tensors. The model
    wrapper normalizes the result back into a torch tensor on the original device.
    """

    def __call__(self, proposal_latent: object) -> object:
        """Decode latent proposals into physical center tensors with trailing dim 2."""
