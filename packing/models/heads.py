"""Optional output heads for the circle packing model."""

from __future__ import annotations

import torch
from torch import nn

from models.features import build_activation


def _apply_fixed_zeroing(
    prediction: torch.Tensor,
    fixed_mask: torch.Tensor | None,
    *,
    zero_fixed_outputs: bool,
) -> torch.Tensor:
    """Zero out fixed-slot outputs when configured."""

    if zero_fixed_outputs and fixed_mask is not None:
        return prediction.masked_fill(fixed_mask.unsqueeze(-1), 0.0)
    return prediction


class _SlotVectorHead(nn.Module):
    """Shared implementation for slot-wise `[B, K, 2]` projection heads."""

    def __init__(self, d_model: int, *, zero_fixed_outputs: bool) -> None:
        super().__init__()
        self.zero_fixed_outputs = zero_fixed_outputs
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 2)

    def forward(self, slot_hidden: torch.Tensor, fixed_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Project `[B, K, d_model]` hidden states into `[B, K, 2]`."""

        prediction = self.proj(self.norm(slot_hidden))
        return _apply_fixed_zeroing(
            prediction,
            fixed_mask,
            zero_fixed_outputs=self.zero_fixed_outputs,
        )


class DDPMHead(_SlotVectorHead):
    """Produce latent-space DDPM targets for each slot."""

    def __init__(self, d_model: int, *, target: str, zero_fixed_outputs: bool) -> None:
        super().__init__(d_model, zero_fixed_outputs=zero_fixed_outputs)
        self.target = target


class FMHead(_SlotVectorHead):
    """Produce latent velocity predictions for each slot."""

    def __init__(self, d_model: int, *, zero_fixed_outputs: bool) -> None:
        super().__init__(d_model, zero_fixed_outputs=zero_fixed_outputs)


class SimpleProposalHead(_SlotVectorHead):
    """Produce lightweight latent proposals for testing and early integration."""

    def __init__(self, d_model: int, *, target: str, zero_fixed_outputs: bool) -> None:
        super().__init__(d_model, zero_fixed_outputs=zero_fixed_outputs)
        self.target = target


class ValueHead(nn.Module):
    """Produce a scalar state-value estimate from a pooled hidden representation."""

    def __init__(self, d_model: int, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            build_activation(activation),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project `[B, d_model]` to a scalar value tensor `[B, 1]`."""

        return self.net(hidden_state)
