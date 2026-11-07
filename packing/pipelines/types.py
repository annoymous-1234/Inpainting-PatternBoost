"""Minimal typed containers used by the training/inference pipeline layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProposalBundle:
    """Container for a set of proposal samples from one inner generation round.

    Attributes
    ----------
    latent_samples:
        Tensor of shape `[N, K, 2]` containing latent proposal samples.
    center_samples:
        Tensor of shape `[N, K, 2]` containing decoded physical center proposal samples.
    aux:
        Dictionary of optional auxiliary tensors produced by the sampler.
    """

    latent_samples: torch.Tensor
    center_samples: torch.Tensor
    aux: dict[str, torch.Tensor]


@dataclass
class BatchedProposalBundle:
    """Proposal samples for multiple independent candidate states.

    Attributes
    ----------
    latent_samples:
        Tensor of shape `[B, N, K, 2]`.
    center_samples:
        Tensor of shape `[B, N, K, 2]`.
    aux:
        Dictionary of optional auxiliary tensors produced by the sampler.
    """

    latent_samples: torch.Tensor
    center_samples: torch.Tensor
    aux: dict[str, torch.Tensor]


@dataclass
class OuterLoopRecord:
    """One controller-side record from a single outer-loop round.

    Attributes
    ----------
    round_idx:
        Zero-based outer round index.
    proposal_bundle:
        Proposal samples produced by the current inner generator.
    raw_signals:
        Raw controller-side signals such as dispersion or cross-round drift.
    env_info:
        Dictionary of environment-side outputs and controller metadata for the round.
    """

    round_idx: int
    proposal_bundle: ProposalBundle
    raw_signals: dict[str, torch.Tensor]
    env_info: dict
