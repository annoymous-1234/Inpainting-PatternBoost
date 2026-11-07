"""Baseline policy that perturbs masked center drafts before re-normalizing them."""

from __future__ import annotations

from typing import Any

import numpy as np

from circle_pack.geometry import normalize_centers, sample_random_centers
from circle_pack.policies.base import ProposalPolicy


class NoisyHintPolicy(ProposalPolicy):
    """Perturb masked center drafts with Gaussian noise and clip them into the unit square."""

    def propose(self, observation: dict[str, Any]) -> np.ndarray:
        centers = np.asarray(observation["centers"], dtype=self.config.numpy_dtype).copy()
        active_mask = np.asarray(observation["active_mask"], dtype=bool)
        masked = ~active_mask
        if not np.any(masked):
            return centers

        masked_centers = centers[masked].copy()
        masked_centers += self.rng.normal(
            loc=0.0,
            scale=self.config.policies.noisy_hint.noise_std_xy,
            size=masked_centers.shape,
        )

        near_zero = np.all(np.isclose(masked_centers, 0.0), axis=1)
        if np.any(near_zero):
            masked_centers[near_zero] = sample_random_centers(
                self.rng,
                int(np.sum(near_zero)),
                dtype=self.config.numpy_dtype,
            )

        centers[masked] = normalize_centers(masked_centers, self.config.safety).astype(
            self.config.numpy_dtype,
            copy=False,
        )
        return centers
