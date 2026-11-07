"""Baseline policy that proposes independent centers for masked slots."""

from __future__ import annotations

from typing import Any

import numpy as np

from circle_pack.geometry import sample_random_centers
from circle_pack.policies.base import ProposalPolicy


class RandomLegalPolicy(ProposalPolicy):
    """Sample independent centers for every masked slot."""

    def propose(self, observation: dict[str, Any]) -> np.ndarray:
        centers = np.asarray(observation["centers"], dtype=self.config.numpy_dtype).copy()
        active_mask = np.asarray(observation["active_mask"], dtype=bool)
        masked = ~active_mask
        if np.any(masked):
            centers[masked] = sample_random_centers(
                self.rng,
                int(np.sum(masked)),
                dtype=self.config.numpy_dtype,
            )
        return centers
