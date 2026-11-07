"""Base interface for proposal policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ProposalPolicy(ABC):
    """Abstract policy that proposes a full `[K, 2]` center array each step."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.project.seed)

    def reset(self, env: Any, seed: int | None = None) -> None:
        """Reset policy state before a new episode."""

        del env
        self.rng = np.random.default_rng(self.config.project.seed if seed is None else seed)

    @abstractmethod
    def propose(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a full proposal array of shape `[K, 2]`."""
