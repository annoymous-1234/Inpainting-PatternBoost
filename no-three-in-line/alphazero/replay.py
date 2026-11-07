from __future__ import annotations

from collections import deque

import numpy as np

from .types import AlphaZeroSample


class ReplayBuffer:
    """Store final learner samples in a fixed-capacity FIFO buffer."""

    def __init__(self, capacity: int) -> None:
        """Initialize an empty replay buffer with the given positive capacity."""

        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._items: deque[AlphaZeroSample] = deque(maxlen=capacity)

    def add_many(self, samples: list[AlphaZeroSample]) -> None:
        """Append multiple final learner samples to the replay buffer."""

        self._items.extend(samples)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[AlphaZeroSample]:
        """Sample ``batch_size`` items uniformly without replacement."""

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size > len(self._items):
            raise ValueError("cannot sample more items than the replay buffer contains")
        indices = rng.choice(len(self._items), size=batch_size, replace=False)
        items = list(self._items)
        return [items[int(index)] for index in indices]

    def __len__(self) -> int:
        """Return the current number of stored learner samples."""

        return len(self._items)

    def get_state(self) -> list[AlphaZeroSample]:
        """Return a serializable copy of the replay contents."""

        return list(self._items)

    def load_state(self, items: list[AlphaZeroSample]) -> None:
        """Replace the replay contents with previously serialized samples."""

        self._items = deque(items, maxlen=self._items.maxlen)
