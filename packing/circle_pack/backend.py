"""Append-only backend for the active packing core."""

from __future__ import annotations

from typing import Any

import numpy as np


class ActiveBackend:
    """Compact append-only storage for active circles.

    The version 1 environment never edits or removes active circles. This backend
    therefore stores only committed circles and grows monotonically across steps.
    """

    def __init__(self, dtype: np.dtype | str = np.float64) -> None:
        self.dtype = np.dtype(dtype)
        self.reset()

    @property
    def n_active(self) -> int:
        """Return the number of currently active circles."""

        return int(self.active_ids.shape[0])

    def reset(self) -> None:
        """Clear the backend to an empty active core."""

        self.active_ids = np.empty(0, dtype=np.int64)
        self.active_centers = np.empty((0, 2), dtype=self.dtype)
        self.active_radii = np.empty(0, dtype=self.dtype)
        self.sum_active_r = float(0.0)

    def append(self, slot_id: int, center: np.ndarray, radius: float) -> None:
        """Append a newly committed active circle to the compact backend."""

        center = np.asarray(center, dtype=self.dtype).reshape(2)
        radius = float(radius)
        self.active_ids = np.append(self.active_ids, np.int64(slot_id))
        self.active_centers = np.vstack((self.active_centers, center.reshape(1, 2)))
        self.active_radii = np.append(self.active_radii, self.dtype.type(radius))
        self.sum_active_r += radius

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copyable backend snapshot."""

        return {
            "active_ids": self.active_ids.copy(),
            "active_centers": self.active_centers.copy(),
            "active_radii": self.active_radii.copy(),
            "sum_active_r": float(self.sum_active_r),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore the backend from a snapshot produced by `snapshot()`."""

        self.active_ids = np.asarray(snapshot["active_ids"], dtype=np.int64).copy()
        self.active_centers = np.asarray(snapshot["active_centers"], dtype=self.dtype).copy()
        self.active_radii = np.asarray(snapshot["active_radii"], dtype=self.dtype).copy()
        self.sum_active_r = float(snapshot["sum_active_r"])
