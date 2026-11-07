"""TopPool for center-only circle-packing PatternBoost."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from circle_pack.geometry import is_valid_packing
from circle_pack.scipy_solver import solve_radii_for_centers
from patternboost.types import CircleTopPoolEntry


def _as_center_array(centers: Any) -> np.ndarray:
    array = np.asarray(centers, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in {2, 3}:
        raise ValueError(f"centers must have shape [K, 2] or legacy [K, 3], got {array.shape}.")
    return array[:, :2].copy()


def canonicalize_centers(centers: Any) -> np.ndarray:
    """Return a deterministic permutation-invariant center ordering."""

    array = _as_center_array(centers)
    order = np.lexsort((array[:, 1], array[:, 0]))
    return array[order].copy()


def canonical_hash(centers: Any, dedup_tol: float) -> str:
    """Hash a quantized canonical center layout."""

    if dedup_tol <= 0.0:
        raise ValueError("dedup_tol must be positive.")
    canonical = canonicalize_centers(centers)
    quantized = np.round(canonical / float(dedup_tol)).astype(np.int64, copy=False)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


class TopPool:
    """Maintain high-scoring complete center layouts."""

    def __init__(
        self,
        max_size: int,
        *,
        dedup_tol: float = 1.0e-5,
        feasibility_tol: float = 1.0e-12,
        radius_max: float = 0.5,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        if dedup_tol <= 0.0:
            raise ValueError("dedup_tol must be positive.")
        self.max_size = int(max_size)
        self.dedup_tol = float(dedup_tol)
        self.feasibility_tol = float(feasibility_tol)
        self.radius_max = float(radius_max)
        self._entries_by_hash: dict[str, CircleTopPoolEntry] = {}

    def __len__(self) -> int:
        return len(self._entries_by_hash)

    def entries(self) -> list[CircleTopPoolEntry]:
        return sorted(self._entries_by_hash.values(), key=lambda entry: (-entry.score, entry.canonical_hash))

    def insert(
        self,
        centers: Any,
        active_mask: Any | None = None,
        *,
        radii: Any | None = None,
        circles: Any | None = None,
        score: float | None = None,
        lp_status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Insert one complete layout if it survives duplicate and top-k filtering."""

        center_array = _as_center_array(centers)
        if active_mask is None:
            mask = np.ones(center_array.shape[0], dtype=bool)
        else:
            mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != (center_array.shape[0],):
            raise ValueError("active_mask must have shape [K].")
        if not np.all(mask):
            return False

        entry_hash = canonical_hash(center_array, self.dedup_tol)
        order = np.lexsort((center_array[:, 1], center_array[:, 0]))
        canonical_centers = center_array[order].copy()

        if radii is not None and circles is not None and score is not None:
            circle_array = np.asarray(circles, dtype=np.float64)
            radius_array = np.asarray(radii, dtype=np.float64)
            if circle_array.shape != (center_array.shape[0], 3) or radius_array.shape != (center_array.shape[0],):
                raise ValueError("cached circles/radii do not match centers.")
            circle_array = circle_array[order].copy()
            radius_array = radius_array[order].copy()
            score_value = float(score)
            status = "cached" if lp_status is None else str(lp_status)
        else:
            result = solve_radii_for_centers(
                canonical_centers,
                radius_max=self.radius_max,
                feasibility_tol=self.feasibility_tol,
            )
            circle_array = result.circles.copy()
            radius_array = circle_array[:, 2].copy()
            score_value = float(result.sum_r)
            status = result.message if result.success else f"failed:{result.status}:{result.message}"

        if not is_valid_packing(circle_array[:, :2], circle_array[:, 2], tol=max(self.feasibility_tol, 1.0e-9)):
            return False
        old_entry = self._entries_by_hash.get(entry_hash)
        if old_entry is not None and old_entry.score >= score_value:
            return False

        self._entries_by_hash[entry_hash] = CircleTopPoolEntry(
            centers=canonical_centers,
            radii=radius_array,
            circles=circle_array,
            active_mask=np.ones(canonical_centers.shape[0], dtype=bool),
            score=score_value,
            lp_status=status,
            canonical_hash=entry_hash,
            metadata={} if metadata is None else dict(metadata),
        )
        self._trim()
        return entry_hash in self._entries_by_hash

    def best(self) -> CircleTopPoolEntry | None:
        ordered = self.entries()
        if not ordered:
            return None
        return _copy_entry(ordered[0])

    def sample_seed(self, rng: np.random.Generator, sampling_config: Any | None = None) -> CircleTopPoolEntry:
        ordered = self.entries()
        if not ordered:
            raise ValueError("cannot sample from an empty TopPool.")
        chosen = self._choose_seed_entry(ordered, rng, sampling_config)
        return _copy_entry(chosen)

    def save(self, path: str | Path) -> None:
        pool_path = Path(path)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "max_size": self.max_size,
                "dedup_tol": self.dedup_tol,
                "feasibility_tol": self.feasibility_tol,
                "radius_max": self.radius_max,
                "entries": self.entries(),
            },
            pool_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TopPool:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        pool = cls(
            max_size=int(payload["max_size"]),
            dedup_tol=float(payload["dedup_tol"]),
            feasibility_tol=float(payload.get("feasibility_tol", 1.0e-12)),
            radius_max=float(payload.get("radius_max", 0.5)),
        )
        for entry in payload["entries"]:
            pool._entries_by_hash[entry.canonical_hash] = entry
        pool._trim()
        return pool

    def _trim(self) -> None:
        self._entries_by_hash = {entry.canonical_hash: entry for entry in self.entries()[: self.max_size]}

    def _choose_seed_entry(
        self,
        ordered: list[CircleTopPoolEntry],
        rng: np.random.Generator,
        sampling_config: Any | None,
    ) -> CircleTopPoolEntry:
        mode = "uniform" if sampling_config is None else _sampling_value(sampling_config, "mode", "uniform")
        if mode == "uniform":
            return ordered[int(rng.integers(0, len(ordered)))]
        if mode != "score_bucket_rank_mixture":
            raise ValueError(f"unsupported TopPool sampling mode: {mode}")
        top_fraction = float(_sampling_value(sampling_config, "top_fraction", 1.0))
        eligible = _top_fraction_prefix(ordered, top_fraction)
        uniform_mix = float(_sampling_value(sampling_config, "uniform_mix", 0.0))
        if uniform_mix > 0.0 and float(rng.random()) < uniform_mix:
            return eligible[int(rng.integers(0, len(eligible)))]
        probabilities = _rank_probabilities(
            eligible,
            alpha=float(_sampling_value(sampling_config, "alpha", 1.0)),
            offset=float(_sampling_value(sampling_config, "offset", 1.0)),
        )
        return eligible[int(rng.choice(len(eligible), p=probabilities))]


def _copy_entry(entry: CircleTopPoolEntry) -> CircleTopPoolEntry:
    return CircleTopPoolEntry(
        centers=entry.centers.copy(),
        radii=entry.radii.copy(),
        circles=entry.circles.copy(),
        active_mask=entry.active_mask.copy(),
        score=float(entry.score),
        lp_status=str(entry.lp_status),
        canonical_hash=entry.canonical_hash,
        metadata=dict(entry.metadata),
    )


def _sampling_value(sampling_config: Any, key: str, default: Any) -> Any:
    if isinstance(sampling_config, dict):
        return sampling_config.get(key, default)
    return getattr(sampling_config, key, default)


def _top_fraction_prefix(ordered: list[CircleTopPoolEntry], top_fraction: float) -> list[CircleTopPoolEntry]:
    target_count = max(1, int(np.ceil(float(top_fraction) * len(ordered))))
    return list(ordered[: min(target_count, len(ordered))])


def _rank_probabilities(
    eligible: list[CircleTopPoolEntry],
    *,
    alpha: float,
    offset: float,
) -> np.ndarray:
    ranks = np.arange(len(eligible), dtype=np.float64)
    weights = 1.0 / np.power(ranks + float(offset), float(alpha))
    probabilities = weights / np.sum(weights)
    return probabilities
