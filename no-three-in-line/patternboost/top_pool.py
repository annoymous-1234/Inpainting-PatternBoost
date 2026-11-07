from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .state import canonical_hash, canonicalize_state_ids, filled_count
from .types import TopPoolEntry


class TopPool:
    """Maintain the highest-scoring canonical PatternBoost constructions.

    The pool stores legal canonical state vectors, deduplicates them by
    canonical hash, and keeps only the highest ``max_size`` scores. Public
    methods mutate only the pool instance and assume callers insert states that
    are legal with respect to no-three-in-line constraints.
    """

    def __init__(self, n: int, max_size: int) -> None:
        """Initialize an empty high-score pool.

        ``n`` is the grid size and ``max_size`` is the maximum retained entry
        count. The constructor returns ``None``, mutates only the new instance,
        and assumes all later entries use shape ``[2n]``.
        """

        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.n = int(n)
        self.max_size = int(max_size)
        self._entries_by_hash: dict[str, TopPoolEntry] = {}
        self._cached_worst_entry: TopPoolEntry | None = None

    def __len__(self) -> int:
        """Return the number of retained pool entries."""

        return len(self._entries_by_hash)

    def entries(self) -> list[TopPoolEntry]:
        """Return pool entries sorted by descending score and hash.

        The method returns a new list, does not mutate the pool, and assumes
        score ties can be resolved deterministically by canonical hash.
        """

        return sorted(
            self._entries_by_hash.values(),
            key=lambda entry: (-entry.score, entry.canonical_hash),
        )

    def insert(self, slot_state_ids: np.ndarray, metadata: dict[str, Any] | None = None) -> bool:
        """Insert one state if it survives score and duplicate filtering.

        ``slot_state_ids`` is canonicalized before insertion and ``metadata`` is
        copied into the entry. The method returns whether a new or better entry
        was retained, mutates the pool, and assumes canonicalization can encode
        the represented points with at most two per column.
        """

        canonical_state = canonicalize_state_ids(slot_state_ids, self.n)
        entry_hash = canonical_hash(canonical_state, self.n)
        score = filled_count(canonical_state)
        old_entry = self._entries_by_hash.get(entry_hash)
        if old_entry is not None and old_entry.score >= score:
            return False

        new_entry = TopPoolEntry(
            slot_state_ids=canonical_state,
            score=score,
            canonical_hash=entry_hash,
            metadata={} if metadata is None else dict(metadata),
        )
        removed_worst_entry = False
        if len(self._entries_by_hash) >= self.max_size and old_entry is None:
            worst_entry = self._worst_entry()
            # Entries are ordered by descending score and then hash; a full pool
            # can reject candidates that would sort after the retained suffix.
            if self._entry_sort_key(new_entry) >= self._entry_sort_key(worst_entry):
                return False
            self._entries_by_hash.pop(worst_entry.canonical_hash)
            self._cached_worst_entry = None
            removed_worst_entry = True

        self._entries_by_hash[entry_hash] = new_entry
        if len(self._entries_by_hash) > self.max_size:
            self._trim()
        elif removed_worst_entry:
            self._worst_entry()
        else:
            self._update_worst_cache(new_entry)
        return True

    def sample_seed(
        self,
        rng: np.random.Generator,
        sampling_config: Any | None = None,
    ) -> TopPoolEntry:
        """Sample one retained seed from the current top pool.

        ``rng`` supplies deterministic sampling and ``sampling_config`` may
        request uniform or score-bucket-rank mixture sampling. The method
        returns a copied entry object, does not mutate the pool, and assumes the
        pool is not empty.
        """

        ordered = self.entries()
        if not ordered:
            raise ValueError("cannot sample from an empty TopPool")
        chosen = self._choose_seed_entry(ordered, rng, sampling_config)
        return TopPoolEntry(
            slot_state_ids=chosen.slot_state_ids.copy(),
            score=chosen.score,
            canonical_hash=chosen.canonical_hash,
            metadata=dict(chosen.metadata),
        )

    def best(self) -> TopPoolEntry | None:
        """Return a copy of the best retained pool entry if present."""

        ordered = self.entries()
        if not ordered:
            return None
        best = ordered[0]
        return TopPoolEntry(
            slot_state_ids=best.slot_state_ids.copy(),
            score=best.score,
            canonical_hash=best.canonical_hash,
            metadata=dict(best.metadata),
        )

    def save(self, path: str | Path) -> None:
        """Serialize the pool to disk with ``torch.save``.

        ``path`` is the output file. The method creates parent directories,
        mutates the filesystem, and does not alter pool contents.
        """

        pool_path = Path(path)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n": self.n,
                "max_size": self.max_size,
                "entries": self.entries(),
            },
            pool_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TopPool:
        """Load a previously serialized pool from disk.

        ``path`` points to a payload written by :meth:`save`. The method
        returns a new ``TopPool``, does not mutate the source file, and assumes
        payload entries are trusted project dataclasses.
        """

        payload = torch.load(Path(path), weights_only=False)
        pool = cls(n=int(payload["n"]), max_size=int(payload["max_size"]))
        for entry in payload["entries"]:
            pool._entries_by_hash[entry.canonical_hash] = entry
        pool._trim()
        return pool

    def _entry_sort_key(self, entry: TopPoolEntry) -> tuple[int, str]:
        """Return the pool ordering key where larger values are worse entries."""

        return (-entry.score, entry.canonical_hash)

    def _worst_entry(self) -> TopPoolEntry:
        """Return the currently retained entry that sorts last in the pool."""

        if self._cached_worst_entry is not None:
            cached = self._entries_by_hash.get(self._cached_worst_entry.canonical_hash)
            if cached is self._cached_worst_entry:
                return cached
        if not self._entries_by_hash:
            raise ValueError("cannot read worst entry from an empty TopPool")
        self._cached_worst_entry = max(
            self._entries_by_hash.values(),
            key=self._entry_sort_key,
        )
        return self._cached_worst_entry

    def _update_worst_cache(self, entry: TopPoolEntry) -> None:
        """Update the cached worst entry after retaining ``entry``."""

        if self._cached_worst_entry is None:
            self._cached_worst_entry = entry
            return
        if self._entry_sort_key(entry) > self._entry_sort_key(self._cached_worst_entry):
            self._cached_worst_entry = entry

    def _trim(self) -> None:
        """Drop entries outside the retained high-score prefix."""

        ordered = self.entries()
        self._entries_by_hash = {
            entry.canonical_hash: entry
            for entry in ordered[: self.max_size]
        }
        self._cached_worst_entry = None
        if self._entries_by_hash:
            self._worst_entry()

    def _choose_seed_entry(
        self,
        ordered: list[TopPoolEntry],
        rng: np.random.Generator,
        sampling_config: Any | None,
    ) -> TopPoolEntry:
        """Choose one ordered entry using uniform or score-bucket mixture sampling."""

        mode = "uniform" if sampling_config is None else _sampling_value(sampling_config, "mode", "uniform")
        if mode == "uniform":
            return ordered[int(rng.integers(0, len(ordered)))]
        if mode != "score_bucket_rank_mixture":
            raise ValueError(f"unsupported TopPool sampling mode: {mode}")

        uniform_mix = float(_sampling_value(sampling_config, "uniform_mix", 0.0))
        if uniform_mix > 0.0 and float(rng.random()) < uniform_mix:
            return ordered[int(rng.integers(0, len(ordered)))]

        eligible, probabilities = _score_bucket_rank_probabilities(
            ordered,
            alpha=float(_sampling_value(sampling_config, "alpha", 1.0)),
            offset=float(_sampling_value(sampling_config, "offset", 1.0)),
            top_fraction=float(_sampling_value(sampling_config, "top_fraction", 1.0)),
        )
        chosen_index = int(rng.choice(len(eligible), p=probabilities))
        return eligible[chosen_index]


def _sampling_value(sampling_config: Any, key: str, default: Any) -> Any:
    """Read one sampling config field from dataclass-like or mapping inputs."""

    if isinstance(sampling_config, dict):
        return sampling_config.get(key, default)
    return getattr(sampling_config, key, default)


def _top_fraction_with_score_ties(
    ordered: list[TopPoolEntry],
    top_fraction: float,
) -> list[TopPoolEntry]:
    """Return the high-score prefix while preserving full cutoff score buckets."""

    if not ordered:
        return []
    target_count = max(1, int(np.ceil(float(top_fraction) * len(ordered))))
    if target_count >= len(ordered):
        return list(ordered)
    cutoff_score = ordered[target_count - 1].score
    return [entry for entry in ordered if entry.score >= cutoff_score]


def _score_bucket_rank_probabilities(
    ordered: list[TopPoolEntry],
    *,
    alpha: float,
    offset: float,
    top_fraction: float,
) -> tuple[list[TopPoolEntry], np.ndarray]:
    """Return eligible entries and per-sample probabilities for bucket-rank sampling."""

    eligible = _top_fraction_with_score_ties(ordered, top_fraction)
    scores_desc = sorted({entry.score for entry in eligible}, reverse=True)
    score_to_bucket_rank = {score: rank for rank, score in enumerate(scores_desc)}
    weights = np.asarray(
        [
            1.0 / float(score_to_bucket_rank[entry.score] + offset) ** float(alpha)
            for entry in eligible
        ],
        dtype=np.float64,
    )
    probabilities = weights / weights.sum()
    return eligible, probabilities
