from __future__ import annotations

import random
from pathlib import Path

import torch

from supervised.config import SupervisedConfig, load_supervised_config
from supervised.preprocess import build_cache_payload, load_raw_solutions_jsonl, preprocess_solution
from supervised.types import RawSolutionRecord


def _split_raw_records(
    records: list[RawSolutionRecord],
    seed: int,
) -> tuple[list[RawSolutionRecord], list[RawSolutionRecord]]:
    """Split raw records into deterministic 80/20 train and validation subsets."""

    shuffled_records = list(records)
    random.Random(seed).shuffle(shuffled_records)
    train_count = int(len(shuffled_records) * 0.8)
    val_count = len(shuffled_records) - train_count
    if train_count == 0 or val_count == 0:
        raise ValueError(
            "at least 2 raw records are required to create non-empty 80/20 train and validation splits"
        )
    return shuffled_records[:train_count], shuffled_records[train_count:]


def _save_preprocessed_split(
    records: list[RawSolutionRecord],
    cache_path: str | Path,
    config: SupervisedConfig,
) -> None:
    """Validate, encode, and save one raw-record split to a preprocessing cache."""

    encoded_records = [
        encoded
        for encoded in (preprocess_solution(raw_record, config) for raw_record in records)
        if encoded is not None
    ]
    payload = build_cache_payload(encoded_records, config)
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_file)


def main() -> None:
    """Load one raw JSONL file, split it 80/20, and save train and validation caches.

    The function reads the raw input path from ``supervised.data_paths.train_jsonl``,
    writes the encoded train split to ``train_cache`` and the encoded validation
    split to ``val_cache``, does not mutate the config object, and assumes the
    raw file is a JSONL stream where each line is an object containing a
    ``points`` field and an optional ``solution_id`` field.
    """

    config = load_supervised_config()
    if not config.data_paths.val_cache:
        raise ValueError("supervised.data_paths.val_cache must be set for preprocess_supervised_data.py")

    raw_records = load_raw_solutions_jsonl(config.data_paths.train_jsonl)
    train_records, val_records = _split_raw_records(raw_records, config.runtime.seed)
    _save_preprocessed_split(train_records, config.data_paths.train_cache, config)
    _save_preprocessed_split(val_records, config.data_paths.val_cache, config)


if __name__ == "__main__":
    main()
