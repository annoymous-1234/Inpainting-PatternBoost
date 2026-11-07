from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import SupervisedConfig
from .encoding import encode_points_to_clean_state
from .types import EncodedSolutionRecord, RawSolutionRecord
from .validation import validate_encoded_solution, validate_raw_solution


CACHE_VERSION = 1


def load_raw_solutions_jsonl(path: str | Path) -> list[RawSolutionRecord]:
    """Load raw final-solution records from a JSONL file."""

    records: list[RawSolutionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise ValueError(f"JSONL line {line_index} must be an object")
            if "points" not in loaded:
                raise ValueError(f"JSONL line {line_index} is missing required key 'points'")
            solution_id = str(loaded.get("solution_id", line_index))
            points = [tuple(point) for point in loaded["points"]]
            records.append(RawSolutionRecord(solution_id=solution_id, points=points))
    return records


def _handle_invalid_record(solution_id: str, errors: list[str], config: SupervisedConfig) -> None:
    """Handle invalid preprocessing records according to validation config."""

    message = f"invalid solution {solution_id}: " + "; ".join(errors)
    if config.validation.fail_on_invalid:
        raise ValueError(message)


def preprocess_solution(
    raw_record: RawSolutionRecord,
    config: SupervisedConfig,
) -> EncodedSolutionRecord | None:
    """Validate and encode one raw final solution."""

    raw_validation = validate_raw_solution(raw_record.points, config.n, config.validation)
    if not raw_validation.is_valid:
        _handle_invalid_record(raw_record.solution_id, raw_validation.errors, config)
        if config.validation.drop_invalid:
            return None

    try:
        encoded = encode_points_to_clean_state(
            solution_id=raw_record.solution_id,
            points=raw_record.points,
            n=config.n,
            num_slot_orders=config.encoding.num_slot_orders,
            slot_order_seed=config.encoding.slot_order_seed,
        )
    except ValueError as exc:
        _handle_invalid_record(raw_record.solution_id, [str(exc)], config)
        if config.validation.drop_invalid:
            return None
        raise

    encoded_validation = validate_encoded_solution(encoded, config.n, config.validation)
    if not encoded_validation.is_valid:
        _handle_invalid_record(raw_record.solution_id, encoded_validation.errors, config)
        if config.validation.drop_invalid:
            return None

    return encoded


def build_cache_payload(records: list[EncodedSolutionRecord], config: SupervisedConfig) -> dict[str, object]:
    """Build the serialized cache payload for preprocessed solutions."""

    fingerprint_source = {
        "n": config.n,
        "validation": asdict(config.validation),
        "encoding": asdict(config.encoding),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "version": CACHE_VERSION,
        "n": config.n,
        "config_fingerprint": fingerprint,
        "records": records,
    }


def preprocess_dataset(
    jsonl_path: str | Path,
    cache_path: str | Path,
    config: SupervisedConfig,
) -> list[EncodedSolutionRecord]:
    """Preprocess raw solutions from JSONL and save a cached encoded dataset."""

    raw_records = load_raw_solutions_jsonl(jsonl_path)
    encoded_records = [
        encoded
        for encoded in (preprocess_solution(raw_record, config) for raw_record in raw_records)
        if encoded is not None
    ]
    payload = build_cache_payload(encoded_records, config)
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_file)
    return encoded_records


def load_preprocessed_cache(cache_path: str | Path) -> list[EncodedSolutionRecord]:
    """Load the encoded solution list from a cached preprocessing artifact."""

    payload = torch.load(Path(cache_path), weights_only=False)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"unsupported cache version: {payload.get('version')}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("cache payload is missing the encoded records list")
    return records
