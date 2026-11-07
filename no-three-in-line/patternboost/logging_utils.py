from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_safe(value: Any) -> Any:
    """Convert common experiment values into JSON-serializable objects.

    ``value`` may contain NumPy scalars/arrays, Torch tensors, dataclasses,
    counters, paths, tuples, lists, or dictionaries. The function returns a
    recursively converted object, does not mutate the input, and assumes tensor
    payloads are small diagnostics rather than model parameters.
    """

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Counter):
        return {str(key): json_safe(count) for key, count in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a UTF-8 JSONL file.

    ``path`` is the destination file and ``record`` is converted with
    :func:`json_safe` before serialization. The function creates parent
    directories, mutates only the destination file, and assumes each record is
    independent.
    """

    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), ensure_ascii=True) + "\n")
