from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a top-level mapping.

    ``path`` points to a YAML config file on disk. The function returns a
    Python ``dict`` parsed with ``yaml.safe_load``, does not mutate caller
    state, and assumes the file contents represent a mapping at the top level.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    return loaded


def load_project_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the root project configuration from ``config.yaml``.

    ``path`` optionally overrides the default project config location. The
    function returns a dictionary, does not mutate external state, and assumes
    the selected file is YAML with a mapping at the top level. If the default
    config file is missing, it returns an empty mapping.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if path is None:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return read_yaml_mapping(config_path)
