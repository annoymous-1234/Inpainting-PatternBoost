"""Placeholder training script for the version 1 scaffold."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Exit with an explicit message about the missing training implementation."""

    raise SystemExit("Training is intentionally not implemented in version 1.")


if __name__ == "__main__":
    main()
