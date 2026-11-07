"""Training placeholders for future work.

Version 1 intentionally excludes model training and RL. This module exists only
to make the project layout explicit for future extensions.
"""

from __future__ import annotations


def train(*args, **kwargs) -> None:
    """Placeholder training entrypoint for future work."""

    raise NotImplementedError("Training is intentionally out of scope for version 1.")
