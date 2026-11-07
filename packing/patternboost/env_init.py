"""Environment initialization helpers for PatternBoost partial episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import normalize_centers


def initialize_partial_env_from_seed(
    env: CirclePackingEnv,
    seed_centers: Any,
    edit_mask: Any,
    *,
    draft_centers: Any | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Initialize `env` with fixed complement center slots active and edited slots free."""

    seed_array = np.asarray(seed_centers, dtype=env.dtype)
    if seed_array.ndim != 2 or seed_array.shape[0] != env.num_slots or seed_array.shape[1] not in {2, 3}:
        raise ValueError(f"seed_centers must have shape ({env.num_slots}, 2) or legacy ({env.num_slots}, 3).")
    seed_array = seed_array[:, :2]
    edit = np.asarray(edit_mask, dtype=bool)
    if edit.shape != (env.num_slots,):
        raise ValueError(f"edit_mask must have shape ({env.num_slots},).")
    if not np.any(edit):
        raise ValueError("edit_mask must select at least one slot.")

    if draft_centers is None:
        centers = seed_array.copy()
    else:
        centers = np.asarray(draft_centers, dtype=env.dtype).copy()
        if centers.ndim != 2 or centers.shape[0] != env.num_slots or centers.shape[1] not in {2, 3}:
            raise ValueError(f"draft_centers must have shape ({env.num_slots}, 2) or legacy ({env.num_slots}, 3).")
        centers = centers[:, :2]
        centers[edit] = normalize_centers(centers[edit], env.config.safety).astype(env.dtype, copy=False)
    fixed_mask = ~edit
    centers[fixed_mask] = seed_array[fixed_mask]

    env.reset(seed=seed)
    env.active_mask = fixed_mask.copy()
    env.centers = centers.astype(env.dtype, copy=True)
    env.step_idx = 0
    env._clear_terminal_cache()
    return env.observe()
