"""Edit-set sampling for circle-packing PatternBoost."""

from __future__ import annotations

from typing import Any

import numpy as np

from patternboost.types import EditSet


def _normalized_probabilities(weights: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a nonempty vector.")
    if np.any(values < 0.0) or float(np.sum(values)) <= 0.0:
        raise ValueError("weights must be nonnegative with positive sum.")
    return values / np.sum(values)


def _eligible_slots(active_mask: Any, num_slots: int) -> np.ndarray:
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (num_slots,):
        raise ValueError("active_mask must have shape [K].")
    eligible = np.flatnonzero(mask)
    if eligible.size == 0:
        raise ValueError("edit set sampling requires at least one active slot.")
    return eligible


def _num_edit_slots(num_eligible: int, ratio: float, knn_k: int | None = None) -> int:
    if knn_k is not None:
        return min(max(int(knn_k), 1), num_eligible)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("edit ratio must be in (0, 1].")
    return min(max(1, int(round(float(ratio) * num_eligible))), num_eligible)


def sample_random_edit(
    circles: np.ndarray,
    active_mask: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> EditSet:
    """Sample a random subset of active slots."""

    circle_array = np.asarray(circles, dtype=np.float64)
    eligible = _eligible_slots(active_mask, circle_array.shape[0])
    count = _num_edit_slots(int(eligible.size), ratio)
    chosen = rng.choice(eligible, size=count, replace=False)
    edit_mask = np.zeros(circle_array.shape[0], dtype=bool)
    edit_mask[chosen] = True
    return EditSet(edit_mask=edit_mask, policy="random", ratio=float(ratio))


def sample_random_point_knn_edit(
    circles: np.ndarray,
    active_mask: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    *,
    knn_k: int | None = None,
) -> EditSet:
    """Sample a random point and edit the nearest active circles."""

    circle_array = np.asarray(circles, dtype=np.float64)
    eligible = _eligible_slots(active_mask, circle_array.shape[0])
    count = _num_edit_slots(int(eligible.size), ratio, knn_k=knn_k)
    query = rng.uniform(0.0, 1.0, size=2)
    centers = circle_array[eligible, :2]
    distances = np.linalg.norm(centers - query.reshape(1, 2), axis=1)
    chosen = eligible[np.argsort(distances)[:count]]
    edit_mask = np.zeros(circle_array.shape[0], dtype=bool)
    edit_mask[chosen] = True
    return EditSet(
        edit_mask=edit_mask,
        policy="random_point_knn",
        ratio=float(ratio),
        query_point=(float(query[0]), float(query[1])),
    )


def sample_edit_set(circles: np.ndarray, active_mask: np.ndarray, edit_config: Any, rng: np.random.Generator) -> EditSet:
    """Sample an edit set from the configured PatternBoost edit policies."""

    policies = list(getattr(edit_config, "policies", ["random", "random_point_knn"]))
    policy_weights = _normalized_probabilities(getattr(edit_config, "policy_weights", [1.0] * len(policies)))
    if len(policy_weights) != len(policies):
        raise ValueError("policy_weights must match policies.")
    ratios = [float(value) for value in getattr(edit_config, "edit_ratios", [0.2])]
    ratio_weights = _normalized_probabilities(getattr(edit_config, "ratio_weights", [1.0] * len(ratios)))
    if len(ratio_weights) != len(ratios):
        raise ValueError("ratio_weights must match edit_ratios.")

    policy = str(rng.choice(policies, p=policy_weights))
    ratio = float(rng.choice(ratios, p=ratio_weights))
    if policy == "random":
        return sample_random_edit(circles, active_mask, ratio, rng)
    if policy == "random_point_knn":
        return sample_random_point_knn_edit(
            circles,
            active_mask,
            ratio,
            rng,
            knn_k=getattr(edit_config, "knn_k", None),
        )
    raise ValueError(f"unsupported edit policy: {policy}")

