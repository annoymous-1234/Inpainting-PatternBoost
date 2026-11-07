"""Geometry helpers for center-only circle packing in the unit square."""

from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic map elementwise."""

    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0.0
    negative = ~positive
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    output[negative] = exp_values / (1.0 + exp_values)
    return output


def logit(values: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Apply a numerically stable inverse-logistic map elementwise."""

    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def box_capacity(centers: np.ndarray) -> np.ndarray:
    """Return the box-based radius upper bound for one or more centers.

    Parameters
    ----------
    centers:
        Array with trailing shape `[2]` storing `(x, y)` coordinates.

    Returns
    -------
    np.ndarray
        Array with shape matching `centers[..., 0]`, containing
        `min(x, 1 - x, y, 1 - y)` for each center.
    """

    centers = np.asarray(centers, dtype=np.float64)
    x = centers[..., 0]
    y = centers[..., 1]
    return np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))


def latent_to_centers(latent: np.ndarray) -> np.ndarray:
    """Map unconstrained 2D latent coordinates `(u, v)` to physical centers `(x, y)`."""

    latent = np.asarray(latent, dtype=np.float64)
    if latent.shape[-1] != 2:
        raise ValueError("latent_to_centers expects trailing dimension 2.")
    return sigmoid(latent)


def centers_to_latent(centers: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Map physical centers `(x, y)` in the unit square to unconstrained latent coordinates."""

    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape[-1] != 2:
        raise ValueError("centers_to_latent expects trailing dimension 2.")
    return logit(centers, eps=eps)


def normalize_centers(
    centers: np.ndarray,
    safety_cfg: Any,
) -> np.ndarray:
    """Apply lightweight clipping to center proposals."""

    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape[-1] != 2:
        raise ValueError("normalize_centers expects trailing dimension 2.")
    normalized = centers.copy()
    if getattr(safety_cfg, "clip_positions", True):
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def sample_random_centers(
    rng: np.random.Generator,
    num_centers: int,
    dtype: np.dtype | str = np.float64,
) -> np.ndarray:
    """Sample independent center locations in the unit square."""

    dtype = np.dtype(dtype)
    return rng.uniform(0.0, 1.0, size=(num_centers, 2)).astype(dtype, copy=False)


def latent_to_physical_circle(latent: np.ndarray) -> np.ndarray:
    """Map unconstrained latent outputs to single-circle legal physical parameters.

    Given latent `(u, v, rho)` this helper uses:

    - `x = sigmoid(u)`
    - `y = sigmoid(v)`
    - `box_cap = min(x, 1 - x, y, 1 - y)`
    - `r = sigmoid(rho) * box_cap`

    Parameters
    ----------
    latent:
        Array with trailing shape `[3]`.

    Returns
    -------
    np.ndarray
        Array with the same leading shape as `latent`, with trailing shape `[3]`.
    """

    latent = np.asarray(latent, dtype=np.float64)
    if latent.shape[-1] != 3:
        raise ValueError("latent_to_physical_circle expects trailing dimension 3.")
    x = sigmoid(latent[..., 0])
    y = sigmoid(latent[..., 1])
    centers = np.stack((x, y), axis=-1)
    caps = box_capacity(centers)
    r = sigmoid(latent[..., 2]) * caps
    return np.stack((x, y, r), axis=-1)


def latent_to_circle_simple(latent: np.ndarray) -> np.ndarray:
    """Legacy alias for the old latent-to-circle mapping.

    Parameters
    ----------
    latent:
        Array with trailing shape `[3]` storing latent vectors `(u, v, rho)`.

    Returns
    -------
    np.ndarray
        Array with the same leading shape and trailing shape `[3]`, storing the
        decoded physical circles `(x, y, r)`.
    """

    return latent_to_physical_circle(latent)


def circle_to_latent_approx(circles: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Approximate the inverse of the simple latent-to-circle mapping.

    Parameters
    ----------
    circles:
        Array with trailing shape `[3]` storing physical circles `(x, y, r)`.
    eps:
        Small clamp value used to keep the inverse-logit numerically stable.

    Returns
    -------
    np.ndarray
        Array with the same leading shape and trailing shape `[3]`, containing
        approximate latent vectors `(u, v, rho)`.

    Notes
    -----
    This is only an approximate inverse for version 1. It is intended for
    training-data construction and outer-loop state conversion, not as an
    authoritative geometry rule.
    """

    circles = np.asarray(circles, dtype=np.float64)
    if circles.shape[-1] != 3:
        raise ValueError("circle_to_latent_approx expects trailing dimension 3.")
    x = np.clip(circles[..., 0], eps, 1.0 - eps)
    y = np.clip(circles[..., 1], eps, 1.0 - eps)
    box_cap = np.clip(box_capacity(circles[..., :2]), eps, None)
    alpha = np.clip(circles[..., 2] / box_cap, eps, 1.0 - eps)
    u = logit(x, eps=eps)
    v = logit(y, eps=eps)
    rho = logit(alpha, eps=eps)
    return np.stack((u, v, rho), axis=-1)


def latent_to_center_simple(latent: np.ndarray) -> np.ndarray:
    """Convenience alias for the center-only latent-to-physical mapping."""

    return latent_to_centers(latent)


def center_to_latent_approx(centers: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convenience alias for the center-only physical-to-latent mapping."""

    return centers_to_latent(centers, eps=eps)


def sample_random_legal_circles(
    rng: np.random.Generator,
    num_circles: int,
    dtype: np.dtype | str = np.float64,
) -> np.ndarray:
    """Sample independent box-legal circles.

    The circles are only guaranteed to satisfy the box constraints individually.
    They are *not* required to be mutually non-overlapping.
    """

    dtype = np.dtype(dtype)
    centers = rng.uniform(0.0, 1.0, size=(num_circles, 2)).astype(dtype, copy=False)
    caps = box_capacity(centers).astype(dtype, copy=False)
    radii = rng.uniform(0.0, 1.0, size=num_circles).astype(dtype, copy=False) * caps
    return np.column_stack((centers, radii)).astype(dtype, copy=False)


def normalize_physical_circles(
    proposal: np.ndarray,
    safety_cfg: Any,
) -> np.ndarray:
    """Apply lightweight clipping to physical proposals.

    This function is intentionally simple. It is a safety normalizer, not the main
    mechanism for reasoning about feasibility against other circles.
    """

    proposal = np.asarray(proposal, dtype=np.float64)
    if proposal.shape[-1] != 3:
        raise ValueError("normalize_physical_circles expects trailing dimension 3.")
    normalized = proposal.copy()
    if getattr(safety_cfg, "clip_positions", True):
        normalized[..., :2] = np.clip(normalized[..., :2], 0.0, 1.0)
    if getattr(safety_cfg, "clip_radii", True):
        radius_max = float(getattr(safety_cfg, "radius_max", 0.5))
        normalized[..., 2] = np.clip(normalized[..., 2], 0.0, radius_max)
    caps = box_capacity(normalized[..., :2])
    normalized[..., 2] = np.minimum(normalized[..., 2], caps)
    return normalized


def compute_initial_caps(
    cand_centers: np.ndarray,
    active_centers: np.ndarray,
    active_radii: np.ndarray,
) -> np.ndarray:
    """Compute per-candidate capacity against the current active core.

    Parameters
    ----------
    cand_centers:
        Array of shape `[M, 2]` for candidate centers.
    active_centers:
        Array of shape `[N, 2]` for active centers.
    active_radii:
        Array of shape `[N]` for active radii.

    Returns
    -------
    np.ndarray
        Array of shape `[M]` containing `min(box_cap, active_cap)` for each
        candidate center.
    """

    cand_centers = np.asarray(cand_centers, dtype=np.float64)
    active_centers = np.asarray(active_centers, dtype=np.float64)
    active_radii = np.asarray(active_radii, dtype=np.float64)
    caps = box_capacity(cand_centers)
    if active_centers.size == 0:
        return caps
    distances = np.linalg.norm(cand_centers[:, None, :] - active_centers[None, :, :], axis=-1)
    active_caps = np.min(distances - active_radii[None, :], axis=1)
    return np.minimum(caps, active_caps)


def lazy_update_caps_from_commit(
    cand_centers: np.ndarray,
    cap: np.ndarray,
    alive: np.ndarray,
    commit_center: np.ndarray,
    commit_radius: float,
) -> np.ndarray:
    """Update remaining candidate caps after one new circle is committed.

    Only distances from the newly committed circle to still-alive candidates are
    computed. This matches the intended lazy-update strategy for version 1.
    """

    cand_centers = np.asarray(cand_centers, dtype=np.float64)
    if not np.any(alive):
        return cap
    distances = np.linalg.norm(cand_centers[alive] - np.asarray(commit_center), axis=1)
    cap[alive] = np.minimum(cap[alive], distances - float(commit_radius))
    return cap


def circles_inside_unit_box(
    centers: np.ndarray,
    radii: np.ndarray,
    tol: float = 1.0e-12,
) -> bool:
    """Return whether all circles lie inside the unit square."""

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if centers.size == 0:
        return True
    x = centers[:, 0]
    y = centers[:, 1]
    return bool(
        np.all(radii >= -tol)
        and np.all(radii <= 0.5 + tol)
        and np.all(x >= radii - tol)
        and np.all(x <= 1.0 - radii + tol)
        and np.all(y >= radii - tol)
        and np.all(y <= 1.0 - radii + tol)
    )


def pairwise_non_overlapping(
    centers: np.ndarray,
    radii: np.ndarray,
    tol: float = 1.0e-12,
) -> bool:
    """Return whether all circle pairs are non-overlapping."""

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    num_circles = radii.shape[0]
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            if distance + tol < radii[i] + radii[j]:
                return False
    return True


def is_valid_packing(
    centers: np.ndarray,
    radii: np.ndarray,
    tol: float = 1.0e-12,
) -> bool:
    """Return whether a set of circles is a legal packing in the unit square."""

    return circles_inside_unit_box(centers, radii, tol=tol) and pairwise_non_overlapping(
        centers, radii, tol=tol
    )


def active_packing_is_feasible(
    circles: np.ndarray,
    active_mask: np.ndarray,
    tol: float = 1.0e-12,
) -> bool:
    """Return whether the active subset of `circles` forms a valid packing."""

    active_circles = np.asarray(circles, dtype=np.float64)[np.asarray(active_mask, dtype=bool)]
    if active_circles.size == 0:
        return True
    return is_valid_packing(active_circles[:, :2], active_circles[:, 2], tol=tol)


def full_recompute_caps(
    cand_centers: np.ndarray,
    active_centers: np.ndarray,
    active_radii: np.ndarray,
    extra_centers: np.ndarray | None = None,
    extra_radii: np.ndarray | None = None,
) -> np.ndarray:
    """Reference implementation for cap recomputation against all accepted circles."""

    active_centers = np.asarray(active_centers, dtype=np.float64)
    active_radii = np.asarray(active_radii, dtype=np.float64)
    if extra_centers is not None and extra_radii is not None:
        extra_centers = np.asarray(extra_centers, dtype=np.float64)
        extra_radii = np.asarray(extra_radii, dtype=np.float64)
        if active_centers.size == 0:
            all_centers = extra_centers
            all_radii = extra_radii
        else:
            all_centers = np.vstack((active_centers, extra_centers))
            all_radii = np.concatenate((active_radii, extra_radii))
    else:
        all_centers = active_centers
        all_radii = active_radii
    return compute_initial_caps(cand_centers, all_centers, all_radii)
