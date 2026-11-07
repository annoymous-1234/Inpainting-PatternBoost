"""Matplotlib-based renderer for circle packing states."""

from __future__ import annotations

from typing import Any

import numpy as np


def render_state(
    circles: np.ndarray,
    active_mask: np.ndarray,
    render_cfg: Any,
    *,
    title: str | None = None,
    save_path: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Render active circles and masked drafts inside the unit square."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    circles = np.asarray(circles, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)

    if ax is None:
        fig, ax = plt.subplots(figsize=tuple(render_cfg.figsize), dpi=int(render_cfg.dpi))
    else:
        fig = ax.figure

    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, edgecolor="black", linewidth=2.0))
    for slot_id, circle in enumerate(circles):
        center = circle[:2]
        radius = float(circle[2])
        is_active = bool(active_mask[slot_id])
        patch = Circle(
            tuple(center),
            radius=radius,
            facecolor=(render_cfg.active_color if is_active else render_cfg.masked_color),
            edgecolor="black",
            linewidth=(1.5 if is_active else 1.0),
            alpha=(render_cfg.active_alpha if is_active else render_cfg.masked_alpha),
            linestyle=("-" if is_active else "--"),
        )
        ax.add_patch(patch)
        if getattr(render_cfg, "show_slot_labels", True):
            ax.text(center[0], center[1], str(slot_id), ha="center", va="center", fontsize=8)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Circle Packing State")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
