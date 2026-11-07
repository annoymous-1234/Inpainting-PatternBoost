from __future__ import annotations

import hashlib
import random

from geometry import Point

from .types import EncodedSolutionRecord


def _stable_seed(*parts: object) -> int:
    """Return a deterministic integer seed derived from the given parts."""

    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_fixed_slot_orders(
    filled_slot_ids: list[int],
    num_slot_orders: int,
    base_seed: int,
    solution_id: str,
) -> list[list[int]]:
    """Build deterministic shuffled slot orders for one encoded solution."""

    slot_orders: list[list[int]] = []
    for order_id in range(num_slot_orders):
        rng = random.Random(_stable_seed(base_seed, solution_id, order_id))
        slot_order = list(filled_slot_ids)
        rng.shuffle(slot_order)
        slot_orders.append(slot_order)
    return slot_orders


def decode_clean_state_to_points(clean_state_ids: list[int], n: int) -> list[Point]:
    """Decode a canonical clean slot state back into a sorted point list."""

    if len(clean_state_ids) != 2 * n:
        raise ValueError(f"clean_state_ids must have length {2 * n}")

    points = [
        (slot_id // 2, state_id - 1)
        for slot_id, state_id in enumerate(clean_state_ids)
        if state_id != 0
    ]
    return sorted(points)


def encode_points_to_clean_state(
    solution_id: str,
    points: list[Point],
    n: int,
    num_slot_orders: int,
    slot_order_seed: int,
) -> EncodedSolutionRecord:
    """Encode a final point set into the canonical two-slots-per-column state."""

    columns: dict[int, list[int]] = {x: [] for x in range(n)}
    for x, y in points:
        if x not in columns:
            raise ValueError(f"point x-coordinate out of range for n={n}: {(x, y)}")
        if not (0 <= y < n):
            raise ValueError(f"point y-coordinate out of range for n={n}: {(x, y)}")
        columns[x].append(y)

    clean_state_ids = [0] * (2 * n)
    for x in range(n):
        rows = sorted(columns[x])
        if len(rows) > 2:
            raise ValueError(f"column {x} contains more than two points")
        if len(rows) != len(set(rows)):
            raise ValueError(f"column {x} contains duplicate row assignments")
        if len(rows) >= 1:
            clean_state_ids[2 * x] = rows[0] + 1
        if len(rows) == 2:
            clean_state_ids[2 * x + 1] = rows[1] + 1

    filled_slot_ids = [slot_id for slot_id, state_id in enumerate(clean_state_ids) if state_id != 0]
    slot_orders = build_fixed_slot_orders(filled_slot_ids, num_slot_orders, slot_order_seed, solution_id)
    return EncodedSolutionRecord(
        solution_id=solution_id,
        points=sorted(points),
        clean_state_ids=clean_state_ids,
        filled_slot_ids=filled_slot_ids,
        point_count=len(filled_slot_ids),
        slot_orders=slot_orders,
    )
