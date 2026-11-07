from __future__ import annotations

from typing import Any

from geometry import Point
from proposal_eval import CandidatePoint


def greedy_repair(
    attempt: dict[str, Any],
    *,
    existing_delete_cost: float = 3.0,
    proposal_delete_cost: float = 1.0,
    allow_existing_deletions: bool = False,
) -> dict[str, Any]:
    """Greedily reject proposal candidates until line conflicts disappear.

    ``attempt`` is one attempt record produced by ``search_repaired_proposals``
    and must contain ``line_conflicts`` plus ``is_feasible``. The cost inputs
    control how expensive it is to delete existing versus proposed points, and
    ``allow_existing_deletions`` enables the legacy path that may remove
    environment points. The function returns rejected slot ids, rejected
    candidates, rejected proposal point tuples, and deleted existing point
    tuples. It does not mutate environment state and assumes the line-conflict
    payload is internally consistent.
    """

    if attempt["is_feasible"]:
        return {
            "rejected_slot_ids": [],
            "rejected_candidates": [],
            "rejected_proposal_points": [],
            "deleted_existing_points": [],
        }

    mutable_lines = [
        {
            "remaining_existing_points": {
                (candidate.x, candidate.y): candidate
                for candidate in conflict_line["existing_points"]
            },
            "remaining_proposal_points": {
                (candidate.x, candidate.y): candidate
                for candidate in conflict_line["proposal_points"]
            },
        }
        for conflict_line in attempt["line_conflicts"]
    ]

    rejected_slot_ids: set[int] = set()
    rejected_candidates_by_point: dict[Point, CandidatePoint] = {}
    deleted_existing_points: set[Point] = set()

    while True:
        active_lines = [
            line
            for line in mutable_lines
            if len(line["remaining_existing_points"]) + len(line["remaining_proposal_points"]) >= 3
        ]
        if not active_lines:
            break

        point_hits: dict[tuple[str, Point], int] = {}
        point_lookup: dict[tuple[str, Point], CandidatePoint] = {}
        for line in active_lines:
            if allow_existing_deletions:
                for point, candidate in line["remaining_existing_points"].items():
                    key = ("existing", point)
                    point_hits[key] = point_hits.get(key, 0) + 1
                    point_lookup[key] = candidate
            for point, candidate in line["remaining_proposal_points"].items():
                key = ("proposal", point)
                point_hits[key] = point_hits.get(key, 0) + 1
                point_lookup[key] = candidate

        best_key: tuple[float, int, float, int, float, int, int] | None = None
        best_point_type: str | None = None
        best_candidate: CandidatePoint | None = None
        best_point: Point | None = None

        for (point_type, point), hits in point_hits.items():
            candidate = point_lookup[(point_type, point)]
            cost = proposal_delete_cost if point_type == "proposal" else existing_delete_cost
            proposal_score_priority = (
                float("inf")
                if point_type == "proposal" and candidate.score is None
                else (float("-inf") if point_type != "proposal" else -candidate.score)
            )
            score_key = (
                hits / cost,
                hits,
                proposal_score_priority,
                1 if point_type == "proposal" else 0,
                float("-inf") if candidate.slot_id is None else -candidate.slot_id,
                -candidate.x,
                -candidate.y,
            )
            if best_key is None or score_key > best_key:
                best_key = score_key
                best_point_type = point_type
                best_candidate = candidate
                best_point = point

        if best_point is None or best_point_type is None or best_candidate is None:
            break

        if best_point_type == "proposal":
            rejected_candidates_by_point[best_point] = best_candidate
            if best_candidate.slot_id is not None:
                rejected_slot_ids.add(best_candidate.slot_id)
        else:
            deleted_existing_points.add(best_point)

        for line in mutable_lines:
            line["remaining_existing_points"].pop(best_point, None)
            line["remaining_proposal_points"].pop(best_point, None)

    rejected_candidates = sorted(
        rejected_candidates_by_point.values(),
        key=lambda candidate: (
            -1 if candidate.slot_id is None else candidate.slot_id,
            candidate.x,
            candidate.y,
        ),
    )
    return {
        "rejected_slot_ids": sorted(rejected_slot_ids),
        "rejected_candidates": rejected_candidates,
        "rejected_proposal_points": sorted(rejected_candidates_by_point),
        "deleted_existing_points": sorted(deleted_existing_points),
    }
