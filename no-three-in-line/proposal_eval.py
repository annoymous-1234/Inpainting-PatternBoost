from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from active_lines import ActiveLineBackend
from geometry import LineKey, Point, line_key


@dataclass
class CandidatePoint:
    """Represent one candidate lattice point and its owning slot metadata.

    ``x`` and ``y`` are integer grid coordinates, ``score`` stores optional
    model-side ranking metadata, and ``slot_id`` identifies the owning proposal
    slot when known. Instances are plain data carriers, mutate only their own
    fields if reassigned by the caller, and assume ``slot_id`` may be ``None``
    only for ad-hoc environment points or legacy construction.
    """

    x: int
    y: int
    score: float | None = None
    slot_id: int | None = None


@dataclass
class ProposalSlot:
    """Group one slot id with an ordered list of candidate points.

    ``slot_id`` identifies the environment slot being proposed and
    ``candidates`` stores candidates in priority order. The dataclass carries
    data only, returns no value on construction, and assumes candidate order is
    meaningful to downstream duplicate resolution.
    """

    slot_id: int
    candidates: list[CandidatePoint]


@dataclass
class ResolvedProposal:
    """Store the concrete point chosen for one slot after duplicate filtering.

    ``slot_id`` identifies the owning slot and ``point`` is the single accepted
    lattice point for evaluation. The dataclass is immutable only by caller
    convention and does not mutate any external state.
    """

    slot_id: int
    point: Point


def _normalized_existing_candidate(
    point: Point,
    existing_point_to_slot: dict[Point, int] | None,
) -> CandidatePoint:
    """Return one environment point in ``CandidatePoint`` form for reporting.

    ``point`` is an existing lattice point already present in the environment
    and ``existing_point_to_slot`` optionally maps that point to its owning
    slot. The function returns a new ``CandidatePoint``, does not mutate caller
    state, and assumes missing slot ownership is acceptable for reporting.
    """

    return CandidatePoint(
        x=point[0],
        y=point[1],
        score=None,
        slot_id=None if existing_point_to_slot is None else existing_point_to_slot.get(point),
    )


def _sorted_candidate_list(slot: ProposalSlot) -> list[CandidatePoint]:
    """Return one slot's candidates with normalized slot ids and score sorting.

    ``slot`` provides the owning ``slot_id`` plus the raw candidate list. The
    function returns a new candidate list sorted by descending score with stable
    tie handling, does not mutate the input slot, and assumes the slot id is
    authoritative when candidate metadata omits it.
    """

    enumerated = list(enumerate(slot.candidates))
    sorted_candidates = sorted(
        enumerated,
        key=lambda item: (
            float("inf") if item[1].score is None else -item[1].score,
            item[0],
        ),
    )
    normalized: list[CandidatePoint] = []
    for _, candidate in sorted_candidates:
        if candidate.slot_id is not None and candidate.slot_id != slot.slot_id:
            raise ValueError(
                f"candidate slot_id {candidate.slot_id} does not match ProposalSlot {slot.slot_id}"
            )
        normalized.append(
            CandidatePoint(
                x=candidate.x,
                y=candidate.y,
                score=candidate.score,
                slot_id=slot.slot_id,
            )
        )
    return normalized


def _normalize_proposal_slots(proposal_slots: list[ProposalSlot]) -> list[ProposalSlot]:
    """Normalize slot candidates for deterministic duplicate resolution.

    ``proposal_slots`` is the ordered list of proposal groups for one step. The
    function returns a new list with per-slot candidates sorted by score and
    carrying slot ids, does not mutate the input proposals, and assumes slot ids
    are unique within the list.
    """

    normalized: list[ProposalSlot] = []
    seen_slot_ids: set[int] = set()
    for proposal_slot in proposal_slots:
        if proposal_slot.slot_id in seen_slot_ids:
            raise ValueError(f"duplicate proposal slot id {proposal_slot.slot_id}")
        seen_slot_ids.add(proposal_slot.slot_id)
        normalized.append(
            ProposalSlot(
                slot_id=proposal_slot.slot_id,
                candidates=_sorted_candidate_list(proposal_slot),
            )
        )
    return normalized


def _empty_repair_result() -> dict[str, Any]:
    """Return the canonical empty repair payload used across the pipeline.

    The function takes no inputs, returns a new dictionary with the standard
    repair keys, does not mutate caller state, and assumes downstream code wants
    list-valued fields even when nothing was rejected or deleted.
    """

    return {
        "rejected_slot_ids": [],
        "rejected_candidates": [],
        "rejected_proposal_points": [],
        "deleted_existing_points": [],
    }


def _new_conflict_report() -> dict[str, Any]:
    """Return one empty step-level conflict report dictionary.

    The function takes no inputs, returns the standard report structure used by
    ``search_repaired_proposals``, does not mutate external state, and assumes
    callers will append attempt records in chronological order.
    """

    return {
        "duplicate_conflicts": [],
        "attempts": [],
        "used_fallback_slot_ids": [],
        "unresolved_slot_ids": [],
        "final_selected_candidates": [],
        "termination_reason": "",
    }


def _selected_candidate_map(
    selected_candidates: list[CandidatePoint],
) -> dict[Point, CandidatePoint]:
    """Index selected proposal candidates by geometric point.

    ``selected_candidates`` is the duplicate-free proposal set for one attempt.
    The function returns a ``Point -> CandidatePoint`` mapping, does not mutate
    the input list, and assumes selected candidates are unique by coordinate.
    """

    return {(candidate.x, candidate.y): candidate for candidate in selected_candidates}


def collect_existing_slot_line_conflict_counts(
    conflict_report: dict[str, Any],
    allowed_slot_ids: set[int] | None = None,
) -> dict[int, int]:
    """Count unique final-attempt line conflicts for existing environment slots.

    ``conflict_report`` is the step-level payload returned by
    ``search_repaired_proposals``. The function inspects only the last recorded
    attempt, counts how many unique ``line_key`` values touch each
    ``existing_points`` slot, optionally filters to ``allowed_slot_ids``, and
    returns a ``slot_id -> count`` mapping with integer counts. The helper does
    not mutate the report, ignores proposal-side points entirely, and assumes
    missing ``slot_id`` metadata on existing points should be skipped.
    """

    attempts = conflict_report.get("attempts", [])
    if not attempts:
        return {}

    last_attempt = attempts[-1]
    slot_to_lines: dict[int, set[LineKey]] = {}
    allowed = None if allowed_slot_ids is None else set(allowed_slot_ids)

    for conflict_line in last_attempt.get("line_conflicts", []):
        line = conflict_line["line_key"]
        for existing_candidate in conflict_line.get("existing_points", []):
            slot_id = existing_candidate.slot_id
            if slot_id is None:
                continue
            if allowed is not None and slot_id not in allowed:
                continue
            slot_to_lines.setdefault(int(slot_id), set()).add(line)

    return {
        slot_id: len(lines)
        for slot_id, lines in sorted(slot_to_lines.items())
        if lines
    }


def compute_existing_legal_rows_by_column(
    existing_points: set[Point] | list[Point],
    n: int,
    candidate_columns: list[int] | set[int] | tuple[int, ...] | None = None,
    backend: ActiveLineBackend | None = None,
) -> dict[int, np.ndarray]:
    """Return existing-only legal row masks for candidate columns.

    ``existing_points`` is the current environment point set, ``n`` is the grid
    size, and ``candidate_columns`` optionally restricts which columns are
    computed. The helper returns a ``dict[int, np.ndarray]`` where each array
    has shape ``[n]`` and dtype ``bool``. Entry ``True`` means the lattice point
    ``(x, y)`` is legal with respect to the *existing* environment points only:
    the row is rejected if the point is already occupied or if it lies on any
    active line induced by two existing points. The function does not mutate the
    caller's backend or point set.
    """

    if candidate_columns is None:
        columns = list(range(int(n)))
    else:
        columns = sorted({int(column) for column in candidate_columns})

    legal_rows_by_column = {
        column: np.ones(int(n), dtype=bool)
        for column in columns
    }
    if not columns:
        return legal_rows_by_column

    existing_point_set = set(existing_points)
    for x, y in existing_point_set:
        if x in legal_rows_by_column and 0 <= y < n:
            legal_rows_by_column[x][y] = False

    scratch_backend = backend
    if scratch_backend is None:
        scratch_backend = ActiveLineBackend()
        scratch_backend.build_from_points(list(existing_point_set))

    for line in scratch_backend.line_to_points:
        A, B, C = line
        if B == 0:
            if A != 0 and C % A == 0:
                x_coord = C // A
                if x_coord in legal_rows_by_column:
                    legal_rows_by_column[x_coord][:] = False
            continue

        for column in columns:
            y_numerator = C - (A * column)
            if y_numerator % B != 0:
                continue
            y_coord = y_numerator // B
            if 0 <= y_coord < n:
                legal_rows_by_column[column][y_coord] = False

    return legal_rows_by_column


def _sorted_candidate_points(candidates: list[CandidatePoint]) -> list[CandidatePoint]:
    """Return candidates sorted by coordinates and slot id for stable reports.

    ``candidates`` is a list of proposal or environment points represented as
    ``CandidatePoint`` objects. The function returns a new sorted list, does not
    mutate the inputs, and assumes lexicographic coordinate order is adequate
    for deterministic conflict-report rendering.
    """

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.x,
            candidate.y,
            -1 if candidate.slot_id is None else candidate.slot_id,
            float("-inf") if candidate.score is None else candidate.score,
        ),
    )


def resolve_slot_candidates_for_duplicates(
    proposal_slots: list[ProposalSlot],
    existing_points: set[Point],
    *,
    existing_point_to_slot: dict[Point, int] | None = None,
    current_index_by_slot: dict[int, int] | None = None,
    attempt_index: int = 0,
    conflict_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one candidate per slot while recording duplicate conflicts.

    ``proposal_slots`` is the ordered proposal list for one step, with each
    slot carrying candidate points. ``existing_points`` is the already occupied
    environment set. ``existing_point_to_slot`` optionally names the owning slot
    of environment points, ``current_index_by_slot`` sets the starting candidate
    index for each slot, ``attempt_index`` labels report entries, and
    ``conflict_report`` receives duplicate-conflict records when provided. The
    function returns resolved proposals, selected candidates, updated slot-index
    state, unresolved slot ids, and fallback slot ids. It does not mutate the
    input proposals or environment state and assumes exact point equality is the
    only duplicate criterion.
    """

    normalized_slots = _normalize_proposal_slots(proposal_slots)
    current_indices = {} if current_index_by_slot is None else dict(current_index_by_slot)
    occupied_points = set(existing_points)
    selected_candidates: list[CandidatePoint] = []
    selected_candidate_by_point: dict[Point, CandidatePoint] = {}
    resolved: list[ResolvedProposal] = []
    unresolved_slot_ids: list[int] = []
    used_fallback_slot_ids: list[int] = []
    resolved_index_by_slot: dict[int, int] = {}

    if conflict_report is not None:
        conflict_report.setdefault("duplicate_conflicts", [])

    for proposal_slot in normalized_slots:
        start_index = current_indices.get(proposal_slot.slot_id, 0)
        if start_index < 0:
            raise ValueError(f"current_index_by_slot[{proposal_slot.slot_id}] must be non-negative")

        chosen_candidate: CandidatePoint | None = None
        chosen_index: int | None = None

        for index in range(start_index, len(proposal_slot.candidates)):
            candidate = proposal_slot.candidates[index]
            point = (candidate.x, candidate.y)

            if point in occupied_points:
                if conflict_report is not None:
                    conflict_report["duplicate_conflicts"].append(
                        {
                            "attempt_index": attempt_index,
                            "source": "existing",
                            "proposal": candidate,
                            "blocked_by": [
                                _normalized_existing_candidate(point, existing_point_to_slot)
                            ],
                        }
                    )
                continue

            proposal_duplicate = selected_candidate_by_point.get(point)
            if proposal_duplicate is not None:
                if conflict_report is not None:
                    conflict_report["duplicate_conflicts"].append(
                        {
                            "attempt_index": attempt_index,
                            "source": "proposal",
                            "proposal": candidate,
                            "blocked_by": [proposal_duplicate],
                        }
                    )
                continue

            chosen_candidate = candidate
            chosen_index = index
            selected_candidates.append(candidate)
            selected_candidate_by_point[point] = candidate
            resolved.append(ResolvedProposal(slot_id=proposal_slot.slot_id, point=point))
            resolved_index_by_slot[proposal_slot.slot_id] = index
            break

        if chosen_candidate is None or chosen_index is None:
            # Keep unresolved slots at their current starting index so they can
            # retry transient proposal-proposal duplicates in a later attempt.
            resolved_index_by_slot[proposal_slot.slot_id] = start_index
            unresolved_slot_ids.append(proposal_slot.slot_id)
            continue

        if chosen_index > 0:
            used_fallback_slot_ids.append(proposal_slot.slot_id)

    return {
        "resolved": resolved,
        "selected_candidates": selected_candidates,
        "current_index_by_slot": resolved_index_by_slot,
        "unresolved_slot_ids": unresolved_slot_ids,
        "used_fallback_slot_ids": used_fallback_slot_ids,
    }


def evaluate_proposals(
    existing_points: set[Point],
    resolved_proposals: list[ResolvedProposal],
    *,
    existing_point_to_slot: dict[Point, int] | None = None,
    selected_candidates: list[CandidatePoint] | None = None,
) -> dict[str, Any]:
    """Evaluate only collinearity conflicts for the resolved proposal set.

    ``existing_points`` is the current environment set and ``resolved_proposals``
    provides one resolved point per proposal slot. ``existing_point_to_slot``
    optionally maps environment points back to slot ids, and
    ``selected_candidates`` optionally supplies proposal scores and slot ids for
    richer conflict reporting. The function returns feasibility, per-line
    conflict records, and per-point conflict counts. It does not mutate
    environment state or the input collections and assumes duplicates were
    removed upstream by ``resolve_slot_candidates_for_duplicates``.
    """

    existing_set = set(existing_points)
    if selected_candidates is None:
        proposal_candidate_by_point = {
            proposal.point: CandidatePoint(
                x=proposal.point[0],
                y=proposal.point[1],
                score=None,
                slot_id=proposal.slot_id,
            )
            for proposal in resolved_proposals
        }
    else:
        proposal_candidate_by_point = _selected_candidate_map(selected_candidates)

    proposal_points = set(proposal_candidate_by_point)
    line_accumulator: dict[LineKey, dict[str, set[Point]]] = {}

    # Step A: proposal-existing lines.
    for proposal_point in sorted(proposal_points):
        for existing_point in sorted(existing_set):
            line = line_key(proposal_point, existing_point)
            bucket = line_accumulator.setdefault(
                line,
                {"existing_points": set(), "proposal_points": set()},
            )
            bucket["existing_points"].add(existing_point)
            bucket["proposal_points"].add(proposal_point)

    # Step B: proposal-proposal lines.
    proposal_list = sorted(proposal_points)
    for index, p_i in enumerate(proposal_list):
        for p_j in proposal_list[index + 1 :]:
            line = line_key(p_i, p_j)
            bucket = line_accumulator.setdefault(
                line,
                {"existing_points": set(), "proposal_points": set()},
            )
            bucket["proposal_points"].update((p_i, p_j))

    conflict_lines: list[dict[str, Any]] = []
    existing_point_conflict_counts: dict[Point, int] = {}
    proposal_point_conflict_counts: dict[Point, int] = {}

    for line in sorted(line_accumulator):
        bucket = line_accumulator[line]
        existing_on_line = bucket["existing_points"]
        proposal_on_line = bucket["proposal_points"]
        total_points = len(existing_on_line) + len(proposal_on_line)
        if total_points < 3:
            continue

        conflict_lines.append(
            {
                "line_key": line,
                "existing_points": _sorted_candidate_points(
                    [
                        _normalized_existing_candidate(point, existing_point_to_slot)
                        for point in existing_on_line
                    ]
                ),
                "proposal_points": _sorted_candidate_points(
                    [proposal_candidate_by_point[point] for point in proposal_on_line]
                ),
                "overflow": total_points - 2,
            }
        )

        for point in existing_on_line:
            existing_point_conflict_counts[point] = (
                existing_point_conflict_counts.get(point, 0) + 1
            )
        for point in proposal_on_line:
            proposal_point_conflict_counts[point] = (
                proposal_point_conflict_counts.get(point, 0) + 1
            )

    return {
        "is_feasible": not conflict_lines,
        "conflict_lines": conflict_lines,
        "existing_point_conflict_counts": existing_point_conflict_counts,
        "proposal_point_conflict_counts": proposal_point_conflict_counts,
    }


def search_repaired_proposals(
    proposal_slots: list[ProposalSlot],
    existing_points: set[Point],
    *,
    existing_point_to_slot: dict[Point, int] | None = None,
    num_search_attempts: int = 1,
    existing_delete_cost: float = 3.0,
    proposal_delete_cost: float = 1.0,
    allow_existing_deletions: bool = False,
) -> dict[str, Any]:
    """Run step-local resolve, evaluate, repair, and candidate advancement.

    ``proposal_slots`` is the ordered list of proposal slots for one env step
    and ``existing_points`` is the already occupied point set. Optional inputs
    provide environment slot ownership, search depth, and repair costs. The
    function returns the final resolved proposals, last repair result, and the
    accumulated step-level conflict report. It does not mutate the environment
    or the input proposals and assumes proposal search should advance only the
    slots rejected by repair.
    """

    from repair import greedy_repair

    if num_search_attempts < 1:
        raise ValueError("num_search_attempts must be >= 1")

    normalized_slots = _normalize_proposal_slots(proposal_slots)
    current_index_by_slot = {proposal_slot.slot_id: 0 for proposal_slot in normalized_slots}
    conflict_report = _new_conflict_report()
    used_fallback_slot_ids: set[int] = set()
    last_resolve_result: dict[str, Any] = {
        "resolved": [],
        "selected_candidates": [],
        "current_index_by_slot": {},
        "unresolved_slot_ids": [],
        "used_fallback_slot_ids": [],
    }
    last_eval_result: dict[str, Any] = {
        "is_feasible": True,
        "conflict_lines": [],
        "existing_point_conflict_counts": {},
        "proposal_point_conflict_counts": {},
    }
    last_repair_result: dict[str, Any] = _empty_repair_result()
    termination_reason = "max_attempts"

    for attempt_index in range(num_search_attempts):
        resolve_result = resolve_slot_candidates_for_duplicates(
            normalized_slots,
            existing_points,
            existing_point_to_slot=existing_point_to_slot,
            current_index_by_slot=current_index_by_slot,
            attempt_index=attempt_index,
            conflict_report=conflict_report,
        )
        used_fallback_slot_ids.update(resolve_result["used_fallback_slot_ids"])

        eval_result = evaluate_proposals(
            existing_points,
            resolve_result["resolved"],
            existing_point_to_slot=existing_point_to_slot,
            selected_candidates=resolve_result["selected_candidates"],
        )
        attempt_record = {
            "attempt_index": attempt_index,
            "selected_candidates": list(resolve_result["selected_candidates"]),
            "line_conflicts": list(eval_result["conflict_lines"]),
            "repair": _empty_repair_result(),
            "is_feasible": bool(eval_result["is_feasible"]),
        }

        last_resolve_result = resolve_result
        last_eval_result = eval_result

        if eval_result["is_feasible"]:
            conflict_report["attempts"].append(attempt_record)
            last_repair_result = _empty_repair_result()
            termination_reason = "feasible"
            break

        repair_result = greedy_repair(
            attempt_record,
            existing_delete_cost=existing_delete_cost,
            proposal_delete_cost=proposal_delete_cost,
            allow_existing_deletions=allow_existing_deletions,
        )
        attempt_record["repair"] = repair_result
        conflict_report["attempts"].append(attempt_record)
        last_repair_result = repair_result

        if attempt_index == num_search_attempts - 1:
            termination_reason = "max_attempts"
            break

        next_indices = dict(resolve_result["current_index_by_slot"])
        advanced_any = False
        for slot_id in repair_result["rejected_slot_ids"]:
            next_index = next_indices.get(slot_id, current_index_by_slot.get(slot_id, 0)) + 1
            next_indices[slot_id] = next_index
            slot_candidates = next(
                proposal_slot.candidates
                for proposal_slot in normalized_slots
                if proposal_slot.slot_id == slot_id
            )
            if next_index < len(slot_candidates):
                advanced_any = True

        current_index_by_slot = next_indices
        if not advanced_any:
            termination_reason = "candidate_exhausted"
            break

    rejected_slot_ids = set(last_repair_result["rejected_slot_ids"])
    final_resolved = [
        proposal
        for proposal in last_resolve_result["resolved"]
        if proposal.slot_id not in rejected_slot_ids
    ]
    final_selected_candidates = [
        candidate
        for candidate in last_resolve_result["selected_candidates"]
        if candidate.slot_id not in rejected_slot_ids
    ]
    final_unresolved_slot_ids = sorted(
        set(last_resolve_result["unresolved_slot_ids"]) | rejected_slot_ids
    )

    conflict_report["used_fallback_slot_ids"] = sorted(used_fallback_slot_ids)
    conflict_report["unresolved_slot_ids"] = final_unresolved_slot_ids
    conflict_report["final_selected_candidates"] = _sorted_candidate_points(final_selected_candidates)
    conflict_report["termination_reason"] = termination_reason

    return {
        "resolved_proposals": last_resolve_result["resolved"],
        "final_resolved_proposals": final_resolved,
        "repair_result": last_repair_result,
        "resolve_result": last_resolve_result,
        "eval_result": last_eval_result,
        "conflict_report": conflict_report,
    }
