from __future__ import annotations

from geometry import Point, line_key
from env import NoThreeInLineEnv

from .config import ValidationConfig
from .encoding import decode_clean_state_to_points
from .types import EncodedSolutionRecord, ValidationResult


def check_duplicates(points: list[Point]) -> list[str]:
    """Return duplicate-point validation errors for a raw point list."""

    if len(points) != len(set(points)):
        return ["solution contains duplicate points"]
    return []


def check_bounds(points: list[Point], n: int) -> list[str]:
    """Return out-of-bounds validation errors for a raw point list."""

    errors: list[str] = []
    for point in points:
        x, y = point
        if not (0 <= x < n and 0 <= y < n):
            errors.append(f"point out of bounds for n={n}: {point}")
    return errors


def check_no_three_collinear(points: list[Point]) -> list[str]:
    """Return collinearity validation errors for a raw point list."""

    point_list = sorted(set(points))
    line_to_points: dict[tuple[int, int, int], set[Point]] = {}
    for index, p in enumerate(point_list):
        for q in point_list[index + 1 :]:
            points_on_line = line_to_points.setdefault(line_key(p, q), set())
            points_on_line.update((p, q))
            if len(points_on_line) >= 3:
                return [f"solution contains three or more collinear points on line {line_key(p, q)}"]
    return []


def _check_clean_encoding(record: EncodedSolutionRecord, n: int) -> list[str]:
    """Return structural validation errors for canonical clean slot states."""

    errors: list[str] = []
    if len(record.clean_state_ids) != 2 * n:
        errors.append(f"clean_state_ids must have length {2 * n}")
        return errors

    expected_filled_slot_ids = [slot_id for slot_id, state_id in enumerate(record.clean_state_ids) if state_id != 0]
    if expected_filled_slot_ids != record.filled_slot_ids:
        errors.append("filled_slot_ids do not match clean_state_ids")
    if record.point_count != len(record.filled_slot_ids):
        errors.append("point_count does not match filled_slot_ids")

    for column in range(n):
        left_state = record.clean_state_ids[2 * column]
        right_state = record.clean_state_ids[2 * column + 1]
        if not (0 <= left_state <= n and 0 <= right_state <= n):
            errors.append(f"column {column} has state ids outside the valid range")
            continue
        if left_state == 0 and right_state != 0:
            errors.append(f"column {column} has a filled second slot with a masked first slot")
        if left_state != 0 and right_state != 0 and left_state >= right_state:
            errors.append(f"column {column} is not canonically sorted")

    return errors


def check_round_trip(record: EncodedSolutionRecord, n: int) -> list[str]:
    """Return errors when clean-state decoding does not recover the original points."""

    decoded_points = decode_clean_state_to_points(record.clean_state_ids, n)
    if decoded_points != sorted(record.points):
        return ["clean_state_ids do not round-trip back to the original point set"]
    return []


def check_env_consistency(record: EncodedSolutionRecord, n: int) -> list[str]:
    """Return env-backed consistency errors for an encoded clean solution."""

    init_points = decode_clean_state_to_points(record.clean_state_ids, n)
    init_slot_assignment = {
        (slot_id // 2, state_id - 1): slot_id
        for slot_id, state_id in enumerate(record.clean_state_ids)
        if state_id != 0
    }
    env = NoThreeInLineEnv(n=n, num_slots=2 * n)
    obs = env.reset(init_points=init_points, init_slot_assignment=init_slot_assignment)

    errors: list[str] = []
    if obs["selected_points"] != sorted(record.points):
        errors.append("env observation does not match the encoded point set")
    if env.point_to_slot != init_slot_assignment:
        errors.append("env point_to_slot mapping does not match the canonical slot assignment")
    for slot_id, state_id in enumerate(record.clean_state_ids):
        slot_state = env.slot_states[slot_id]
        if state_id == 0:
            if slot_state.status != "MASKED" or slot_state.point is not None:
                errors.append(f"env slot {slot_id} is inconsistent with a masked clean token")
        else:
            expected_point = (slot_id // 2, state_id - 1)
            if slot_state.status != "FILLED" or slot_state.point != expected_point:
                errors.append(f"env slot {slot_id} is inconsistent with clean_state_ids")
    return errors


def validate_raw_solution(points: list[Point], n: int, config: ValidationConfig) -> ValidationResult:
    """Validate a raw point-set solution before canonical encoding."""

    if not config.enabled:
        return ValidationResult(is_valid=True, errors=[])

    errors = check_duplicates(points)
    errors.extend(check_bounds(points, n))
    errors.extend(check_no_three_collinear(points))
    return ValidationResult(is_valid=not errors, errors=errors)


def validate_encoded_solution(
    record: EncodedSolutionRecord,
    n: int,
    config: ValidationConfig,
) -> ValidationResult:
    """Validate one encoded clean solution and optional env consistency checks."""

    if not config.enabled:
        return ValidationResult(is_valid=True, errors=[])

    errors = check_duplicates(record.points)
    errors.extend(check_bounds(record.points, n))
    errors.extend(check_no_three_collinear(record.points))
    errors.extend(_check_clean_encoding(record, n))
    if config.round_trip_check:
        errors.extend(check_round_trip(record, n))
    if config.validate_with_env:
        errors.extend(check_env_consistency(record, n))
    return ValidationResult(is_valid=not errors, errors=errors)
