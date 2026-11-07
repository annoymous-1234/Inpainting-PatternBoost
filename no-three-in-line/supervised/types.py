from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from geometry import Point


@dataclass
class RawSolutionRecord:
    """Store one raw final-solution record loaded from JSONL."""

    solution_id: str
    points: list[Point]


@dataclass
class ValidationResult:
    """Store validation status plus any collected error messages."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class EncodedSolutionRecord:
    """Store one canonical clean solution plus precomputed slot orders."""

    solution_id: str
    points: list[Point]
    clean_state_ids: list[int]
    filled_slot_ids: list[int]
    point_count: int
    slot_orders: list[list[int]]


@dataclass
class CorruptionResult:
    """Store one dynamically corrupted training input derived from a clean state."""

    input_state_ids: list[int]
    masked_slot_ids: list[int]
    missing_slot_ids: list[int]
    step_id: int
    q_t: float
    current_point_count: int
    masked_ratio_actual: float


@dataclass
class SupervisedSample:
    """Store one fully materialized supervised training sample.

    The tensor fields describe one unbatched example on CPU before collation.
    ``input_state_ids`` and ``target_state_ids`` are ``LongTensor[num_slots]``
    with a shared slot ordering. ``step_id`` is a scalar ``LongTensor[]`` and
    ``current_point_count`` is a scalar ``FloatTensor[]``. ``token_weights``
    and ``supervision_mask`` are ``FloatTensor[num_slots]``; the supervision
    mask uses ``1.0`` for supervised slots and ``0.0`` elsewhere.

    Constructing the dataclass returns a new container, does not mutate
    external state, and assumes all tensor fields refer to the same example.
    """

    input_state_ids: Tensor
    target_state_ids: Tensor
    step_id: Tensor
    current_point_count: Tensor
    token_weights: Tensor
    supervision_mask: Tensor
    task_type: str
    metadata: dict[str, object]
