from __future__ import annotations

import unittest

from alphazero.state import (
    apply_executed_action_to_slot_state,
    build_empty_root_state,
    filled_count,
    masked_count,
    slot_state_ids_to_points_and_assignment,
)
from alphazero.types import ExecutedAction


class AlphaZeroStateTests(unittest.TestCase):
    def test_state_helpers_round_trip_slot_assignment(self) -> None:
        state = build_empty_root_state(4)
        executed = ExecutedAction(active_slots=[1, 6], values={1: 3, 6: 0})
        next_slot_state = apply_executed_action_to_slot_state(state.slot_state_ids, executed, n=4)
        points, point_to_slot = slot_state_ids_to_points_and_assignment(next_slot_state)

        self.assertEqual(filled_count(next_slot_state), 2)
        self.assertEqual(masked_count(next_slot_state), 6)
        self.assertEqual(sorted(points), [(0, 3), (3, 0)])
        self.assertEqual(point_to_slot[(0, 3)], 1)
        self.assertEqual(point_to_slot[(3, 0)], 6)
