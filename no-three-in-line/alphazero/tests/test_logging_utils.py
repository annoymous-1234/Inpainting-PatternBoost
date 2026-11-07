from __future__ import annotations

import unittest

import numpy as np

from alphazero.logging_utils import validate_candidate_state


class AlphaZeroLoggingUtilsTests(unittest.TestCase):
    def test_validate_candidate_state_rejects_three_collinear_points(self) -> None:
        slot_state_ids = np.zeros(8, dtype=np.int64)
        slot_state_ids[0] = 1
        slot_state_ids[2] = 2
        slot_state_ids[4] = 3

        result = validate_candidate_state(slot_state_ids, n=4)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any("collinear" in error for error in result["validation_errors"]))
