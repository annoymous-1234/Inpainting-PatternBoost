from __future__ import annotations

import unittest

import numpy as np
import torch

from alphazero.config import AlphaZeroConfig
from alphazero.losses import collate_alphazero_samples, compute_policy_losses, compute_value_loss
from alphazero.types import AlphaZeroSample


class AlphaZeroLossTests(unittest.TestCase):
    def test_policy_and_value_losses_are_finite(self) -> None:
        config = AlphaZeroConfig(n=10)
        samples = [
            AlphaZeroSample(
                slot_state_ids=np.zeros(20, dtype=np.int64),
                episode_step=0,
                candidate_slot_ids=[0, 1],
                fill_targets=[1.0, 0.0],
                value_pool_ids=[[2, 3], [4]],
                value_targets_on_support=[[0.7, 0.3], [0.0]],
                value_weights=[1.0, 0.0],
                value_target=0.5,
            ),
            AlphaZeroSample(
                slot_state_ids=np.zeros(20, dtype=np.int64),
                episode_step=1,
                candidate_slot_ids=[2],
                fill_targets=[0.5],
                value_pool_ids=[[5, 6]],
                value_targets_on_support=[[0.4, 0.6]],
                value_weights=[0.5],
                value_target=0.25,
            ),
        ]
        batch = collate_alphazero_samples(samples, torch.device("cpu"))
        unified_logits = torch.randn(2, 20, 11, dtype=torch.float32)
        losses = compute_policy_losses(unified_logits, batch, config)
        value_loss = compute_value_loss(torch.sigmoid(torch.randn(2)), batch["value_target"])

        self.assertTrue(torch.isfinite(losses["fill_loss"]))
        self.assertTrue(torch.isfinite(losses["val_loss"]))
        self.assertTrue(torch.isfinite(losses["policy_loss"]))
        self.assertTrue(torch.isfinite(value_loss))
