from __future__ import annotations

import unittest

import numpy as np

from alphazero.config import AlphaZeroConfig
from alphazero.targets import attach_value_targets, build_root_marginal_targets, terminal_reward_from_points
from alphazero.types import ExecutedAction, EpisodeTransition, JointAction, RootSearchResult, SearchNode, SearchState


class AlphaZeroTargetTests(unittest.TestCase):
    def test_capped_exp2_terminal_reward(self) -> None:
        config = AlphaZeroConfig(n=10)

        self.assertAlmostEqual(terminal_reward_from_points(10, config, terminal_reward_baseline_points=10.0), 2.0**-6)
        self.assertAlmostEqual(terminal_reward_from_points(16, config, terminal_reward_baseline_points=10.0), 1.0)
        self.assertAlmostEqual(terminal_reward_from_points(18, config, terminal_reward_baseline_points=10.0), 1.0)

    def test_normalized_terminal_reward_mode_is_preserved(self) -> None:
        config = AlphaZeroConfig(n=10)
        config.reward.terminal_reward_mode = "normalized_points"

        self.assertAlmostEqual(terminal_reward_from_points(12, config, terminal_reward_baseline_points=10.0), 0.6)

    def test_value_targets_use_terminal_reward(self) -> None:
        config = AlphaZeroConfig(n=10)
        transition = EpisodeTransition(
            slot_state_ids=np.zeros(20, dtype=np.int64),
            episode_step=0,
            candidate_slot_ids=[0],
            fill_targets=[1.0],
            value_pool_ids=[[2]],
            value_targets_on_support=[[1.0]],
            value_weights=[1.0],
            step_reward=1.0,
            terminal_score=0.5,
            terminal_reward=0.25,
        )

        samples = attach_value_targets([transition], config, n=10)

        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0].value_target, 0.25)

    def test_root_marginals_use_executed_actions(self) -> None:
        config = AlphaZeroConfig(n=10)
        root_node = SearchNode(
            state=SearchState(slot_state_ids=np.zeros(20, dtype=np.int64), episode_step=0),
            candidate_slots=[0, 1],
            value_pools={0: [2, 3], 1: [4]},
            sampled_actions=[
                JointAction(active_slots=[0], values={0: 2}),
                JointAction(active_slots=[1], values={1: 4}),
            ],
            prior_logits=np.array([0.0, 0.0]),
            visit_counts=np.array([1, 1]),
            value_sums=np.array([0.5, 0.5]),
            executed_actions=[
                ExecutedAction(active_slots=[0], values={0: 2}),
                ExecutedAction(active_slots=[], values={}),
            ],
            step_rewards=[0.1, 0.0],
            stop_reasons=[None, "no_progress"],
            value=0.5,
        )
        result = RootSearchResult(
            root_node=root_node,
            improved_policy=np.array([0.75, 0.25]),
            best_action_index=0,
            completed_q=np.array([0.5, 0.5]),
        )
        marginals = build_root_marginal_targets(result, config)

        self.assertEqual(marginals["candidate_slot_ids"], [0, 1])
        self.assertAlmostEqual(marginals["fill_targets"][0], 0.75)
        self.assertAlmostEqual(marginals["fill_targets"][1], 0.0)
        self.assertEqual(marginals["value_targets_on_support"][0], [1.0, 0.0])
        self.assertEqual(marginals["value_weights"][1], 0.0)
