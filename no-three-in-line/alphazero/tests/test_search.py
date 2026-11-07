from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from alphazero.config import load_alphazero_config
from alphazero.model_adapter import build_alphazero_model_from_project_config
from alphazero.search import (
    _select_nonroot_action,
    compute_completed_q,
    compute_improved_policy_logits,
    compute_slot_select_scores,
    expand_nonroot_node,
    run_root_search,
    run_simulation_from_action,
    softmax_over_sampled_actions,
    simulate_action_from_state,
)
from alphazero.state import build_empty_root_state
from alphazero.tests.test_utils import write_test_config
from alphazero.types import JointAction, SearchNode, SearchState


class AlphaZeroSearchTests(unittest.TestCase):
    def test_simulate_action_and_no_progress_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            empty_state = build_empty_root_state(10)

            first = simulate_action_from_state(
                empty_state,
                JointAction(active_slots=[0], values={0: 4}),
                n=10,
                config=config,
            )
            self.assertEqual(first.executed_action.active_slots, [0])
            self.assertAlmostEqual(first.step_reward, 1.0 / 20.0)
            self.assertIsNone(first.stop_reason)

            duplicate = simulate_action_from_state(
                first.next_state,
                JointAction(active_slots=[1], values={1: 4}),
                n=10,
                config=config,
            )
            self.assertEqual(duplicate.executed_action.active_slots, [])
            self.assertEqual(duplicate.stop_reason, "no_progress")

    def test_simulate_action_can_disable_no_progress_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={"search": {"terminate_on_no_progress": False}},
                )
            )
            empty_state = build_empty_root_state(10)

            first = simulate_action_from_state(
                empty_state,
                JointAction(active_slots=[0], values={0: 4}),
                n=10,
                config=config,
            )
            duplicate = simulate_action_from_state(
                first.next_state,
                JointAction(active_slots=[1], values={1: 4}),
                n=10,
                config=config,
            )

            self.assertEqual(duplicate.executed_action.active_slots, [])
            self.assertIsNone(duplicate.stop_reason)
            self.assertEqual(duplicate.next_state.episode_step, first.next_state.episode_step + 1)

    def test_simulate_action_insufficient_slots_uses_nonroot_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={"search": {"J_root": 1, "J_nonroot": 2, "J_root_prime": 3}},
                )
            )
            with mock.patch("alphazero.search.count_feasible_masked_slots", return_value=1):
                result = simulate_action_from_state(
                    build_empty_root_state(10),
                    JointAction(active_slots=[0], values={0: 4}),
                    n=10,
                    config=config,
                )
            self.assertEqual(result.stop_reason, "insufficient_feasible_slots")

    def test_mcts_terminal_leaf_uses_configured_terminal_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=4,
                    device="cpu",
                    alphazero_overrides={"search": {"J_root": 1, "J_nonroot": 8, "J_root_prime": 2}},
                )
            )
            model = build_alphazero_model_from_project_config(config)
            node = SearchNode(
                state=build_empty_root_state(4),
                sampled_actions=[JointAction(active_slots=[0], values={0: 0})],
            )

            total_return = run_simulation_from_action(
                node,
                0,
                ply_from_root=1,
                model=model,
                rng=np.random.default_rng(0),
                config=config,
                terminal_reward_baseline_points=0.0,
            )

            self.assertAlmostEqual(total_return, 2.0 ** (1.0 - 0.0 - 6.0))

    def test_expand_nonroot_node_caches_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            rng = np.random.default_rng(0)
            node = SearchNode(
                state=build_empty_root_state(10),
                unified_logits=np.zeros((20, 11), dtype=np.float32),
                value=0.5,
            )
            expand_nonroot_node(node, rng, config)
            first_fixed_slots = list(node.fixed_slots)
            first_value_pools = {slot_id: list(pool) for slot_id, pool in node.value_pools.items()}
            first_actions = [(list(action.active_slots), dict(action.values)) for action in node.sampled_actions]

            expand_nonroot_node(node, np.random.default_rng(123), config)
            self.assertEqual(node.fixed_slots, first_fixed_slots)
            self.assertEqual(node.value_pools, first_value_pools)
            self.assertEqual(
                [(list(action.active_slots), dict(action.values)) for action in node.sampled_actions],
                first_actions,
            )

    def test_root_and_nonroot_action_widths_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={
                        "search": {
                            "J_root": 2,
                            "J_nonroot": 3,
                            "J_root_prime": 4,
                        }
                    },
                )
            )
            model = build_alphazero_model_from_project_config(config)
            root_result = run_root_search(build_empty_root_state(10), model, np.random.default_rng(0), config)
            self.assertTrue(root_result.root_node.sampled_actions)
            self.assertTrue(
                all(len(action.active_slots) == config.search.J_root for action in root_result.root_node.sampled_actions)
            )

            node = SearchNode(
                state=build_empty_root_state(10),
                unified_logits=np.zeros((20, 11), dtype=np.float32),
                value=0.5,
            )
            expand_nonroot_node(node, np.random.default_rng(0), config)
            self.assertEqual(len(node.fixed_slots), config.search.J_nonroot)
            self.assertTrue(all(len(action.active_slots) == config.search.J_nonroot for action in node.sampled_actions))

    def test_expand_nonroot_node_computes_legality_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            node = SearchNode(
                state=build_empty_root_state(10),
                unified_logits=np.zeros((20, 11), dtype=np.float32),
                value=0.5,
            )

            with mock.patch(
                "alphazero.search.compute_existing_legal_rows_by_column",
                return_value={column: np.ones(10, dtype=bool) for column in range(10)},
            ) as legality_mock:
                expand_nonroot_node(node, np.random.default_rng(0), config)
                expand_nonroot_node(node, np.random.default_rng(1), config)

            self.assertEqual(legality_mock.call_count, 1)

    def test_value_pools_respect_existing_action_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            node = SearchNode(
                state=build_empty_root_state(10),
                unified_logits=np.zeros((20, 11), dtype=np.float32),
                value=0.5,
            )

            legal_rows = {column: np.zeros(10, dtype=bool) for column in range(10)}
            legal_rows[0][3] = True
            legal_rows[1][5] = True

            with mock.patch("alphazero.search.compute_existing_legal_rows_by_column", return_value=legal_rows):
                expand_nonroot_node(node, np.random.default_rng(0), config)

            self.assertTrue(node.fixed_slots)
            for slot_id, pool in node.value_pools.items():
                self.assertEqual(pool, [3] if slot_id // 2 == 0 else [5])
            for action in node.sampled_actions:
                for slot_id, value in action.values.items():
                    self.assertTrue(bool(legal_rows[slot_id // 2][value]))

    def test_select_nonroot_action_matches_improved_policy_visits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            node = SearchNode(
                state=build_empty_root_state(10),
                value=0.5,
                sampled_actions=[
                    JointAction(active_slots=[0], values={0: 0}),
                    JointAction(active_slots=[2], values={2: 1}),
                    JointAction(active_slots=[4], values={4: 2}),
                ],
                prior_logits=np.asarray([0.0, 2.0, 1.0], dtype=np.float64),
                visit_counts=np.asarray([3, 0, 0], dtype=np.int64),
                value_sums=np.asarray([2.4, 0.0, 0.0], dtype=np.float64),
            )

            completed_q = compute_completed_q(node.visit_counts, node.value_sums, float(node.value))
            logits = compute_improved_policy_logits(
                node.prior_logits,
                completed_q,
                config.search.tau_pi_nonroot,
                config.search.c_Q,
            )
            improved_policy = softmax_over_sampled_actions(logits)
            expected = int(
                np.argmax(improved_policy - node.visit_counts / float(1 + int(node.visit_counts.sum())))
            )

            self.assertEqual(_select_nonroot_action(node, config), expected)
            self.assertEqual(expected, 1)

    def test_select_nonroot_action_uses_value_for_unvisited_completed_q(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            node = SearchNode(
                state=build_empty_root_state(10),
                value=0.75,
                sampled_actions=[
                    JointAction(active_slots=[0], values={0: 0}),
                    JointAction(active_slots=[2], values={2: 1}),
                ],
                prior_logits=np.zeros((2,), dtype=np.float64),
                visit_counts=np.asarray([1, 0], dtype=np.int64),
                value_sums=np.asarray([0.1, 0.0], dtype=np.float64),
            )

            completed_q = compute_completed_q(node.visit_counts, node.value_sums, float(node.value))

            self.assertAlmostEqual(float(completed_q[1]), 0.75)
            self.assertEqual(_select_nonroot_action(node, config), 1)
            self.assertEqual(_select_nonroot_action(node, config), 1)

    def test_run_root_search_returns_probability_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            model = build_alphazero_model_from_project_config(config)
            rng = np.random.default_rng(0)
            result = run_root_search(build_empty_root_state(10), model, rng, config)

            self.assertGreater(len(result.root_node.sampled_actions), 0)
            self.assertAlmostEqual(float(result.improved_policy.sum()), 1.0, places=6)
            self.assertGreaterEqual(result.best_action_index, 0)
            self.assertEqual(result.root_mcts_conflict_counts, {})

    def test_run_root_search_reports_root_level_mcts_conflict_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            model = build_alphazero_model_from_project_config(config)
            rng = np.random.default_rng(0)
            root_state_ids = np.zeros(20, dtype=np.int64)
            root_state_ids[0] = 1
            root_state_ids[2] = 2
            root_state = SearchState(slot_state_ids=root_state_ids, episode_step=2)
            sampled_actions = [
                JointAction(active_slots=[4], values={4: 2}),
                JointAction(active_slots=[6], values={6: 0}),
                JointAction(active_slots=[8], values={8: 0}),
                JointAction(active_slots=[10], values={10: 0}),
            ]

            with (
                mock.patch("alphazero.search.sample_root_candidate_slots", return_value=[4, 6, 8, 10]),
                mock.patch(
                    "alphazero.search.sample_value_pools_for_slots",
                    return_value={4: [2], 6: [0], 8: [0], 10: [0]},
                ),
                mock.patch("alphazero.search.sample_root_joint_actions", return_value=sampled_actions),
                mock.patch("alphazero.search.get_considered_visit_sequence", return_value=[]),
            ):
                result = run_root_search(root_state, model, rng, config)

            self.assertEqual(result.root_mcts_conflict_counts, {0: 0.25, 2: 0.25})

    def test_mcts_internal_best_update_threshold_skips_non_improving_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            model = build_alphazero_model_from_project_config(config)
            rng = np.random.default_rng(0)
            observed: list[tuple[int, int]] = []

            run_root_search(
                build_empty_root_state(10),
                model,
                rng,
                config,
                state_observer=lambda slot_state_ids, ply_from_root: observed.append(
                    (int((slot_state_ids != 0).sum()), ply_from_root)
                ),
                best_overall_filled_count_fn=lambda: 100,
            )
            self.assertEqual(observed, [])

            run_root_search(
                build_empty_root_state(10),
                model,
                np.random.default_rng(0),
                config,
                state_observer=lambda slot_state_ids, ply_from_root: observed.append(
                    (int((slot_state_ids != 0).sum()), ply_from_root)
                ),
                best_overall_filled_count_fn=lambda: 0,
            )
            self.assertTrue(observed)
            self.assertTrue(all(points > 0 for points, _ in observed))

    def test_root_search_returns_no_actions_when_feasible_slots_are_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            model = build_alphazero_model_from_project_config(config)
            rng = np.random.default_rng(0)

            with mock.patch(
                "alphazero.search.compute_existing_legal_rows_by_column",
                return_value={column: np.zeros(10, dtype=bool) for column in range(10)},
            ):
                result = run_root_search(build_empty_root_state(10), model, rng, config)

            self.assertEqual(result.best_action_index, -1)
            self.assertEqual(len(result.root_node.sampled_actions), 0)
            self.assertEqual(result.improved_policy.size, 0)

    def test_root_search_insufficient_slots_uses_root_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={"search": {"J_root": 3, "J_nonroot": 1, "J_root_prime": 4}},
                )
            )
            model = build_alphazero_model_from_project_config(config)
            legal_rows = {column: np.zeros(10, dtype=bool) for column in range(10)}
            legal_rows[0][0] = True

            with mock.patch("alphazero.search.compute_existing_legal_rows_by_column", return_value=legal_rows):
                result = run_root_search(build_empty_root_state(10), model, np.random.default_rng(0), config)

            self.assertEqual(result.best_action_index, -1)
            self.assertEqual(len(result.root_node.sampled_actions), 0)

    def test_slot_select_scores_include_mask_scarcity_and_entropy_terms(self) -> None:
        mask_logits = np.asarray([2.0, 0.0], dtype=np.float64)
        value_logits = np.asarray(
            [
                [2.0, 2.0, 2.0],
                [8.0, -8.0, -8.0],
            ],
            dtype=np.float64,
        )
        legal_row_counts = np.asarray([2.0, 1.0], dtype=np.float64)

        scores = compute_slot_select_scores(
            mask_logits=mask_logits,
            value_logits=value_logits,
            legal_row_counts=legal_row_counts,
            tau=1.0,
            coef_mask_logit=1.0,
            coef_scarcity=2.0,
            coef_entropy=0.5,
        )

        entropy = []
        for row_logits in value_logits:
            stable = row_logits - np.max(row_logits)
            probs = np.exp(stable) / np.sum(np.exp(stable))
            entropy.append(-float(np.sum(probs * np.log(probs))))
        expected = np.asarray(
            [
                -2.0 + 2.0 * (1.0 / 2.0) - 0.5 * entropy[0],
                0.0 + 2.0 * (1.0 / 1.0) - 0.5 * entropy[1],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(scores, expected, atol=1e-6)
