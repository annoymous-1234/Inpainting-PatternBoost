from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from env import NoThreeInLineEnv

from alphazero.config import load_alphazero_config
from alphazero.self_play import _compute_episode_stop_reason, run_self_play_episode_with_trace
from alphazero.state import build_empty_root_state, search_state_from_env
from alphazero.tests.test_utils import write_test_config
from alphazero.types import ExecutedAction, JointAction, RootSearchResult, SearchNode, SearchState


def _stub_root_result(
    state: SearchState,
    *,
    action: JointAction,
    candidate_slot_ids: list[int],
    value_pool_ids: dict[int, list[int]],
    executed_action: ExecutedAction,
    step_reward: float,
    root_mcts_conflict_counts: dict[int, float],
) -> RootSearchResult:
    """Build one minimal deterministic root-search result for self-play tests."""

    root_node = SearchNode(
        state=state,
        expanded=True,
        value=0.5,
        candidate_slots=list(candidate_slot_ids),
        value_pools={slot_id: list(pool) for slot_id, pool in value_pool_ids.items()},
        sampled_actions=[action],
        prior_logits=np.zeros(1, dtype=np.float64),
        visit_counts=np.zeros(1, dtype=np.int64),
        value_sums=np.zeros(1, dtype=np.float64),
        executed_actions=[executed_action],
        step_rewards=[step_reward],
        stop_reasons=[None],
        root_action_gumbels=np.zeros(1, dtype=np.float64),
    )
    return RootSearchResult(
        root_node=root_node,
        improved_policy=np.asarray([1.0], dtype=np.float64),
        best_action_index=0,
        completed_q=np.asarray([0.5], dtype=np.float64),
        root_mcts_conflict_counts=dict(root_mcts_conflict_counts),
    )


class AlphaZeroSelfPlayTests(unittest.TestCase):
    def test_real_rollout_insufficient_slots_uses_root_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={"search": {"J_root": 2, "J_nonroot": 3, "J_root_prime": 4}},
                )
            )

            with mock.patch("alphazero.self_play.count_feasible_masked_slots", return_value=2):
                stop_reason = _compute_episode_stop_reason(
                    np.zeros(20, dtype=np.int64),
                    env_done=False,
                    accepted_any=True,
                    config=config,
                )

            self.assertIsNone(stop_reason)

    def test_self_play_stops_when_root_has_no_feasible_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            empty_root = RootSearchResult(
                root_node=SearchNode(
                    state=build_empty_root_state(10),
                    expanded=True,
                    value=0.5,
                    candidate_slots=[],
                    value_pools={},
                    sampled_actions=[],
                    prior_logits=np.zeros((0,), dtype=np.float64),
                    visit_counts=np.zeros((0,), dtype=np.int64),
                    value_sums=np.zeros((0,), dtype=np.float64),
                    executed_actions=[],
                    step_rewards=[],
                    stop_reasons=[],
                    root_action_gumbels=np.zeros((0,), dtype=np.float64),
                ),
                improved_policy=np.zeros((0,), dtype=np.float64),
                best_action_index=-1,
                completed_q=np.zeros((0,), dtype=np.float64),
                root_mcts_conflict_counts={},
            )

            with mock.patch("alphazero.self_play.run_root_search", return_value=empty_root):
                samples, trace, episode_summary = run_self_play_episode_with_trace(
                    model=object(),
                    rng=np.random.default_rng(0),
                    config=config,
                )

            self.assertEqual(samples, [])
            self.assertEqual(trace, [])
            self.assertEqual(episode_summary["stop_reason"], "insufficient_feasible_slots")

    def test_self_play_value_targets_use_terminal_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={"self_play": {"max_episode_steps": 1}},
                )
            )
            root_result = _stub_root_result(
                SearchState(slot_state_ids=np.zeros(20, dtype=np.int64), episode_step=0),
                action=JointAction(active_slots=[0], values={0: 0}),
                candidate_slot_ids=[0],
                value_pool_ids={0: [0]},
                executed_action=ExecutedAction(active_slots=[0], values={0: 0}),
                step_reward=1.0 / float(2 * config.n),
                root_mcts_conflict_counts={},
            )

            with mock.patch("alphazero.self_play.run_root_search", return_value=root_result):
                samples, _, episode_summary = run_self_play_episode_with_trace(
                    model=object(),
                    rng=np.random.default_rng(0),
                    config=config,
                    terminal_reward_baseline_points=0.0,
                )

            expected_reward = 2.0 ** (1.0 - 0.0 - 6.0)
            self.assertEqual(len(samples), 1)
            self.assertAlmostEqual(episode_summary["terminal_reward"], expected_reward)
            self.assertAlmostEqual(samples[0].value_target, expected_reward)

    def test_self_play_uses_live_env_and_applies_remask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={
                        "self_play": {"max_episode_steps": 2},
                        "local_anneal": {
                            "enabled": True,
                            "alpha_env": 1.0,
                            "alpha_mcts": 1.0,
                            "trigger_steps": [2],
                            "num_slots_to_remask": 1,
                            "min_active_age_for_remask": 1,
                            "sampling_temperature": 1.0,
                        },
                    },
                )
            )
            step_reward = 1.0 / float(2 * config.n)
            captured_envs: list[NoThreeInLineEnv] = []

            class CapturingEnv(NoThreeInLineEnv):
                """Record the live env instance created by self-play."""

                def __init__(self, *args, **kwargs) -> None:
                    super().__init__(*args, **kwargs)
                    captured_envs.append(self)

            root_results: list[RootSearchResult] = [
                _stub_root_result(
                    SearchState(slot_state_ids=np.zeros(20, dtype=np.int64), episode_step=0),
                    action=JointAction(active_slots=[0], values={0: 0}),
                    candidate_slot_ids=[0],
                    value_pool_ids={0: [0]},
                    executed_action=ExecutedAction(active_slots=[0], values={0: 0}),
                    step_reward=step_reward,
                    root_mcts_conflict_counts={},
                ),
                _stub_root_result(
                    SearchState(
                        slot_state_ids=np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        episode_step=1,
                    ),
                    action=JointAction(active_slots=[2], values={2: 1}),
                    candidate_slot_ids=[2],
                    value_pool_ids={2: [1]},
                    executed_action=ExecutedAction(active_slots=[2], values={2: 1}),
                    step_reward=step_reward,
                    root_mcts_conflict_counts={0: 1.0},
                ),
            ]

            with (
                mock.patch("alphazero.self_play.NoThreeInLineEnv", CapturingEnv),
                mock.patch("alphazero.self_play.run_root_search", side_effect=root_results),
            ):
                samples, trace, episode_summary = run_self_play_episode_with_trace(
                    model=object(),
                    rng=np.random.default_rng(0),
                    config=config,
                )

            self.assertEqual(len(samples), 2)
            self.assertEqual(len(trace), 2)
            self.assertEqual(trace[1]["remasked_slots"], [0])
            self.assertEqual(trace[1]["mcts_conflict_counts"], {0: 1.0})
            self.assertEqual(trace[1]["points_after_env_step"], 2)
            self.assertEqual(trace[1]["points_after_remask"], 1)
            self.assertEqual(episode_summary["num_remask_events"], 1)
            self.assertEqual(episode_summary["total_remasked_slots"], 1)
            self.assertEqual(float(episode_summary["final_slot_conflict_mass"][0]), 0.0)
            self.assertEqual(float(episode_summary["final_slot_active_age"][0]), 0.0)
            self.assertEqual(float(episode_summary["final_slot_active_age"][2]), 1.0)

            final_env_state = search_state_from_env(captured_envs[0])
            np.testing.assert_array_equal(
                final_env_state.slot_state_ids,
                episode_summary["terminal_slot_state_ids"],
            )

    def test_self_play_can_continue_after_no_progress_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(
                write_test_config(
                    tmpdir,
                    n=10,
                    device="cpu",
                    alphazero_overrides={
                        "search": {"terminate_on_no_progress": False},
                        "self_play": {"max_episode_steps": 3},
                    },
                )
            )
            step_reward = 1.0 / float(2 * config.n)
            root_results: list[RootSearchResult] = [
                _stub_root_result(
                    SearchState(slot_state_ids=np.zeros(20, dtype=np.int64), episode_step=0),
                    action=JointAction(active_slots=[0], values={0: 0}),
                    candidate_slot_ids=[0],
                    value_pool_ids={0: [0]},
                    executed_action=ExecutedAction(active_slots=[0], values={0: 0}),
                    step_reward=step_reward,
                    root_mcts_conflict_counts={},
                ),
                _stub_root_result(
                    SearchState(
                        slot_state_ids=np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        episode_step=1,
                    ),
                    action=JointAction(active_slots=[1], values={1: 0}),
                    candidate_slot_ids=[1],
                    value_pool_ids={1: [0]},
                    executed_action=ExecutedAction(active_slots=[], values={}),
                    step_reward=0.0,
                    root_mcts_conflict_counts={},
                ),
                _stub_root_result(
                    SearchState(
                        slot_state_ids=np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        episode_step=2,
                    ),
                    action=JointAction(active_slots=[2], values={2: 1}),
                    candidate_slot_ids=[2],
                    value_pool_ids={2: [1]},
                    executed_action=ExecutedAction(active_slots=[2], values={2: 1}),
                    step_reward=step_reward,
                    root_mcts_conflict_counts={},
                ),
            ]

            with mock.patch("alphazero.self_play.run_root_search", side_effect=root_results):
                samples, trace, episode_summary = run_self_play_episode_with_trace(
                    model=object(),
                    rng=np.random.default_rng(0),
                    config=config,
                )

            self.assertEqual(len(samples), 3)
            self.assertEqual(len(trace), 3)
            self.assertIsNone(trace[1]["stop_reason"])
            self.assertEqual(trace[1]["executed_action"].active_slots, [])
            self.assertEqual(trace[-1]["stop_reason"], "episode_step_limit")
            self.assertEqual(episode_summary["final_points"], 2)
