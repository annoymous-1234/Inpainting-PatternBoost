from __future__ import annotations

import tempfile
import unittest

from alphazero.config import load_alphazero_config
from alphazero.model_adapter import build_alphazero_model_from_project_config
from alphazero.tests.test_utils import write_test_config


class AlphaZeroConfigTests(unittest.TestCase):
    def test_load_config_and_force_model_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(tmpdir, n=10, device="cpu")
            config = load_alphazero_config(config_path)
            self.assertEqual(config.n, 10)
            self.assertEqual(config.search.J_root, 1)
            self.assertEqual(config.search.J_nonroot, 1)
            self.assertTrue(config.search.terminate_on_no_progress)
            self.assertTrue(config.search.use_existing_action_mask)
            self.assertEqual(config.search.slot_score_coef_mask_logit, 1.0)
            self.assertEqual(config.search.slot_score_coef_scarcity, 0.0)
            self.assertEqual(config.search.slot_score_coef_entropy, 0.0)
            self.assertEqual(config.reward.step_weight, 0.0)
            self.assertEqual(config.reward.terminal_weight, 1.0)
            self.assertEqual(config.reward.terminal_reward_mode, "capped_exp2_advantage")
            self.assertEqual(config.reward.terminal_reward_beta, 1.0)
            self.assertEqual(config.reward.terminal_reward_cap, 6.0)
            self.assertIsNone(config.reward.initial_terminal_baseline_points)
            self.assertEqual(config.reward.initial_terminal_baseline_fraction, 0.75)
            self.assertEqual(config.self_play.sample_prefix_steps, 4)
            self.assertFalse(config.local_anneal.enabled)
            self.assertEqual(config.local_anneal.trigger_steps, [])

            model = build_alphazero_model_from_project_config(config)
            self.assertTrue(model.use_global_token)
            self.assertTrue(model.use_value_head)
            self.assertEqual(model.token_embedding.num_steps, 21)

    def test_invalid_root_candidate_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(tmpdir, n=10, device="cpu")
            text = config_path.read_text(encoding="utf-8").replace("J_root_prime: 2", "J_root_prime: 0")
            config_path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_alphazero_config(config_path)

    def test_invalid_nonroot_action_width_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={"search": {"J_nonroot": 0}},
            )
            with self.assertRaisesRegex(ValueError, "search.J_nonroot"):
                load_alphazero_config(config_path)

    def test_invalid_local_anneal_trigger_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={"local_anneal": {"trigger_steps": [0]}},
            )
            with self.assertRaises(ValueError):
                load_alphazero_config(config_path)

    def test_can_override_terminate_on_no_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={"search": {"terminate_on_no_progress": False}},
            )
            config = load_alphazero_config(config_path)
            self.assertFalse(config.search.terminate_on_no_progress)

    def test_negative_slot_score_coef_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={"search": {"slot_score_coef_scarcity": -0.1}},
            )
            with self.assertRaises(ValueError):
                load_alphazero_config(config_path)

    def test_invalid_terminal_reward_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={"reward": {"terminal_reward_mode": "raw_exp"}},
            )
            with self.assertRaisesRegex(ValueError, "terminal_reward_mode"):
                load_alphazero_config(config_path)

    def test_resume_and_model_init_paths_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={
                    "runtime": {
                        "resume_from": "checkpoints/run/latest.pt",
                        "init_from_model": "checkpoints/source/latest.pt",
                    }
                },
            )
            with self.assertRaisesRegex(ValueError, "resume_from and runtime.init_from_model"):
                load_alphazero_config(config_path)
