from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from alphazero.checkpoint import load_checkpoint
from alphazero.config import load_alphazero_config
from alphazero.tests.test_utils import write_test_config
from alphazero.trainer import AlphaZeroTrainer


class AlphaZeroTrainerTests(unittest.TestCase):
    def test_run_iteration_logs_terminal_state_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_alphazero_config(write_test_config(tmpdir, n=10, device="cpu"))
            trainer = AlphaZeroTrainer(config)
            metrics = trainer.run_iteration()

            self.assertGreaterEqual(metrics["new_samples"], 1.0)
            self.assertGreaterEqual(metrics["replay_size"], 1.0)
            self.assertIn("checkpoint", metrics)
            self.assertEqual(int(metrics["episodes_completed"]), 1)
            self.assertEqual(metrics["episode_terminal_points"], [int(metrics["max_terminal_points"])])
            self.assertEqual(len(metrics["episode_stop_reasons"]), 1)
            self.assertEqual(len(metrics["episode_terminal_rewards"]), 1)
            self.assertAlmostEqual(float(metrics["terminal_reward_baseline_points"]), 15.0)
            self.assertAlmostEqual(trainer.terminal_reward_baseline_points, float(metrics["mean_terminal_points"]))
            self.assertTrue((trainer.run_dir / "latest.pt").exists())
            self.assertTrue((trainer.run_dir / "rollout_terminal_states.jsonl").exists())
            self.assertTrue((trainer.run_dir / "metrics.jsonl").exists())

            terminal_records = [
                json.loads(line)
                for line in (trainer.run_dir / "rollout_terminal_states.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(terminal_records), 1)
            self.assertEqual(terminal_records[0]["event_type"], "rollout_terminal")
            self.assertEqual(terminal_records[0]["filled_count"], int(metrics["max_terminal_points"]))
            self.assertIn("terminal_reward", terminal_records[0])
            self.assertEqual(terminal_records[0]["terminal_reward_baseline_points"], 15)

    def test_checkpoint_roundtrip_and_resume_logging_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(tmpdir, n=10, device="cpu")
            config = load_alphazero_config(config_path)
            trainer = AlphaZeroTrainer(config)
            first_metrics = trainer.run_iteration()
            first_run_dir = trainer.run_dir
            latest_checkpoint = Path(first_metrics["checkpoint"])

            resumed_config = load_alphazero_config(config_path)
            resumed_config.runtime.resume_from = str(latest_checkpoint)
            resumed_trainer = AlphaZeroTrainer(resumed_config)

            self.assertEqual(resumed_trainer.iteration, trainer.iteration)
            self.assertEqual(resumed_trainer.episodes_completed, trainer.episodes_completed)
            self.assertEqual(len(resumed_trainer.replay_buffer), len(trainer.replay_buffer))
            self.assertEqual(resumed_trainer.best_state_tracker, trainer.best_state_tracker)
            self.assertEqual(resumed_trainer.terminal_reward_baseline_points, trainer.terminal_reward_baseline_points)
            self.assertEqual(resumed_trainer.run_dir, first_run_dir)

            second_metrics = resumed_trainer.run_iteration()
            self.assertEqual(int(second_metrics["episodes_completed"]), 2)
            self.assertAlmostEqual(
                float(second_metrics["terminal_reward_baseline_points"]),
                float(first_metrics["mean_terminal_points"]),
            )

            metrics_lines = [
                json.loads(line)
                for line in (first_run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rollout_lines = [
                json.loads(line)
                for line in (first_run_dir / "rollout_terminal_states.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(metrics_lines), 2)
            self.assertEqual(len(rollout_lines), 2)
            self.assertEqual(metrics_lines[-1]["episodes_completed"], 2)
            self.assertIn("episode_terminal_points", metrics_lines[-1])
            self.assertIn("episode_stop_reasons", metrics_lines[-1])

    def test_init_from_model_loads_weights_only_into_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(tmpdir, n=10, device="cpu")
            source_config = load_alphazero_config(config_path)
            source_trainer = AlphaZeroTrainer(source_config)
            source_metrics = source_trainer.run_iteration()
            source_checkpoint = Path(source_metrics["checkpoint"])
            source_payload = load_checkpoint(source_checkpoint)

            init_config = load_alphazero_config(config_path)
            init_config.runtime.run_name = "init_only_run"
            init_config.runtime.init_from_model = str(source_checkpoint)
            init_trainer = AlphaZeroTrainer(init_config)

            source_state = source_payload["model_state_dict"]
            loaded_state = init_trainer.model.state_dict()
            comparable_key = next(
                key
                for key, value in source_state.items()
                if key in loaded_state and torch.is_tensor(value) and loaded_state[key].shape == value.shape
            )
            self.assertTrue(torch.equal(loaded_state[comparable_key].cpu(), source_state[comparable_key].cpu()))
            self.assertEqual(init_trainer.iteration, 0)
            self.assertEqual(init_trainer.episodes_completed, 0)
            self.assertEqual(len(init_trainer.replay_buffer), 0)
            self.assertEqual(len(init_trainer.optimizer.state), 0)
            self.assertNotEqual(init_trainer.run_dir, source_trainer.run_dir)
            self.assertEqual(init_trainer.best_state_tracker["best_overall"], None)
