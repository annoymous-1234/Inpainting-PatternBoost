from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from alphazero.demo import run_demo
from alphazero.trainer import AlphaZeroTrainer
from alphazero.config import load_alphazero_config
from alphazero.tests.test_utils import write_test_config


class AlphaZeroDemoSmokeTests(unittest.TestCase):
    def test_cpu_demo_runs_full_rollout_remask_training_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(
                tmpdir,
                n=10,
                device="cpu",
                alphazero_overrides={
                    "self_play": {"max_episode_steps": 8},
                    "local_anneal": {
                        "enabled": True,
                        "alpha_env": 1.0,
                        "alpha_mcts": 1.0,
                        "trigger_steps": [3, 4, 5, 6, 7, 8],
                        "num_slots_to_remask": 1,
                        "min_active_age_for_remask": 1,
                        "sampling_temperature": 1.0,
                    },
                },
            )
            result = run_demo(config_path=config_path, device="cpu")

            self.assertGreaterEqual(result["rollout_num_samples"], 1)
            self.assertGreaterEqual(len(result["rollout_trace"]), 1)
            self.assertTrue(result["resolved_device"].startswith("cpu"))
            self.assertTrue(0 <= result["final_points"] <= 20)
            self.assertTrue(
                all(
                    {"env_conflict_counts", "mcts_conflict_counts", "remasked_slots", "points_after_env_step", "points_after_remask"}
                    <= set(step_trace)
                    for step_trace in result["rollout_trace"]
                )
            )
            self.assertTrue(math.isfinite(result["iteration_metrics"]["total_loss"]))
            self.assertIn("episode_terminal_points", result["iteration_metrics"])
            self.assertIn("episode_stop_reasons", result["iteration_metrics"])
            self.assertGreaterEqual(result["iteration_metrics"]["num_remask_events"], 0)
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "latest.pt").exists())
            self.assertTrue((run_dir / "rollout_terminal_states.jsonl").exists())
            self.assertTrue((run_dir / "best_state_updates.jsonl").exists())
            self.assertTrue((run_dir / "metrics.jsonl").exists())

            resumed_config = load_alphazero_config(config_path)
            resumed_config.runtime.resume_from = str(run_dir / "latest.pt")
            resumed_config.runtime.device = "cpu"
            resumed_trainer = AlphaZeroTrainer(resumed_config)
            resumed_metrics = resumed_trainer.run_iteration()
            self.assertTrue(math.isfinite(resumed_metrics["total_loss"]))

    def test_gpu_demo_runs_rollout_and_training(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the GPU demo smoke test")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_test_config(tmpdir, n=10, device="cuda")
            result = run_demo(config_path=config_path, device="cuda")

            self.assertGreaterEqual(result["rollout_num_samples"], 1)
            self.assertGreaterEqual(len(result["rollout_trace"]), 1)
            self.assertTrue(result["resolved_device"].startswith("cuda"))
            self.assertTrue(0 <= result["final_points"] <= 20)
            self.assertTrue(math.isfinite(result["iteration_metrics"]["total_loss"]))
            self.assertIn("episode_terminal_points", result["iteration_metrics"])
            self.assertIn("episode_stop_reasons", result["iteration_metrics"])
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "latest.pt").exists())
