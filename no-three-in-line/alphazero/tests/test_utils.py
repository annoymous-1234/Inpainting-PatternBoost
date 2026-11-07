from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return the mutated mapping."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_test_config(
    directory: str | Path,
    *,
    n: int = 10,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    run_name: str | None = "test_run",
    alphazero_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write one minimal project config file for AlphaZero tests."""

    directory_path = Path(directory)
    mapping: dict[str, Any] = {
        "n": n,
        "model": {
            "d_model": 32,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 64,
            "dropout": 0.0,
            "use_global_token": False,
            "use_value_head": False,
            "state_head_type": "classifier",
        },
        "point_features": {
            "use_uv": False,
            "use_uv_sumdiff": False,
            "use_uv_product": False,
            "use_fourier_u": False,
            "use_fourier_v": False,
            "use_fourier_sumdiff": False,
            "fourier_frequencies": [1.0, 2.0, 4.0, 8.0],
            "use_y_onehot": True,
            "use_x_onehot": True,
            "use_mod_features": False,
            "mod_bases": [2, 3, 5, 7],
            "use_mod_x": False,
            "use_mod_y": False,
            "use_mod_sum": False,
            "use_mod_diff": False,
        },
        "alphazero": {
            "search": {
                "J_root": 1,
                "J_nonroot": 1,
                "J_root_prime": 2,
                "L_root": 2,
                "K_root": 4,
                "L_nonroot": 2,
                "K_nonroot": 2,
                "root_budget": 4,
                "max_search_plies": 4,
                "tau_slot_root": 1.0,
                "tau_val_root": 1.0,
                "tau_slot_nonroot": 1.0,
                "tau_val_nonroot": 1.0,
                "tau_pi_root": 1.0,
                "tau_pi_nonroot": 1.0,
                "tau_target": 1.0,
                "c_Q": 2.0,
                "gamma_step": 0.95,
                "eps": 1e-8,
                "terminate_on_no_progress": True,
                "use_existing_action_mask": True,
                "slot_score_coef_mask_logit": 1.0,
                "slot_score_coef_scarcity": 0.0,
                "slot_score_coef_entropy": 0.0,
            },
            "reward": {
                "step_weight": 0.0,
                "terminal_weight": 1.0,
                "terminal_reward_mode": "capped_exp2_advantage",
                "terminal_reward_beta": 1.0,
                "terminal_reward_cap": 6.0,
                "initial_terminal_baseline_points": None,
                "initial_terminal_baseline_fraction": 0.75,
            },
            "self_play": {
                "num_iterations": 1,
                "episodes_per_iteration": 1,
                "max_episode_steps": 6,
                "root_temperature": 1.0,
                "sample_prefix_steps": 4,
            },
            "training": {
                "lr": 1e-4,
                "weight_decay": 0.0,
                "grad_clip_norm": 10.0,
                "batch_size": 2,
                "replay_capacity": 128,
                "updates_per_iteration": 1,
                "lambda_fill": 1.0,
                "lambda_val": 1.0,
                "value_loss_weight": 1.0,
                "save_every": 1,
            },
            "runtime": {
                "seed": 0,
                "device": device,
                "checkpoint_dir": checkpoint_dir or str(directory_path / "checkpoints"),
                "run_name": run_name,
                "resume_from": None,
                "init_from_model": None,
                "keep_iteration_checkpoints": True,
                "log_jsonl": True,
            },
            "local_anneal": {
                "enabled": False,
                "alpha_env": 1.0,
                "alpha_mcts": 1.0,
                "trigger_steps": [],
                "num_slots_to_remask": 1,
                "min_active_age_for_remask": 2,
                "sampling_temperature": 1.0,
            },
        },
    }
    if alphazero_overrides is not None:
        _deep_update(mapping["alphazero"], alphazero_overrides)
    config_path = directory_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return config_path
