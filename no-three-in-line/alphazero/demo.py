from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_alphazero_config
from .model_adapter import resolve_runtime_device
from .self_play import run_self_play_episode_with_trace
from .trainer import AlphaZeroTrainer


def run_demo(config_path: str | Path | None = None, device: str | None = None) -> dict[str, object]:
    """Run a minimal AlphaZero rollout plus one training iteration demo."""

    config = load_alphazero_config(config_path)
    if device is not None:
        config.runtime.device = device
    resolved_device = resolve_runtime_device(config.runtime.device)
    trainer = AlphaZeroTrainer(config)
    trainer.model.to(resolved_device)

    print(f"resolved_device={resolved_device}")
    print(
        "demo_config n={n} J_root={J_root} J_nonroot={J_nonroot} J_root_prime={J_root_prime} "
        "K_root={K_root} K_nonroot={K_nonroot} root_budget={root_budget}".format(
            n=config.n,
            J_root=config.search.J_root,
            J_nonroot=config.search.J_nonroot,
            J_root_prime=config.search.J_root_prime,
            K_root=config.search.K_root,
            K_nonroot=config.search.K_nonroot,
            root_budget=config.search.root_budget,
        )
    )

    rollout_samples, rollout_trace, rollout_summary = run_self_play_episode_with_trace(
        trainer.model,
        trainer.rng,
        trainer.config,
    )
    for step_trace in rollout_trace:
        print(
            "rollout_step={episode_step} chosen_action_index={chosen_action_index} "
            "executed_slots={executed_slots} executed_values={executed_values} "
            "root_value={root_value:.6f} env_conflict_counts={env_conflict_counts} "
            "mcts_conflict_counts={mcts_conflict_counts} remasked_slots={remasked_slots} "
            "points_after_env_step={points_after_env_step} points_after_remask={points_after_remask} "
            "stop_reason={stop_reason}".format(
                episode_step=step_trace["episode_step"],
                chosen_action_index=step_trace["chosen_action_index"],
                executed_slots=step_trace["executed_action"].active_slots,
                executed_values=step_trace["executed_action"].values,
                root_value=step_trace["root_value"],
                env_conflict_counts=step_trace["env_conflict_counts"],
                mcts_conflict_counts=step_trace["mcts_conflict_counts"],
                remasked_slots=step_trace["remasked_slots"],
                points_after_env_step=step_trace["points_after_env_step"],
                points_after_remask=step_trace["points_after_remask"],
                stop_reason=step_trace["stop_reason"],
            )
        )
    final_points = int(rollout_summary["final_points"])
    print(f"episode_final_points={final_points}")
    print(
        "rollout_anneal num_remask_events={num_remask_events} total_remasked_slots={total_remasked_slots}".format(
            **rollout_summary
        )
    )

    iteration_metrics = trainer.run_iteration()
    print(f"run_dir={trainer.run_dir}")
    print(f"replay_size={int(iteration_metrics['replay_size'])}")
    print(
        "training_update num_remask_events={num_remask_events:.0f} total_remasked_slots={total_remasked_slots:.0f} "
        "total_loss={total_loss:.6f} fill_loss={fill_loss:.6f} "
        "val_loss={val_loss:.6f} value_loss={value_loss:.6f}".format(**iteration_metrics)
    )
    for episode_index, (terminal_points, stop_reason) in enumerate(
        zip(iteration_metrics["episode_terminal_points"], iteration_metrics["episode_stop_reasons"])
    ):
        print(
            f"  training_episode_{episode_index}: terminal_points={terminal_points} "
            f"stop_reason={stop_reason}"
        )
    return {
        "resolved_device": str(resolved_device),
        "rollout_trace": rollout_trace,
        "rollout_num_samples": len(rollout_samples),
        "rollout_summary": rollout_summary,
        "final_points": final_points,
        "iteration_metrics": iteration_metrics,
        "run_dir": str(trainer.run_dir),
    }


def main() -> None:
    """Run the minimal AlphaZero demo from the command line."""

    parser = argparse.ArgumentParser(description="Run a minimal AlphaZero demo.")
    parser.add_argument("--config-path", type=str, default=None, help="Optional path to a project config.yaml file.")
    parser.add_argument("--device", type=str, default=None, help="Optional runtime device override.")
    args = parser.parse_args()
    run_demo(config_path=args.config_path, device=args.device)


if __name__ == "__main__":
    main()
