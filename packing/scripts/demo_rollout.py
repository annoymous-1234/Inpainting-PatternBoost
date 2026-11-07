"""Run a complete rollout with a simple baseline policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.collect import collect_episode
from circle_pack.config import load_config
from circle_pack.env import CirclePackingEnv
from circle_pack.policies import make_policy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the rollout demo."""

    parser = argparse.ArgumentParser(description="Circle packing rollout demo.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        choices=["random_legal", "noisy_hint"],
        help="Optional policy override.",
    )
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help="Force saving the final render even if rollout.save_figure is false.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, run one rollout, print summary, and optionally save a figure."""

    args = parse_args()
    config = load_config(args.config)
    env = CirclePackingEnv(config)
    policy_name = args.policy or config.rollout.policy_name
    policy = make_policy(policy_name, config)
    transitions, summary = collect_episode(
        env,
        policy,
        seed=config.project.seed,
        record_snapshots=config.rollout.record_snapshots,
    )

    print(f"policy={policy_name}")
    print(f"steps={summary['num_steps']}")
    print(f"num_active={summary['num_active']}")
    print(f"total_reward={summary['total_reward']:.6f}")
    print(f"final_layout_score={summary['final_sum_active_r']:.6f}")
    for step_idx, transition in enumerate(transitions):
        print(
            f"step={step_idx} reward={transition.reward:.6f} "
            f"commits={transition.info['committed_ids']}"
        )

    should_save = args.save_figure or config.rollout.save_figure
    if config.rollout.render_final or should_save:
        save_path = config.rollout.figure_path if should_save else None
        env.render(title="Final rollout state", save_path=save_path)
        if save_path is not None:
            print(f"saved_figure={save_path}")


if __name__ == "__main__":
    main()
