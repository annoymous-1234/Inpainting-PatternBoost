from __future__ import annotations

import argparse
from pathlib import Path

from patternboost.config import load_patternboost_config
from patternboost.loop import PatternBoostRunner


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the PatternBoost training entrypoint.

    The returned namespace contains an optional ``config_path`` pointing to a
    unified project YAML. The function reads only CLI arguments, does not
    mutate state, and preserves the default root ``config.yaml`` behavior when
    no config path is supplied.
    """

    parser = argparse.ArgumentParser(description="Run the PatternBoost train-generate loop.")
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=None,
        help="Optional path to a unified project YAML config. Defaults to config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a PatternBoost config from CLI arguments and run the full loop.

    The function mutates training checkpoints, TopPool files, and JSONL logs
    through ``PatternBoostRunner``. It assumes the selected YAML is compatible
    with ``load_patternboost_config`` and prints a compact completion summary.
    """

    args = parse_args()
    config = load_patternboost_config(args.config_path)
    runner = PatternBoostRunner(config)
    summary = runner.run()
    print(
        "patternboost complete "
        f"initial_best={summary['initial_best_score']} "
        f"best={summary['best_score']} "
        f"pool_entries={summary['num_pool_entries']} "
        f"run_dir={summary['run_dir']} "
        f"metrics={summary['metrics_log_path']} "
        f"pool={summary['pool_save_path']}"
    )


if __name__ == "__main__":
    main()
