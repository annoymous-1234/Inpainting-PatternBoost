"""Run Circle Packing PatternBoost training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import load_config
from patternboost.runner import PatternBoostRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circle packing PatternBoost runner.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to unified config YAML.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Device override.")
    parser.add_argument("--mode", type=str, default=None, choices=["fm", "ddpm"], help="PatternBoost training mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config.patternboost.runtime
    training = config.patternboost.training
    if args.device is not None:
        runtime = replace(runtime, device=args.device)
    if args.mode is not None:
        training = replace(training, mode=args.mode)
    config = replace(config, patternboost=replace(config.patternboost, runtime=runtime, training=training))

    runner = PatternBoostRunner(config)
    summary = runner.run()
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

