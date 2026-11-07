from __future__ import annotations

import argparse

from .config import load_alphazero_config
from .trainer import AlphaZeroTrainer


def main() -> None:
    """Run AlphaZero training from the project ``config.yaml``."""

    parser = argparse.ArgumentParser(description="Train AlphaZero for no-three-in-line.")
    parser.add_argument("--config-path", type=str, default=None, help="Optional path to a project config.yaml file.")
    parser.add_argument("--device", type=str, default=None, help="Optional runtime device override.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path for boundary resume.")
    parser.add_argument("--init-from-model", type=str, default=None, help="Optional checkpoint path for model-only initialization.")
    args = parser.parse_args()

    config = load_alphazero_config(args.config_path)
    if args.device is not None:
        config.runtime.device = args.device
    if args.resume_from is not None:
        config.runtime.resume_from = args.resume_from
    if args.init_from_model is not None:
        config.runtime.init_from_model = args.init_from_model
    trainer = AlphaZeroTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
