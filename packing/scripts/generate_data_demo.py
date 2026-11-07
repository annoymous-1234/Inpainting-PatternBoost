"""Generate teacher-layout demo data from environment rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import load_config
from pipelines.data import generate_synthetic_teacher_layouts, save_teacher_layouts


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for teacher-layout demo generation."""

    parser = argparse.ArgumentParser(description="Generate teacher-layout demo data from environment rollouts.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    parser.add_argument("--out", type=Path, required=True, help="Output data path. Supports .pt or .jsonl.")
    parser.add_argument("--num-layouts", type=int, required=True, help="Number of teacher layouts to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed override. Defaults to config.project.seed.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "pt", "jsonl"],
        help="Optional output format override. 'auto' infers from --out.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate a teacher-layout bank and save it for supervised-training demos."""

    args = parse_args()
    if args.num_layouts <= 0:
        raise SystemExit("--num-layouts must be positive.")

    config = load_config(args.config)
    base_seed = config.project.seed if args.seed is None else args.seed
    teacher_layouts = generate_synthetic_teacher_layouts(config, count=args.num_layouts, seed=base_seed)
    resolved_format = args.format if args.format != "auto" else args.out.suffix.lstrip(".").lower()
    save_teacher_layouts(
        teacher_layouts,
        args.out,
        format_name=args.format,
        metadata={
            "base_seed": int(base_seed),
            "num_layouts": int(args.num_layouts),
            "policy_name": config.rollout.policy_name,
        },
    )

    print(f"out={args.out}")
    print(f"format={resolved_format}")
    print(f"num_layouts={teacher_layouts.shape[0]}")
    print(f"num_slots={teacher_layouts.shape[1]}")
    print(f"seed={base_seed}")


if __name__ == "__main__":
    main()
