"""Small CPU-friendly PatternBoost demo."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import load_config
from circle_pack.render import render_state
from models.factory import build_model
from patternboost.runner import PatternBoostRunner
from pipelines.data import generate_synthetic_teacher_layouts, save_teacher_layouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circle packing PatternBoost smoke demo.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to unified config YAML.")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device override.")
    parser.add_argument("--mode", type=str, default="fm", choices=["fm", "ddpm"], help="PatternBoost training mode.")
    parser.add_argument("--K", type=int, default=None, help="Optional slot-count override.")
    parser.add_argument("--num-generations", type=int, default=1, help="Demo PatternBoost generations.")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates per demo generation.")
    parser.add_argument("--render-best", type=Path, default=None, help="Optional output figure for the best layout.")
    return parser.parse_args()


def _prepare_demo_config(args: argparse.Namespace):
    config = load_config(args.config)
    K = int(config.problem.K if args.K is None else args.K)
    model = replace(
        config.model,
        preset="custom",
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2.0,
        use_global_token=False,
    )
    config = replace(
        config,
        problem=replace(config.problem, K=K, max_steps=max(2, min(config.problem.max_steps, K))),
        project=replace(config.project, dtype="float32"),
        model=model,
        env=replace(config.env, activation_budget=min(config.env.activation_budget, max(1, K))),
        fm=replace(config.fm, num_samples=2, num_steps=2, source_noise_std=0.02),
        ddpm=replace(config.ddpm, num_samples=2, num_steps=4, infer_num_steps=2),
        outer_loop=replace(config.outer_loop, max_rounds=1),
    )
    pb = config.patternboost
    pb = replace(
        pb,
        pool=replace(pb.pool, max_size=16, greedy_num_seeds=8),
        training=replace(
            pb.training,
            mode=args.mode,
            samples_per_generation=8,
            batch_size=2,
            epochs_per_generation=1,
            lr=1.0e-3,
        ),
        generation=replace(
            pb.generation,
            num_generations=int(args.num_generations),
            candidates_per_generation=int(args.candidates),
            modes=["full_episode"],
            mode_weights=[1.0],
        ),
        polish=replace(pb.polish, refresh_lp_cache=False),
        runtime=replace(pb.runtime, device=args.device, checkpoint_dir="checkpoints/patternboost_demo"),
    )
    return replace(config, patternboost=pb)


def _ensure_demo_teacher_layouts(config) -> None:
    dataset_path = Path(config.train.dataset_path)
    if dataset_path.exists():
        return
    layouts = generate_synthetic_teacher_layouts(config, count=4, seed=config.project.seed)
    save_teacher_layouts(layouts, dataset_path)


def main() -> None:
    args = parse_args()
    config = _prepare_demo_config(args)
    _ensure_demo_teacher_layouts(config)

    model = build_model(config)
    runner = PatternBoostRunner(config, model=model)
    summary = runner.run()
    for key, value in summary.items():
        print(f"{key}={value}")

    if args.render_best is not None:
        best = runner.top_pool.best()
        if best is not None:
            render_state(
                best.circles,
                best.active_mask,
                config.render,
                title="PatternBoost best",
                save_path=str(args.render_best),
            )


if __name__ == "__main__":
    main()
