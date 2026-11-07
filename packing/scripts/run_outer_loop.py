"""Run the version-1 multi-round outer-loop inference pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import load_config
from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import center_to_latent_approx, latent_to_center_simple
from models.factory import build_model
from pipelines.outer_loop import run_outer_loop_episode, run_outer_round_segmented_demo


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the outer-loop demo script."""

    parser = argparse.ArgumentParser(description="Circle packing outer-loop inference demo.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to unified config YAML.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["ddpm", "fm"],
        help="Optional inference-mode override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path containing model_state_dict.",
    )
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default=None,
        choices=["all_free", "variance_demo"],
        help="Optional candidate-mode override.",
    )
    parser.add_argument(
        "--segmented-demo",
        action="store_true",
        help="Run the illustrative segmented two-stage demo instead of a standard episode.",
    )
    return parser.parse_args()


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested execution device."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in the current environment.")
    return torch.device(requested_device)


def _maybe_load_checkpoint(model, checkpoint_path: Path | None) -> None:
    """Load a saved model checkpoint if one was requested."""

    if checkpoint_path is None:
        return
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)


def _print_record(record) -> None:
    """Print one compact outer-round summary line."""

    dispersion = record.raw_signals["dispersion"].detach().cpu().numpy()
    drift = record.raw_signals.get("drift")
    drift_mean = float(drift.detach().cpu().mean().item()) if drift is not None else 0.0
    fm_signal = record.raw_signals.get("fm_signal")
    fm_mean = float(fm_signal.detach().cpu().mean().item()) if fm_signal is not None else 0.0
    print(
        f"round={record.round_idx} reward={record.env_info['reward']:.6f} done={record.env_info['done']} "
        f"committed={record.env_info['committed_ids']} candidates={int(np.asarray(record.env_info['candidate_mask']).sum())} "
        f"dispersion_mean={float(dispersion.mean()):.6f} drift_mean={drift_mean:.6f} fm_motion_mean={fm_mean:.6f}"
    )


def main() -> None:
    """Build env/model, run a multi-round episode, and print debug summaries."""

    args = parse_args()
    config = load_config(args.config)
    if args.candidate_mode is not None:
        config = replace(config, outer_loop=replace(config.outer_loop, candidate_mode=args.candidate_mode))

    mode = args.mode or config.train.mode
    device = _resolve_device(args.device)

    env = CirclePackingEnv(config)
    env.reset(seed=config.project.seed)
    model = build_model(config).to(device=device)
    _maybe_load_checkpoint(model, args.checkpoint)
    model.eval()

    print(f"device={device}")
    print(f"mode={mode}")
    print(f"candidate_mode={config.outer_loop.candidate_mode}")

    if args.segmented_demo or config.outer_loop.segmented_demo:
        records = run_outer_round_segmented_demo(
            env,
            model,
            mode,
            config,
            latent_to_center_fn=latent_to_center_simple,
            center_to_latent_fn=center_to_latent_approx,
        )
    else:
        records = run_outer_loop_episode(
            env,
            model,
            mode,
            config,
            latent_to_center_fn=latent_to_center_simple,
            center_to_latent_fn=center_to_latent_approx,
        )

    for record in records:
        _print_record(record)

    final_obs = env.observe()
    print(f"final_step_idx={final_obs['step_idx']}")
    print(f"final_num_active={int(final_obs['active_mask'].sum())}")
    print(f"final_layout_score={float(final_obs['sum_active_r']):.6f}")
    print(f"final_sum_active_r={float(final_obs['sum_active_r']):.6f}")


if __name__ == "__main__":
    main()
