"""Run a model-side smoke pass on CPU or CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import load_config
from models.factory import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the model smoke script."""

    parser = argparse.ArgumentParser(description="Circle packing model smoke script.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to unified config YAML.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' picks CUDA when available.",
    )
    return parser.parse_args()


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested execution device."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in the current environment.")
    return torch.device(requested_device)


def _make_inputs(config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create lightweight random inputs for one smoke forward pass."""

    torch.manual_seed(config.project.seed)
    dtype = config.torch_dtype
    batch_size = 2
    num_slots = config.problem.K
    slot_latent = torch.randn(batch_size, num_slots, 2, dtype=dtype, device=device)
    slot_centers = torch.rand(batch_size, num_slots, 2, dtype=dtype, device=device)
    fixed_mask = (torch.rand(batch_size, num_slots, device=device) > 0.5).to(torch.bool)
    inner_time = torch.rand(batch_size, dtype=dtype, device=device)
    return slot_latent, slot_centers, fixed_mask, inner_time


def main() -> None:
    """Run a single forward pass, proposal prediction, and decode smoke check."""

    args = parse_args()
    config = load_config(args.config)
    device = _resolve_device(args.device)

    model = build_model(config).to(device=device)
    slot_latent, slot_centers, fixed_mask, inner_time = _make_inputs(config, device=device)

    outputs = model(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
        predict_ddpm=True,
        predict_fm=True,
        predict_proposal=True,
        predict_value=True,
        return_hidden=True,
    )
    proposal_latent = model.predict_proposal_latent(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
    )
    proposal_centers = model.predict_proposal_centers_for_test(
        slot_latent,
        slot_centers,
        fixed_mask,
        inner_time=inner_time,
    )

    print(f"device={device}")
    print(f"torch_dtype={config.torch_dtype}")
    print(f"ddpm_pred_shape={tuple(outputs['ddpm_pred'].shape) if outputs['ddpm_pred'] is not None else None}")
    print(f"fm_pred_shape={tuple(outputs['fm_pred'].shape) if outputs['fm_pred'] is not None else None}")
    print(
        "proposal_head_shape="
        f"{tuple(outputs['proposal_latent'].shape) if outputs['proposal_latent'] is not None else None}"
    )
    print(f"value_shape={tuple(outputs['value'].shape) if outputs['value'] is not None else None}")
    print(f"slot_hidden_shape={tuple(outputs['slot_hidden'].shape) if outputs['slot_hidden'] is not None else None}")
    print(
        f"global_hidden_shape={tuple(outputs['global_hidden'].shape) if outputs['global_hidden'] is not None else None}"
    )
    print(f"proposal_latent_shape={tuple(proposal_latent.shape)}")
    print(f"proposal_centers_shape={tuple(proposal_centers.shape)}")


if __name__ == "__main__":
    main()
