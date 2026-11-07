"""Run a minimal supervised DDPM or FM training loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_pack.config import load_config
from models.factory import build_model
from models.objectives import ddpm_supervised_loss, fm_supervised_loss
from pipelines.data import CirclePackingSupervisedDataset, load_teacher_layouts


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the supervised training script."""

    parser = argparse.ArgumentParser(description="Circle packing supervised training script.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to unified config YAML.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["ddpm", "fm"],
        help="Optional training-mode override.",
    )
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


def _load_teacher_bank(config):
    """Load the teacher-layout bank used by the dataset."""

    return load_teacher_layouts(config.train.dataset_path)


def main() -> None:
    """Load config, build data/model/loss, and run a simple optimizer loop."""

    args = parse_args()
    config = load_config(args.config)
    device = _resolve_device(args.device)
    mode = args.mode or config.train.mode

    teacher_layouts = _load_teacher_bank(config)
    dataset = CirclePackingSupervisedDataset(
        teacher_layouts,
        mode=mode,
        cfg=config,
        seed=config.project.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    loss_fn = ddpm_supervised_loss if mode == "ddpm" else fm_supervised_loss

    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"mode={mode}")
    print(f"dataset_path={config.train.dataset_path}")
    print(f"num_teacher_layouts={len(dataset)}")
    print(f"batch_size={config.train.batch_size}")

    for epoch_idx in range(config.train.num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = loss_fn(model, batch, config)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.detach().cpu().item())
            num_batches += 1

        mean_epoch_loss = epoch_loss_sum / max(num_batches, 1)
        checkpoint = {
            "epoch": epoch_idx,
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "mean_epoch_loss": mean_epoch_loss,
        }
        epoch_path = checkpoint_dir / f"epoch_{epoch_idx:03d}.pt"
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, latest_path)
        print(f"epoch={epoch_idx} mean_loss={mean_epoch_loss:.6f} saved={epoch_path}")


if __name__ == "__main__":
    main()
