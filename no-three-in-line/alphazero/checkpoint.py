from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture NumPy and Torch RNG state for checkpoint-boundary resume.

    The returned mapping contains NumPy generator state, CPU Torch RNG state,
    and all CUDA RNG states when CUDA is available. The function does not
    mutate global RNG state and returns only Python and Torch-serializable
    values.
    """

    import numpy as np

    return {
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state_cpu": torch.get_rng_state(),
        "torch_rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore global Torch RNG state from a checkpoint payload."""

    if "torch_rng_state_cpu" in state:
        torch.set_rng_state(state["torch_rng_state_cpu"])
    cuda_states = state.get("torch_rng_state_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(trainer, path: str | Path) -> Path:
    """Serialize the minimal trainer state needed for boundary resume.

    ``trainer`` must provide model, optimizer, replay buffer, iteration, best
    tracker, run directory, and RNG attributes. The function writes a single
    checkpoint file, returns its ``Path``, and does not mutate repo-tracked
    files beyond the requested checkpoint path.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": trainer.iteration,
            "episodes_completed": trainer.episodes_completed,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "replay_buffer_items": trainer.replay_buffer.get_state(),
            "numpy_rng_state": trainer.rng.bit_generator.state,
            "torch_rng_state_cpu": torch.get_rng_state(),
            "torch_rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "best_state_tracker": trainer.best_state_tracker,
            "terminal_reward_baseline_points": trainer.terminal_reward_baseline_points,
            "run_dir": str(trainer.run_dir),
            "config_path": trainer.config.config_path,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a previously saved AlphaZero checkpoint payload."""

    return torch.load(Path(path), map_location="cpu", weights_only=False)
