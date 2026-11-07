"""Smoke tests for teacher-layout generation and supervised-training scripts."""

from __future__ import annotations

from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from circle_pack.config import load_config
from pipelines.data import load_teacher_layouts, save_teacher_layouts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path, config) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    return path


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_load_config_requires_train_dataset_path(tmp_path, make_config) -> None:
    config = make_config()
    config = replace(config, train=replace(config.train, dataset_path=""))
    config_path = _write_config(tmp_path / "missing_dataset_path.yaml", config)

    with pytest.raises(ValueError, match="train.dataset_path must be set"):
        load_config(config_path)


@pytest.mark.parametrize("suffix", [".pt", ".jsonl"])
def test_generate_data_demo_script_outputs_teacher_bank(tmp_path, make_config, suffix: str) -> None:
    output_path = tmp_path / f"teacher_layouts{suffix}"
    config = make_config(K=4, max_steps=2, activation_budget=1, policy_name="random_legal")
    config = replace(
        config,
        project=replace(config.project, dtype="float32"),
        train=replace(config.train, dataset_path=str(output_path)),
    )
    config_path = _write_config(tmp_path / f"generate_data_demo_{suffix[1:]}.yaml", config)

    result = _run_script(
        "scripts/generate_data_demo.py",
        "--config",
        str(config_path),
        "--out",
        str(output_path),
        "--num-layouts",
        "3",
        "--seed",
        "17",
    )

    teacher_layouts = load_teacher_layouts(output_path)
    assert teacher_layouts.shape == (3, 4, 2)
    assert np.all((teacher_layouts >= 0.0) & (teacher_layouts <= 1.0))
    assert f"out={output_path}" in result.stdout


@pytest.mark.parametrize(
    ("suffix", "mode"),
    [
        (".pt", "fm"),
        (".jsonl", "ddpm"),
    ],
)
def test_train_supervised_script_reads_teacher_bank(tmp_path, make_config, suffix: str, mode: str) -> None:
    dataset_path = tmp_path / f"teacher_layouts{suffix}"
    checkpoint_dir = tmp_path / f"checkpoints_{mode}"
    teacher = np.array(
        [
            [0.70, 0.80, 0.10],
            [0.20, 0.60, 0.08],
            [0.40, 0.30, 0.05],
            [0.10, 0.10, 0.00],
        ],
        dtype=np.float32,
    )
    teacher_bank = np.stack([teacher, teacher[::-1]], axis=0)
    save_teacher_layouts(teacher_bank, dataset_path)

    config = make_config(K=4, max_steps=2)
    small_model = replace(
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
        project=replace(config.project, dtype="float32"),
        model=small_model,
        train=replace(
            config.train,
            mode=mode,
            dataset_path=str(dataset_path),
            batch_size=2,
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
            num_workers=0,
        ),
    )
    config_path = _write_config(tmp_path / f"train_supervised_{mode}_{suffix[1:]}.yaml", config)

    result = _run_script(
        "scripts/train_supervised.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--mode",
        mode,
    )

    assert checkpoint_dir.joinpath("latest.pt").exists()
    assert f"dataset_path={dataset_path}" in result.stdout
    assert f"mode={mode}" in result.stdout


def test_ddpm_demo_chain_generates_trains_and_runs_outer_loop(tmp_path, make_config) -> None:
    dataset_path = tmp_path / "demo_teacher_layouts.pt"
    checkpoint_dir = tmp_path / "checkpoints_ddpm"
    config = make_config(K=4, max_steps=2, activation_budget=1, policy_name="random_legal")
    small_model = replace(
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
        project=replace(config.project, dtype="float32"),
        model=small_model,
        train=replace(
            config.train,
            mode="ddpm",
            dataset_path=str(dataset_path),
            batch_size=2,
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
            num_workers=0,
        ),
        ddpm=replace(
            config.ddpm,
            num_samples=2,
            num_steps=4,
            beta_start=1.0e-4,
            beta_end=0.08,
            tau_start_infer=1.0,
            infer_num_steps=2,
        ),
        outer_loop=replace(config.outer_loop, max_rounds=1),
    )
    config_path = _write_config(tmp_path / "ddpm_demo_chain.yaml", config)

    generate_result = _run_script(
        "scripts/generate_data_demo.py",
        "--config",
        str(config_path),
        "--out",
        str(dataset_path),
        "--num-layouts",
        "3",
        "--seed",
        "31",
    )
    assert "num_layouts=3" in generate_result.stdout

    train_result = _run_script(
        "scripts/train_supervised.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--mode",
        "ddpm",
    )
    checkpoint_path = checkpoint_dir / "latest.pt"
    assert checkpoint_path.exists()
    assert "mode=ddpm" in train_result.stdout

    rollout_result = _run_script(
        "scripts/run_outer_loop.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--mode",
        "ddpm",
        "--checkpoint",
        str(checkpoint_path),
    )

    assert "final_sum_active_r=" in rollout_result.stdout
    assert "final_num_active=" in rollout_result.stdout
