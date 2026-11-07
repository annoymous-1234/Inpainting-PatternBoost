#!/usr/bin/env python3
"""Run PatternBoost experiments for YAML configs in a folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "run_patternboost_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripts/run_patternboost_experiment.py once for each YAML config in a folder."
    )
    parser.add_argument("folder", type=Path, help="Folder containing .yaml/.yml experiment configs.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"], help="Device passed to the runner.")
    parser.add_argument("--force", action="store_true", help="Pass --force to each experiment run.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively below the folder instead of only direct children.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later configs after a failed experiment.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help="Experiment runner script to execute for each config.",
    )
    return parser.parse_args()


def find_yaml_configs(folder: Path, *, recursive: bool) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    configs = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    ]
    return sorted(configs, key=lambda path: path.relative_to(folder).as_posix())


def run_one(*, runner: Path, config_path: Path, device: str, force: bool) -> int:
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--device",
        str(device),
    ]
    if force:
        command.append("--force")

    print("=" * 80, flush=True)
    print(f"config={config_path}", flush=True)
    print("command=" + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    runner = args.runner.resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Runner does not exist: {runner}")

    folder = args.folder.resolve()
    configs = find_yaml_configs(folder, recursive=bool(args.recursive))
    if not configs:
        raise FileNotFoundError(f"No .yaml/.yml configs found in {folder}")

    failures: list[tuple[Path, int]] = []
    print(f"Found {len(configs)} config(s) in {folder}", flush=True)
    for idx, config_path in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] running {config_path.relative_to(folder)}", flush=True)
        returncode = run_one(
            runner=runner,
            config_path=config_path,
            device=str(args.device),
            force=bool(args.force),
        )
        if returncode != 0:
            failures.append((config_path, returncode))
            print(f"failed: {config_path} returncode={returncode}", flush=True)
            if not args.continue_on_error:
                break

    print("=" * 80, flush=True)
    if failures:
        print(f"Completed with {len(failures)} failure(s).", flush=True)
        for config_path, returncode in failures:
            print(f"failed config={config_path} returncode={returncode}", flush=True)
        return failures[0][1]

    print(f"Completed {len(configs)} config(s) successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
