"""Minimal AlphaZero package entry points for no-three-in-line."""

from .config import load_alphazero_config
from .demo import run_demo
from .trainer import AlphaZeroTrainer

__all__ = [
    "AlphaZeroTrainer",
    "load_alphazero_config",
    "run_demo",
]
