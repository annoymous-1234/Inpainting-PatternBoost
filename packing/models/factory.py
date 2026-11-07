"""Factory helpers for building configured model instances."""

from __future__ import annotations

from circle_pack.config import MODEL_PRESET_SPECS, ProjectConfig
from models.model import CirclePackingModel


MODEL_PRESETS = MODEL_PRESET_SPECS


def resolve_model_config(config: ProjectConfig):
    """Return the validated model config section from a project config."""

    return config.model


def build_model(config: ProjectConfig) -> CirclePackingModel:
    """Construct the top-level circle packing model from a project config."""

    return CirclePackingModel(config)
