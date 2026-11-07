"""Tests for project configuration defaults and validation."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from circle_pack.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path, config) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    return path


def test_default_config_uses_patternboost_ddpm_defaults() -> None:
    config = load_config(REPO_ROOT / "config.yaml")

    assert config.model.slot_features.use_latent
    assert not config.model.slot_features.use_fixed_flag
    assert not config.model.slot_features.use_physical
    assert not config.model.slot_features.use_wall_slack
    assert not config.model.slot_features.use_extra_slot_features
    assert not config.model.pair_bias.enabled
    assert isinstance(config.model.use_outer_step_embedding, bool)
    assert config.model.outer_step_embedding_type in {"learned", "sin"}
    assert config.env.commit_order in {"slot_index", "random"}
    assert config.env.center_conflict_policy == "none"
    assert config.ddpm.num_steps == 64
    assert config.ddpm.beta_end == pytest.approx(0.08)
    assert config.ddpm.noising_center == "target"
    assert config.ddpm.draft_strength == pytest.approx(1.0)
    assert not config.ddpm.use_timestep_subsampling
    assert config.ddpm.infer_num_steps == 16
    assert not config.fm.use_draft_feature


@pytest.mark.parametrize(
    ("ddpm_field", "value", "message"),
    [
        ("noising_center", "bad", "ddpm.noising_center"),
        ("draft_strength", -0.1, "ddpm.draft_strength"),
        ("draft_strength", 1.1, "ddpm.draft_strength"),
        ("infer_num_steps", 0, "ddpm.infer_num_steps"),
    ],
)
def test_ddpm_config_validation_rejects_invalid_values(tmp_path, make_config, ddpm_field, value, message) -> None:
    config = make_config()
    config = replace(config, ddpm=replace(config.ddpm, **{ddpm_field: value}))
    config_path = _write_config(tmp_path / f"invalid_{ddpm_field}.yaml", config)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("fm_field", "value", "message"),
    [
        ("use_draft_feature", 1, "fm.use_draft_feature"),
    ],
)
def test_fm_config_validation_rejects_invalid_values(tmp_path, make_config, fm_field, value, message) -> None:
    config = make_config()
    config = replace(config, fm=replace(config.fm, **{fm_field: value}))
    config_path = _write_config(tmp_path / f"invalid_{fm_field}.yaml", config)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_model_outer_step_embedding_type_validation(tmp_path, make_config) -> None:
    config = make_config()
    config = replace(config, model=replace(config.model, outer_step_embedding_type="bad"))
    config_path = _write_config(tmp_path / "invalid_outer_step_embedding_type.yaml", config)

    with pytest.raises(ValueError, match="model.outer_step_embedding_type"):
        load_config(config_path)


def test_env_commit_order_validation(tmp_path, make_config) -> None:
    config = make_config()
    config = replace(config, env=replace(config.env, commit_order="bad"))
    config_path = _write_config(tmp_path / "invalid_commit_order.yaml", config)

    with pytest.raises(ValueError, match="env.commit_order"):
        load_config(config_path)


@pytest.mark.parametrize("embedding_type", ["learned", "sin"])
def test_model_outer_step_embedding_type_accepts_valid_values(tmp_path, make_config, embedding_type) -> None:
    config = make_config()
    config = replace(config, model=replace(config.model, outer_step_embedding_type=embedding_type))
    config_path = _write_config(tmp_path / f"valid_outer_step_{embedding_type}.yaml", config)

    loaded = load_config(config_path)

    assert loaded.model.outer_step_embedding_type == embedding_type


@pytest.mark.parametrize(
    ("train_field", "value", "message"),
    [
        ("draft_center_jitter_std", -0.01, "train.draft_center_jitter_std"),
    ],
)
def test_train_draft_config_validation_rejects_invalid_values(tmp_path, make_config, train_field, value, message) -> None:
    config = make_config()
    config = replace(config, train=replace(config.train, **{train_field: value}))
    config_path = _write_config(tmp_path / f"invalid_{train_field}.yaml", config)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)
