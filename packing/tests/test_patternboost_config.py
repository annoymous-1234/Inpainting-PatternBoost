"""Tests for PatternBoost config defaults and validation."""

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


def test_default_config_loads_patternboost_section() -> None:
    config = load_config(REPO_ROOT / "config.yaml")

    assert config.patternboost.pool.max_size == 2048
    assert config.patternboost.pool.sampling.mode == "score_bucket_rank_mixture"
    assert config.patternboost.pool.sampling.uniform_mix == pytest.approx(0.1)
    assert config.patternboost.edit.edit_ratios == [0.1, 0.2, 0.4, 0.6, 1.0]
    assert config.patternboost.training.mode == "ddpm"
    assert config.patternboost.generation.greedy_complete_unfinished
    assert not config.patternboost.polish.refresh_lp_cache
    assert not config.patternboost.polish.srp_enabled
    assert config.patternboost.polish.srp_top_k == 64
    assert config.patternboost.polish.srp_num_workers == 1
    assert config.patternboost.polish.srp_Imax == 400
    assert config.patternboost.polish.srp_m == 30
    assert config.patternboost.polish.srp_beta == pytest.approx(0.99)
    assert config.patternboost.polish.srp_step_radius == pytest.approx(0.001)
    assert config.patternboost.polish.srp_w_overlap == pytest.approx(1000.0)
    assert config.patternboost.polish.srp_alpha_sum_r == pytest.approx(0.2)
    assert config.patternboost.polish.srp_strict_shrink
    assert config.patternboost.polish.srp_strict_shrink_margin == pytest.approx(1.0e-8)
    assert not config.patternboost.runtime.save_checkpoints
    assert not config.patternboost.runtime.log_candidates_jsonl
    assert config.patternboost.runtime.top_pool_snapshot_every_generations == 0
    assert not config.patternboost.resume.enabled
    assert config.patternboost.resume.checkpoint_path is None
    assert config.patternboost.resume.top_pool_path is None
    assert config.patternboost.resume.load_optimizer
    assert not config.fm.use_draft_feature


def test_legacy_cap_expansion_maps_to_refresh_lp_cache(tmp_path, make_config) -> None:
    config = make_config()
    payload = asdict(config)
    payload["patternboost"]["polish"] = {"cap_expansion": True}
    config_path = tmp_path / "legacy_polish.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    loaded = load_config(config_path)

    assert loaded.patternboost.polish.refresh_lp_cache


@pytest.mark.parametrize(
    ("section_path", "value", "message"),
    [
        (("pool", "sampling", "uniform_mix"), 1.5, "patternboost.pool.sampling.uniform_mix"),
        (("edit", "edit_ratios"), [0.0], "patternboost.edit.edit_ratios"),
        (("polish", "refresh_lp_cache"), 1, "patternboost.polish.refresh_lp_cache"),
        (("polish", "srp_enabled"), 1, "patternboost.polish.srp_enabled"),
        (("polish", "srp_top_k"), -1, "patternboost.polish.srp_top_k"),
        (("polish", "srp_num_workers"), 0, "patternboost.polish.srp_num_workers"),
        (("polish", "srp_strict_shrink"), 1, "patternboost.polish.srp_strict_shrink"),
        (
            ("runtime", "top_pool_snapshot_every_generations"),
            -1,
            "patternboost.runtime.top_pool_snapshot_every_generations",
        ),
        (("runtime", "save_checkpoints"), 1, "patternboost.runtime.save_checkpoints"),
        (("runtime", "log_candidates_jsonl"), 1, "patternboost.runtime.log_candidates_jsonl"),
        (("resume", "enabled"), 1, "patternboost.resume.enabled"),
        (("resume", "load_optimizer"), 1, "patternboost.resume.load_optimizer"),
    ],
)
def test_patternboost_config_validation_rejects_invalid_values(
    tmp_path,
    make_config,
    section_path,
    value,
    message,
) -> None:
    config = make_config()
    pb = config.patternboost
    if section_path[0] == "pool":
        sampling = replace(pb.pool.sampling, **{section_path[2]: value})
        pb = replace(pb, pool=replace(pb.pool, sampling=sampling))
    elif section_path[0] == "edit":
        pb = replace(pb, edit=replace(pb.edit, **{section_path[1]: value}))
    elif section_path[0] == "runtime":
        pb = replace(pb, runtime=replace(pb.runtime, **{section_path[1]: value}))
    elif section_path[0] == "resume":
        pb = replace(pb, resume=replace(pb.resume, **{section_path[1]: value}))
    else:
        pb = replace(pb, polish=replace(pb.polish, **{section_path[1]: value}))
    config = replace(config, patternboost=pb)
    config_path = _write_config(tmp_path / "invalid_patternboost.yaml", config)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("checkpoint_path", "top_pool_path", "message"),
    [
        (None, "top_pool.pt", "patternboost.resume.checkpoint_path"),
        ("latest.pt", None, "patternboost.resume.top_pool_path"),
    ],
)
def test_patternboost_resume_requires_paths(tmp_path, make_config, checkpoint_path, top_pool_path, message) -> None:
    config = make_config()
    resume = replace(
        config.patternboost.resume,
        enabled=True,
        checkpoint_path=checkpoint_path,
        top_pool_path=top_pool_path,
    )
    config = replace(config, patternboost=replace(config.patternboost, resume=resume))
    config_path = _write_config(tmp_path / "invalid_resume.yaml", config)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_patternboost_fm_mode_requires_enabled_fm_head(tmp_path, make_config) -> None:
    config = make_config()
    model = replace(config.model, heads=replace(config.model.heads, enable_fm_head=False))
    pb = replace(config.patternboost, training=replace(config.patternboost.training, mode="fm"))
    config = replace(config, model=model, patternboost=pb)
    config_path = _write_config(tmp_path / "invalid_fm_head.yaml", config)

    with pytest.raises(ValueError, match="enable_fm_head"):
        load_config(config_path)


@pytest.mark.parametrize("extra_dim", [0, 5])
def test_fm_draft_feature_requires_extra_slot_feature_dim_four(tmp_path, make_config, extra_dim) -> None:
    config = make_config()
    slot_features = replace(
        config.model.slot_features,
        use_extra_slot_features=extra_dim > 0,
        extra_slot_feature_dim=extra_dim,
    )
    model = replace(config.model, slot_features=slot_features)
    config = replace(config, model=model, fm=replace(config.fm, use_draft_feature=True))
    config_path = _write_config(tmp_path / f"invalid_fm_draft_feature_{extra_dim}.yaml", config)

    with pytest.raises(ValueError, match="fm.use_draft_feature"):
        load_config(config_path)


def test_patternboost_fm_mode_rejects_unserved_extra_slot_features(tmp_path, make_config) -> None:
    config = make_config()
    slot_features = replace(
        config.model.slot_features,
        use_extra_slot_features=True,
        extra_slot_feature_dim=4,
    )
    model = replace(config.model, slot_features=slot_features)
    pb = replace(config.patternboost, training=replace(config.patternboost.training, mode="fm"))
    config = replace(config, model=model, fm=replace(config.fm, use_draft_feature=False), patternboost=pb)
    config_path = _write_config(tmp_path / "invalid_fm_extra_without_draft.yaml", config)

    with pytest.raises(ValueError, match="fm.use_draft_feature"):
        load_config(config_path)


def test_completion_operator_defaults_to_learned() -> None:
    config = load_config(REPO_ROOT / "config.yaml")

    assert config.patternboost.generation.completion_operator == "learned"
    assert config.patternboost.generation.random_center_jitter_std > 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("completion_operator", "random", "completion_operator"),
        ("completion_operator", "", "completion_operator"),
        ("random_center_jitter_std", 0.0, "random_center_jitter_std"),
        ("random_center_jitter_std", -0.1, "random_center_jitter_std"),
    ],
)
def test_completion_operator_validation_rejects_invalid_values(
    tmp_path, make_config, field, value, message
) -> None:
    config = make_config()
    generation = replace(config.patternboost.generation, **{field: value})
    config = replace(config, patternboost=replace(config.patternboost, generation=generation))
    config_path = _write_config(tmp_path / f"invalid_{field}.yaml", config)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize("operator", ["random_perturb", "random_scratch"])
def test_random_completion_operators_are_accepted(tmp_path, make_config, operator) -> None:
    config = make_config()
    generation = replace(config.patternboost.generation, completion_operator=operator)
    config = replace(config, patternboost=replace(config.patternboost, generation=generation))
    config_path = _write_config(tmp_path / f"valid_{operator}.yaml", config)

    assert load_config(config_path).patternboost.generation.completion_operator == operator
