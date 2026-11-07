"""Integration tests for the multi-round outer-loop inference pipeline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from circle_pack.config import FMSection, ModelSection, OuterLoopSection
from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import center_to_latent_approx, latent_to_center_simple
from models.factory import build_model
from pipelines.outer_loop import run_outer_loop_episode, run_outer_round_segmented_demo


def _tiny_model() -> ModelSection:
    return ModelSection(
        preset="tiny",
        d_model=192,
        n_heads=6,
        n_layers=4,
        mlp_ratio=4.0,
        dropout=0.0,
    )


def test_outer_loop_episode_runs_and_computes_drift(make_config) -> None:
    config = make_config(
        K=6,
        activation_budget=1,
        model=_tiny_model(),
        fm=FMSection(
            num_samples=3,
            num_steps=6,
            source_noise_std=0.05,
            fixed_mode="frozen_full_state",
            return_terminal_velocity_norm=True,
            return_short_displacement=True,
            short_displacement_fraction=0.25,
        ),
        outer_loop=OuterLoopSection(
            max_rounds=2,
            candidate_mode="all_free",
            segmented_demo=False,
            segmented_split=0.5,
            representative_reduction="first_sample",
        ),
    )
    env = CirclePackingEnv(config)
    model = build_model(config)

    records = run_outer_loop_episode(
        env,
        model,
        mode="fm",
        cfg=config,
        latent_to_center_fn=latent_to_center_simple,
        center_to_latent_fn=center_to_latent_approx,
        max_outer_rounds=2,
    )

    assert len(records) >= 1
    assert "dispersion" in records[0].raw_signals
    assert int(np.asarray(records[0].env_info["candidate_mask"]).sum()) == config.problem.K
    if len(records) >= 2:
        assert "drift" in records[1].raw_signals


def test_outer_loop_episode_runs_with_outer_step_embedding(make_config) -> None:
    config = make_config(
        K=5,
        activation_budget=1,
        model=replace(
            _tiny_model(),
            use_outer_step_embedding=True,
            outer_step_embedding_type="learned",
        ),
        outer_loop=OuterLoopSection(
            max_rounds=2,
            candidate_mode="all_free",
            segmented_demo=False,
            segmented_split=0.5,
            representative_reduction="first_sample",
        ),
    )
    env = CirclePackingEnv(config)
    model = build_model(config)

    records = run_outer_loop_episode(
        env,
        model,
        mode="fm",
        cfg=config,
        latent_to_center_fn=latent_to_center_simple,
        center_to_latent_fn=center_to_latent_approx,
        max_outer_rounds=2,
    )

    assert len(records) >= 1
    assert "dispersion" in records[0].raw_signals


def test_outer_loop_variance_demo_candidate_mode(make_config) -> None:
    config = make_config(
        K=6,
        activation_budget=2,
        model=_tiny_model(),
        outer_loop=OuterLoopSection(
            max_rounds=2,
            candidate_mode="variance_demo",
            segmented_demo=False,
            segmented_split=0.5,
            representative_reduction="first_sample",
        ),
    )
    env = CirclePackingEnv(config)
    model = build_model(config)
    records = run_outer_loop_episode(
        env,
        model,
        mode="fm",
        cfg=config,
        latent_to_center_fn=latent_to_center_simple,
        center_to_latent_fn=center_to_latent_approx,
        max_outer_rounds=1,
    )

    candidate_count = int(np.asarray(records[0].env_info["candidate_mask"]).sum())
    assert candidate_count <= config.env.activation_budget


def test_segmented_demo_wrapper_runs(make_config) -> None:
    config = make_config(
        K=5,
        activation_budget=1,
        model=_tiny_model(),
        outer_loop=OuterLoopSection(
            max_rounds=2,
            candidate_mode="all_free",
            segmented_demo=True,
            segmented_split=0.5,
            representative_reduction="first_sample",
        ),
    )
    env = CirclePackingEnv(config)
    env.reset(seed=config.project.seed)
    model = build_model(config)

    records = run_outer_round_segmented_demo(
        env,
        model,
        mode="fm",
        cfg=config,
        latent_to_center_fn=latent_to_center_simple,
        center_to_latent_fn=center_to_latent_approx,
    )

    assert len(records) >= 1
    assert records[0].proposal_bundle.center_samples.ndim == 3
