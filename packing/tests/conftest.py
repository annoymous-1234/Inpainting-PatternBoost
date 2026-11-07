"""Shared fixtures for the circle packing test suite."""

from __future__ import annotations

import pytest

from circle_pack.config import (
    DDPMSection,
    DemoPrioritySection,
    EnvSection,
    FMSection,
    ModelHeadsSection,
    ModelPairBiasSection,
    ModelSection,
    ModelSlotFeaturesSection,
    NoisyHintPolicySection,
    OuterLoopSection,
    PoliciesSection,
    ProblemSection,
    ProjectConfig,
    ProjectSection,
    RandomLegalPolicySection,
    RenderSection,
    RLSection,
    RolloutSection,
    SafetySection,
    TrainSection,
)


@pytest.fixture
def make_config():
    """Return a small config factory for deterministic tests."""

    def _make_config(
        *,
        seed: int = 123,
        K: int = 5,
        max_steps: int = 4,
        commit_policy: str = "shrink_to_fit",
        activation_budget: int = 2,
        r_min_commit: float = 0.02,
        commit_order: str = "slot_index",
        center_conflict_policy: str = "none",
        center_duplicate_eps: float = 1.0e-8,
        done_on_all_active: bool = True,
        policy_name: str = "random_legal",
        noise_std_xy: float = 0.08,
        noise_std_r: float = 0.05,
        model: ModelSection | None = None,
        train: TrainSection | None = None,
        ddpm: DDPMSection | None = None,
        fm: FMSection | None = None,
        outer_loop: OuterLoopSection | None = None,
        demo_priority: DemoPrioritySection | None = None,
    ) -> ProjectConfig:
        del commit_policy, r_min_commit
        return ProjectConfig(
            project=ProjectSection(seed=seed, dtype="float64"),
            problem=ProblemSection(K=K, max_steps=max_steps),
            env=EnvSection(
                activation_budget=activation_budget,
                commit_order=commit_order,
                center_conflict_policy=center_conflict_policy,
                center_duplicate_eps=center_duplicate_eps,
                initialization_mode="random_legal",
                done_on_all_active=done_on_all_active,
            ),
            safety=SafetySection(
                clip_positions=True,
                radius_max=0.5,
                feasibility_tol=1.0e-12,
            ),
            rollout=RolloutSection(
                policy_name=policy_name,
                record_snapshots=False,
                render_final=False,
                save_figure=False,
                figure_path="test_rollout.png",
            ),
            render=RenderSection(
                figsize=(4.0, 4.0),
                dpi=100,
                show_slot_labels=False,
                active_alpha=0.85,
                masked_alpha=0.20,
                active_color="#1f77b4",
                masked_color="#7f7f7f",
            ),
            model=model
            or ModelSection(
                preset="base",
                d_model=256,
                n_heads=8,
                n_layers=6,
                mlp_ratio=4.0,
                dropout=0.0,
                activation="gelu",
                use_global_token=True,
                use_outer_step_embedding=False,
                outer_step_embedding_type="learned",
                use_inner_time_embedding=True,
                slot_features=ModelSlotFeaturesSection(
                    use_fixed_flag=False,
                    use_latent=True,
                    use_physical=False,
                    use_wall_slack=False,
                    use_extra_slot_features=False,
                    extra_slot_feature_dim=0,
                ),
                pair_bias=ModelPairBiasSection(
                    enabled=False,
                    pair_bias_mode="scalar",
                    use_pair_dxdy=True,
                    use_pair_distance=True,
                    use_pair_gap=False,
                    use_pair_radii=False,
                    use_pair_role_bits=False,
                    use_pair_abs_delta=False,
                    use_pair_sqdist=False,
                ),
                heads=ModelHeadsSection(
                    enable_ddpm_head=True,
                    enable_fm_head=True,
                    enable_simple_proposal_head=True,
                    enable_value_head=True,
                    ddpm_target="eps",
                    ddpm_zero_fixed_outputs=True,
                    fm_zero_fixed_outputs=True,
                    proposal_target="delta_z",
                    proposal_zero_fixed_outputs=True,
                ),
            ),
            policies=PoliciesSection(
                random_legal=RandomLegalPolicySection(),
                noisy_hint=NoisyHintPolicySection(
                    noise_std_xy=noise_std_xy,
                    noise_std_r=noise_std_r,
                ),
            ),
            train=train
            or TrainSection(
                mode="fm",
                dataset_path="test_teacher_layouts.pt",
                batch_size=4,
                num_epochs=1,
                lr=1.0e-3,
                weight_decay=0.0,
                fixed_ratio_min=0.0,
                fixed_ratio_max=0.5,
                assignment_mode_no_draft="xy",
                assignment_mode_with_draft="draft_match",
                synthetic_draft_mode="mixed",
                synthetic_draft_scale=0.15,
                draft_center_jitter_std=0.03,
                draft_radius_scale_down=0.25,
                draft_radius_scale_up=0.10,
                checkpoint_dir="test_checkpoints",
                num_workers=0,
            ),
            ddpm=ddpm
            or DDPMSection(
                num_samples=3,
                num_steps=8,
                beta_start=1.0e-4,
                beta_end=0.08,
                tau_start_infer=1.0,
                use_draft_warmstart=True,
                use_draft_feature=False,
                fixed_mode="frozen_full_state",
                noising_center="draft",
                draft_strength=1.0,
                use_timestep_subsampling=False,
                infer_num_steps=4,
            ),
            fm=fm
            or FMSection(
                num_samples=3,
                num_steps=8,
                source_noise_std=0.05,
                fixed_mode="frozen_full_state",
                return_terminal_velocity_norm=True,
                return_short_displacement=True,
                short_displacement_fraction=0.25,
            ),
            outer_loop=outer_loop
            or OuterLoopSection(
                max_rounds=3,
                candidate_mode="all_free",
                segmented_demo=False,
                segmented_split=0.5,
                representative_reduction="first_sample",
            ),
            demo_priority=demo_priority or DemoPrioritySection(lambda_var=1.0, lambda_drift=0.25, lambda_motion=0.25),
            rl=RLSection(placeholder=True),
        )

    return _make_config
