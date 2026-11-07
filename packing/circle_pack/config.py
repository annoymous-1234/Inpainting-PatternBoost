"""Configuration loading and validation for the circle packing project.

All user-facing runtime options live in a single YAML file. This module turns the
raw YAML dictionary into typed dataclasses so the rest of the codebase can rely
on explicit fields instead of unstructured dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml


T = TypeVar("T")

MODEL_PRESET_SPECS: dict[str, dict[str, float | int]] = {
    "tiny": {
        "d_model": 192,
        "n_heads": 6,
        "n_layers": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
    },
    "base": {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
    },
}


@dataclass(frozen=True)
class ProjectSection:
    """Global project settings shared across modules."""

    seed: int = 1234
    dtype: str = "float32"


@dataclass(frozen=True)
class ProblemSection:
    """Static problem definition for one environment instance."""

    K: int = 8
    max_steps: int = 6


@dataclass(frozen=True)
class EnvSection:
    """Environment behavior and center-only commit settings."""

    activation_budget: int = 2
    commit_order: str = "slot_index"
    center_conflict_policy: str = "none"
    center_duplicate_eps: float = 1.0e-8
    initialization_mode: str = "random_legal"
    done_on_all_active: bool = True


@dataclass(frozen=True)
class SafetySection:
    """Lightweight clipping and feasibility tolerance parameters."""

    clip_positions: bool = True
    radius_max: float = 0.5
    feasibility_tol: float = 1.0e-12


@dataclass(frozen=True)
class RolloutSection:
    """Environment demo rollout defaults."""

    policy_name: str = "noisy_hint"
    record_snapshots: bool = False
    render_final: bool = True
    save_figure: bool = True
    figure_path: str = "demo_rollout.png"


@dataclass(frozen=True)
class RenderSection:
    """Renderer defaults for debug figures."""

    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 140
    show_slot_labels: bool = True
    active_alpha: float = 0.85
    masked_alpha: float = 0.20
    active_color: str = "#1f77b4"
    masked_color: str = "#7f7f7f"


@dataclass(frozen=True)
class ModelSlotFeaturesSection:
    """Feature toggles for the per-slot input builder."""

    use_fixed_flag: bool = False
    use_latent: bool = True
    use_physical: bool = False
    use_wall_slack: bool = False
    use_extra_slot_features: bool = False
    extra_slot_feature_dim: int = 0


@dataclass(frozen=True)
class ModelPairBiasSection:
    """Configuration for geometry-aware attention pair bias."""

    enabled: bool = False
    pair_bias_mode: str = "scalar"
    use_pair_dxdy: bool = True
    use_pair_distance: bool = True
    use_pair_gap: bool = False
    use_pair_radii: bool = False
    use_pair_role_bits: bool = False
    use_pair_abs_delta: bool = False
    use_pair_sqdist: bool = False


@dataclass(frozen=True)
class ModelHeadsSection:
    """Configuration for optional output heads."""

    enable_ddpm_head: bool = True
    enable_fm_head: bool = True
    enable_simple_proposal_head: bool = True
    enable_value_head: bool = True
    ddpm_target: str = "eps"
    ddpm_zero_fixed_outputs: bool = True
    fm_zero_fixed_outputs: bool = True
    proposal_target: str = "delta_z"
    proposal_zero_fixed_outputs: bool = True


@dataclass(frozen=True)
class ModelSection:
    """Transformer backbone and head configuration for the model stack."""

    preset: str = "base"
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    activation: str = "gelu"
    use_global_token: bool = True
    use_outer_step_embedding: bool = False
    outer_step_embedding_type: str = "learned"
    use_inner_time_embedding: bool = True
    slot_features: ModelSlotFeaturesSection = field(default_factory=ModelSlotFeaturesSection)
    pair_bias: ModelPairBiasSection = field(default_factory=ModelPairBiasSection)
    heads: ModelHeadsSection = field(default_factory=ModelHeadsSection)


@dataclass(frozen=True)
class TrainSection:
    """Settings for supervised data construction and optimization."""

    mode: str = "fm"
    dataset_path: str = ""
    batch_size: int = 8
    num_epochs: int = 2
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    fixed_ratio_min: float = 0.0
    fixed_ratio_max: float = 0.5
    assignment_mode_no_draft: str = "xy"
    assignment_mode_with_draft: str = "draft_match"
    synthetic_draft_mode: str = "mixed"
    synthetic_draft_scale: float = 0.15
    draft_center_jitter_std: float = 0.03
    # Deprecated center-only no-ops retained for older config files.
    draft_radius_scale_down: float = 0.25
    draft_radius_scale_up: float = 0.10
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0


@dataclass(frozen=True)
class DDPMSection:
    """Configuration for DDPM training helpers and proposal sampling."""

    num_samples: int = 4
    num_steps: int = 64
    beta_start: float = 1.0e-4
    beta_end: float = 0.08
    tau_start_infer: float = 1.0
    use_draft_warmstart: bool = True
    use_draft_feature: bool = False
    fixed_mode: str = "frozen_full_state"
    noising_center: str = "draft"
    draft_strength: float = 1.0
    use_timestep_subsampling: bool = False
    infer_num_steps: int = 16


@dataclass(frozen=True)
class FMSection:
    """Configuration for FM training helpers and proposal sampling."""

    num_samples: int = 4
    num_steps: int = 16
    source_noise_std: float = 0.05
    fixed_mode: str = "frozen_full_state"
    return_terminal_velocity_norm: bool = True
    return_short_displacement: bool = True
    short_displacement_fraction: float = 0.25
    use_draft_feature: bool = False


@dataclass(frozen=True)
class OuterLoopSection:
    """Configuration for the multi-round inference controller layer."""

    max_rounds: int = 4
    candidate_mode: str = "all_free"
    segmented_demo: bool = False
    segmented_split: float = 0.5
    representative_reduction: str = "first_sample"


@dataclass(frozen=True)
class DemoPrioritySection:
    """Weights for the tiny variance-based demo priority heuristic."""

    lambda_var: float = 1.0
    lambda_drift: float = 0.25
    lambda_motion: float = 0.25


@dataclass(frozen=True)
class PatternBoostPoolSamplingSection:
    """TopPool seed sampling settings for PatternBoost."""

    mode: str = "score_bucket_rank_mixture"
    alpha: float = 1.0
    offset: float = 1.0
    top_fraction: float = 0.5
    uniform_mix: float = 0.1


@dataclass(frozen=True)
class PatternBoostPoolSection:
    """TopPool storage, initialization, and deduplication settings."""

    max_size: int = 2048
    dedup_tol: float = 1.0e-5
    init_sources: list[str] = field(default_factory=lambda: ["teacher_layouts", "greedy"])
    greedy_num_seeds: int = 256
    save_path: str | None = None
    sampling: PatternBoostPoolSamplingSection = field(default_factory=PatternBoostPoolSamplingSection)


@dataclass(frozen=True)
class PatternBoostEditSection:
    """PatternBoost edit-set sampling settings."""

    policies: list[str] = field(default_factory=lambda: ["random", "random_point_knn"])
    policy_weights: list[float] = field(default_factory=lambda: [0.4, 0.6])
    edit_ratios: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.6, 1.0])
    ratio_weights: list[float] = field(default_factory=lambda: [0.30, 0.30, 0.20, 0.15, 0.05])
    knn_k: int | None = None


@dataclass(frozen=True)
class PatternBoostTrainingSection:
    """PatternBoost per-generation supervised fine-tuning settings."""

    mode: str = "fm"
    samples_per_generation: int = 8192
    epochs_per_generation: int = 1
    batch_size: int = 32
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    fine_tune_from_previous: bool = True
    use_draft_match: bool = True
    checkpoint_every_generations: int = 1


@dataclass(frozen=True)
class PatternBoostGenerationSection:
    """PatternBoost candidate-generation settings."""

    num_generations: int = 20
    candidates_per_generation: int = 1024
    candidate_batch_size: int = 1
    modes: list[str] = field(default_factory=lambda: ["full_episode", "one_shot"])
    mode_weights: list[float] = field(default_factory=lambda: [0.8, 0.2])
    greedy_complete_unfinished: bool = True
    completion_operator: str = "learned"
    random_center_jitter_std: float = 0.04


@dataclass(frozen=True)
class PatternBoostPolishSection:
    """PatternBoost post-generation polishing settings."""

    refresh_lp_cache: bool = False
    srp_enabled: bool = False
    srp_top_k: int = 64
    srp_num_workers: int = 1
    srp_min_improvement: float = 1.0e-12
    srp_Imax: int = 400
    srp_m: int = 30
    srp_beta: float = 0.99
    srp_backtrack: int = 5
    srp_step_center: float = 0.05
    srp_step_radius: float = 0.001
    srp_w_overlap: float = 1000.0
    srp_w_wall: float = 1000.0
    srp_alpha_sum_r: float = 0.2
    srp_lbfgs_gtol: float = 1.0e-8
    srp_lbfgs_ftol: float = 1.0e-12
    srp_lbfgs_maxiter: int = 1000
    srp_lbfgs_maxcor: int = 20
    srp_use_small_radius_branch: bool = True
    srp_init_r0_small: float = 0.01
    srp_init_rjitter_small: float = 0.005
    srp_small_branch_step_scale: float = 0.2
    srp_strict_shrink: bool = True
    srp_strict_shrink_margin: float = 1.0e-8


@dataclass(frozen=True)
class PatternBoostRuntimeSection:
    """PatternBoost runtime output and logging settings."""

    seed: int = 1234
    device: str = "auto"
    checkpoint_dir: str = "checkpoints/patternboost"
    run_name: str | None = None
    log_jsonl: bool = True
    save_checkpoints: bool = True
    log_candidates_jsonl: bool = True
    top_pool_snapshot_every_generations: int = 0


@dataclass(frozen=True)
class PatternBoostResumeSection:
    """PatternBoost checkpoint and TopPool resume settings."""

    enabled: bool = False
    checkpoint_path: str | None = None
    top_pool_path: str | None = None
    load_optimizer: bool = True


@dataclass(frozen=True)
class PatternBoostSection:
    """PatternBoost outer-loop configuration."""

    enabled: bool = True
    pool: PatternBoostPoolSection = field(default_factory=PatternBoostPoolSection)
    edit: PatternBoostEditSection = field(default_factory=PatternBoostEditSection)
    training: PatternBoostTrainingSection = field(default_factory=PatternBoostTrainingSection)
    generation: PatternBoostGenerationSection = field(default_factory=PatternBoostGenerationSection)
    polish: PatternBoostPolishSection = field(default_factory=PatternBoostPolishSection)
    runtime: PatternBoostRuntimeSection = field(default_factory=PatternBoostRuntimeSection)
    resume: PatternBoostResumeSection = field(default_factory=PatternBoostResumeSection)


@dataclass(frozen=True)
class RandomLegalPolicySection:
    """Configuration for the random legal baseline policy."""


@dataclass(frozen=True)
class NoisyHintPolicySection:
    """Configuration for the noisy hint baseline policy."""

    noise_std_xy: float = 0.08
    noise_std_r: float = 0.05


@dataclass(frozen=True)
class PoliciesSection:
    """Per-policy configuration grouped in one section."""

    random_legal: RandomLegalPolicySection = field(default_factory=RandomLegalPolicySection)
    noisy_hint: NoisyHintPolicySection = field(default_factory=NoisyHintPolicySection)


@dataclass(frozen=True)
class RLSection:
    """Placeholder RL section for future expansion."""

    placeholder: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    """Fully typed configuration object for the project."""

    project: ProjectSection = field(default_factory=ProjectSection)
    problem: ProblemSection = field(default_factory=ProblemSection)
    env: EnvSection = field(default_factory=EnvSection)
    safety: SafetySection = field(default_factory=SafetySection)
    rollout: RolloutSection = field(default_factory=RolloutSection)
    render: RenderSection = field(default_factory=RenderSection)
    model: ModelSection = field(default_factory=ModelSection)
    train: TrainSection = field(default_factory=TrainSection)
    ddpm: DDPMSection = field(default_factory=DDPMSection)
    fm: FMSection = field(default_factory=FMSection)
    outer_loop: OuterLoopSection = field(default_factory=OuterLoopSection)
    demo_priority: DemoPrioritySection = field(default_factory=DemoPrioritySection)
    patternboost: PatternBoostSection = field(default_factory=PatternBoostSection)
    policies: PoliciesSection = field(default_factory=PoliciesSection)
    rl: RLSection = field(default_factory=RLSection)

    @property
    def numpy_dtype(self) -> np.dtype:
        """Return the numpy dtype configured for environment state arrays."""

        return np.dtype(self.project.dtype)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype configured for model tensors and parameters."""

        dtype_name = self.project.dtype.lower()
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch dtype mapping for project.dtype={self.project.dtype!r}.")
        return mapping[dtype_name]


def _build_section(section_cls: type[T], payload: dict[str, Any] | None) -> T:
    """Construct a dataclass section from a partial dictionary."""

    payload = payload or {}
    valid_names = {field_info.name for field_info in fields(section_cls)}
    kwargs = {key: value for key, value in payload.items() if key in valid_names}

    if section_cls is RenderSection and "figsize" in kwargs:
        kwargs["figsize"] = tuple(kwargs["figsize"])
    if section_cls is PoliciesSection:
        kwargs = {
            "random_legal": _build_section(RandomLegalPolicySection, payload.get("random_legal")),
            "noisy_hint": _build_section(NoisyHintPolicySection, payload.get("noisy_hint")),
        }
    if section_cls is ModelSection:
        kwargs = {
            key: value
            for key, value in payload.items()
            if key in valid_names and key not in {"slot_features", "pair_bias", "heads"}
        }
        kwargs.update(
            {
                "slot_features": _build_section(ModelSlotFeaturesSection, payload.get("slot_features")),
                "pair_bias": _build_section(ModelPairBiasSection, payload.get("pair_bias")),
                "heads": _build_section(ModelHeadsSection, payload.get("heads")),
            }
        )
    if section_cls is PatternBoostSection:
        kwargs = {
            "enabled": payload.get("enabled", PatternBoostSection.enabled),
            "pool": _build_section(PatternBoostPoolSection, payload.get("pool")),
            "edit": _build_section(PatternBoostEditSection, payload.get("edit")),
            "training": _build_section(PatternBoostTrainingSection, payload.get("training")),
            "generation": _build_section(PatternBoostGenerationSection, payload.get("generation")),
            "polish": _build_section(PatternBoostPolishSection, payload.get("polish")),
            "runtime": _build_section(PatternBoostRuntimeSection, payload.get("runtime")),
            "resume": _build_section(PatternBoostResumeSection, payload.get("resume")),
        }
    if section_cls is PatternBoostPoolSection:
        kwargs = {
            key: value
            for key, value in payload.items()
            if key in valid_names and key != "sampling"
        }
        kwargs["sampling"] = _build_section(PatternBoostPoolSamplingSection, payload.get("sampling"))
    if section_cls is PatternBoostPolishSection:
        kwargs = {key: value for key, value in payload.items() if key in valid_names}
        if "refresh_lp_cache" not in kwargs and "cap_expansion" in payload:
            kwargs["refresh_lp_cache"] = bool(payload["cap_expansion"])
    return section_cls(**kwargs)


def _validate_model_config(model: ModelSection) -> None:
    """Validate model-side invariants and preset semantics."""

    if model.preset not in {"tiny", "base", "custom"}:
        raise ValueError("model.preset must be 'tiny', 'base', or 'custom'.")
    if model.d_model <= 0 or model.n_heads <= 0 or model.n_layers <= 0:
        raise ValueError("model dimensions and layer counts must be positive.")
    if model.d_model % model.n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads.")
    if model.mlp_ratio <= 0.0:
        raise ValueError("model.mlp_ratio must be positive.")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("model.dropout must be in the interval [0, 1).")
    if model.activation not in {"gelu", "silu"}:
        raise ValueError("model.activation must be 'gelu' or 'silu'.")
    if model.outer_step_embedding_type not in {"learned", "sin"}:
        raise ValueError("model.outer_step_embedding_type must be 'learned' or 'sin'.")

    if model.preset in MODEL_PRESET_SPECS:
        preset_spec = MODEL_PRESET_SPECS[model.preset]
        for key, expected in preset_spec.items():
            actual = getattr(model, key)
            if actual != expected:
                raise ValueError(
                    f"model.preset={model.preset!r} requires {key}={expected!r}, got {actual!r}. "
                    "Use preset='custom' for non-preset values."
                )

    slot_cfg = model.slot_features
    if slot_cfg.extra_slot_feature_dim < 0:
        raise ValueError("model.slot_features.extra_slot_feature_dim must be non-negative.")
    if slot_cfg.use_extra_slot_features and slot_cfg.extra_slot_feature_dim <= 0:
        raise ValueError(
            "model.slot_features.extra_slot_feature_dim must be positive when "
            "use_extra_slot_features is enabled."
        )
    if not any(
        (
            slot_cfg.use_fixed_flag,
            slot_cfg.use_latent,
            slot_cfg.use_physical,
            slot_cfg.use_wall_slack,
            slot_cfg.use_extra_slot_features,
        )
    ):
        raise ValueError("At least one slot feature must be enabled.")

    pair_cfg = model.pair_bias
    if pair_cfg.pair_bias_mode not in {"scalar", "per_head"}:
        raise ValueError("model.pair_bias.pair_bias_mode must be 'scalar' or 'per_head'.")
    if pair_cfg.use_pair_gap or pair_cfg.use_pair_radii:
        raise ValueError("Radius-dependent pair gap/radii features are disabled in center-only mode.")

    heads_cfg = model.heads
    if heads_cfg.ddpm_target not in {"eps", "x0", "v"}:
        raise ValueError("model.heads.ddpm_target must be 'eps', 'x0', or 'v'.")
    if heads_cfg.proposal_target not in {"delta_z", "z_prop"}:
        raise ValueError("model.heads.proposal_target must be 'delta_z' or 'z_prop'.")


def _validate_train_config(train: TrainSection) -> None:
    """Validate supervised training configuration."""

    if train.mode not in {"ddpm", "fm"}:
        raise ValueError("train.mode must be 'ddpm' or 'fm'.")
    if not train.dataset_path:
        raise ValueError("train.dataset_path must be set.")
    if train.batch_size <= 0 or train.num_epochs <= 0:
        raise ValueError("train.batch_size and train.num_epochs must be positive.")
    if train.lr <= 0.0:
        raise ValueError("train.lr must be positive.")
    if train.weight_decay < 0.0:
        raise ValueError("train.weight_decay must be non-negative.")
    if not 0.0 <= train.fixed_ratio_min <= 1.0 or not 0.0 <= train.fixed_ratio_max <= 1.0:
        raise ValueError("train fixed ratio bounds must lie in [0, 1].")
    if train.fixed_ratio_min > train.fixed_ratio_max:
        raise ValueError("train.fixed_ratio_min must be <= train.fixed_ratio_max.")
    if train.assignment_mode_no_draft not in {"random", "xy"}:
        raise ValueError("train.assignment_mode_no_draft must be 'random' or 'xy'.")
    if train.assignment_mode_with_draft not in {"draft_match", "random"}:
        raise ValueError("train.assignment_mode_with_draft must be 'draft_match' or 'random'.")
    if train.synthetic_draft_mode not in {"gaussian_latent", "center_jitter", "radius_shrink_and_jitter", "mixed"}:
        raise ValueError(
            "train.synthetic_draft_mode must be one of "
            "{'gaussian_latent', 'center_jitter', 'radius_shrink_and_jitter', 'mixed'}."
        )
    if train.synthetic_draft_scale < 0.0:
        raise ValueError("train.synthetic_draft_scale must be non-negative.")
    if train.draft_center_jitter_std < 0.0:
        raise ValueError("train.draft_center_jitter_std must be non-negative.")
    if not 0.0 <= train.draft_radius_scale_down <= 1.0:
        raise ValueError("train.draft_radius_scale_down must be in [0, 1].")
    if train.draft_radius_scale_up < 0.0:
        raise ValueError("train.draft_radius_scale_up must be non-negative.")
    if train.num_workers < 0:
        raise ValueError("train.num_workers must be non-negative.")


def _validate_ddpm_config(ddpm: DDPMSection) -> None:
    """Validate DDPM training and inference configuration."""

    if ddpm.num_samples <= 0 or ddpm.num_steps <= 0:
        raise ValueError("ddpm.num_samples and ddpm.num_steps must be positive.")
    if not 0.0 < ddpm.beta_start < ddpm.beta_end < 1.0:
        raise ValueError("ddpm.beta_start and ddpm.beta_end must satisfy 0 < beta_start < beta_end < 1.")
    if not 0.0 <= ddpm.tau_start_infer <= 1.0:
        raise ValueError("ddpm.tau_start_infer must be in [0, 1].")
    if ddpm.fixed_mode not in {"conditional", "frozen_full_state"}:
        raise ValueError("ddpm.fixed_mode must be 'conditional' or 'frozen_full_state'.")
    if ddpm.noising_center not in {"draft", "target"}:
        raise ValueError("ddpm.noising_center must be 'draft' or 'target'.")
    if not 0.0 <= ddpm.draft_strength <= 1.0:
        raise ValueError("ddpm.draft_strength must be in [0, 1].")
    if ddpm.infer_num_steps <= 0:
        raise ValueError("ddpm.infer_num_steps must be positive.")


def _validate_fm_config(fm: FMSection) -> None:
    """Validate FM training and inference configuration."""

    if fm.num_samples <= 0 or fm.num_steps <= 0:
        raise ValueError("fm.num_samples and fm.num_steps must be positive.")
    if fm.source_noise_std < 0.0:
        raise ValueError("fm.source_noise_std must be non-negative.")
    if fm.fixed_mode not in {"conditional", "frozen_full_state"}:
        raise ValueError("fm.fixed_mode must be 'conditional' or 'frozen_full_state'.")
    if not 0.0 <= fm.short_displacement_fraction <= 1.0:
        raise ValueError("fm.short_displacement_fraction must be in [0, 1].")
    if not isinstance(fm.use_draft_feature, bool):
        raise ValueError("fm.use_draft_feature must be boolean.")


def _validate_outer_loop_config(outer_loop: OuterLoopSection) -> None:
    """Validate outer-loop controller configuration."""

    if outer_loop.max_rounds <= 0:
        raise ValueError("outer_loop.max_rounds must be positive.")
    if outer_loop.candidate_mode not in {"all_free", "variance_demo"}:
        raise ValueError("outer_loop.candidate_mode must be 'all_free' or 'variance_demo'.")
    if not 0.0 < outer_loop.segmented_split < 1.0:
        raise ValueError("outer_loop.segmented_split must be strictly between 0 and 1.")
    if outer_loop.representative_reduction not in {"first_sample", "mean"}:
        raise ValueError("outer_loop.representative_reduction must be 'first_sample' or 'mean'.")


def _validate_positive_weights(values: list[float], name: str, *, expected_len: int | None = None) -> None:
    """Validate a nonnegative finite weight vector with positive sum."""

    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}.")
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if any(float(value) < 0.0 for value in values):
        raise ValueError(f"{name} values must be non-negative.")
    if sum(float(value) for value in values) <= 0.0:
        raise ValueError(f"{name} must have positive sum.")


def _validate_patternboost_config(patternboost: PatternBoostSection) -> None:
    """Validate PatternBoost configuration."""

    pool = patternboost.pool
    if pool.max_size <= 0:
        raise ValueError("patternboost.pool.max_size must be positive.")
    if pool.dedup_tol <= 0.0:
        raise ValueError("patternboost.pool.dedup_tol must be positive.")
    if pool.greedy_num_seeds < 0:
        raise ValueError("patternboost.pool.greedy_num_seeds must be non-negative.")
    if any(source not in {"teacher_layouts", "greedy", "top_pool"} for source in pool.init_sources):
        raise ValueError("patternboost.pool.init_sources supports teacher_layouts, greedy, and top_pool.")

    sampling = pool.sampling
    if sampling.mode not in {"uniform", "score_bucket_rank_mixture"}:
        raise ValueError("patternboost.pool.sampling.mode must be 'uniform' or 'score_bucket_rank_mixture'.")
    if sampling.alpha < 0.0:
        raise ValueError("patternboost.pool.sampling.alpha must be non-negative.")
    if sampling.offset <= 0.0:
        raise ValueError("patternboost.pool.sampling.offset must be positive.")
    if not 0.0 < sampling.top_fraction <= 1.0:
        raise ValueError("patternboost.pool.sampling.top_fraction must be in (0, 1].")
    if not 0.0 <= sampling.uniform_mix <= 1.0:
        raise ValueError("patternboost.pool.sampling.uniform_mix must be in [0, 1].")

    edit = patternboost.edit
    if not edit.policies:
        raise ValueError("patternboost.edit.policies must not be empty.")
    if any(policy not in {"random", "random_point_knn"} for policy in edit.policies):
        raise ValueError("patternboost.edit.policies supports random and random_point_knn.")
    _validate_positive_weights(edit.policy_weights, "patternboost.edit.policy_weights", expected_len=len(edit.policies))
    if not edit.edit_ratios:
        raise ValueError("patternboost.edit.edit_ratios must not be empty.")
    if any(float(ratio) <= 0.0 or float(ratio) > 1.0 for ratio in edit.edit_ratios):
        raise ValueError("patternboost.edit.edit_ratios must be in (0, 1].")
    _validate_positive_weights(edit.ratio_weights, "patternboost.edit.ratio_weights", expected_len=len(edit.edit_ratios))
    if edit.knn_k is not None and edit.knn_k <= 0:
        raise ValueError("patternboost.edit.knn_k must be positive when set.")

    training = patternboost.training
    if training.mode not in {"fm", "ddpm"}:
        raise ValueError("patternboost.training.mode must be 'fm' or 'ddpm'.")
    if training.samples_per_generation <= 0 or training.epochs_per_generation <= 0:
        raise ValueError("patternboost.training sample and epoch counts must be positive.")
    if training.batch_size <= 0:
        raise ValueError("patternboost.training.batch_size must be positive.")
    if training.lr <= 0.0:
        raise ValueError("patternboost.training.lr must be positive.")
    if training.weight_decay < 0.0:
        raise ValueError("patternboost.training.weight_decay must be non-negative.")
    if training.checkpoint_every_generations <= 0:
        raise ValueError("patternboost.training.checkpoint_every_generations must be positive.")

    generation = patternboost.generation
    if generation.num_generations <= 0 or generation.candidates_per_generation <= 0:
        raise ValueError("patternboost.generation counts must be positive.")
    if generation.candidate_batch_size <= 0:
        raise ValueError("patternboost.generation.candidate_batch_size must be positive.")
    if any(mode not in {"full_episode", "one_shot"} for mode in generation.modes):
        raise ValueError("patternboost.generation.modes supports full_episode and one_shot.")
    _validate_positive_weights(
        generation.mode_weights,
        "patternboost.generation.mode_weights",
        expected_len=len(generation.modes),
    )
    if generation.completion_operator not in {"learned", "random_perturb", "random_scratch"}:
        raise ValueError(
            "patternboost.generation.completion_operator must be 'learned', 'random_perturb', or 'random_scratch'."
        )
    if generation.random_center_jitter_std <= 0.0:
        raise ValueError("patternboost.generation.random_center_jitter_std must be positive.")

    polish = patternboost.polish
    if not isinstance(polish.refresh_lp_cache, bool):
        raise ValueError("patternboost.polish.refresh_lp_cache must be boolean.")
    if not isinstance(polish.srp_enabled, bool):
        raise ValueError("patternboost.polish.srp_enabled must be boolean.")
    if polish.srp_top_k < 0:
        raise ValueError("patternboost.polish.srp_top_k must be non-negative.")
    if polish.srp_num_workers <= 0:
        raise ValueError("patternboost.polish.srp_num_workers must be positive.")
    if polish.srp_min_improvement < 0.0:
        raise ValueError("patternboost.polish.srp_min_improvement must be non-negative.")
    if polish.srp_Imax <= 0 or polish.srp_m <= 0:
        raise ValueError("patternboost.polish SRP iteration counts must be positive.")
    if not 0.0 < polish.srp_beta <= 1.0:
        raise ValueError("patternboost.polish.srp_beta must be in (0, 1].")
    if polish.srp_backtrack < 0:
        raise ValueError("patternboost.polish.srp_backtrack must be non-negative.")
    if polish.srp_step_center <= 0.0 or polish.srp_step_radius <= 0.0:
        raise ValueError("patternboost.polish SRP step sizes must be positive.")
    if polish.srp_w_overlap < 0.0 or polish.srp_w_wall < 0.0:
        raise ValueError("patternboost.polish SRP penalty weights must be non-negative.")
    if polish.srp_alpha_sum_r < 0.0:
        raise ValueError("patternboost.polish.srp_alpha_sum_r must be non-negative.")
    if polish.srp_lbfgs_gtol <= 0.0 or polish.srp_lbfgs_ftol <= 0.0:
        raise ValueError("patternboost.polish L-BFGS tolerances must be positive.")
    if polish.srp_lbfgs_maxiter <= 0 or polish.srp_lbfgs_maxcor <= 0:
        raise ValueError("patternboost.polish L-BFGS iteration settings must be positive.")
    if not isinstance(polish.srp_use_small_radius_branch, bool):
        raise ValueError("patternboost.polish.srp_use_small_radius_branch must be boolean.")
    if polish.srp_init_r0_small <= 0.0 or polish.srp_init_rjitter_small < 0.0:
        raise ValueError("patternboost.polish small-radius initialization must be valid.")
    if polish.srp_small_branch_step_scale <= 0.0:
        raise ValueError("patternboost.polish.srp_small_branch_step_scale must be positive.")
    if not isinstance(polish.srp_strict_shrink, bool):
        raise ValueError("patternboost.polish.srp_strict_shrink must be boolean.")
    if polish.srp_strict_shrink_margin < 0.0:
        raise ValueError("patternboost.polish.srp_strict_shrink_margin must be non-negative.")

    runtime = patternboost.runtime
    if runtime.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("patternboost.runtime.device must be 'auto', 'cpu', or 'cuda'.")
    if not isinstance(runtime.log_jsonl, bool):
        raise ValueError("patternboost.runtime.log_jsonl must be boolean.")
    if not isinstance(runtime.save_checkpoints, bool):
        raise ValueError("patternboost.runtime.save_checkpoints must be boolean.")
    if not isinstance(runtime.log_candidates_jsonl, bool):
        raise ValueError("patternboost.runtime.log_candidates_jsonl must be boolean.")
    if runtime.top_pool_snapshot_every_generations < 0:
        raise ValueError("patternboost.runtime.top_pool_snapshot_every_generations must be non-negative.")

    resume = patternboost.resume
    if not isinstance(resume.enabled, bool):
        raise ValueError("patternboost.resume.enabled must be boolean.")
    if not isinstance(resume.load_optimizer, bool):
        raise ValueError("patternboost.resume.load_optimizer must be boolean.")
    if resume.enabled:
        if not resume.checkpoint_path:
            raise ValueError("patternboost.resume.checkpoint_path is required when resume is enabled.")
        if not resume.top_pool_path:
            raise ValueError("patternboost.resume.top_pool_path is required when resume is enabled.")


def _validate_config(config: ProjectConfig) -> None:
    """Validate key invariants for the v1 research scaffold."""

    if config.problem.K <= 0:
        raise ValueError("problem.K must be positive.")
    if config.problem.max_steps <= 0:
        raise ValueError("problem.max_steps must be positive.")
    if config.env.activation_budget < 1:
        raise ValueError("env.activation_budget must be at least 1.")
    if config.env.commit_order not in {"slot_index", "random"}:
        raise ValueError("env.commit_order must be 'slot_index' or 'random'.")
    if config.env.center_conflict_policy not in {"none", "jitter", "resample", "reject"}:
        raise ValueError("env.center_conflict_policy must be one of {'none', 'jitter', 'resample', 'reject'}.")
    if config.env.center_duplicate_eps < 0.0:
        raise ValueError("env.center_duplicate_eps must be non-negative.")
    if config.env.initialization_mode != "random_legal":
        raise ValueError("Version 1 only supports env.initialization_mode='random_legal'.")
    if config.safety.radius_max <= 0.0 or config.safety.radius_max > 0.5:
        raise ValueError("safety.radius_max must be in the interval (0, 0.5].")
    if config.safety.feasibility_tol < 0.0:
        raise ValueError("safety.feasibility_tol must be non-negative.")
    if config.rollout.policy_name not in {"random_legal", "noisy_hint"}:
        raise ValueError("rollout.policy_name must be 'random_legal' or 'noisy_hint'.")
    np.dtype(config.project.dtype)
    _ = config.torch_dtype
    _validate_model_config(config.model)
    _validate_train_config(config.train)
    _validate_ddpm_config(config.ddpm)
    _validate_fm_config(config.fm)
    _validate_outer_loop_config(config.outer_loop)
    _validate_patternboost_config(config.patternboost)
    if config.ddpm.use_draft_feature:
        slot_cfg = config.model.slot_features
        if not slot_cfg.use_extra_slot_features or int(slot_cfg.extra_slot_feature_dim) != 4:
            raise ValueError(
                "ddpm.use_draft_feature=True requires model.slot_features.use_extra_slot_features=True "
                "and model.slot_features.extra_slot_feature_dim=4."
            )
    if config.fm.use_draft_feature:
        slot_cfg = config.model.slot_features
        if not slot_cfg.use_extra_slot_features or int(slot_cfg.extra_slot_feature_dim) != 4:
            raise ValueError(
                "fm.use_draft_feature=True requires model.slot_features.use_extra_slot_features=True "
                "and model.slot_features.extra_slot_feature_dim=4."
            )
    if config.patternboost.training.mode == "fm":
        if not config.model.heads.enable_fm_head:
            raise ValueError("patternboost.training.mode='fm' requires model.heads.enable_fm_head=True.")
        if config.model.slot_features.use_extra_slot_features and not config.fm.use_draft_feature:
            raise ValueError(
                "patternboost.training.mode='fm' with model.slot_features.use_extra_slot_features=True "
                "requires fm.use_draft_feature=True."
            )


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project config from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ProjectConfig(
        project=_build_section(ProjectSection, raw.get("project")),
        problem=_build_section(ProblemSection, raw.get("problem")),
        env=_build_section(EnvSection, raw.get("env")),
        safety=_build_section(SafetySection, raw.get("safety")),
        rollout=_build_section(RolloutSection, raw.get("rollout")),
        render=_build_section(RenderSection, raw.get("render")),
        model=_build_section(ModelSection, raw.get("model")),
        train=_build_section(TrainSection, raw.get("train")),
        ddpm=_build_section(DDPMSection, raw.get("ddpm")),
        fm=_build_section(FMSection, raw.get("fm")),
        outer_loop=_build_section(OuterLoopSection, raw.get("outer_loop")),
        demo_priority=_build_section(DemoPrioritySection, raw.get("demo_priority")),
        patternboost=_build_section(PatternBoostSection, raw.get("patternboost")),
        policies=_build_section(PoliciesSection, raw.get("policies")),
        rl=_build_section(RLSection, raw.get("rl")),
    )
    _validate_config(config)
    return config
