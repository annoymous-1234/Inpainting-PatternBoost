from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_config import load_project_config


@dataclass
class SearchConfig:
    """Configure root and non-root sampled-action search hyperparameters."""

    J_root: int = 1
    J_nonroot: int = 1
    J_root_prime: int = 2
    L_root: int = 2
    K_root: int = 4
    L_nonroot: int = 2
    K_nonroot: int = 2
    root_budget: int = 4
    max_search_plies: int = 6
    tau_slot_root: float = 1.0
    tau_val_root: float = 1.0
    tau_slot_nonroot: float = 1.0
    tau_val_nonroot: float = 1.0
    tau_pi_root: float = 1.0
    tau_pi_nonroot: float = 1.0
    tau_target: float = 1.0
    c_Q: float = 2.0
    gamma_step: float = 0.95
    eps: float = 1e-8
    terminate_on_no_progress: bool = True
    use_existing_action_mask: bool = True
    slot_score_coef_mask_logit: float = 1.0
    slot_score_coef_scarcity: float = 0.0
    slot_score_coef_entropy: float = 0.0


@dataclass
class RewardConfig:
    """Configure the relative weight of step rewards and terminal utility."""

    step_weight: float = 0.0
    terminal_weight: float = 1.0
    terminal_reward_mode: str = "capped_exp2_advantage"
    terminal_reward_beta: float = 1.0
    terminal_reward_cap: float = 6.0
    initial_terminal_baseline_points: float | None = None
    initial_terminal_baseline_fraction: float = 0.75


@dataclass
class SelfPlayConfig:
    """Configure AlphaZero self-play rollout generation."""

    num_iterations: int = 1
    episodes_per_iteration: int = 1
    max_episode_steps: int = 6
    root_temperature: float = 1.0
    sample_prefix_steps: int = 4


@dataclass
class TrainingConfig:
    """Configure learner-side optimization and replay usage."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 10.0
    batch_size: int = 2
    replay_capacity: int = 128
    updates_per_iteration: int = 1
    lambda_fill: float = 1.0
    lambda_val: float = 1.0
    value_loss_weight: float = 1.0
    save_every: int = 1


@dataclass
class RuntimeConfig:
    """Configure device, seeds, and checkpoint output."""

    seed: int = 0
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints/alphazero"
    run_name: str | None = None
    resume_from: str | None = None
    init_from_model: str | None = None
    keep_iteration_checkpoints: bool = True
    log_jsonl: bool = True


@dataclass
class LocalAnnealConfig:
    """Configure rollout-side local anneal / remask behavior."""

    enabled: bool = False
    alpha_env: float = 1.0
    alpha_mcts: float = 1.0
    trigger_steps: list[int] = field(default_factory=list)
    num_slots_to_remask: int = 1
    min_active_age_for_remask: int = 2
    sampling_temperature: float = 1.0


@dataclass
class AlphaZeroConfig:
    """Bundle the AlphaZero section plus project-level fields it depends on."""

    n: int = 8
    search: SearchConfig = field(default_factory=SearchConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    local_anneal: LocalAnnealConfig = field(default_factory=LocalAnnealConfig)
    config_path: str | None = None


def _mapping_section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one nested mapping section or an empty mapping when absent."""

    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"alphazero.{key} must be a mapping")
    return value


def _load_dataclass(dataclass_type: type, mapping: dict[str, Any]) -> Any:
    """Instantiate one config dataclass from a plain mapping."""

    return dataclass_type(**mapping)


def load_alphazero_config(config_path: str | Path | None = None) -> AlphaZeroConfig:
    """Load and validate the AlphaZero project configuration.

    ``config_path`` optionally overrides the repository ``config.yaml`` path.
    The function returns an ``AlphaZeroConfig``, does not mutate external
    state, and assumes the top-level config is a YAML mapping.
    """

    loaded = load_project_config(config_path)
    alphazero = loaded.get("alphazero", {})
    if alphazero is None:
        alphazero = {}
    if not isinstance(alphazero, dict):
        raise ValueError("alphazero config must be a mapping")

    config = AlphaZeroConfig(
        n=int(loaded.get("n", AlphaZeroConfig.n)),
        search=_load_dataclass(SearchConfig, _mapping_section(alphazero, "search")),
        reward=_load_dataclass(RewardConfig, _mapping_section(alphazero, "reward")),
        self_play=_load_dataclass(SelfPlayConfig, _mapping_section(alphazero, "self_play")),
        training=_load_dataclass(TrainingConfig, _mapping_section(alphazero, "training")),
        runtime=_load_dataclass(RuntimeConfig, _mapping_section(alphazero, "runtime")),
        local_anneal=_load_dataclass(LocalAnnealConfig, _mapping_section(alphazero, "local_anneal")),
        config_path=str(config_path) if config_path is not None else None,
    )
    validate_alphazero_config(config)
    return config


def validate_alphazero_config(config: AlphaZeroConfig) -> None:
    """Validate high-impact AlphaZero configuration invariants."""

    if config.n < 1:
        raise ValueError("n must be >= 1")
    if config.search.J_root < 1:
        raise ValueError("search.J_root must be >= 1")
    if config.search.J_nonroot < 1:
        raise ValueError("search.J_nonroot must be >= 1")
    if config.search.J_root_prime <= config.search.J_root:
        raise ValueError("search.J_root_prime must be > search.J_root")
    if config.search.L_root < 1 or config.search.L_nonroot < 1:
        raise ValueError("value pool sizes must be >= 1")
    if config.search.K_root < 1 or config.search.K_nonroot < 1:
        raise ValueError("sampled action counts must be >= 1")
    if config.search.root_budget < 1:
        raise ValueError("search.root_budget must be >= 1")
    if config.search.max_search_plies < 1:
        raise ValueError("search.max_search_plies must be >= 1")
    if not 0.0 <= config.search.gamma_step <= 1.0:
        raise ValueError("search.gamma_step must be in [0, 1]")
    if config.search.tau_slot_root <= 0 or config.search.tau_val_root <= 0:
        raise ValueError("root temperatures must be > 0")
    if config.search.tau_slot_nonroot <= 0 or config.search.tau_val_nonroot <= 0:
        raise ValueError("non-root temperatures must be > 0")
    if config.search.tau_pi_root <= 0 or config.search.tau_pi_nonroot <= 0 or config.search.tau_target <= 0:
        raise ValueError("policy temperatures must be > 0")
    if config.search.c_Q <= 0:
        raise ValueError("search.c_Q must be > 0")
    if config.search.eps <= 0:
        raise ValueError("search.eps must be > 0")
    if config.search.slot_score_coef_mask_logit < 0:
        raise ValueError("search.slot_score_coef_mask_logit must be >= 0")
    if config.search.slot_score_coef_scarcity < 0:
        raise ValueError("search.slot_score_coef_scarcity must be >= 0")
    if config.search.slot_score_coef_entropy < 0:
        raise ValueError("search.slot_score_coef_entropy must be >= 0")
    if config.reward.step_weight < 0 or config.reward.terminal_weight < 0:
        raise ValueError("reward weights must be >= 0")
    if config.reward.step_weight == 0 and config.reward.terminal_weight == 0:
        raise ValueError("at least one reward weight must be positive")
    if config.reward.terminal_reward_mode not in {"normalized_points", "capped_exp2_advantage"}:
        raise ValueError("reward.terminal_reward_mode must be 'normalized_points' or 'capped_exp2_advantage'")
    if config.reward.terminal_reward_beta <= 0:
        raise ValueError("reward.terminal_reward_beta must be > 0")
    if config.reward.terminal_reward_cap < 0:
        raise ValueError("reward.terminal_reward_cap must be >= 0")
    if config.reward.initial_terminal_baseline_points is not None:
        if not 0.0 <= config.reward.initial_terminal_baseline_points <= 2 * config.n:
            raise ValueError("reward.initial_terminal_baseline_points must be in [0, 2n]")
    if not 0.0 <= config.reward.initial_terminal_baseline_fraction <= 1.0:
        raise ValueError("reward.initial_terminal_baseline_fraction must be in [0, 1]")
    if config.self_play.num_iterations < 1 or config.self_play.episodes_per_iteration < 1:
        raise ValueError("self_play iteration and episode counts must be >= 1")
    if config.self_play.max_episode_steps < 1:
        raise ValueError("self_play.max_episode_steps must be >= 1")
    if config.self_play.root_temperature <= 0:
        raise ValueError("self_play.root_temperature must be > 0")
    if config.self_play.sample_prefix_steps < 0:
        raise ValueError("self_play.sample_prefix_steps must be >= 0")
    if config.training.lr <= 0:
        raise ValueError("training.lr must be > 0")
    if config.training.weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0")
    if config.training.grad_clip_norm is not None and config.training.grad_clip_norm <= 0:
        raise ValueError("training.grad_clip_norm must be > 0 when set")
    if config.training.batch_size < 1 or config.training.replay_capacity < 1:
        raise ValueError("training.batch_size and replay_capacity must be >= 1")
    if config.training.updates_per_iteration < 1:
        raise ValueError("training.updates_per_iteration must be >= 1")
    if config.training.lambda_fill < 0 or config.training.lambda_val < 0 or config.training.value_loss_weight < 0:
        raise ValueError("loss weights must be >= 0")
    if config.training.save_every < 1:
        raise ValueError("training.save_every must be >= 1")
    if config.runtime.resume_from is not None and config.runtime.init_from_model is not None:
        raise ValueError("runtime.resume_from and runtime.init_from_model cannot both be set")
    if config.local_anneal.alpha_env < 0 or config.local_anneal.alpha_mcts < 0:
        raise ValueError("local_anneal alpha weights must be >= 0")
    if config.local_anneal.num_slots_to_remask < 1:
        raise ValueError("local_anneal.num_slots_to_remask must be >= 1")
    if config.local_anneal.min_active_age_for_remask < 1:
        raise ValueError("local_anneal.min_active_age_for_remask must be >= 1")
    if config.local_anneal.sampling_temperature <= 0:
        raise ValueError("local_anneal.sampling_temperature must be > 0")
    if any(step < 1 for step in config.local_anneal.trigger_steps):
        raise ValueError("local_anneal.trigger_steps must contain only positive integers")


def reward_mixture_weights(config: AlphaZeroConfig) -> tuple[float, float]:
    """Return normalized step and terminal reward weights.

    The returned tuple is ``(alpha_step, alpha_term)`` with both values in
    ``[0, 1]`` and summing to ``1``. The function does not mutate ``config``.
    """

    total = config.reward.step_weight + config.reward.terminal_weight
    return config.reward.step_weight / total, config.reward.terminal_weight / total
