from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_config import load_project_config


@dataclass
class ValidationConfig:
    """Configure optional dataset validation before training.

    The fields control whether validation runs, whether env-backed checks are
    included, and how invalid records are handled during preprocessing.
    """

    enabled: bool = True
    validate_with_env: bool = True
    fail_on_invalid: bool = True
    drop_invalid: bool = False
    round_trip_check: bool = True


@dataclass
class EncodingConfig:
    """Configure canonical slot encoding and fixed slot-order generation."""

    num_slot_orders: int = 4
    slot_order_seed: int = 0


@dataclass
class CorruptionConfig:
    """Configure noisy-input construction for supervised denoising."""

    total_steps: int = 8
    q_min: float = 0.0
    q_max: float = 1.0
    sample_step_min: int = 0
    corruption_mode: str = "exact_ratio"


@dataclass
class PartialTargetConfig:
    """Configure partial-denoising target construction."""

    restore_schedule_type: str = "fixed"
    restore_k_fixed: int = 1
    restore_k_min: int = 1
    restore_k_max: int = 4
    use_multiple_orders: bool = True


@dataclass
class DatasetConfig:
    """Configure task sampling for the dynamic supervised dataset."""

    mode: str = "full"
    partial_prob: float = 0.5


@dataclass
class LossConfig:
    """Configure weighting and supervision scope in the supervised loss."""

    filled_weight: float = 1.0
    mask_weight: float = 1.0
    full_loss_weight: float = 1.0
    partial_loss_weight: float = 1.0
    full_supervision_mode: str = "masked_filled_only"
    partial_supervision_mode: str = "all_missing"


@dataclass
class TrainerConfig:
    """Configure optimizer and training-loop cadence."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 1
    grad_clip_norm: float | None = 1.0
    eval_every: int = 1
    rollout_eval_every: int = 0
    save_every: int = 1


@dataclass
class RolloutEvalConfig:
    """Configure evaluation-time denoising rollouts and decoding ablations."""

    enabled: bool = False
    eval_types: list[str] = field(default_factory=lambda: ["standard"])
    num_episodes: int = 0
    seed: int = 0
    record_step_metrics: bool = False
    max_denoise_steps: int = 0
    k_schedule_mode: str = "fixed"
    fixed_k: int = 4
    k_min: int = 1
    k_max: int = 4
    partial_start_step_ids: list[int] = field(default_factory=lambda: [0])
    partial_corruption_mode: str | None = None
    env_num_search_attempts: int = 1
    env_allow_existing_deletions: bool = False
    decode_ablation_configs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DataPathConfig:
    """Configure raw-data and cache paths for supervised training."""

    train_jsonl: str = "data/train_solutions.jsonl"
    train_cache: str = "data/train_solutions.pt"
    val_jsonl: str | None = None
    val_cache: str | None = None


@dataclass
class RuntimeConfig:
    """Configure runtime details such as batching, device, and checkpoints."""

    batch_size: int = 8
    num_workers: int = 0
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"


@dataclass
class CurriculumConfig:
    """Configure the two-stage supervised training curriculum."""

    stage1_epochs: int = 1
    stage2_epochs: int = 1
    stage2_mode: str = "mixed"
    stage2_partial_prob: float = 0.5


@dataclass
class SupervisedConfig:
    """Bundle the full supervised-training configuration."""

    n: int = 8
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    partial_target: PartialTargetConfig = field(default_factory=PartialTargetConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    rollout_eval: RolloutEvalConfig = field(default_factory=RolloutEvalConfig)
    data_paths: DataPathConfig = field(default_factory=DataPathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    config_path: str | None = None


def _mapping_section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mapping section or an empty mapping when absent."""

    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"supervised.{key} must be a mapping")
    return value


def _load_dataclass(dataclass_type: type, mapping: dict[str, Any]) -> Any:
    """Instantiate a config dataclass from a plain mapping."""

    return dataclass_type(**mapping)


def load_supervised_config(config_path: str | Path | None = None) -> SupervisedConfig:
    """Load and validate the supervised section of the project config."""

    loaded = load_project_config(config_path)
    supervised = loaded.get("supervised", {})
    if supervised is None:
        supervised = {}
    if not isinstance(supervised, dict):
        raise ValueError("supervised config must be a mapping")

    config = SupervisedConfig(
        n=int(loaded.get("n", SupervisedConfig.n)),
        validation=_load_dataclass(ValidationConfig, _mapping_section(supervised, "validation")),
        encoding=_load_dataclass(EncodingConfig, _mapping_section(supervised, "encoding")),
        corruption=_load_dataclass(CorruptionConfig, _mapping_section(supervised, "corruption")),
        partial_target=_load_dataclass(PartialTargetConfig, _mapping_section(supervised, "partial_target")),
        dataset=_load_dataclass(DatasetConfig, _mapping_section(supervised, "dataset")),
        loss=_load_dataclass(LossConfig, _mapping_section(supervised, "loss")),
        trainer=_load_dataclass(TrainerConfig, _mapping_section(supervised, "trainer")),
        rollout_eval=_load_dataclass(RolloutEvalConfig, _mapping_section(supervised, "rollout_eval")),
        data_paths=_load_dataclass(DataPathConfig, _mapping_section(supervised, "data_paths")),
        runtime=_load_dataclass(RuntimeConfig, _mapping_section(supervised, "runtime")),
        curriculum=_load_dataclass(CurriculumConfig, _mapping_section(supervised, "curriculum")),
        config_path=str(config_path) if config_path is not None else None,
    )
    validate_supervised_config(config)
    return config


def validate_supervised_config(config: SupervisedConfig) -> None:
    """Validate high-impact constraints for the supervised config."""

    if config.n < 1:
        raise ValueError("n must be >= 1")
    if config.encoding.num_slot_orders < 1:
        raise ValueError("encoding.num_slot_orders must be >= 1")
    if config.corruption.total_steps < 1:
        raise ValueError("corruption.total_steps must be >= 1")
    if not 0.0 <= config.corruption.q_min <= config.corruption.q_max <= 1.0:
        raise ValueError("corruption.q_min and q_max must satisfy 0 <= q_min <= q_max <= 1")
    if not 0 <= config.corruption.sample_step_min < config.corruption.total_steps:
        raise ValueError("corruption.sample_step_min must satisfy 0 <= sample_step_min < total_steps")
    if config.corruption.corruption_mode not in {"exact_ratio", "bernoulli"}:
        raise ValueError("corruption.corruption_mode must be 'exact_ratio' or 'bernoulli'")
    if config.dataset.mode not in {"full", "partial", "mixed"}:
        raise ValueError("dataset.mode must be 'full', 'partial', or 'mixed'")
    if not 0.0 <= config.dataset.partial_prob <= 1.0:
        raise ValueError("dataset.partial_prob must be in [0, 1]")
    if config.loss.full_supervision_mode not in {"masked_filled_only", "all_tokens"}:
        raise ValueError("loss.full_supervision_mode must be 'masked_filled_only' or 'all_tokens'")
    if config.loss.partial_supervision_mode not in {"all_missing", "restored_only", "all_tokens"}:
        raise ValueError(
            "loss.partial_supervision_mode must be 'all_missing', 'restored_only', or 'all_tokens'"
        )
    if config.partial_target.restore_schedule_type == "fixed":
        if config.partial_target.restore_k_fixed < 0:
            raise ValueError("partial_target.restore_k_fixed must be >= 0")
    elif config.partial_target.restore_schedule_type == "front_loaded":
        if not 0 <= config.partial_target.restore_k_min <= config.partial_target.restore_k_max:
            raise ValueError("partial_target.restore_k_min/max must satisfy 0 <= min <= max")
    else:
        raise ValueError("partial_target.restore_schedule_type must be 'fixed' or 'front_loaded'")
    if config.curriculum.stage2_mode not in {"full", "partial", "mixed"}:
        raise ValueError("curriculum.stage2_mode must be 'full', 'partial', or 'mixed'")
    if not 0.0 <= config.curriculum.stage2_partial_prob <= 1.0:
        raise ValueError("curriculum.stage2_partial_prob must be in [0, 1]")
    if config.validation.enabled and not (config.validation.fail_on_invalid or config.validation.drop_invalid):
        raise ValueError("validation must either fail_on_invalid or drop_invalid when enabled")
    if config.runtime.batch_size < 1:
        raise ValueError("runtime.batch_size must be >= 1")
    if config.runtime.num_workers < 0:
        raise ValueError("runtime.num_workers must be >= 0")
    if config.trainer.eval_every < 1:
        raise ValueError("trainer.eval_every must be >= 1")
    if config.trainer.save_every < 1:
        raise ValueError("trainer.save_every must be >= 1")
    allowed_eval_types = {"standard", "partial_state", "decode_ablation"}
    if any(eval_type not in allowed_eval_types for eval_type in config.rollout_eval.eval_types):
        raise ValueError("rollout_eval.eval_types must contain only standard, partial_state, or decode_ablation")
    if config.rollout_eval.num_episodes < 0:
        raise ValueError("rollout_eval.num_episodes must be >= 0")
    if config.rollout_eval.max_denoise_steps < 0:
        raise ValueError("rollout_eval.max_denoise_steps must be >= 0")
    if config.rollout_eval.k_schedule_mode == "fixed":
        if config.rollout_eval.fixed_k < 0:
            raise ValueError("rollout_eval.fixed_k must be >= 0")
    elif config.rollout_eval.k_schedule_mode == "front_loaded":
        if not 0 <= config.rollout_eval.k_min <= config.rollout_eval.k_max:
            raise ValueError("rollout_eval.k_min/max must satisfy 0 <= k_min <= k_max")
    else:
        raise ValueError("rollout_eval.k_schedule_mode must be 'fixed' or 'front_loaded'")
    if config.rollout_eval.partial_corruption_mode not in {None, "exact_ratio", "bernoulli"}:
        raise ValueError("rollout_eval.partial_corruption_mode must be null, 'exact_ratio', or 'bernoulli'")
    if config.rollout_eval.env_num_search_attempts < 1:
        raise ValueError("rollout_eval.env_num_search_attempts must be >= 1")
    if any(step_id < 0 or step_id >= config.corruption.total_steps for step_id in config.rollout_eval.partial_start_step_ids):
        raise ValueError("rollout_eval.partial_start_step_ids must be valid corruption step ids")
    if "partial_state" in config.rollout_eval.eval_types and not config.rollout_eval.partial_start_step_ids:
        raise ValueError("rollout_eval.partial_start_step_ids must be non-empty for partial_state evaluation")
    if "decode_ablation" in config.rollout_eval.eval_types:
        if not config.rollout_eval.decode_ablation_configs:
            raise ValueError("rollout_eval.decode_ablation_configs must be non-empty for decode_ablation evaluation")
        if any(not isinstance(value, dict) for value in config.rollout_eval.decode_ablation_configs.values()):
            raise ValueError("rollout_eval.decode_ablation_configs values must be mappings")
