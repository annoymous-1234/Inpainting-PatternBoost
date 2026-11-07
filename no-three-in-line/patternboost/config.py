from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_config import load_project_config


@dataclass
class PoolConfig:
    """Configure PatternBoost TopPool output and initialization sources.

    ``init_sources`` may read supervised caches, greedy seeds, or serialized
    TopPool files listed in ``top_pool_paths``. The dataclass stores
    configuration only and does not mutate external state.
    """

    max_size: int = 5000
    save_path: str | None = None
    init_sources: list[str] = field(default_factory=lambda: ["supervised_cache", "greedy"])
    supervised_cache_paths: list[str] = field(default_factory=lambda: ["data/n15/train_solutions.pt"])
    top_pool_paths: list[str] = field(default_factory=list)
    greedy_num_seeds: int = 200


@dataclass
class PoolSamplingConfig:
    """Configure TopPool seed sampling for PatternBoost training data.

    ``mode`` selects uniform or score-bucket-rank mixture sampling. ``alpha``,
    ``offset``, and ``top_fraction`` control the high-score biased component,
    while ``uniform_mix`` keeps part of the draws uniform over the full pool.
    The dataclass stores configuration only and does not mutate external state.
    """

    mode: str = "uniform"
    alpha: float = 1.0
    offset: float = 1.0
    top_fraction: float = 1.0
    uniform_mix: float = 0.0


@dataclass
class MaskRatioScheduleEntry:
    """Configure active mask ratios starting at one PatternBoost generation.

    ``start`` is a 1-based generation number and ``mask_ratios`` lists ratios
    in ``[0, 1]`` used for both training noise and generation remask from that
    generation onward until another entry starts. The dataclass stores config
    only and does not mutate external state.
    """

    start: int
    mask_ratios: list[float]


@dataclass
class RemaskWeightScheduleEntry:
    """Configure generation remask weights starting at one generation.

    ``start`` is a 1-based generation number and ``weights`` is aligned with
    ``generation.remask_modes``. The dataclass stores config only and assumes
    validation checks length and nonnegative positive-sum weights.
    """

    start: int
    weights: list[float]


@dataclass
class ScheduleConfig:
    """Configure optional generation-indexed PatternBoost schedules.

    ``mask_ratios_by_generation`` controls both training noise and generation
    remask ratios, while ``remask_mode_weights_by_generation`` controls only
    generation remask mode sampling. The dataclass stores configuration only.
    """

    mask_ratios_by_generation: list[MaskRatioScheduleEntry] | None = None
    remask_mode_weights_by_generation: list[RemaskWeightScheduleEntry] | None = None


@dataclass
class TrainingConfig:
    """Configure PatternBoost denoising fine-tuning."""

    samples_per_generation: int = 2048
    epochs_per_generation: int = 1
    checkpoint_every_generations: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 100.0
    fine_tune_from_previous: bool = True
    init_checkpoint_path: str | None = None
    init_optimizer_from_checkpoint: bool = True
    pool_sampling: PoolSamplingConfig = field(default_factory=PoolSamplingConfig)


@dataclass
class GenerationConfig:
    """Configure PatternBoost candidate generation from TopPool seeds.

    ``candidates_per_generation`` counts generated candidates per outer-loop
    generation, ``candidate_batch_size`` controls how many candidates share one
    batched model-forward loop, and ``remask_modes`` lists allowed generation
    perturbations. The dataclass stores configuration only and does not mutate
    external state.
    """

    num_generations: int = 20
    candidates_per_generation: int = 1000
    candidate_batch_size: int = 1
    remask_modes: list[str] = field(default_factory=lambda: ["random", "replaceability"])
    remask_mode_weights: list[float] | None = None
    completion_operator: str = "learned"


@dataclass
class DecodeConfig:
    """Configure iterative decoding and environment proposal search."""

    num_steps: int = 6
    quota_ratios: list[float] = field(default_factory=lambda: [0.25, 0.20, 0.15, 0.10, 0.15, 0.15])
    num_candidates_per_slot: int = 3
    y_temperature: float = 1.0
    env_num_search_attempts: int = 5
    env_allow_existing_deletions: bool = False


@dataclass
class SelectorConfig:
    """Configure slot-priority scoring and stochastic proposal sampling."""

    confidence_metric: str = "margin"
    legal_choice_metric: str = "mrv"
    confidence_weight: float = 1.0
    legal_weight: float = 0.15
    noise_std: float = 0.01


@dataclass
class SaturationConfig:
    """Configure optional greedy saturation after model decoding."""

    enabled: bool = True
    max_steps: int | None = None
    slot_order: str = "mrv"
    y_order: str = "random"


@dataclass
class RuntimeConfig:
    """Configure PatternBoost random seed, device, workers, output path, and logs."""

    seed: int = 0
    device: str = "cuda"
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints/patternboost"
    run_name: str | None = None
    log_jsonl: bool = True
    log_candidates_jsonl: bool = True
    top_pool_snapshot_every_generations: int = 0


@dataclass
class TopPoolAugmentationConfig:
    """Configure optional symmetry augmentation of newly inserted TopPool entries.

    ``enabled`` controls whether generation-time augmentation runs and
    ``max_source_entries`` optionally caps the number of new global-best
    entries augmented per generation. The dataclass stores configuration only
    and does not mutate external state.
    """

    enabled: bool = False
    max_source_entries: int | None = None


@dataclass
class PatternBoostConfig:
    """Bundle PatternBoost configuration loaded from the project YAML."""

    n: int = 15
    mask_ratios: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    pool: PoolConfig = field(default_factory=PoolConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    schedules: ScheduleConfig = field(default_factory=ScheduleConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    saturation: SaturationConfig = field(default_factory=SaturationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    top_pool_augmentation: TopPoolAugmentationConfig = field(default_factory=TopPoolAugmentationConfig)
    config_path: str | None = None


def _mapping_section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested PatternBoost config mapping or an empty mapping."""

    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"patternboost.{key} must be a mapping")
    return value


def _load_dataclass(dataclass_type: type, mapping: dict[str, Any]) -> Any:
    """Instantiate one config dataclass from a plain mapping."""

    return dataclass_type(**mapping)


def _load_training_config(patternboost: dict[str, Any]) -> TrainingConfig:
    """Instantiate training config while parsing nested pool-sampling fields."""

    training_mapping = dict(_mapping_section(patternboost, "training"))
    pool_sampling = _load_dataclass(
        PoolSamplingConfig,
        _mapping_section(training_mapping, "pool_sampling"),
    )
    training_mapping.pop("pool_sampling", None)
    return TrainingConfig(**training_mapping, pool_sampling=pool_sampling)


def _load_schedule_config(patternboost: dict[str, Any]) -> ScheduleConfig:
    """Instantiate optional generation-indexed schedule config.

    ``patternboost`` is the top-level PatternBoost mapping. The function
    returns a ``ScheduleConfig``, does not mutate the source mapping, and
    accepts absent schedule sections as no-op schedules.
    """

    schedule_mapping = _mapping_section(patternboost, "schedules")
    mask_entries = schedule_mapping.get("mask_ratios_by_generation")
    weight_entries = schedule_mapping.get("remask_mode_weights_by_generation")
    return ScheduleConfig(
        mask_ratios_by_generation=None
        if mask_entries is None
        else [MaskRatioScheduleEntry(start=int(entry["start"]), mask_ratios=[float(value) for value in entry["mask_ratios"]]) for entry in mask_entries],
        remask_mode_weights_by_generation=None
        if weight_entries is None
        else [RemaskWeightScheduleEntry(start=int(entry["start"]), weights=[float(value) for value in entry["weights"]]) for entry in weight_entries],
    )


def load_patternboost_config(config_path: str | Path | None = None) -> PatternBoostConfig:
    """Load the PatternBoost section from the unified project config.

    ``config_path`` optionally points to a YAML config file. The function
    returns a validated ``PatternBoostConfig``, does not mutate the loaded
    mapping, and assumes absent PatternBoost sections should use v1 defaults.
    """

    loaded = load_project_config(config_path)
    patternboost = loaded.get("patternboost", {})
    if patternboost is None:
        patternboost = {}
    if not isinstance(patternboost, dict):
        raise ValueError("patternboost config must be a mapping")

    config = PatternBoostConfig(
        n=int(loaded.get("n", PatternBoostConfig.n)),
        mask_ratios=[float(ratio) for ratio in patternboost.get("mask_ratios", [0.25, 0.5, 0.75])],
        pool=_load_dataclass(PoolConfig, _mapping_section(patternboost, "pool")),
        training=_load_training_config(patternboost),
        generation=_load_dataclass(GenerationConfig, _mapping_section(patternboost, "generation")),
        schedules=_load_schedule_config(patternboost),
        decode=_load_dataclass(DecodeConfig, _mapping_section(patternboost, "decode")),
        selector=_load_dataclass(SelectorConfig, _mapping_section(patternboost, "selector")),
        saturation=_load_dataclass(SaturationConfig, _mapping_section(patternboost, "saturation")),
        runtime=_load_dataclass(RuntimeConfig, _mapping_section(patternboost, "runtime")),
        top_pool_augmentation=_load_dataclass(
            TopPoolAugmentationConfig,
            _mapping_section(patternboost, "top_pool_augmentation"),
        ),
        config_path=str(config_path) if config_path is not None else None,
    )
    validate_patternboost_config(config)
    return config


def validate_patternboost_config(config: PatternBoostConfig) -> None:
    """Validate high-impact PatternBoost configuration constraints."""

    if config.n < 1:
        raise ValueError("n must be >= 1")
    if not config.mask_ratios:
        raise ValueError("patternboost.mask_ratios must not be empty")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in config.mask_ratios):
        raise ValueError("patternboost.mask_ratios must be in [0, 1]")
    if config.pool.max_size < 1:
        raise ValueError("patternboost.pool.max_size must be >= 1")
    if config.pool.save_path is not None and not isinstance(config.pool.save_path, str):
        raise ValueError("patternboost.pool.save_path must be null or a string")
    if config.pool.greedy_num_seeds < 0:
        raise ValueError("patternboost.pool.greedy_num_seeds must be >= 0")
    if any(source not in {"supervised_cache", "greedy", "top_pool"} for source in config.pool.init_sources):
        raise ValueError("patternboost.pool.init_sources supports supervised_cache, greedy, and top_pool")
    if config.training.samples_per_generation < 1:
        raise ValueError("patternboost.training.samples_per_generation must be >= 1")
    if config.training.epochs_per_generation < 1:
        raise ValueError("patternboost.training.epochs_per_generation must be >= 1")
    if config.training.checkpoint_every_generations < 1:
        raise ValueError("patternboost.training.checkpoint_every_generations must be >= 1")
    if config.training.batch_size < 1:
        raise ValueError("patternboost.training.batch_size must be >= 1")
    if config.training.lr <= 0.0 or config.training.weight_decay < 0.0:
        raise ValueError("patternboost.training lr must be > 0 and weight_decay must be >= 0")
    if config.training.grad_clip_norm is not None and config.training.grad_clip_norm <= 0.0:
        raise ValueError("patternboost.training.grad_clip_norm must be > 0 when set")
    if config.training.init_checkpoint_path is not None and not isinstance(config.training.init_checkpoint_path, str):
        raise ValueError("patternboost.training.init_checkpoint_path must be null or a string")
    if not isinstance(config.training.init_optimizer_from_checkpoint, bool):
        raise ValueError("patternboost.training.init_optimizer_from_checkpoint must be a boolean")
    sampling = config.training.pool_sampling
    if sampling.mode not in {"uniform", "score_bucket_rank_mixture"}:
        raise ValueError("patternboost.training.pool_sampling.mode is invalid")
    if sampling.alpha < 0.0:
        raise ValueError("patternboost.training.pool_sampling.alpha must be >= 0")
    if sampling.offset <= 0.0:
        raise ValueError("patternboost.training.pool_sampling.offset must be > 0")
    if not 0.0 < sampling.top_fraction <= 1.0:
        raise ValueError("patternboost.training.pool_sampling.top_fraction must be in (0, 1]")
    if not 0.0 <= sampling.uniform_mix <= 1.0:
        raise ValueError("patternboost.training.pool_sampling.uniform_mix must be in [0, 1]")
    if config.generation.num_generations < 1 or config.generation.candidates_per_generation < 1:
        raise ValueError("patternboost.generation counts must be >= 1")
    if config.generation.candidate_batch_size < 1:
        raise ValueError("patternboost.generation.candidate_batch_size must be >= 1")
    if any(mode not in {"random", "replaceability"} for mode in config.generation.remask_modes):
        raise ValueError("patternboost.generation.remask_modes supports random and replaceability")
    if config.generation.completion_operator not in {"learned", "random"}:
        raise ValueError("patternboost.generation.completion_operator supports learned and random")
    if config.generation.completion_operator == "random" and not config.saturation.enabled:
        # Saturation *is* the random completion operator, so disabling it would
        # leave remasked slots unfilled instead of randomly completed.
        raise ValueError(
            "patternboost.generation.completion_operator='random' requires patternboost.saturation.enabled"
        )
    if config.generation.remask_mode_weights is not None:
        _validate_remask_weights(
            config.generation.remask_mode_weights,
            len(config.generation.remask_modes),
            "patternboost.generation.remask_mode_weights",
        )
    _validate_schedule_config(config)
    if config.decode.num_steps < 1:
        raise ValueError("patternboost.decode.num_steps must be >= 1")
    if len(config.decode.quota_ratios) != config.decode.num_steps:
        raise ValueError("patternboost.decode.quota_ratios length must equal decode.num_steps")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in config.decode.quota_ratios):
        raise ValueError("patternboost.decode.quota_ratios must be in [0, 1]")
    if config.decode.num_candidates_per_slot < 1:
        raise ValueError("patternboost.decode.num_candidates_per_slot must be >= 1")
    if config.decode.y_temperature <= 0.0:
        raise ValueError("patternboost.decode.y_temperature must be > 0")
    if config.decode.env_num_search_attempts < 1:
        raise ValueError("patternboost.decode.env_num_search_attempts must be >= 1")
    if config.selector.confidence_metric not in {"margin", "top_prob", "negative_entropy"}:
        raise ValueError("patternboost.selector.confidence_metric is invalid")
    if config.selector.legal_choice_metric not in {"mrv", "easy_first"}:
        raise ValueError("patternboost.selector.legal_choice_metric is invalid")
    if config.selector.confidence_weight < 0.0 or config.selector.legal_weight < 0.0:
        raise ValueError("patternboost.selector weights must be >= 0")
    if config.selector.noise_std < 0.0:
        raise ValueError("patternboost.selector.noise_std must be >= 0")
    if config.saturation.slot_order not in {"mrv", "random"}:
        raise ValueError("patternboost.saturation.slot_order supports mrv and random")
    if config.saturation.y_order not in {"random", "ascending"}:
        raise ValueError("patternboost.saturation.y_order supports random and ascending")
    if config.saturation.max_steps is not None and config.saturation.max_steps < 0:
        raise ValueError("patternboost.saturation.max_steps must be >= 0 when set")
    if config.runtime.num_workers < 0:
        raise ValueError("patternboost.runtime.num_workers must be >= 0")
    if not isinstance(config.runtime.log_jsonl, bool):
        raise ValueError("patternboost.runtime.log_jsonl must be a boolean")
    if not isinstance(config.runtime.log_candidates_jsonl, bool):
        raise ValueError("patternboost.runtime.log_candidates_jsonl must be a boolean")
    if config.runtime.top_pool_snapshot_every_generations < 0:
        raise ValueError("patternboost.runtime.top_pool_snapshot_every_generations must be >= 0")
    augmentation = config.top_pool_augmentation
    if not isinstance(augmentation.enabled, bool):
        raise ValueError("patternboost.top_pool_augmentation.enabled must be a boolean")
    if augmentation.max_source_entries is not None and augmentation.max_source_entries < 1:
        raise ValueError("patternboost.top_pool_augmentation.max_source_entries must be null or >= 1")


def _validate_remask_weights(weights: list[float], expected_length: int, field_name: str) -> None:
    """Validate one remask-weight list against configured remask modes."""

    if len(weights) != expected_length:
        raise ValueError(f"{field_name} length must match remask_modes")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError(f"{field_name} must be nonnegative with positive sum")


def _validate_schedule_config(config: PatternBoostConfig) -> None:
    """Validate optional generation-indexed PatternBoost schedules."""

    schedules = config.schedules
    if schedules.mask_ratios_by_generation is not None:
        previous_start = 0
        for entry in schedules.mask_ratios_by_generation:
            if entry.start < 1 or entry.start <= previous_start:
                raise ValueError("patternboost.schedules.mask_ratios_by_generation starts must be strictly increasing and >= 1")
            if not entry.mask_ratios:
                raise ValueError("patternboost.schedules.mask_ratios_by_generation mask_ratios must not be empty")
            if any(ratio < 0.0 or ratio > 1.0 for ratio in entry.mask_ratios):
                raise ValueError("patternboost.schedules.mask_ratios_by_generation mask_ratios must be in [0, 1]")
            previous_start = entry.start
    if schedules.remask_mode_weights_by_generation is not None:
        previous_start = 0
        for entry in schedules.remask_mode_weights_by_generation:
            if entry.start < 1 or entry.start <= previous_start:
                raise ValueError("patternboost.schedules.remask_mode_weights_by_generation starts must be strictly increasing and >= 1")
            _validate_remask_weights(
                entry.weights,
                len(config.generation.remask_modes),
                "patternboost.schedules.remask_mode_weights_by_generation weights",
            )
            previous_start = entry.start
