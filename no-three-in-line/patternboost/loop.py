from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from env import NoThreeInLineEnv
from grid_symmetry import generate_grid_symmetry_state_ids
from models import ColumnPairModel
from project_config import load_project_config
from saturation import greedy_saturate
from supervised.config import load_supervised_config
from supervised.dataset import collate_supervised_batch
from supervised.preprocess import load_preprocessed_cache
from supervised.trainer import SupervisedTrainer, build_optimizer

from .config import PatternBoostConfig
from .dataset import PatternBoostDenoisingDataset
from .generator import PatternBoostGenerator
from .logging_utils import append_jsonl
from .state import canonical_hash, canonicalize_state_ids, state_ids_to_env_reset_args
from .top_pool import TopPool
from .types import GenerationResult


class PatternBoostRunner:
    """Run the PatternBoost train-generate-insert outer loop."""

    def __init__(
        self,
        config: PatternBoostConfig,
        model: ColumnPairModel | None = None,
    ) -> None:
        """Initialize config, RNG, run outputs, trainer, and TopPool.

        ``config`` is the validated PatternBoost configuration and ``model`` can
        override the model built from the supervised/project config. The
        constructor mutates only the new runner and assumes all configuration
        comes from the unified project YAML.
        """

        self.config = config
        self.rng = np.random.default_rng(config.runtime.seed)
        self.device = self._resolve_device(config.runtime.device)
        torch.manual_seed(config.runtime.seed)
        self.run_dir = self._initialize_new_run_dir()
        self.run_id = self.run_dir.name
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        self.candidates_log_path = self.run_dir / "candidates.jsonl"
        self.config_snapshot_path = self.run_dir / "config_snapshot.yaml"
        self.top_pool_path = self._resolve_top_pool_path()
        self.latest_checkpoint_path = self.run_dir / "latest.pt"
        self._write_config_snapshot()
        # The completion-operator ablation replaces the learned decoder with the
        # environment's greedy saturation, so it builds no model and trains
        # nothing: the run is pure search and needs only a CPU.
        self.uses_learned_operator = config.generation.completion_operator == "learned"
        self.init_checkpoint_loaded = False
        self.init_optimizer_loaded = False
        if self.uses_learned_operator:
            self.supervised_config = self._build_supervised_config()
            self.trainer = SupervisedTrainer(
                config=self.supervised_config,
                model=model,
                device=self.device,
            )
            self._initialize_from_training_checkpoint()
            self._initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in self.trainer.model.state_dict().items()
            }
        else:
            self.supervised_config = None
            self.trainer = None
            self._initial_model_state = {}
        self.top_pool = TopPool(n=config.n, max_size=config.pool.max_size)
        self.generation_results: list[GenerationResult] = []

    def _initialize_new_run_dir(self) -> Path:
        """Create and return one fresh PatternBoost run directory.

        The directory lives below ``runtime.checkpoint_dir`` and is named by
        ``runtime.run_name`` or a UTC timestamp plus seed. The method mutates
        the filesystem, returns the new path, and assumes existing explicit run
        names should not be reused.
        """

        checkpoint_root = Path(self.config.runtime.checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if self.config.runtime.run_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            run_name = f"pb_{timestamp}_seed{self.config.runtime.seed}"
        else:
            run_name = self.config.runtime.run_name
        run_dir = checkpoint_root / run_name
        if run_dir.exists():
            raise FileExistsError(f"PatternBoost run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _resolve_top_pool_path(self) -> Path:
        """Return the TopPool output path for this run.

        The current default stores ``top_pool.pt`` inside ``run_dir``. A
        configured ``pool.save_path`` is honored for backwards-compatible
        experiments; this method only returns the resolved path and does not
        write files.
        """

        if self.config.pool.save_path is None:
            return self.run_dir / "top_pool.pt"
        return Path(self.config.pool.save_path)

    def _save_top_pool_snapshot(self, generation_index: int) -> None:
        """Save a per-generation copy of the pool when snapshots are enabled.

        ``top_pool.pt`` is rewritten in place every generation, so without
        snapshots only the final pool survives. Post-hoc analyses that need the
        pool a candidate was actually generated from -- the learning-vs-
        inheritance control in ``analysis/`` -- need these copies. ``interval``
        of 0 disables snapshots, preserving the previous behaviour.
        """

        interval = int(self.config.runtime.top_pool_snapshot_every_generations)
        if interval <= 0:
            return
        generation_number = generation_index + 1
        is_final = generation_number == int(self.config.generation.num_generations)
        if generation_number % interval != 0 and not is_final:
            return
        self.top_pool.save(self.run_dir / f"top_pool_generation_{generation_number:03d}.pt")

    def _write_config_snapshot(self) -> None:
        """Write the loaded project config into this run directory.

        The snapshot captures the YAML config as loaded before runtime path
        overrides. The method mutates only ``config_snapshot.yaml`` and assumes
        ``self.run_dir`` already exists.
        """

        snapshot_mapping = load_project_config(self.config.config_path)
        self.config_snapshot_path.write_text(
            yaml.safe_dump(snapshot_mapping, sort_keys=False),
            encoding="utf-8",
        )

    def run(self) -> dict[str, object]:
        """Run all configured PatternBoost generations and return a summary.

        The method initializes TopPool if empty, repeatedly fine-tunes the
        model, generates candidates, inserts them, saves the pool, writes JSONL
        logs, and prints per-generation score/decode summaries. It mutates the
        runner, model, optimizer, TopPool, and configured output files.
        """

        self.initialize_top_pool()
        initial_best = 0 if self.top_pool.best() is None else self.top_pool.best().score
        self.top_pool.save(self.top_pool_path)
        self._log_run_start(initial_best)
        for generation_index in range(self.config.generation.num_generations):
            active_mask_ratios = self._active_mask_ratios(generation_index)
            active_remask_weights = self._active_remask_weights(generation_index)
            best_before = 0 if self.top_pool.best() is None else self.top_pool.best().score
            pool_size_before = len(self.top_pool)
            pool_score_counts_before = Counter(entry.score for entry in self.top_pool.entries())
            if self.uses_learned_operator:
                training_metrics, checkpoint_path = self._train_generation(generation_index, active_mask_ratios)
            else:
                training_metrics, checkpoint_path = {}, None
            generation_results = self._generate_generation(
                generation_index,
                active_mask_ratios,
                active_remask_weights,
            )
            augmentation_metrics = self._augment_new_global_best_entries(generation_index, generation_results)
            self.top_pool.save(self.top_pool_path)
            self._save_top_pool_snapshot(generation_index)
            generation_summary = self._build_generation_summary(
                generation_index,
                best_before=best_before,
                pool_size_before=pool_size_before,
                pool_score_counts_before=pool_score_counts_before,
                generation_results=generation_results,
                training_metrics=training_metrics,
                checkpoint_path=checkpoint_path,
                active_mask_ratios=active_mask_ratios,
                active_remask_weights=active_remask_weights,
                augmentation_metrics=augmentation_metrics,
            )
            self._log_metrics_record(generation_summary)
            self._print_generation_summary(generation_summary)
        best = self.top_pool.best()
        run_complete_record = self._build_run_complete_record(initial_best)
        self._log_metrics_record(run_complete_record)
        return {
            "initial_best_score": initial_best,
            "best_score": 0 if best is None else best.score,
            "num_pool_entries": len(self.top_pool),
            "pool_save_path": str(self.top_pool_path),
            "num_generation_results": len(self.generation_results),
            "run_dir": str(self.run_dir),
            "metrics_log_path": str(self.metrics_log_path),
            "candidates_log_path": str(self.candidates_log_path),
            "config_snapshot_path": str(self.config_snapshot_path),
        }

    def initialize_top_pool(self) -> None:
        """Populate TopPool from configured supervised caches and greedy seeds."""

        if "top_pool" in self.config.pool.init_sources:
            for pool_path in self.config.pool.top_pool_paths:
                path = Path(pool_path)
                if not path.exists():
                    continue
                source_pool = TopPool.load(path)
                if source_pool.n != self.config.n:
                    continue
                for entry in source_pool.entries():
                    self.top_pool.insert(
                        entry.slot_state_ids,
                        metadata={"source": "top_pool", "path": str(path), "source_hash": entry.canonical_hash},
                    )

        if "supervised_cache" in self.config.pool.init_sources:
            for cache_path in self.config.pool.supervised_cache_paths:
                path = Path(cache_path)
                if not path.exists():
                    continue
                for record in load_preprocessed_cache(path):
                    state = np.asarray(record.clean_state_ids, dtype=np.int64)
                    if state.shape != (2 * self.config.n,):
                        continue
                    self.top_pool.insert(state, metadata={"source": "supervised_cache", "path": str(path)})

        if "greedy" in self.config.pool.init_sources:
            for seed_index in range(self.config.pool.greedy_num_seeds):
                env = NoThreeInLineEnv(n=self.config.n, num_slots=2 * self.config.n)
                env.reset()
                greedy_saturate(
                    env,
                    self.rng,
                    max_steps=self.config.saturation.max_steps,
                    slot_order=self.config.saturation.slot_order,
                    y_order=self.config.saturation.y_order,
                )
                state = canonicalize_state_ids(
                    torch.as_tensor(
                        [
                            0
                            if env.slot_states[slot_id].status != "FILLED"
                            else int(env.slot_states[slot_id].point[1]) + 1
                            for slot_id in range(env.num_slots)
                        ],
                        dtype=torch.long,
                    ).numpy(),
                    self.config.n,
                )
                self.top_pool.insert(state, metadata={"source": "greedy", "seed_index": seed_index})

        if len(self.top_pool) == 0:
            raise ValueError("PatternBoost TopPool initialization produced no entries")

    def _build_supervised_config(self):
        """Return a supervised config overridden with PatternBoost training fields."""

        supervised_config = load_supervised_config(self.config.config_path)
        supervised_config.n = self.config.n
        supervised_config.runtime = replace(
            supervised_config.runtime,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.runtime.num_workers,
            seed=self.config.runtime.seed,
            device=self.config.runtime.device,
            checkpoint_dir=str(self.run_dir),
        )
        supervised_config.trainer = replace(
            supervised_config.trainer,
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            grad_clip_norm=self.config.training.grad_clip_norm,
            save_every=1,
        )
        supervised_config.loss = replace(
            supervised_config.loss,
            full_supervision_mode="masked_filled_only",
        )
        return supervised_config

    def _initialize_from_training_checkpoint(self) -> None:
        """Load an optional PatternBoost model/optimizer checkpoint.

        The configured checkpoint must contain a strict-compatible
        ``model_state_dict``. Optimizer state is loaded only when requested and
        compatible with the current model parameter shapes. The method mutates
        model and possibly optimizer state, and assumes the trainer has already
        been constructed.
        """

        checkpoint_path = self.config.training.init_checkpoint_path
        if checkpoint_path is None:
            return
        payload = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        if "model_state_dict" not in payload:
            raise ValueError("PatternBoost init checkpoint must contain model_state_dict")
        self.trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.trainer.model.to(self.device)
        self.init_checkpoint_loaded = True
        if self.config.training.init_optimizer_from_checkpoint and "optimizer_state_dict" in payload:
            optimizer_state = payload["optimizer_state_dict"]
            if self._optimizer_state_is_compatible(optimizer_state):
                self.trainer.optimizer.load_state_dict(optimizer_state)
                self._move_optimizer_state_to_device()
                self.init_optimizer_loaded = True

    def _optimizer_state_is_compatible(self, optimizer_state: dict[str, object]) -> bool:
        """Return whether a checkpoint optimizer state matches current tensors.

        ``optimizer_state`` is a serialized Torch optimizer state dictionary.
        The method returns a boolean, does not mutate state, and checks tensor
        slots such as Adam moments against the current model parameter shapes in
        optimizer order.
        """

        param_groups = optimizer_state.get("param_groups", [])
        state = optimizer_state.get("state", {})
        if not isinstance(param_groups, list) or not isinstance(state, dict):
            return False
        saved_param_ids = [param_id for group in param_groups for param_id in group.get("params", [])]
        current_params = [param for group in self.trainer.optimizer.param_groups for param in group["params"]]
        if len(saved_param_ids) != len(current_params):
            return False
        for saved_param_id, current_param in zip(saved_param_ids, current_params):
            saved_state = state.get(saved_param_id, {})
            if not isinstance(saved_state, dict):
                return False
            for value in saved_state.values():
                if torch.is_tensor(value) and value.ndim > 0 and tuple(value.shape) != tuple(current_param.shape):
                    return False
        return True

    def _move_optimizer_state_to_device(self) -> None:
        """Move tensor-valued optimizer state to the active PatternBoost device."""

        for state in self.trainer.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _active_mask_ratios(self, generation_index: int) -> list[float]:
        """Return mask ratios active for one zero-based generation index."""

        generation_number = generation_index + 1
        active = list(self.config.mask_ratios)
        schedule = self.config.schedules.mask_ratios_by_generation
        if schedule is not None:
            for entry in schedule:
                if entry.start <= generation_number:
                    active = list(entry.mask_ratios)
        return active

    def _active_remask_weights(self, generation_index: int) -> list[float] | None:
        """Return generation remask weights active for one generation index."""

        generation_number = generation_index + 1
        active = self.config.generation.remask_mode_weights
        schedule = self.config.schedules.remask_mode_weights_by_generation
        if schedule is not None:
            for entry in schedule:
                if entry.start <= generation_number:
                    active = list(entry.weights)
        return None if active is None else list(active)

    def _train_generation(
        self,
        generation_index: int,
        active_mask_ratios: list[float],
    ) -> tuple[dict[str, float], str | None]:
        """Fine-tune the model on TopPool-derived denoising samples.

        ``generation_index`` is zero-based and ``active_mask_ratios`` controls
        training noise for this generation. The method returns latest training
        metrics and an optional checkpoint path, mutates model/optimizer state,
        and writes checkpoints only on configured generation boundaries.
        """

        if generation_index > 0 and not self.config.training.fine_tune_from_previous:
            self.trainer.model.load_state_dict(self._initial_model_state)
            self.trainer.model.to(self.device)
            self.trainer.optimizer = build_optimizer(self.trainer.model, self.supervised_config)
        dataset = PatternBoostDenoisingDataset(
            top_pool=self.top_pool,
            mask_ratios=active_mask_ratios,
            loss_config=self.supervised_config.loss,
            num_samples=self.config.training.samples_per_generation,
            seed=self.config.runtime.seed + generation_index,
            pool_sampling_config=self.config.training.pool_sampling,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.runtime.num_workers,
            collate_fn=collate_supervised_batch,
        )
        self._print_training_sampling_summary(active_mask_ratios)
        training_metrics = self.trainer.fit_loader(
            loader,
            stage_name=f"patternboost_gen_{generation_index + 1}",
            num_epochs=self.config.training.epochs_per_generation,
            save_checkpoints=False,
        )
        checkpoint_path: str | None = None
        if self._should_save_training_checkpoint(generation_index):
            saved_path = self.trainer.save_checkpoint(
                stage_name=f"patternboost_gen_{generation_index + 1}",
                epoch_index=self.config.training.epochs_per_generation - 1,
                extra_metrics=self.trainer.latest_metrics,
            )
            shutil.copy2(saved_path, self.latest_checkpoint_path)
            checkpoint_path = str(saved_path)
        return dict(training_metrics), checkpoint_path

    def _generate_generation(
        self,
        generation_index: int,
        active_mask_ratios: list[float],
        active_remask_weights: list[float] | None,
    ) -> list[GenerationResult]:
        """Generate, insert, and return candidates for one PatternBoost generation."""

        generator = PatternBoostGenerator(
            None if self.trainer is None else self.trainer.model,
            self.config,
            self.device,
            self.rng,
            mask_ratios=active_mask_ratios,
            remask_mode_weights=active_remask_weights,
        )
        generation_results: list[GenerationResult] = []
        candidate_batch_size = int(self.config.generation.candidate_batch_size)
        if candidate_batch_size == 1:
            for candidate_index in range(self.config.generation.candidates_per_generation):
                seed_entry = self.top_pool.sample_seed(self.rng)
                result = generator.generate(seed_entry)
                self._insert_generation_result(generation_results, generation_index, candidate_index, result)
            return generation_results

        candidate_index = 0
        while candidate_index < self.config.generation.candidates_per_generation:
            batch_size = min(
                candidate_batch_size,
                self.config.generation.candidates_per_generation - candidate_index,
            )
            seed_entries = [self.top_pool.sample_seed(self.rng) for _ in range(batch_size)]
            batch_results = generator.generate_batch(seed_entries)
            for batch_offset, result in enumerate(batch_results):
                self._insert_generation_result(
                    generation_results,
                    generation_index,
                    candidate_index + batch_offset,
                    result,
                )
            candidate_index += len(batch_results)
        return generation_results

    def _insert_generation_result(
        self,
        generation_results: list[GenerationResult],
        generation_index: int,
        candidate_index: int,
        result: GenerationResult,
    ) -> None:
        """Insert and log one generated candidate.

        ``generation_results`` is the current generation accumulator,
        ``generation_index`` is zero-based, ``candidate_index`` is the
        zero-based index within that generation, and ``result`` is the generated
        candidate. The method mutates TopPool, ``result.inserted``, runner
        diagnostics, and optional candidate logs.
        """

        inserted = self.top_pool.insert(
            result.slot_state_ids,
            metadata={
                "source": "patternboost",
                "generation": generation_index + 1,
                "candidate_index": candidate_index,
                "seed_hash": result.seed_hash,
                "remask_mode": result.remask_mode,
                "mask_ratio": result.mask_ratio,
            },
        )
        result.inserted = inserted
        self.generation_results.append(result)
        generation_results.append(result)
        self._log_candidate_record(generation_index, candidate_index, result)

    def _augment_new_global_best_entries(
        self,
        generation_index: int,
        generation_results: list[GenerationResult],
    ) -> dict[str, object]:
        """Augment newly inserted global-best entries by fixed grid symmetries.

        ``generation_index`` is zero-based and ``generation_results`` contains
        only model-generated candidates from the current generation. The method
        mutates TopPool when augmentation is enabled, returns JSON-friendly
        metrics, and never recursively augments entries created by augmentation.
        """

        augmentation_config = self.config.top_pool_augmentation
        if not augmentation_config.enabled:
            return {"enabled": False}

        best = self.top_pool.best()
        source_score = 0 if best is None else int(best.score)
        retained_hashes = {entry.canonical_hash for entry in self.top_pool.entries()}
        source_results: list[GenerationResult] = []
        for result in generation_results:
            if not result.inserted or int(result.score) != source_score:
                continue
            result_hash = canonical_hash(result.slot_state_ids, self.config.n)
            if result_hash in retained_hashes:
                source_results.append(result)

        if augmentation_config.max_source_entries is not None:
            source_results = source_results[: augmentation_config.max_source_entries]

        inserted_score_counts: Counter[int] = Counter()
        candidate_count = 0
        inserted_count = 0
        for result in source_results:
            source_hash = canonical_hash(result.slot_state_ids, self.config.n)
            for transform_name, transformed_state in generate_grid_symmetry_state_ids(
                result.slot_state_ids,
                self.config.n,
            ):
                candidate_count += 1
                inserted = self.top_pool.insert(
                    transformed_state,
                    metadata={
                        "source": "top_pool_augmentation",
                        "generation": generation_index + 1,
                        "transform": transform_name,
                        "source_hash": source_hash,
                        "source_score": result.score,
                    },
                )
                if inserted:
                    inserted_count += 1
                    inserted_score_counts[int(result.score)] += 1

        return {
            "enabled": True,
            "source_score": source_score,
            "source_count": len(source_results),
            "candidate_count": candidate_count,
            "inserted_count": inserted_count,
            "duplicate_or_trimmed_count": candidate_count - inserted_count,
            "inserted_score_counts": inserted_score_counts,
        }

    def _should_save_training_checkpoint(self, generation_index: int) -> bool:
        """Return whether the just-trained generation should write a checkpoint."""

        generation_number = generation_index + 1
        is_interval = generation_number % self.config.training.checkpoint_every_generations == 0
        is_final = generation_number == self.config.generation.num_generations
        return is_interval or is_final

    def _log_run_start(self, initial_best: int) -> None:
        """Append the initial pool and config summary to ``metrics.jsonl``."""

        initial_score_counts = Counter(entry.score for entry in self.top_pool.entries())
        self._log_metrics_record(
            {
                "event": "run_start",
                "run_id": self.run_id,
                "created_at_utc": self.created_at_utc,
                "run_dir": str(self.run_dir),
                "config_snapshot_path": str(self.config_snapshot_path),
                "top_pool_path": str(self.top_pool_path),
                "n": self.config.n,
                "seed": self.config.runtime.seed,
                "num_generations": self.config.generation.num_generations,
                "candidates_per_generation": self.config.generation.candidates_per_generation,
                "candidate_batch_size": self.config.generation.candidate_batch_size,
                "samples_per_generation": self.config.training.samples_per_generation,
                "epochs_per_generation": self.config.training.epochs_per_generation,
                "pool_max_size": self.config.pool.max_size,
                "initial_best_score": initial_best,
                "initial_pool_size": len(self.top_pool),
                "initial_score_counts": initial_score_counts,
                "mask_ratios": list(self.config.mask_ratios),
                "generation_remask_mode_weights": self.config.generation.remask_mode_weights,
                "pool_sampling": asdict(self.config.training.pool_sampling),
                "schedules": asdict(self.config.schedules),
                "top_pool_augmentation": asdict(self.config.top_pool_augmentation),
                "init_checkpoint_path": self.config.training.init_checkpoint_path,
                "init_optimizer_from_checkpoint": self.config.training.init_optimizer_from_checkpoint,
                "init_checkpoint_loaded": self.init_checkpoint_loaded,
                "init_optimizer_loaded": self.init_optimizer_loaded,
            }
        )

    def _log_metrics_record(self, record: dict[str, object]) -> None:
        """Append one metrics record when JSONL metrics logging is enabled."""

        if self.config.runtime.log_jsonl:
            append_jsonl(self.metrics_log_path, record)

    def _log_candidate_record(
        self,
        generation_index: int,
        candidate_index: int,
        result: GenerationResult,
    ) -> None:
        """Append one generated-candidate record when candidate logging is enabled."""

        if not self.config.runtime.log_candidates_jsonl:
            return
        append_jsonl(
            self.candidates_log_path,
            {
                "event": "candidate",
                "run_id": self.run_id,
                "generation": generation_index + 1,
                "candidate_index": candidate_index,
                "score": result.score,
                "seed_score": result.seed_score,
                "delta_score": result.score - result.seed_score,
                "inserted": result.inserted,
                "seed_hash": result.seed_hash,
                "remask_mode": result.remask_mode,
                "mask_ratio": result.mask_ratio,
                "num_decode_steps": len(result.step_stats),
                "selected_total": sum(len(step.selected_slot_ids) for step in result.step_stats),
                "accepted_total": sum(len(step.accepted_points) for step in result.step_stats),
                "rejected_total": sum(len(step.rejected_points) for step in result.step_stats),
                "point_gain_total": sum(
                    step.point_count_after - step.point_count_before
                    for step in result.step_stats
                ),
                "step_stats": result.step_stats,
            },
        )

    def _build_run_complete_record(self, initial_best: int) -> dict[str, object]:
        """Return the final run-complete metrics record."""

        best = self.top_pool.best()
        final_score_counts = Counter(entry.score for entry in self.top_pool.entries())
        inserted_count = sum(1 for result in self.generation_results if result.inserted)
        return {
            "event": "run_complete",
            "run_id": self.run_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(self.run_dir),
            "initial_best_score": initial_best,
            "best_score": 0 if best is None else best.score,
            "num_pool_entries": len(self.top_pool),
            "num_generation_results": len(self.generation_results),
            "num_inserted_results": inserted_count,
            "final_score_counts": final_score_counts,
            "top_pool_path": str(self.top_pool_path),
            "latest_checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path.exists() else None,
        }

    def _print_training_sampling_summary(self, active_mask_ratios: list[float]) -> None:
        """Print the configured TopPool training-sampling policy."""

        sampling = self.config.training.pool_sampling
        print(
            "patternboost | training sampling "
            f"{sampling.mode} alpha={sampling.alpha:g} "
            f"offset={sampling.offset:g} "
            f"top_fraction={sampling.top_fraction:g} "
            f"uniform_mix={sampling.uniform_mix:g} "
            f"mask_ratios={active_mask_ratios}",
            flush=True,
        )

    def _build_generation_summary(
        self,
        generation_index: int,
        *,
        best_before: int,
        pool_size_before: int,
        pool_score_counts_before: Counter,
        generation_results: list[GenerationResult],
        training_metrics: dict[str, float],
        checkpoint_path: str | None,
        active_mask_ratios: list[float],
        active_remask_weights: list[float] | None,
        augmentation_metrics: dict[str, object],
    ) -> dict[str, object]:
        """Build score, insertion, training, and decode metrics for one generation."""

        best_after = 0 if self.top_pool.best() is None else self.top_pool.best().score
        pool_score_counts_after = Counter(entry.score for entry in self.top_pool.entries())
        inserted_results = [result for result in generation_results if result.inserted]
        generated_score_counts = Counter(result.score for result in generation_results)
        inserted_score_counts = Counter(result.score for result in inserted_results)
        remask_counts = Counter(result.remask_mode for result in generation_results)
        mask_ratio_counts = Counter(f"{result.mask_ratio:g}" for result in generation_results)
        step_counts = [len(result.step_stats) for result in generation_results]
        selected_counts = [
            len(step.selected_slot_ids)
            for result in generation_results
            for step in result.step_stats
        ]
        accepted_counts = [
            len(step.accepted_points)
            for result in generation_results
            for step in result.step_stats
        ]
        rejected_counts = [
            len(step.rejected_points)
            for result in generation_results
            for step in result.step_stats
        ]
        point_gain_counts = [
            step.point_count_after - step.point_count_before
            for result in generation_results
            for step in result.step_stats
        ]
        return {
            "event": "generation",
            "run_id": self.run_id,
            "generation": generation_index + 1,
            "num_generations": self.config.generation.num_generations,
            "active_mask_ratios": list(active_mask_ratios),
            "active_remask_mode_weights": None if active_remask_weights is None else list(active_remask_weights),
            "best_before": best_before,
            "best_after": best_after,
            "best_delta": best_after - best_before,
            "pool_size_before": pool_size_before,
            "pool_size_after": len(self.top_pool),
            "pool_score_counts_before": pool_score_counts_before,
            "pool_score_counts_after": pool_score_counts_after,
            "inserted_count": len(inserted_results),
            "generated_count": len(generation_results),
            "generated_score_counts": generated_score_counts,
            "inserted_score_counts": inserted_score_counts,
            "remask_counts": remask_counts,
            "mask_ratio_counts": mask_ratio_counts,
            "decode_steps_mean": self._mean(step_counts),
            "selected_mean": self._mean(selected_counts),
            "accepted_mean": self._mean(accepted_counts),
            "rejected_mean": self._mean(rejected_counts),
            "point_gain_mean": self._mean(point_gain_counts),
            "training_metrics": dict(training_metrics),
            "checkpoint_path": checkpoint_path,
            "top_pool_path": str(self.top_pool_path),
            "top_pool_augmentation": augmentation_metrics,
        }

    def _print_generation_summary(self, summary: dict[str, object]) -> None:
        """Print one generation summary already prepared for JSONL logging."""

        print(
            "patternboost | "
            f"generation {summary['generation']}/{summary['num_generations']} | "
            f"best {summary['best_before']}->{summary['best_after']} | "
            f"pool_entries {summary['pool_size_before']}->{summary['pool_size_after']} | "
            f"inserted {summary['inserted_count']}/{summary['generated_count']} | "
            f"generated_scores {self._format_counter(summary['generated_score_counts'])} | "
            f"inserted_scores {self._format_counter(summary['inserted_score_counts'])}",
            flush=True,
        )
        print(
            "patternboost | "
            f"generation {summary['generation']}/{summary['num_generations']} detail | "
            f"remask {self._format_counter(summary['remask_counts'])} | "
            f"mask_ratios {self._format_counter(summary['mask_ratio_counts'])} | "
            f"decode_steps_mean {summary['decode_steps_mean']:.2f} | "
            f"selected_mean {summary['selected_mean']:.2f} | "
            f"accepted_mean {summary['accepted_mean']:.2f} | "
            f"rejected_mean {summary['rejected_mean']:.2f} | "
            f"point_gain_mean {summary['point_gain_mean']:.2f}",
            flush=True,
        )
        augmentation = summary.get("top_pool_augmentation", {"enabled": False})
        if isinstance(augmentation, dict) and augmentation.get("enabled"):
            print(
                "patternboost | "
                f"generation {summary['generation']}/{summary['num_generations']} augmentation | "
                f"inserted {augmentation['inserted_count']}/{augmentation['candidate_count']} | "
                f"source_score {augmentation['source_score']} | "
                f"sources {augmentation['source_count']}",
                flush=True,
            )

    def _format_counter(self, counter: Counter) -> str:
        """Return a compact stable ``key:count`` histogram string."""

        if not counter:
            return "{}"
        return "{" + ", ".join(f"{key}:{counter[key]}" for key in sorted(counter)) + "}"

    def _mean(self, values: list[int]) -> float:
        """Return the arithmetic mean for integer metrics, using zero when empty."""

        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    def _resolve_device(self, requested_device: str) -> torch.device:
        """Resolve the runtime torch device with CUDA fallback."""

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested_device)
