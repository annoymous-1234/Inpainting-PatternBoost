"""PatternBoost train-generate-polish runner for circle packing."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from circle_pack.config import ProjectConfig
from circle_pack.env import CirclePackingEnv
from models.factory import build_model
from models.objectives import ddpm_supervised_loss, fm_supervised_loss
from patternboost.dataset import PatternBoostSupervisedDataset
from patternboost.generator import PatternBoostGenerator
from patternboost.logging_utils import append_jsonl
from patternboost.polish import candidate_final_score, cap_expand_candidate, srp_polish_candidates
from patternboost.top_pool import TopPool
from patternboost.types import GenerationStats, PatternBoostCandidate
from pipelines.data import load_teacher_layouts


class PatternBoostRunner:
    """Coordinate PatternBoost fine-tuning, generation, polishing, and TopPool updates."""

    def __init__(self, cfg: ProjectConfig, model: torch.nn.Module | None = None) -> None:
        self.cfg = cfg
        self.pb_cfg = cfg.patternboost
        # The completion-operator ablation runs the whole outer loop without any
        # learned component: no network is built, trained, or evaluated.
        self.uses_learned_operator = str(self.pb_cfg.generation.completion_operator) == "learned"
        if self.uses_learned_operator:
            self._validate_training_mode_config()
        self.rng = np.random.default_rng(self.pb_cfg.runtime.seed)
        torch.manual_seed(self.pb_cfg.runtime.seed)
        self.device = self._resolve_device(self.pb_cfg.runtime.device)
        self.run_dir = self._initialize_run_dir()
        self.metrics_log_path = self.run_dir / "metrics.jsonl"
        self.candidates_log_path = self.run_dir / "candidates.jsonl"
        self.top_pool_path = self._resolve_top_pool_path()
        self.latest_checkpoint_path = self.run_dir / "latest.pt"
        self.top_pool = TopPool(
            self.pb_cfg.pool.max_size,
            dedup_tol=self.pb_cfg.pool.dedup_tol,
            feasibility_tol=self.cfg.safety.feasibility_tol,
            radius_max=self.cfg.safety.radius_max,
        )
        if self.uses_learned_operator:
            self.model = (model if model is not None else build_model(cfg)).to(device=self.device)
            self.optimizer = self._build_optimizer()
        else:
            self.model = None
            self.optimizer = None
        self.resume_from_generation = 0
        self.resume_loaded_optimizer = False
        self.resume_checkpoint_path: str | None = None
        self.resume_top_pool_path: str | None = None
        if self.pb_cfg.resume.enabled:
            self._load_resume_state()
        self._initial_model_state = (
            {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            if self.model is not None
            else {}
        )

    def _validate_training_mode_config(self) -> None:
        if self.pb_cfg.training.mode == "fm":
            if not self.cfg.model.heads.enable_fm_head:
                raise ValueError("patternboost.training.mode='fm' requires model.heads.enable_fm_head=True.")
            if self.cfg.model.slot_features.use_extra_slot_features and not self.cfg.fm.use_draft_feature:
                raise ValueError(
                    "patternboost.training.mode='fm' with model.slot_features.use_extra_slot_features=True "
                    "requires fm.use_draft_feature=True."
                )

    def run(self) -> dict[str, Any]:
        """Run all configured PatternBoost generations and return a summary."""

        self.initialize_top_pool()
        initial_best = 0.0 if self.top_pool.best() is None else float(self.top_pool.best().score)
        self.top_pool.save(self.top_pool_path)
        resume_info = self._resume_info()
        self._log_metrics(
            {
                "event": "run_start",
                "initial_best_score": initial_best,
                **resume_info,
                **self._pool_score_summary(),
            }
        )
        latest_loss = 0.0
        for generation_offset in range(int(self.pb_cfg.generation.num_generations)):
            generation_idx = int(self.resume_from_generation) + generation_offset
            if (
                self.uses_learned_operator
                and generation_offset > 0
                and not self.pb_cfg.training.fine_tune_from_previous
            ):
                self.model.load_state_dict(self._initial_model_state)
                self.model.to(device=self.device)
                self.optimizer = self._build_optimizer()
            best_before = 0.0 if self.top_pool.best() is None else float(self.top_pool.best().score)
            latest_loss = self._train_generation(generation_idx) if self.uses_learned_operator else 0.0
            candidates = self._generate_candidates(generation_idx)
            candidates = self._polish_candidates(candidates, generation_idx)
            num_inserted = self._insert_candidates(candidates, generation_idx)
            self.top_pool.save(self.top_pool_path)
            self._save_top_pool_snapshot(generation_idx)
            self._save_checkpoint(generation_idx, latest_loss)
            best_after = 0.0 if self.top_pool.best() is None else float(self.top_pool.best().score)
            pool_stats = self._pool_score_summary()
            stats = GenerationStats(
                generation=generation_idx + 1,
                best_before=best_before,
                best_after=best_after,
                num_candidates=len(candidates),
                num_inserted=num_inserted,
                num_solver_polished=sum(1 for candidate in candidates if candidate.polish_score is not None),
            )
            candidate_stats = self._candidate_summary(candidates)
            self._log_metrics(
                {
                    "event": "generation",
                    **asdict(stats),
                    "mean_train_loss": latest_loss,
                    **candidate_stats,
                    **pool_stats,
                }
            )
            print_stats = {**asdict(stats), **candidate_stats, **pool_stats}
            final_score_mean = print_stats.get("final_score_mean")
            final_score_mean_text = "none" if final_score_mean is None else f"{final_score_mean:.6f}"
            print(
                "generation={generation} best_before={best_before:.6f} best_after={best_after:.6f} "
                "candidates={num_candidates} inserted={num_inserted} solver_polished={num_solver_polished} "
                f"final_score_mean={final_score_mean_text} "
                "full_active_rate={full_active_rate:.3f} legal_rate={legal_rate:.3f} "
                "generation_errors={generation_error_count} pool_size={pool_size} "
                "pool_score_best={pool_score_best:.6f} pool_score_mean={pool_score_mean:.6f} "
                "pool_score_p50={pool_score_p50:.6f} pool_score_p75={pool_score_p75:.6f} "
                "pool_score_p90={pool_score_p90:.6f}".format(**print_stats)
            )

        best = self.top_pool.best()
        final_pool_stats = self._pool_score_summary()
        summary = {
            "initial_best_score": initial_best,
            "best_score": 0.0 if best is None else float(best.score),
            "num_pool_entries": len(self.top_pool),
            **final_pool_stats,
            "top_pool_path": str(self.top_pool_path),
            "latest_checkpoint_path": str(self.latest_checkpoint_path)
            if self.pb_cfg.runtime.save_checkpoints and self.model is not None
            else None,
            "metrics_log_path": str(self.metrics_log_path),
            "candidates_log_path": str(self.candidates_log_path)
            if self.pb_cfg.runtime.log_jsonl and self.pb_cfg.runtime.log_candidates_jsonl
            else None,
            "run_dir": str(self.run_dir),
            **resume_info,
        }
        self._log_metrics({"event": "run_complete", **summary})
        return summary

    def _pool_score_summary(self) -> dict[str, float | int | None]:
        """Return compact score-distribution statistics for the current TopPool."""

        scores = np.asarray([entry.score for entry in self.top_pool.entries()], dtype=np.float64)
        if scores.size == 0:
            return {
                "pool_size": 0,
                "pool_score_best": None,
                "pool_score_mean": None,
                "pool_score_p50": None,
                "pool_score_p75": None,
                "pool_score_p90": None,
            }
        return {
            "pool_size": int(scores.size),
            "pool_score_best": float(np.max(scores)),
            "pool_score_mean": float(np.mean(scores)),
            "pool_score_p50": float(np.percentile(scores, 50)),
            "pool_score_p75": float(np.percentile(scores, 75)),
            "pool_score_p90": float(np.percentile(scores, 90)),
        }

    def _candidate_summary(self, candidates: list[PatternBoostCandidate]) -> dict[str, Any]:
        """Return generation-level candidate stats for metrics.jsonl."""

        def score_summary(prefix: str, scores: np.ndarray) -> dict[str, float | None]:
            if scores.size == 0:
                return {
                    f"{prefix}_mean": None,
                    f"{prefix}_std": None,
                    f"{prefix}_min": None,
                    f"{prefix}_p50": None,
                    f"{prefix}_p75": None,
                    f"{prefix}_p90": None,
                    f"{prefix}_max": None,
                }
            return {
                f"{prefix}_mean": float(np.mean(scores)),
                f"{prefix}_std": float(np.std(scores)),
                f"{prefix}_min": float(np.min(scores)),
                f"{prefix}_p50": float(np.percentile(scores, 50)),
                f"{prefix}_p75": float(np.percentile(scores, 75)),
                f"{prefix}_p90": float(np.percentile(scores, 90)),
                f"{prefix}_max": float(np.max(scores)),
            }

        def distance_summary(distances: np.ndarray) -> dict[str, float | None]:
            return score_summary("inference_center_distance", distances)

        if not candidates:
            return {
                "num_candidates": 0,
                "num_inserted": 0,
                "full_active_rate": None,
                "legal_rate": None,
                "generation_error_count": 0,
                "greedy_completion_success_rate": None,
                "raw_score_mean": None,
                "raw_score_max": None,
                "polish_score_mean": None,
                "polish_score_max": None,
                "final_score_mean": None,
                "final_score_max": None,
                **score_summary("candidate_raw_score", np.asarray([], dtype=np.float64)),
                **score_summary("candidate_polish_score", np.asarray([], dtype=np.float64)),
                **score_summary("candidate_final_score", np.asarray([], dtype=np.float64)),
                "inference_center_distance_count": 0,
                "inference_center_distance_candidate_count": 0,
                **distance_summary(np.asarray([], dtype=np.float64)),
                "edit_policy_counts": {},
                "generation_mode_counts": {},
            }

        raw_scores = np.asarray(
            [float(candidate.raw_score) for candidate in candidates if candidate.raw_score is not None],
            dtype=np.float64,
        )
        polish_scores = np.asarray(
            [float(candidate.polish_score) for candidate in candidates if candidate.polish_score is not None],
            dtype=np.float64,
        )
        final_scores = np.asarray(
            [score for candidate in candidates if (score := candidate_final_score(candidate)) is not None],
            dtype=np.float64,
        )
        distance_arrays = [
            np.asarray(candidate.inference_center_distances, dtype=np.float64)
            for candidate in candidates
            if candidate.inference_center_distances is not None
            and int(candidate.inference_center_distance_count) > 0
        ]
        inference_distances = (
            np.concatenate(distance_arrays).astype(np.float64, copy=False)
            if distance_arrays
            else np.asarray([], dtype=np.float64)
        )
        greedy_seen = 0
        greedy_success = 0
        generation_error_count = 0
        edit_policy_counts: Counter[str] = Counter()
        generation_mode_counts: Counter[str] = Counter()
        for candidate in candidates:
            metadata = candidate.metadata or {}
            if metadata.get("generation_error"):
                generation_error_count += 1
            completion = metadata.get("greedy_completion")
            if isinstance(completion, dict):
                greedy_seen += 1
                greedy_success += int(bool(completion.get("success")))
            edit_policy_counts[str(metadata.get("edit_policy", "unknown"))] += 1
            generation_mode_counts[str(metadata.get("generation_mode", "unknown"))] += 1

        return {
            "num_candidates": int(len(candidates)),
            "num_inserted": int(sum(bool(candidate.inserted) for candidate in candidates)),
            "full_active_rate": float(np.mean([candidate.full_active for candidate in candidates])),
            "legal_rate": float(np.mean([candidate.legal for candidate in candidates])),
            "generation_error_count": int(generation_error_count),
            "greedy_completion_success_rate": float(greedy_success / greedy_seen) if greedy_seen else None,
            "raw_score_mean": float(np.mean(raw_scores)) if raw_scores.size else None,
            "raw_score_max": float(np.max(raw_scores)) if raw_scores.size else None,
            "polish_score_mean": float(np.mean(polish_scores)) if polish_scores.size else None,
            "polish_score_max": float(np.max(polish_scores)) if polish_scores.size else None,
            "final_score_mean": float(np.mean(final_scores)) if final_scores.size else None,
            "final_score_max": float(np.max(final_scores)) if final_scores.size else None,
            **score_summary("candidate_raw_score", raw_scores),
            **score_summary("candidate_polish_score", polish_scores),
            **score_summary("candidate_final_score", final_scores),
            "inference_center_distance_count": int(inference_distances.size),
            "inference_center_distance_candidate_count": int(len(distance_arrays)),
            **distance_summary(inference_distances),
            "edit_policy_counts": dict(edit_policy_counts),
            "generation_mode_counts": dict(generation_mode_counts),
        }

    def initialize_top_pool(self) -> None:
        """Populate TopPool from configured teacher layouts and greedy completions."""

        if self.pb_cfg.resume.enabled:
            if len(self.top_pool) == 0:
                raise ValueError("PatternBoost resume TopPool is empty.")
            return

        if "teacher_layouts" in self.pb_cfg.pool.init_sources:
            dataset_path = Path(self.cfg.train.dataset_path)
            if dataset_path.exists():
                for layout_idx, layout in enumerate(load_teacher_layouts(dataset_path)):
                    active_mask = np.ones(layout.shape[0], dtype=bool)
                    self.top_pool.insert(layout, active_mask, metadata={"source": "teacher_layouts", "index": layout_idx})

        if "greedy" in self.pb_cfg.pool.init_sources:
            for seed_idx in range(int(self.pb_cfg.pool.greedy_num_seeds)):
                env = CirclePackingEnv(self.cfg)
                env.reset(seed=self.pb_cfg.runtime.seed + seed_idx)
                info = env.greedy_fill_masked_slots()
                if info["success"]:
                    obs = env.observe()
                    self.top_pool.insert(
                        obs["centers"],
                        obs["active_mask"],
                        radii=obs["radii"],
                        circles=obs["circles"],
                        score=obs["layout_score"],
                        lp_status=obs["lp_status"],
                        metadata={"source": "greedy", "seed_index": seed_idx},
                    )

        if len(self.top_pool) == 0:
            raise ValueError("PatternBoost TopPool initialization produced no entries.")

    def _generate_candidates(self, generation_idx: int) -> list[PatternBoostCandidate]:
        batch_size = int(self.pb_cfg.generation.candidate_batch_size)
        if batch_size > 1 and self.pb_cfg.training.mode in {"ddpm", "fm"}:
            return self._generate_candidates_batched(generation_idx, batch_size=batch_size)
        return self._generate_candidates_serial(generation_idx)

    def _generate_candidates_serial(self, generation_idx: int) -> list[PatternBoostCandidate]:
        generator = PatternBoostGenerator(self.model, self.cfg, self.rng)
        modes = list(self.pb_cfg.generation.modes)
        weights = np.asarray(self.pb_cfg.generation.mode_weights, dtype=np.float64)
        probabilities = weights / np.sum(weights)
        candidates: list[PatternBoostCandidate] = []
        inference_mode = str(self.pb_cfg.training.mode)
        was_training = self.model is not None and self.model.training
        if self.model is not None:
            self.model.eval()
        try:
            for candidate_idx in range(int(self.pb_cfg.generation.candidates_per_generation)):
                seed_entry = self.top_pool.sample_seed(self.rng, self.pb_cfg.pool.sampling)
                mode = str(self.rng.choice(modes, p=probabilities))
                result = generator.generate(
                    seed_entry,
                    mode=mode,
                    inference_mode=inference_mode,
                    candidate_seed=self.pb_cfg.runtime.seed + 100000 * generation_idx + candidate_idx,
                )
                result.candidate.metadata.update(
                    {
                        "generation": generation_idx + 1,
                        "candidate_index": candidate_idx,
                        "edit_ratio": result.edit_set.ratio,
                        "edit_policy": result.edit_set.policy,
                        "generation_mode": result.generation_mode,
                        "seed_score": float(seed_entry.score),
                    }
                )
                candidates.append(result.candidate)
        finally:
            if was_training:
                self.model.train()
        return candidates

    def _generate_candidates_batched(self, generation_idx: int, *, batch_size: int) -> list[PatternBoostCandidate]:
        generator = PatternBoostGenerator(self.model, self.cfg, self.rng)
        modes = list(self.pb_cfg.generation.modes)
        weights = np.asarray(self.pb_cfg.generation.mode_weights, dtype=np.float64)
        probabilities = weights / np.sum(weights)
        total_candidates = int(self.pb_cfg.generation.candidates_per_generation)
        candidates: list[PatternBoostCandidate] = []
        inference_mode = str(self.pb_cfg.training.mode)
        was_training = self.model is not None and self.model.training
        if self.model is not None:
            self.model.eval()
        try:
            for chunk_start in range(0, total_candidates, int(batch_size)):
                chunk_end = min(chunk_start + int(batch_size), total_candidates)
                work_items = []
                for candidate_idx in range(chunk_start, chunk_end):
                    seed_entry = self.top_pool.sample_seed(self.rng, self.pb_cfg.pool.sampling)
                    mode = str(self.rng.choice(modes, p=probabilities))
                    work_items.append(
                        generator.prepare_batch_work_item(
                            seed_entry,
                            mode=mode,
                            candidate_seed=self.pb_cfg.runtime.seed + 100000 * generation_idx + candidate_idx,
                            candidate_index=candidate_idx,
                        )
                    )
                try:
                    results = generator.generate_batch(work_items, inference_mode=inference_mode)
                except Exception as exc:
                    results = []
                    for item in work_items:
                        result = generator.generate(
                            item.seed_entry,
                            mode=item.mode,
                            inference_mode=inference_mode,
                            candidate_seed=item.candidate_seed,
                        )
                        result.candidate.metadata["batch_fallback_error"] = str(exc)
                        result.candidate.metadata["candidate_index"] = item.candidate_index
                        result.candidate.metadata["edit_ratio"] = result.edit_set.ratio
                        result.candidate.metadata["edit_policy"] = result.edit_set.policy
                        result.candidate.metadata["generation_mode"] = result.generation_mode
                        result.candidate.metadata["seed_score"] = float(item.seed_entry.score)
                        results.append(result)
                for result in results:
                    result.candidate.metadata.update(
                        {
                            "generation": generation_idx + 1,
                            "candidate_index": int(result.candidate.metadata.get("candidate_index", len(candidates))),
                            "edit_ratio": result.candidate.metadata.get("edit_ratio", result.edit_set.ratio),
                            "edit_policy": result.candidate.metadata.get("edit_policy", result.edit_set.policy),
                            "generation_mode": result.candidate.metadata.get(
                                "generation_mode",
                                result.generation_mode,
                            ),
                        }
                    )
                results.sort(key=lambda item: int(item.candidate.metadata.get("candidate_index", 0)))
                candidates.extend(result.candidate for result in results)
        finally:
            if was_training:
                self.model.train()
        candidates.sort(key=lambda candidate: int(candidate.metadata.get("candidate_index", 0)))
        return candidates

    def _polish_candidates(
        self,
        candidates: list[PatternBoostCandidate],
        generation_idx: int,
    ) -> list[PatternBoostCandidate]:
        if self.pb_cfg.polish.refresh_lp_cache:
            for candidate in candidates:
                if candidate.full_active:
                    cap_expand_candidate(
                        candidate,
                        feasibility_tol=self.cfg.safety.feasibility_tol,
                        radius_max=self.cfg.safety.radius_max,
                    )
        srp_polish_candidates(
            candidates,
            polish_cfg=self.pb_cfg.polish,
            generation_idx=generation_idx,
            runtime_seed=self.pb_cfg.runtime.seed,
        )
        return candidates

    def _insert_candidates(self, candidates: list[PatternBoostCandidate], generation_idx: int) -> int:
        num_inserted = 0
        for candidate_idx, candidate in enumerate(candidates):
            final_score = candidate_final_score(candidate)
            if final_score is None:
                candidate.inserted = False
                self._log_candidate(generation_idx, candidate_idx, candidate)
                continue
            inserted = self.top_pool.insert(
                candidate.centers,
                candidate.active_mask,
                radii=candidate.radii,
                circles=candidate.circles,
                score=final_score,
                lp_status=candidate.lp_status,
                metadata=dict(candidate.metadata),
            )
            candidate.inserted = bool(inserted)
            num_inserted += int(inserted)
            self._log_candidate(generation_idx, candidate_idx, candidate)
        return num_inserted

    def _train_generation(self, generation_idx: int) -> float:
        dataset = PatternBoostSupervisedDataset(
            self.top_pool,
            self.cfg,
            mode=self.pb_cfg.training.mode,
            num_samples=self.pb_cfg.training.samples_per_generation,
            seed=self.pb_cfg.runtime.seed + generation_idx,
        )
        loader = DataLoader(dataset, batch_size=self.pb_cfg.training.batch_size, shuffle=True, num_workers=0)
        loss_fn = ddpm_supervised_loss if self.pb_cfg.training.mode == "ddpm" else fm_supervised_loss
        total_loss = 0.0
        num_batches = 0
        self.model.train()
        for _epoch in range(int(self.pb_cfg.training.epochs_per_generation)):
            for batch in loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss, _stats = loss_fn(self.model, batch, self.cfg)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.pb_cfg.training.lr),
            weight_decay=float(self.pb_cfg.training.weight_decay),
        )

    def _save_checkpoint(self, generation_idx: int, mean_loss: float) -> None:
        if not self.pb_cfg.runtime.save_checkpoints or self.model is None:
            return
        generation_number = int(generation_idx) + 1
        final_generation_number = int(self.resume_from_generation) + int(self.pb_cfg.generation.num_generations)
        should_save = generation_number % int(self.pb_cfg.training.checkpoint_every_generations) == 0
        should_save = should_save or generation_number == final_generation_number
        if not should_save:
            return
        payload = {
            "generation": generation_number,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "mean_loss": float(mean_loss),
            "top_pool_path": str(self.top_pool_path),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        epoch_path = self.run_dir / f"generation_{generation_number:03d}.pt"
        torch.save(payload, epoch_path)
        torch.save(payload, self.latest_checkpoint_path)

    def _save_top_pool_snapshot(self, generation_idx: int) -> None:
        interval = int(self.pb_cfg.runtime.top_pool_snapshot_every_generations)
        if interval <= 0:
            return
        generation_number = int(generation_idx) + 1
        final_generation_number = int(self.resume_from_generation) + int(self.pb_cfg.generation.num_generations)
        is_final_generation = generation_number == final_generation_number
        if generation_number % interval != 0 and not is_final_generation:
            return
        snapshot_path = self.run_dir / f"top_pool_generation_{generation_number:03d}.pt"
        self.top_pool.save(snapshot_path)

    def _initialize_run_dir(self) -> Path:
        root = Path(self.pb_cfg.runtime.checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        if self.pb_cfg.runtime.run_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            run_name = f"pb_{timestamp}_seed{self.pb_cfg.runtime.seed}"
        else:
            run_name = self.pb_cfg.runtime.run_name
        run_dir = root / run_name
        if run_dir.exists():
            raise FileExistsError(f"PatternBoost run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _resolve_top_pool_path(self) -> Path:
        if self.pb_cfg.pool.save_path is None:
            return self.run_dir / "top_pool.pt"
        return Path(self.pb_cfg.pool.save_path)

    def _load_resume_state(self) -> None:
        resume_cfg = self.pb_cfg.resume
        checkpoint_path = Path(str(resume_cfg.checkpoint_path))
        top_pool_path = Path(str(resume_cfg.top_pool_path))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PatternBoost resume checkpoint not found: {checkpoint_path}")
        if not top_pool_path.exists():
            raise FileNotFoundError(f"PatternBoost resume TopPool not found: {top_pool_path}")
        if top_pool_path.resolve() == self.top_pool_path.resolve(strict=False):
            raise ValueError("PatternBoost resume TopPool path must differ from the current output top_pool_path.")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "generation" not in payload:
            raise ValueError("PatternBoost resume checkpoint is missing generation.")
        if "model_state_dict" not in payload:
            raise ValueError("PatternBoost resume checkpoint is missing model_state_dict.")
        self.resume_from_generation = int(payload["generation"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device=self.device)

        if bool(resume_cfg.load_optimizer):
            if "optimizer_state_dict" not in payload:
                raise ValueError(
                    "PatternBoost resume checkpoint is missing optimizer_state_dict; "
                    "set patternboost.resume.load_optimizer=false to rebuild the optimizer."
                )
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self._move_optimizer_state_to_device()
            self.resume_loaded_optimizer = True

        loaded_top_pool = TopPool.load(top_pool_path)
        self._validate_resume_top_pool(loaded_top_pool)
        self.top_pool = loaded_top_pool
        self.resume_checkpoint_path = str(checkpoint_path)
        self.resume_top_pool_path = str(top_pool_path)

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=self.device)

    def _validate_resume_top_pool(self, top_pool: TopPool) -> None:
        if len(top_pool) == 0:
            raise ValueError("PatternBoost resume TopPool is empty.")
        expected_k = int(self.cfg.problem.K)
        for entry in top_pool.entries():
            if entry.centers.shape[0] != expected_k:
                raise ValueError(
                    f"PatternBoost resume TopPool entry K mismatch: expected {expected_k}, "
                    f"got {entry.centers.shape[0]}."
                )

    def _resume_info(self) -> dict[str, Any]:
        return {
            "resume_enabled": bool(self.pb_cfg.resume.enabled),
            "resume_checkpoint_path": self.resume_checkpoint_path,
            "resume_top_pool_path": self.resume_top_pool_path,
            "resume_from_generation": int(self.resume_from_generation),
            "resume_loaded_optimizer": bool(self.resume_loaded_optimizer),
        }

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(requested_device)

    def _log_metrics(self, record: dict[str, Any]) -> None:
        if self.pb_cfg.runtime.log_jsonl:
            append_jsonl(self.metrics_log_path, record)

    def _log_candidate(self, generation_idx: int, candidate_idx: int, candidate: PatternBoostCandidate) -> None:
        if not self.pb_cfg.runtime.log_jsonl or not self.pb_cfg.runtime.log_candidates_jsonl:
            return
        append_jsonl(
            self.candidates_log_path,
            {
                "generation": generation_idx + 1,
                "candidate_index": candidate_idx,
                "raw_score": candidate.raw_score,
                "polish_score": candidate.polish_score,
                "final_score": candidate_final_score(candidate),
                "legal": candidate.legal,
                "full_active": candidate.full_active,
                "inserted": candidate.inserted,
                "metadata": candidate.metadata,
            },
        )
