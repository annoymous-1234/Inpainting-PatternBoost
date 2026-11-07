"""Candidate generation for circle-packing PatternBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from circle_pack.config import ProjectConfig
from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import center_to_latent_approx, is_valid_packing, latent_to_center_simple
from models.sampling import (
    sample_ddpm_proposals_batched,
    sample_fm_proposals_batched,
    sample_random_proposals_batched,
)
from circle_pack.scipy_solver import solve_radii_for_centers
from patternboost.edit import sample_edit_set
from patternboost.env_init import initialize_partial_env_from_seed
from patternboost.types import CircleTopPoolEntry, GenerationResult, PatternBoostCandidate
from pipelines.data import build_jittered_draft
from pipelines.outer_loop import run_outer_loop_episode, run_outer_round
from pipelines.priority import (
    compute_cross_round_drift_batched,
    compute_slot_dispersion_batched,
    demo_priority_score,
)


def compute_inference_center_distances(
    start_centers: Any,
    end_centers: Any,
    edit_mask: Any,
) -> np.ndarray:
    """Return physical center displacements on edited/free slots."""

    start = np.asarray(start_centers, dtype=np.float64)
    end = np.asarray(end_centers, dtype=np.float64)
    mask = np.asarray(edit_mask, dtype=bool)
    if start.ndim != 2 or start.shape[-1] != 2:
        raise ValueError("start_centers must have shape [K, 2].")
    if end.shape != start.shape:
        raise ValueError("end_centers must have the same shape as start_centers.")
    if mask.shape != (start.shape[0],):
        raise ValueError("edit_mask must have shape [K].")
    if not np.any(mask):
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(end[mask] - start[mask], axis=1).astype(np.float64, copy=False)


def _distance_candidate_fields(distances: np.ndarray) -> dict[str, Any]:
    """Build non-JSONL candidate fields for inference center distances."""

    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        return {
            "inference_center_distances": None,
            "inference_center_distance_count": 0,
            "inference_center_distance_mean": None,
            "inference_center_distance_max": None,
        }
    return {
        "inference_center_distances": values.copy(),
        "inference_center_distance_count": int(values.size),
        "inference_center_distance_mean": float(np.mean(values)),
        "inference_center_distance_max": float(np.max(values)),
    }


@dataclass
class PatternBoostBatchWorkItem:
    """One candidate-generation item prepared for batched DDPM/FM inference."""

    seed_entry: CircleTopPoolEntry
    edit_set: Any
    env: CirclePackingEnv
    inference_start_centers: np.ndarray
    edit_mask: np.ndarray
    mode: str
    candidate_seed: int | None
    candidate_index: int
    generation_error: str | None = None


@dataclass
class PatternBoostGenerator:
    """Generate candidates from PatternBoost TopPool seeds."""

    model: Any
    cfg: ProjectConfig
    rng: np.random.Generator

    def generate(
        self,
        seed_entry: CircleTopPoolEntry,
        *,
        mode: str,
        inference_mode: str,
        candidate_seed: int | None = None,
    ) -> GenerationResult:
        if mode == "full_episode":
            return self.generate_full_episode(
                seed_entry,
                inference_mode=inference_mode,
                candidate_seed=candidate_seed,
            )
        if mode == "one_shot":
            return self.generate_one_shot(
                seed_entry,
                inference_mode=inference_mode,
                candidate_seed=candidate_seed,
            )
        raise ValueError("mode must be 'full_episode' or 'one_shot'.")

    def prepare_batch_work_item(
        self,
        seed_entry: CircleTopPoolEntry,
        *,
        mode: str,
        candidate_seed: int | None,
        candidate_index: int,
    ) -> PatternBoostBatchWorkItem:
        if mode not in {"full_episode", "one_shot"}:
            raise ValueError("mode must be 'full_episode' or 'one_shot'.")
        edit_set, env = self._initialize_candidate_env(seed_entry, candidate_seed)
        start_centers = np.asarray(env.observe()["centers"], dtype=np.float64)
        return PatternBoostBatchWorkItem(
            seed_entry=seed_entry,
            edit_set=edit_set,
            env=env,
            inference_start_centers=start_centers.copy(),
            edit_mask=np.asarray(edit_set.edit_mask, dtype=bool).copy(),
            mode=mode,
            candidate_seed=candidate_seed,
            candidate_index=int(candidate_index),
        )

    def generate_batch(
        self,
        work_items: list[PatternBoostBatchWorkItem],
        *,
        inference_mode: str,
    ) -> list[GenerationResult]:
        """Generate candidates with batched DDPM or FM proposal sampling."""

        if inference_mode not in {"ddpm", "fm"}:
            return [
                self.generate(
                    item.seed_entry,
                    mode=item.mode,
                    inference_mode=inference_mode,
                    candidate_seed=item.candidate_seed,
                )
                for item in work_items
            ]
        results: list[GenerationResult] = []
        for mode in ["full_episode", "one_shot"]:
            group = [item for item in work_items if item.mode == mode]
            if group:
                results.extend(self._generate_group_batched(group, mode=mode, inference_mode=inference_mode))
        results.sort(key=lambda result: int(result.candidate.metadata.get("candidate_index", 0)))
        return results

    def _generate_group_batched(
        self,
        work_items: list[PatternBoostBatchWorkItem],
        *,
        mode: str,
        inference_mode: str,
    ) -> list[GenerationResult]:
        round_limit = 1 if mode == "one_shot" else int(self.cfg.outer_loop.max_rounds)
        prev_samples_by_id: dict[int, torch.Tensor] = {}
        for round_idx in range(round_limit):
            active_items = [
                item
                for item in work_items
                if item.generation_error is None
                and int(item.env.step_idx) < int(self.cfg.problem.max_steps)
                and not bool(np.all(item.env.active_mask))
            ]
            if not active_items:
                break
            try:
                self._run_batched_outer_round(active_items, round_idx, prev_samples_by_id, inference_mode=inference_mode)
            except AssertionError as exc:
                for item in active_items:
                    item.generation_error = str(exc)
                break

        results: list[GenerationResult] = []
        for item in work_items:
            candidate = self._finalize_env_candidate(
                item.env,
                item.seed_entry,
                mode=item.mode,
                edit_policy=item.edit_set.policy,
                generation_error=item.generation_error,
                inference_start_centers=item.inference_start_centers,
                edit_mask=item.edit_mask,
            )
            candidate.metadata.update(
                {
                    "candidate_index": item.candidate_index,
                    "edit_ratio": item.edit_set.ratio,
                    "edit_policy": item.edit_set.policy,
                    "generation_mode": item.mode,
                    "seed_score": float(item.seed_entry.score),
                }
            )
            results.append(
                GenerationResult(
                    seed_hash=item.seed_entry.canonical_hash,
                    generation_mode=item.mode,
                    edit_set=item.edit_set,
                    candidate=candidate,
                    records=[],
                )
            )
        return results

    def _run_batched_outer_round(
        self,
        work_items: list[PatternBoostBatchWorkItem],
        round_idx: int,
        prev_samples_by_id: dict[int, torch.Tensor],
        *,
        inference_mode: str,
    ) -> None:
        observations = [item.env.observe() for item in work_items]
        centers_np = np.stack([np.asarray(obs["centers"], dtype=np.float64) for obs in observations], axis=0)
        latent_np = np.asarray(center_to_latent_approx(centers_np), dtype=np.float64)
        fixed_np = np.stack([np.asarray(obs["active_mask"], dtype=bool) for obs in observations], axis=0)
        # The random completion operator must not touch the model at all, so it
        # reads device/dtype from the config instead of from self.model.
        completion_operator = str(self.cfg.patternboost.generation.completion_operator)
        if completion_operator != "learned":
            proposal_bundle = sample_random_proposals_batched(centers_np, fixed_np, self.cfg)
        elif inference_mode == "ddpm":
            proposal_bundle = sample_ddpm_proposals_batched(
                self.model,
                torch.as_tensor(latent_np, device=self.model.device, dtype=self.model.dtype),
                torch.as_tensor(centers_np, device=self.model.device, dtype=self.model.dtype),
                torch.as_tensor(fixed_np, device=self.model.device, dtype=torch.bool),
                self.cfg,
                latent_to_center_simple,
            )
        elif inference_mode == "fm":
            proposal_bundle = sample_fm_proposals_batched(
                self.model,
                torch.as_tensor(latent_np, device=self.model.device, dtype=self.model.dtype),
                torch.as_tensor(centers_np, device=self.model.device, dtype=self.model.dtype),
                torch.as_tensor(fixed_np, device=self.model.device, dtype=torch.bool),
                self.cfg,
                latent_to_center_simple,
            )
        else:
            raise ValueError("inference_mode must be 'ddpm' or 'fm'.")
        center_samples = proposal_bundle.center_samples
        dispersion = compute_slot_dispersion_batched(center_samples)
        prev_stack: list[torch.Tensor] = []
        has_prev = True
        for item in work_items:
            prev = prev_samples_by_id.get(id(item))
            if prev is None:
                has_prev = False
                break
            prev_stack.append(prev)
        drift = (
            compute_cross_round_drift_batched(torch.stack(prev_stack, dim=0), center_samples)
            if has_prev and prev_stack
            else None
        )
        fm_signal = None
        if inference_mode == "fm" and proposal_bundle.aux:
            aux = proposal_bundle.aux
            if "short_displacement" in aux:
                fm_signal = aux["short_displacement"].mean(dim=1)
            elif "terminal_velocity_norm" in aux:
                fm_signal = aux["terminal_velocity_norm"].mean(dim=1)
        candidate_masks = self._select_batched_candidate_masks(
            work_items,
            observations,
            dispersion,
            drift,
            fm_signal=fm_signal,
        )
        representative = self._representative_batched(center_samples)
        for row_idx, item in enumerate(work_items):
            env = item.env
            observation = observations[row_idx]
            candidate_mask = candidate_masks[row_idx]
            current_centers = np.asarray(observation["centers"], dtype=env.dtype)
            fixed = np.asarray(observation["active_mask"], dtype=bool)
            free = ~fixed
            candidate_mask = np.asarray(candidate_mask, dtype=bool) & free
            proposal = current_centers.copy()
            representative_np = representative[row_idx].detach().cpu().numpy().astype(env.dtype, copy=False)
            proposal[candidate_mask] = representative_np[candidate_mask]
            proposal[fixed] = current_centers[fixed]
            _next_obs, _reward, _done, _info = env.step(proposal, candidate_mask=candidate_mask)
            prev_samples_by_id[id(item)] = center_samples[row_idx].detach()
        del round_idx

    def _select_batched_candidate_masks(
        self,
        work_items: list[PatternBoostBatchWorkItem],
        observations: list[dict[str, Any]],
        dispersion: torch.Tensor,
        drift: torch.Tensor | None,
        fm_signal: torch.Tensor | None = None,
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for row_idx, item in enumerate(work_items):
            env = item.env
            free_mask = ~np.asarray(observations[row_idx]["active_mask"], dtype=bool)
            if self.cfg.outer_loop.candidate_mode == "all_free":
                masks.append(free_mask.copy())
                continue
            score = demo_priority_score(
                dispersion[row_idx],
                drift=drift[row_idx] if drift is not None else None,
                fm_signal=fm_signal[row_idx] if fm_signal is not None else None,
                free_mask=torch.as_tensor(free_mask),
                cfg=self.cfg,
            )
            budget = min(int(env.config.env.activation_budget), int(free_mask.sum()))
            candidate_mask = np.zeros_like(free_mask, dtype=bool)
            if budget > 0:
                score_np = score.detach().cpu().numpy()
                free_indices = np.flatnonzero(free_mask)
                ranked = free_indices[np.argsort(score_np[free_indices])[::-1]]
                candidate_mask[ranked[:budget]] = True
            masks.append(candidate_mask)
        return masks

    def _representative_batched(self, center_samples: torch.Tensor) -> torch.Tensor:
        reduction = self.cfg.outer_loop.representative_reduction
        if reduction == "first_sample":
            return center_samples[:, 0]
        if reduction == "mean":
            return center_samples.mean(dim=1)
        raise ValueError("representative reduction must be 'first_sample' or 'mean'.")

    def generate_full_episode(
        self,
        seed_entry: CircleTopPoolEntry,
        *,
        inference_mode: str,
        candidate_seed: int | None = None,
    ) -> GenerationResult:
        edit_set, env = self._initialize_candidate_env(seed_entry, candidate_seed)
        inference_start_centers = np.asarray(env.observe()["centers"], dtype=np.float64).copy()
        snapshot = env.snapshot()
        generation_error: str | None = None
        try:
            records = run_outer_loop_episode(
                env,
                self.model,
                inference_mode,
                self.cfg,
                latent_to_center_fn=latent_to_center_simple,
                center_to_latent_fn=center_to_latent_approx,
            )
        except AssertionError as exc:
            env.restore(snapshot)
            records = []
            generation_error = str(exc)
        candidate = self._finalize_env_candidate(
            env,
            seed_entry,
            mode="full_episode",
            edit_policy=edit_set.policy,
            generation_error=generation_error,
            inference_start_centers=inference_start_centers,
            edit_mask=edit_set.edit_mask,
        )
        return GenerationResult(
            seed_hash=seed_entry.canonical_hash,
            generation_mode="full_episode",
            edit_set=edit_set,
            candidate=candidate,
            records=records,
        )

    def generate_one_shot(
        self,
        seed_entry: CircleTopPoolEntry,
        *,
        inference_mode: str,
        candidate_seed: int | None = None,
    ) -> GenerationResult:
        edit_set, env = self._initialize_candidate_env(seed_entry, candidate_seed)
        inference_start_centers = np.asarray(env.observe()["centers"], dtype=np.float64).copy()
        snapshot = env.snapshot()
        generation_error: str | None = None
        try:
            record, _ = run_outer_round(
                env,
                self.model,
                inference_mode,
                self.cfg,
                latent_to_center_fn=latent_to_center_simple,
                center_to_latent_fn=center_to_latent_approx,
            )
            records = [record]
        except AssertionError as exc:
            env.restore(snapshot)
            records = []
            generation_error = str(exc)
        candidate = self._finalize_env_candidate(
            env,
            seed_entry,
            mode="one_shot",
            edit_policy=edit_set.policy,
            generation_error=generation_error,
            inference_start_centers=inference_start_centers,
            edit_mask=edit_set.edit_mask,
        )
        return GenerationResult(
            seed_hash=seed_entry.canonical_hash,
            generation_mode="one_shot",
            edit_set=edit_set,
            candidate=candidate,
            records=records,
        )

    def _initialize_candidate_env(
        self,
        seed_entry: CircleTopPoolEntry,
        candidate_seed: int | None,
    ) -> tuple[Any, CirclePackingEnv]:
        edit_set = sample_edit_set(seed_entry.centers, seed_entry.active_mask, self.cfg.patternboost.edit, self.rng)
        fixed_mask = ~edit_set.edit_mask
        _, draft_centers = build_jittered_draft(seed_entry.centers, fixed_mask, self.rng, cfg=self.cfg)
        env = CirclePackingEnv(self.cfg)
        initialize_partial_env_from_seed(
            env,
            seed_entry.centers,
            edit_set.edit_mask,
            draft_centers=draft_centers,
            seed=candidate_seed,
        )
        return edit_set, env

    def _finalize_env_candidate(
        self,
        env: CirclePackingEnv,
        seed_entry: CircleTopPoolEntry,
        *,
        mode: str,
        edit_policy: str,
        generation_error: str | None = None,
        inference_start_centers: np.ndarray | None = None,
        edit_mask: np.ndarray | None = None,
    ) -> PatternBoostCandidate:
        pre_completion_observation = env.observe()
        pre_completion_centers = np.asarray(pre_completion_observation["centers"], dtype=np.float64)
        if generation_error is None and inference_start_centers is not None and edit_mask is not None:
            distances = compute_inference_center_distances(inference_start_centers, pre_completion_centers, edit_mask)
        else:
            distances = np.asarray([], dtype=np.float64)
        distance_fields = _distance_candidate_fields(distances)

        if self.cfg.patternboost.generation.greedy_complete_unfinished and not bool(np.all(env.active_mask)):
            completion_info = env.greedy_fill_masked_slots()
        else:
            completion_info = {"success": bool(np.all(env.active_mask))}
        observation = env.observe()
        active_mask = np.asarray(observation["active_mask"], dtype=bool)
        centers = np.asarray(observation["centers"], dtype=np.float64)
        full_active = bool(np.all(active_mask))
        if full_active:
            result = solve_radii_for_centers(
                centers,
                radius_max=self.cfg.safety.radius_max,
                feasibility_tol=self.cfg.safety.feasibility_tol,
            )
            circles = result.circles
            radii = circles[:, 2].copy()
            legal = bool(result.success) and is_valid_packing(
                circles[:, :2],
                circles[:, 2],
                tol=max(self.cfg.safety.feasibility_tol, 1.0e-9),
            )
            raw_score = float(result.sum_r) if legal else None
            if result.success:
                lp_status = result.message if legal else f"invalid:{result.message}"
            else:
                lp_status = f"failed:{result.status}:{result.message}"
        else:
            circles = np.column_stack((centers, np.zeros(centers.shape[0], dtype=np.float64)))
            radii = None
            legal = False
            raw_score = None
            lp_status = None
        return PatternBoostCandidate(
            centers=centers,
            radii=radii,
            circles=circles,
            active_mask=active_mask,
            raw_score=raw_score,
            lp_status=lp_status,
            legal=bool(legal),
            full_active=full_active,
            **distance_fields,
            metadata={
                "source": "patternboost",
                "seed_hash": seed_entry.canonical_hash,
                "generation_mode": mode,
                "edit_policy": edit_policy,
                "greedy_completion": completion_info,
                "generation_error": generation_error,
            },
        )
