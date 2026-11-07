"""Main numpy-based environment for center-only circle packing."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from circle_pack.config import ProjectConfig
from circle_pack.geometry import normalize_centers, sample_random_centers
from circle_pack.render import render_state
from circle_pack.scipy_solver import solve_radii_for_centers
from circle_pack.types import CommitDecision, EnvSnapshot, RoundCache


class CirclePackingEnv:
    """Environment whose persistent slot state stores only circle centers."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dtype = config.numpy_dtype
        self.num_slots = int(config.problem.K)
        self.rng = np.random.default_rng(config.project.seed)
        self.active_mask: np.ndarray | None = None
        self.centers: np.ndarray | None = None
        self.step_idx = 0
        self.radii: np.ndarray | None = None
        self.circles: np.ndarray | None = None
        self.layout_score: float | None = None
        self.lp_status: str | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment to an empty active center state."""

        if self.config.env.initialization_mode != "random_legal":
            raise ValueError("Version 1 only supports initialization_mode='random_legal'.")
        self.rng = np.random.default_rng(self.config.project.seed if seed is None else seed)
        self.active_mask = np.zeros(self.num_slots, dtype=bool)
        self.centers = sample_random_centers(self.rng, self.num_slots, dtype=self.dtype)
        self.step_idx = 0
        self._clear_terminal_cache()
        return self.observe()

    def _require_initialized(self) -> None:
        if self.active_mask is None or self.centers is None:
            raise RuntimeError("Call reset() before using the environment.")

    def _clear_terminal_cache(self) -> None:
        self.radii = None
        self.circles = None
        self.layout_score = None
        self.lp_status = None

    def _ensure_terminal_cache(self) -> None:
        """Compute and cache LP-derived terminal circles if all slots are active."""

        self._require_initialized()
        if not bool(np.all(self.active_mask)):
            self._clear_terminal_cache()
            return
        if self.circles is not None and self.radii is not None and self.layout_score is not None:
            return
        result = solve_radii_for_centers(
            self.centers,
            radius_max=float(self.config.safety.radius_max),
            feasibility_tol=float(self.config.safety.feasibility_tol),
        )
        self.circles = result.circles.astype(self.dtype, copy=False)
        self.radii = self.circles[:, 2].astype(self.dtype, copy=True)
        self.layout_score = float(result.sum_r)
        self.lp_status = result.message if result.success else f"failed:{result.status}:{result.message}"

    def observe(self) -> dict[str, Any]:
        """Return a copy-safe observation of the current environment state."""

        self._require_initialized()
        if bool(np.all(self.active_mask)):
            self._ensure_terminal_cache()
        return {
            "active_mask": self.active_mask.copy(),
            "centers": self.centers.copy(),
            "step_idx": int(self.step_idx),
            "radii": None if self.radii is None else self.radii.copy(),
            "circles": None if self.circles is None else self.circles.copy(),
            "layout_score": None if self.layout_score is None else float(self.layout_score),
            "lp_status": self.lp_status,
            # Backward-compatible scalar name for rollout summaries.
            "sum_active_r": 0.0 if self.layout_score is None else float(self.layout_score),
        }

    def resolve(self, proposal: np.ndarray) -> np.ndarray:
        """Resolve a full `[K, 2]` center proposal against environment safety rules."""

        self._require_initialized()
        proposal = np.asarray(proposal, dtype=self.dtype)
        if proposal.shape != (self.num_slots, 2):
            raise ValueError(f"proposal must have shape ({self.num_slots}, 2), got {proposal.shape}.")
        resolved = self.centers.copy()
        masked = ~self.active_mask
        if np.any(masked):
            resolved[masked] = normalize_centers(proposal[masked], self.config.safety).astype(
                self.dtype,
                copy=False,
            )
        return self._apply_center_conflict_policy(resolved)

    def _find_duplicate_mask(self, centers: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
        eps = float(self.config.env.center_duplicate_eps)
        duplicate = np.zeros(candidate_ids.shape[0], dtype=bool)
        if eps <= 0.0 or candidate_ids.size == 0:
            return duplicate
        active_ids = np.flatnonzero(self.active_mask)
        for local_idx, slot_id in enumerate(candidate_ids):
            compare_ids = np.concatenate((active_ids, candidate_ids[:local_idx]))
            if compare_ids.size == 0:
                continue
            distances = np.linalg.norm(centers[compare_ids] - centers[int(slot_id)], axis=1)
            duplicate[local_idx] = bool(np.any(distances <= eps))
        return duplicate

    def _apply_center_conflict_policy(self, centers: np.ndarray) -> np.ndarray:
        policy = str(self.config.env.center_conflict_policy)
        if policy == "none":
            return centers
        candidate_ids = np.flatnonzero(~self.active_mask).astype(np.int64, copy=False)
        if candidate_ids.size == 0:
            return centers

        resolved = centers.copy()
        duplicate = self._find_duplicate_mask(resolved, candidate_ids)
        if not np.any(duplicate):
            return resolved
        if policy == "reject":
            return resolved

        eps = max(float(self.config.env.center_duplicate_eps), 1.0e-12)
        for slot_id in candidate_ids[duplicate]:
            if policy == "jitter":
                resolved[int(slot_id)] = np.clip(
                    resolved[int(slot_id)] + self.rng.normal(scale=eps, size=2),
                    0.0,
                    1.0,
                )
            elif policy == "resample":
                resolved[int(slot_id)] = sample_random_centers(self.rng, 1, dtype=self.dtype)[0]
            else:
                raise ValueError(f"unsupported center conflict policy: {policy}")
        return resolved

    def build_round_cache(self, resolved_proposal: np.ndarray, candidate_mask: np.ndarray | None = None) -> RoundCache:
        """Build the temporary candidate cache for one environment step."""

        self._require_initialized()
        free_mask = ~self.active_mask
        if candidate_mask is not None:
            candidate = np.asarray(candidate_mask, dtype=bool)
            if candidate.shape != (self.num_slots,):
                raise ValueError(f"candidate_mask must have shape ({self.num_slots},).")
            free_mask &= candidate
        cand_ids = np.flatnonzero(free_mask).astype(np.int64, copy=False)
        cand_centers = resolved_proposal[cand_ids].astype(self.dtype, copy=True)
        alive = np.ones(cand_ids.shape[0], dtype=bool)
        if str(self.config.env.center_conflict_policy) == "reject":
            alive &= ~self._find_duplicate_mask(resolved_proposal, cand_ids)
        return RoundCache(cand_ids=cand_ids, cand_centers=cand_centers, alive=alive)

    def _ordered_candidate_indices(self, round_cache: RoundCache) -> np.ndarray:
        alive = np.flatnonzero(round_cache.alive).astype(np.int64, copy=False)
        if str(self.config.env.commit_order) == "slot_index":
            order = np.argsort(round_cache.cand_ids[alive], kind="stable")
            return alive[order]
        if str(self.config.env.commit_order) == "random":
            return self.rng.permutation(alive)
        raise ValueError(f"unsupported commit_order: {self.config.env.commit_order}")

    def compute_gain(self, round_cache: RoundCache, local_idx: int) -> CommitDecision | None:
        """Return a center-only commit decision for one candidate."""

        if not round_cache.alive[local_idx]:
            return None
        return CommitDecision(
            slot_id=int(round_cache.cand_ids[local_idx]),
            center=round_cache.cand_centers[local_idx].copy(),
        )

    def commit_round(self, round_cache: RoundCache) -> list[CommitDecision]:
        """Select up to `activation_budget` center candidates for commitment."""

        commits: list[CommitDecision] = []
        budget = min(int(self.config.env.activation_budget), int(round_cache.cand_ids.shape[0]))
        for local_idx in self._ordered_candidate_indices(round_cache)[:budget]:
            decision = self.compute_gain(round_cache, int(local_idx))
            if decision is None:
                continue
            commits.append(decision)
            round_cache.alive[int(local_idx)] = False
        return commits

    def apply_commits(self, commits: list[CommitDecision]) -> float:
        """Apply committed centers to the persistent environment state."""

        self._require_initialized()
        self._clear_terminal_cache()
        for commit in commits:
            self.centers[commit.slot_id] = commit.center
            self.active_mask[commit.slot_id] = True
        if bool(np.all(self.active_mask)):
            self._ensure_terminal_cache()
            return float(self.layout_score or 0.0)
        return 0.0

    def _greedy_fill_info(self, *, success: bool, filled_ids: list[int]) -> dict[str, Any]:
        self._require_initialized()
        return {
            "success": bool(success),
            "filled_ids": list(filled_ids),
            "remaining_masked_ids": np.flatnonzero(~self.active_mask).astype(int).tolist(),
            "layout_score": None if self.layout_score is None else float(self.layout_score),
            "lp_status": self.lp_status,
        }

    def greedy_fill_masked_slots(self, *, grid_size: int | None = None) -> dict[str, Any]:
        """Fill all masked slots with random centers and score the terminal layout."""

        del grid_size
        self._require_initialized()
        masked_ids = np.flatnonzero(~self.active_mask).astype(np.int64, copy=False)
        filled_ids: list[int] = []
        for slot_id in masked_ids:
            self.centers[int(slot_id)] = sample_random_centers(self.rng, 1, dtype=self.dtype)[0]
            self.active_mask[int(slot_id)] = True
            filled_ids.append(int(slot_id))
        self._ensure_terminal_cache()
        return self._greedy_fill_info(success=bool(np.all(self.active_mask)), filled_ids=filled_ids)

    def step(
        self,
        proposal: np.ndarray,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Advance the environment by one center proposal step."""

        self._require_initialized()
        resolved = self.resolve(proposal)
        masked = ~self.active_mask
        if np.any(masked):
            self.centers[masked] = resolved[masked]

        round_cache = self.build_round_cache(self.centers, candidate_mask=candidate_mask)
        commits = self.commit_round(round_cache)
        reward = self.apply_commits(commits)
        self.step_idx += 1

        done = bool(self.step_idx >= self.config.problem.max_steps)
        if self.config.env.done_on_all_active:
            done = done or bool(np.all(self.active_mask))

        info = {
            "committed_ids": [commit.slot_id for commit in commits],
            "num_committed": len(commits),
            "layout_score": None if self.layout_score is None else float(self.layout_score),
            "lp_status": self.lp_status,
            "candidate_count": int(round_cache.cand_ids.shape[0]),
        }
        return self.observe(), float(reward), done, info

    def snapshot(self) -> EnvSnapshot:
        """Return a full environment snapshot for later restore."""

        self._require_initialized()
        return EnvSnapshot(
            active_mask=self.active_mask.copy(),
            centers=self.centers.copy(),
            step_idx=int(self.step_idx),
            radii=None if self.radii is None else self.radii.copy(),
            circles=None if self.circles is None else self.circles.copy(),
            layout_score=None if self.layout_score is None else float(self.layout_score),
            lp_status=self.lp_status,
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, snapshot: EnvSnapshot) -> dict[str, Any]:
        """Restore the environment from a previous snapshot."""

        self.active_mask = np.asarray(snapshot.active_mask, dtype=bool).copy()
        self.centers = np.asarray(snapshot.centers, dtype=self.dtype).copy()
        self.step_idx = int(snapshot.step_idx)
        self.radii = None if snapshot.radii is None else np.asarray(snapshot.radii, dtype=self.dtype).copy()
        self.circles = None if snapshot.circles is None else np.asarray(snapshot.circles, dtype=self.dtype).copy()
        self.layout_score = None if snapshot.layout_score is None else float(snapshot.layout_score)
        self.lp_status = snapshot.lp_status
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
        return self.observe()

    def render(self, **kwargs: Any) -> Any:
        """Render terminal circles when available; otherwise render zero-radius center markers."""

        self._require_initialized()
        circles = self.circles
        if circles is None:
            radii = np.zeros(self.num_slots, dtype=self.dtype)
            circles = np.column_stack((self.centers, radii))
        return render_state(circles, self.active_mask, self.config.render, **kwargs)
