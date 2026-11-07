"""Multi-round outer-loop inference utilities for version 1.

This module belongs to the controller/pipeline layer. It coordinates:
- observing the current environment state
- converting centers to latent state
- calling the DDPM or FM proposal samplers
- extracting lightweight raw signals
- reducing the `N` sampled center proposals to one environment-facing proposal

The environment remains authoritative for commit decisions. The model side only
produces proposal samples and optional raw per-slot signals.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import torch

from circle_pack.config import ProjectConfig
from models.sampling import sample_ddpm_proposals, sample_fm_proposals, sample_random_proposals
from pipelines.priority import (
    compute_cross_round_drift,
    compute_slot_dispersion,
    demo_priority_score,
    reduce_fm_aux_signals,
)
from pipelines.types import OuterLoopRecord, ProposalBundle


CandidateSelector = Callable[[dict[str, Any], ProposalBundle, dict[str, torch.Tensor], ProjectConfig], Any]


def _to_model_inputs(
    env_observation: dict[str, Any],
    model: Any,
    center_to_latent_fn: Callable[[Any], Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one environment observation to model-side batch tensors."""

    centers_np = np.asarray(env_observation["centers"], dtype=np.float64)
    latent_np = np.asarray(center_to_latent_fn(centers_np), dtype=np.float64)
    fixed_np = np.asarray(env_observation["active_mask"], dtype=bool)

    slot_centers = torch.as_tensor(centers_np, device=model.device, dtype=model.dtype).unsqueeze(0)
    slot_latent = torch.as_tensor(latent_np, device=model.device, dtype=model.dtype).unsqueeze(0)
    fixed_mask = torch.as_tensor(fixed_np, device=model.device, dtype=torch.bool).unsqueeze(0)
    return slot_latent, slot_centers, fixed_mask


def _representative_center_sample(
    center_samples: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Reduce `[N, K, 2]` center samples to one representative `[K, 2]` sample."""

    if reduction == "first_sample":
        return center_samples[0]
    if reduction == "mean":
        return center_samples.mean(dim=0)
    raise ValueError("representative reduction must be 'first_sample' or 'mean'.")


def _select_candidate_mask(
    env: Any,
    observation: dict[str, Any],
    proposal_bundle: ProposalBundle,
    raw_signals: dict[str, torch.Tensor],
    cfg: ProjectConfig,
    candidate_selector: CandidateSelector | None,
) -> np.ndarray:
    """Choose which free slots remain candidate-eligible in the current round."""

    free_mask_np = ~np.asarray(observation["active_mask"], dtype=bool)
    if candidate_selector is not None:
        candidate_mask = np.asarray(candidate_selector(observation, proposal_bundle, raw_signals, cfg), dtype=bool)
        if candidate_mask.shape != free_mask_np.shape:
            raise ValueError("candidate_selector must return a boolean mask of shape [K].")
        return candidate_mask & free_mask_np

    if cfg.outer_loop.candidate_mode == "all_free":
        return free_mask_np.copy()

    score = demo_priority_score(
        raw_signals["dispersion"],
        drift=raw_signals.get("drift"),
        fm_signal=raw_signals.get("fm_signal"),
        free_mask=torch.as_tensor(free_mask_np),
        cfg=cfg,
    )
    budget = min(int(env.config.env.activation_budget), int(free_mask_np.sum()))
    candidate_mask = np.zeros_like(free_mask_np, dtype=bool)
    if budget <= 0:
        return candidate_mask

    score_np = score.detach().cpu().numpy()
    free_indices = np.flatnonzero(free_mask_np)
    ranked = free_indices[np.argsort(score_np[free_indices])[::-1]]
    candidate_mask[ranked[:budget]] = True
    return candidate_mask


def _restore_blocked_drafts(
    env: Any,
    snapshot_before: Any,
    blocked_free_mask: np.ndarray,
) -> None:
    """Restore masked draft rows for controller-blocked slots after `env.step()`.

    The current environment consumes a full physical proposal array for all masked
    slots. To keep candidate masking explicit without redesigning the environment,
    the controller temporarily writes blocked free slots with zero radius so they
    cannot be committed in the current round, then restores their previous draft
    rows if they remain masked after the step.
    """

    if not np.any(blocked_free_mask):
        return
    still_masked = blocked_free_mask & (~np.asarray(env.active_mask, dtype=bool))
    if np.any(still_masked):
        env.centers[still_masked] = np.asarray(snapshot_before.centers)[still_masked]


def _build_env_proposal(
    env: Any,
    observation: dict[str, Any],
    representative_sample: torch.Tensor,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single environment-facing proposal and blocked-slot mask.

    Returns
    -------
    proposal:
        Array of shape `[K, 2]` to submit to `env.step`.
    blocked_free_mask:
        Boolean mask over free slots that were intentionally blocked from being
        updated in this round.
    """

    current_centers = np.asarray(observation["centers"], dtype=env.dtype)
    fixed_mask = np.asarray(observation["active_mask"], dtype=bool)
    free_mask = ~fixed_mask
    candidate_mask = np.asarray(candidate_mask, dtype=bool) & free_mask
    blocked_free_mask = free_mask & (~candidate_mask)

    proposal = current_centers.copy()
    representative_np = representative_sample.detach().cpu().numpy().astype(env.dtype, copy=False)
    proposal[candidate_mask] = representative_np[candidate_mask]
    proposal[blocked_free_mask] = current_centers[blocked_free_mask]
    proposal[fixed_mask] = current_centers[fixed_mask]
    return proposal, blocked_free_mask


def _sampler_for_mode(mode: str):
    """Return the model-side sampler function for the requested inference mode."""

    if mode == "ddpm":
        return sample_ddpm_proposals
    if mode == "fm":
        return sample_fm_proposals
    raise ValueError("mode must be 'ddpm' or 'fm'.")


def _run_outer_round_impl(
    env: Any,
    model: Any,
    mode: str,
    cfg: ProjectConfig,
    latent_to_center_fn: Callable[[Any], Any],
    center_to_latent_fn: Callable[[Any], Any],
    *,
    prev_center_samples: torch.Tensor | None = None,
    candidate_selector: CandidateSelector | None = None,
    round_idx: int = 0,
    ddpm_num_steps: int | None = None,
    ddpm_tau_start: float | None = None,
    fm_num_steps: int | None = None,
    fm_t_start: float = 0.0,
    fm_t_end: float = 1.0,
) -> tuple[OuterLoopRecord, torch.Tensor]:
    """Internal implementation shared by the standard and segmented outer rounds."""

    observation = env.observe()

    # The random completion operator replaces the learned sampler outright: it
    # never builds model inputs, so a random-operator run needs no model at all.
    if str(cfg.patternboost.generation.completion_operator) != "learned":
        proposal_bundle = sample_random_proposals(
            np.asarray(observation["centers"], dtype=np.float64),
            np.asarray(observation["active_mask"], dtype=bool),
            cfg,
        )
    elif mode == "ddpm":
        slot_latent, slot_centers, fixed_mask = _to_model_inputs(observation, model, center_to_latent_fn)
        proposal_bundle = sample_ddpm_proposals(
            model,
            slot_latent,
            slot_centers,
            fixed_mask,
            cfg,
            latent_to_center_fn,
            num_steps=ddpm_num_steps,
            tau_start=ddpm_tau_start,
        )
    else:
        slot_latent, slot_centers, fixed_mask = _to_model_inputs(observation, model, center_to_latent_fn)
        proposal_bundle = sample_fm_proposals(
            model,
            slot_latent,
            slot_centers,
            fixed_mask,
            cfg,
            latent_to_center_fn,
            num_steps=fm_num_steps,
            t_start=fm_t_start,
            t_end=fm_t_end,
        )

    dispersion = compute_slot_dispersion(proposal_bundle.center_samples)
    raw_signals: dict[str, torch.Tensor] = {"dispersion": dispersion}
    if prev_center_samples is not None:
        raw_signals["drift"] = compute_cross_round_drift(prev_center_samples, proposal_bundle.center_samples)
    if mode == "fm" and proposal_bundle.aux:
        reduced_aux = reduce_fm_aux_signals(proposal_bundle.aux, reduction="mean_over_samples")
        raw_signals.update(reduced_aux)
        if "short_displacement" in reduced_aux:
            raw_signals["fm_signal"] = reduced_aux["short_displacement"]
        elif "terminal_velocity_norm" in reduced_aux:
            raw_signals["fm_signal"] = reduced_aux["terminal_velocity_norm"]

    candidate_mask = _select_candidate_mask(
        env,
        observation,
        proposal_bundle,
        raw_signals,
        cfg,
        candidate_selector,
    )
    representative = _representative_center_sample(
        proposal_bundle.center_samples,
        cfg.outer_loop.representative_reduction,
    )
    raw_signals["candidate_mask"] = torch.as_tensor(candidate_mask, dtype=torch.bool)

    snapshot_before = env.snapshot()
    env_proposal, blocked_free_mask = _build_env_proposal(env, observation, representative, candidate_mask)
    next_obs, reward, done, info = env.step(env_proposal, candidate_mask=candidate_mask)
    _restore_blocked_drafts(env, snapshot_before, blocked_free_mask)
    next_obs = env.observe()

    env_info = dict(info)
    env_info.update(
        {
            "reward": float(reward),
            "done": bool(done),
            "candidate_mask": candidate_mask.copy(),
            "selected_slot_ids": np.flatnonzero(candidate_mask).tolist(),
            "representative_reduction": cfg.outer_loop.representative_reduction,
            "round_idx": int(round_idx),
            "post_step_sum_active_r": float(next_obs["sum_active_r"]),
        }
    )
    record = OuterLoopRecord(
        round_idx=int(round_idx),
        proposal_bundle=proposal_bundle,
        raw_signals=raw_signals,
        env_info=env_info,
    )
    return record, proposal_bundle.center_samples


def run_outer_round(
    env: Any,
    model: Any,
    mode: str,
    cfg: ProjectConfig,
    latent_to_center_fn: Callable[[Any], Any],
    center_to_latent_fn: Callable[[Any], Any],
    prev_center_samples: torch.Tensor | None = None,
    candidate_selector: CandidateSelector | None = None,
) -> tuple[OuterLoopRecord, torch.Tensor]:
    """Run one whole outer denoising/refinement round.

    This is the main version-1 inference path. It observes the current env state,
    samples `N` proposals, computes lightweight controller-side signals, reduces
    the proposals to one representative sample, submits that sample to the env,
    and returns a record plus the current round's sample bank for drift tracking.
    """

    return _run_outer_round_impl(
        env,
        model,
        mode,
        cfg,
        latent_to_center_fn,
        center_to_latent_fn,
        prev_center_samples=prev_center_samples,
        candidate_selector=candidate_selector,
        round_idx=int(env.observe()["step_idx"]),
    )


def run_outer_loop_episode(
    env: Any,
    model: Any,
    mode: str,
    cfg: ProjectConfig,
    latent_to_center_fn: Callable[[Any], Any],
    center_to_latent_fn: Callable[[Any], Any],
    candidate_selector: CandidateSelector | None = None,
    max_outer_rounds: int | None = None,
) -> list[OuterLoopRecord]:
    """Run repeated outer rounds until the env is done or the round limit is met."""

    if getattr(env, "active_mask", None) is None or getattr(env, "centers", None) is None:
        env.reset()

    records: list[OuterLoopRecord] = []
    prev_center_samples: torch.Tensor | None = None
    round_limit = int(cfg.outer_loop.max_rounds if max_outer_rounds is None else max_outer_rounds)

    for round_idx in range(round_limit):
        record, prev_center_samples = _run_outer_round_impl(
            env,
            model,
            mode,
            cfg,
            latent_to_center_fn,
            center_to_latent_fn,
            prev_center_samples=prev_center_samples,
            candidate_selector=candidate_selector,
            round_idx=round_idx,
        )
        records.append(record)
        if bool(record.env_info.get("done", False)):
            break
    return records


def run_outer_round_segmented_demo(
    env: Any,
    model: Any,
    mode: str,
    cfg: ProjectConfig,
    latent_to_center_fn: Callable[[Any], Any],
    center_to_latent_fn: Callable[[Any], Any],
    prev_center_samples: torch.Tensor | None = None,
    candidate_selector: CandidateSelector | None = None,
) -> list[OuterLoopRecord]:
    """Run a minimal two-segment illustrative outer-round demo.

    This is not the main inference path. It exists only to show how future
    segmented coordination could be layered around the existing samplers.
    """

    split = float(cfg.outer_loop.segmented_split)
    records: list[OuterLoopRecord] = []

    if mode == "ddpm":
        first_steps = max(1, int(round(cfg.ddpm.num_steps * split)))
        second_steps = max(1, cfg.ddpm.num_steps - first_steps)
        record_a, prev_center_samples = _run_outer_round_impl(
            env,
            model,
            mode,
            cfg,
            latent_to_center_fn,
            center_to_latent_fn,
            prev_center_samples=prev_center_samples,
            candidate_selector=candidate_selector,
            round_idx=0,
            ddpm_num_steps=first_steps,
            ddpm_tau_start=cfg.ddpm.tau_start_infer,
        )
        records.append(record_a)
        if not bool(record_a.env_info.get("done", False)):
            record_b, _ = _run_outer_round_impl(
                env,
                model,
                mode,
                cfg,
                latent_to_center_fn,
                center_to_latent_fn,
                prev_center_samples=prev_center_samples,
                candidate_selector=candidate_selector,
                round_idx=1,
                ddpm_num_steps=second_steps,
                ddpm_tau_start=split,
            )
            records.append(record_b)
        return records

    if mode != "fm":
        raise ValueError("mode must be 'ddpm' or 'fm'.")

    first_steps = max(1, int(round(cfg.fm.num_steps * split)))
    second_steps = max(1, cfg.fm.num_steps - first_steps)
    record_a, prev_center_samples = _run_outer_round_impl(
        env,
        model,
        mode,
        cfg,
        latent_to_center_fn,
        center_to_latent_fn,
        prev_center_samples=prev_center_samples,
        candidate_selector=candidate_selector,
        round_idx=0,
        fm_num_steps=first_steps,
        fm_t_start=0.0,
        fm_t_end=split,
    )
    records.append(record_a)
    if not bool(record_a.env_info.get("done", False)):
        record_b, _ = _run_outer_round_impl(
            env,
            model,
            mode,
            cfg,
            latent_to_center_fn,
            center_to_latent_fn,
            prev_center_samples=prev_center_samples,
            candidate_selector=candidate_selector,
            round_idx=1,
            fm_num_steps=second_steps,
            fm_t_start=split,
            fm_t_end=1.0,
        )
        records.append(record_b)
    return records
