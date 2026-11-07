from __future__ import annotations

import random
from dataclasses import replace
from statistics import mean, pstdev
from typing import Any

import torch

from env import NoThreeInLineEnv
from geometry import Point
from models import (
    DecodingConfig,
    SelectionStrategyConfig,
    current_point_count_from_slot_states,
    load_decoding_config,
    logits_to_proposal_slots,
    slot_states_to_state_ids,
)

from .config import RolloutEvalConfig, SupervisedConfig
from .corruption import corrupt_clean_state_at_step
from .types import EncodedSolutionRecord


def _compute_k_for_step(step_id: int, total_steps: int, rollout_cfg: RolloutEvalConfig) -> int:
    """Return the rollout decoding width for one denoising step.

    ``step_id`` is the current denoising step in ``[0, total_steps - 1]`` and
    ``rollout_cfg`` supplies either a fixed width or a front-loaded schedule.
    The function returns a non-negative integer, does not mutate state, and
    assumes the rollout config has already been validated.
    """
    if rollout_cfg.k_schedule_mode == "fixed":
        return rollout_cfg.fixed_k

    raw_k = rollout_cfg.k_min + step_id * (rollout_cfg.k_max - rollout_cfg.k_min) / max(total_steps - 1, 1)
    return max(rollout_cfg.k_min, min(rollout_cfg.k_max, round(raw_k)))


def _state_ids_to_reset_args(state_ids: list[int]) -> tuple[list[Point], dict[Point, int]]:
    """Convert a partially visible slot state into env reset arguments.

    ``state_ids`` is a length-``2n`` list where ``0`` means ``[MASK]`` and
    ``y + 1`` stores a visible point. The function returns ``(init_points,
    init_slot_assignment)``, does not mutate the input list, and assumes the
    state ids already follow the column-pair slot convention.
    """
    init_points: list[Point] = []
    init_slot_assignment: dict[Point, int] = {}
    for slot_id, state_id in enumerate(state_ids):
        if state_id == 0:
            continue
        point = (slot_id // 2, state_id - 1)
        init_points.append(point)
        init_slot_assignment[point] = slot_id
    return sorted(init_points), init_slot_assignment


def _resolve_device(
    model: torch.nn.Module,
    config: SupervisedConfig,
    device: str | torch.device | None,
) -> torch.device:
    """Resolve the device used for rollout model inference.

    ``device`` overrides the inferred device when provided. The function
    returns a ``torch.device``, does not mutate the model, and assumes the
    model parameters already live on the chosen device when a parameter exists.
    """
    if device is not None:
        return torch.device(device)

    first_parameter = next(model.parameters(), None)
    if first_parameter is not None:
        return first_parameter.device
    return torch.device(config.runtime.device)


def _build_env(config: SupervisedConfig) -> NoThreeInLineEnv:
    """Construct a fresh environment for one rollout episode.

    ``config`` supplies the grid size, number of slots, and proposal-search
    options. The function returns a new environment instance, does not mutate
    caller state, and assumes rollout owns the denoising-step budget rather
    than the environment's ``max_steps`` field.
    """
    return NoThreeInLineEnv(
        n=config.n,
        num_slots=2 * config.n,
        max_steps=None,
        proposal_search_config={
            "num_search_attempts": config.rollout_eval.env_num_search_attempts,
            "allow_existing_deletions": config.rollout_eval.env_allow_existing_deletions,
        },
    )


def _infer_proposal_slots(
    model: torch.nn.Module,
    env: NoThreeInLineEnv,
    denoise_step_id: int,
    decoding_config: DecodingConfig,
    k_override: int | None,
    device: torch.device,
) -> list:
    """Run the existing model-decoding path and return proposal slots.

    ``env`` provides the current slot states, ``denoise_step_id`` controls
    rollout scheduling, and the model receives current filled-slot progress as
    its step embedding input. The function returns one env-compatible
    ``ProposalSlot`` list, does not mutate the model or environment, and
    assumes the decoder should remain the single proposal-generation path.
    """
    slot_state_ids = slot_states_to_state_ids(env.slot_states, env.n).unsqueeze(0).to(device)
    current_point_count = current_point_count_from_slot_states(env.slot_states)
    step_ids = torch.tensor([current_point_count], dtype=torch.long, device=device)
    current_point_counts = torch.tensor([float(current_point_count)], dtype=torch.float32, device=device)

    outputs = model(slot_state_ids, step_ids, current_point_counts)
    proposal_batches = logits_to_proposal_slots(
        outputs["state_logits"],
        slot_state_ids,
        k=k_override,
        decoding_config=decoding_config,
    )
    return proposal_batches[0]


def _run_rollout_step(
    model: torch.nn.Module,
    env: NoThreeInLineEnv,
    denoise_step_id: int,
    total_steps: int,
    decoding_config: DecodingConfig,
    rollout_cfg: RolloutEvalConfig,
    device: torch.device,
) -> dict[str, object]:
    """Run one inference-plus-env step and collect step-level diagnostics.

    The function reuses the existing inference path and environment transition.
    It returns a new metrics dictionary, mutates the environment through
    ``env.step(...)``, and assumes rollout stopping is based on accepted
    proposals rather than score thresholds.
    """
    point_count_before = int(env.observe()["num_selected"])
    k_override = _compute_k_for_step(denoise_step_id, total_steps, rollout_cfg)
    proposal_slots = _infer_proposal_slots(model, env, denoise_step_id, decoding_config, k_override, device)
    observation, reward, done, info = env.step(proposal_slots)

    num_accepted = len(info["accepted_proposal_points"])
    num_deleted_existing = len(info["deleted_existing_points"])
    num_proposed_slots = len(proposal_slots)
    num_invalid_or_rejected = max(num_proposed_slots - num_accepted, 0)
    net_gain = num_accepted - num_deleted_existing

    return {
        "denoise_step_id": denoise_step_id,
        "point_count_before": point_count_before,
        "point_count_after": int(observation["num_selected"]),
        "num_proposed_slots": num_proposed_slots,
        "num_resolved": int(info["num_resolved"]),
        "num_accepted": num_accepted,
        "num_invalid_or_rejected": num_invalid_or_rejected,
        "num_deleted_existing": num_deleted_existing,
        "net_gain": net_gain,
        "effective_recovery": num_accepted > 0,
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


def _merge_decoding_config(
    base_config: DecodingConfig,
    override: dict[str, object],
) -> DecodingConfig:
    """Merge one decode-ablation override onto the base decoding config.

    ``override`` may replace fields inside ``slot_selection`` and
    ``candidate_selection`` only. The function returns a new ``DecodingConfig``
    without mutating the base config and assumes override values are JSON/YAML
    compatible scalars.
    """
    allowed_top_level = {"slot_selection", "candidate_selection"}
    unknown_keys = set(override) - allowed_top_level
    if unknown_keys:
        raise ValueError(f"unsupported decode ablation keys: {sorted(unknown_keys)}")

    slot_override = override.get("slot_selection", {})
    candidate_override = override.get("candidate_selection", {})
    if not isinstance(slot_override, dict) or not isinstance(candidate_override, dict):
        raise ValueError("decode ablation slot_selection and candidate_selection must be mappings")

    slot_selection = SelectionStrategyConfig(
        strategy=str(slot_override.get("strategy", base_config.slot_selection.strategy)),
        top_k=int(slot_override.get("top_k", base_config.slot_selection.top_k)),
        top_n=(
            base_config.slot_selection.top_n
            if "top_n" not in slot_override
            else None if slot_override["top_n"] is None else int(slot_override["top_n"])
        ),
    )
    candidate_selection = SelectionStrategyConfig(
        strategy=str(candidate_override.get("strategy", base_config.candidate_selection.strategy)),
        top_k=int(candidate_override.get("top_k", base_config.candidate_selection.top_k)),
        top_n=(
            base_config.candidate_selection.top_n
            if "top_n" not in candidate_override
            else None if candidate_override["top_n"] is None else int(candidate_override["top_n"])
        ),
    )
    return DecodingConfig(slot_selection=slot_selection, candidate_selection=candidate_selection)


def _summarize_episodes(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate episode-level rollout results into simple summary statistics.

    ``episodes`` is a list of per-episode dictionaries produced by the rollout
    loops. The function returns a new summary dictionary, does not mutate the
    episode records, and assumes missing episodes should yield zero-valued
    aggregate metrics.
    """
    if not episodes:
        return {
            "episodes": [],
            "num_episodes": 0,
            "mean_final_point_count": 0.0,
            "std_final_point_count": 0.0,
            "mean_gain": 0.0,
            "std_gain": 0.0,
            "mean_steps_used": 0.0,
            "mean_invalid_total": 0.0,
            "mean_accept_rate": 0.0,
        }

    final_point_counts = [int(episode["final_point_count"]) for episode in episodes]
    gains = [int(episode["gain"]) for episode in episodes]
    steps_used = [int(episode["steps_used"]) for episode in episodes]
    invalid_totals = [int(episode["invalid_total"]) for episode in episodes]
    accept_rates = [float(episode["accept_rate"]) for episode in episodes]
    return {
        "episodes": episodes,
        "num_episodes": len(episodes),
        "mean_final_point_count": mean(final_point_counts),
        "std_final_point_count": pstdev(final_point_counts),
        "mean_gain": mean(gains),
        "std_gain": pstdev(gains),
        "mean_steps_used": mean(steps_used),
        "mean_invalid_total": mean(invalid_totals),
        "mean_accept_rate": mean(accept_rates),
    }


def _run_episode_loop(
    model: torch.nn.Module,
    env: NoThreeInLineEnv,
    start_step_id: int,
    decoding_config: DecodingConfig,
    config: SupervisedConfig,
    device: torch.device,
    *,
    mode: str,
    masked_ratio_actual: float | None = None,
) -> dict[str, object]:
    """Run one multi-step rollout episode and summarize its metrics.

    ``env`` is already initialized at the desired start state and
    ``start_step_id`` is the first denoising step passed to the model. The
    function returns one episode summary dictionary, mutates the environment
    through repeated ``env.step(...)`` calls, and assumes stopping occurs after
    a step with zero accepted proposals or after the denoising budget expires.
    """
    rollout_cfg = config.rollout_eval
    start_point_count = int(env.observe()["num_selected"])
    step_records: list[dict[str, object]] = []
    accepted_total = 0
    invalid_total = 0
    proposed_total = 0
    repair_deletions_total = 0
    net_gain_total = 0
    steps_used = 0
    current_step_id = start_step_id
    step_budget = (
        min(rollout_cfg.max_denoise_steps, start_step_id + 1)
        if rollout_cfg.max_denoise_steps > 0
        else start_step_id + 1
    )

    for _ in range(step_budget):
        step_result = _run_rollout_step(
            model,
            env,
            current_step_id,
            config.corruption.total_steps,
            decoding_config,
            rollout_cfg,
            device,
        )
        steps_used += 1
        accepted_total += int(step_result["num_accepted"])
        invalid_total += int(step_result["num_invalid_or_rejected"])
        proposed_total += int(step_result["num_proposed_slots"])
        repair_deletions_total += int(step_result["num_deleted_existing"])
        net_gain_total += int(step_result["net_gain"])

        if rollout_cfg.record_step_metrics:
            step_records.append(step_result)
        if int(step_result["num_accepted"]) == 0:
            break
        if current_step_id == 0:
            break
        current_step_id -= 1

    final_point_count = int(env.observe()["num_selected"])
    episode_summary: dict[str, object] = {
        "mode": mode,
        "start_point_count": start_point_count,
        "final_point_count": final_point_count,
        "gain": final_point_count - start_point_count,
        "steps_used": steps_used,
        "accepted_total": accepted_total,
        "invalid_total": invalid_total,
        "accept_rate": accepted_total / max(1, proposed_total),
        "repair_deletions_total": repair_deletions_total,
        "net_gain_total": net_gain_total,
    }
    if mode == "partial_state":
        episode_summary["start_step_id"] = start_step_id
        episode_summary["masked_ratio_actual"] = 0.0 if masked_ratio_actual is None else masked_ratio_actual
    if rollout_cfg.record_step_metrics:
        episode_summary["step_records"] = step_records
    return episode_summary


def run_rollout_evaluation(
    model: torch.nn.Module,
    config: SupervisedConfig,
    eval_records: list[EncodedSolutionRecord] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Run configured standard, partial-state, and decode-ablation rollouts.

    ``model`` is the trained denoiser, ``config`` provides rollout settings,
    and ``eval_records`` optionally supplies clean solutions for partial-state
    starts. The function returns nested summary dictionaries, mutates only
    temporary environments during rollout, and assumes proposal generation must
    reuse the existing model-forward plus decoder path.
    """
    rollout_cfg = config.rollout_eval
    if not rollout_cfg.enabled:
        return {"skipped_reason": "disabled"}
    if rollout_cfg.num_episodes == 0:
        return {"skipped_reason": "num_episodes_zero"}

    resolved_device = _resolve_device(model, config, device)
    base_decoding_config = load_decoding_config(config.config_path)
    results: dict[str, object] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if "standard" in rollout_cfg.eval_types:
                standard_episodes: list[dict[str, object]] = []
                for _ in range(rollout_cfg.num_episodes):
                    env = _build_env(config)
                    env.reset()
                    standard_episodes.append(
                        _run_episode_loop(
                            model,
                            env,
                            start_step_id=config.corruption.total_steps - 1,
                            decoding_config=base_decoding_config,
                            config=config,
                            device=resolved_device,
                            mode="standard",
                        )
                    )
                results["standard"] = _summarize_episodes(standard_episodes)

            if "partial_state" in rollout_cfg.eval_types:
                if not eval_records:
                    results["partial_state"] = {"skipped_reason": "no_eval_records"}
                else:
                    corruption_config = config.corruption
                    if rollout_cfg.partial_corruption_mode is not None:
                        corruption_config = replace(
                            corruption_config,
                            corruption_mode=rollout_cfg.partial_corruption_mode,
                        )

                    partial_episodes: list[dict[str, object]] = []
                    for episode_index in range(rollout_cfg.num_episodes):
                        rng = random.Random(rollout_cfg.seed + episode_index)
                        start_step_id = rollout_cfg.partial_start_step_ids[
                            episode_index % len(rollout_cfg.partial_start_step_ids)
                        ]
                        record = eval_records[rng.randrange(len(eval_records))]
                        corruption_result = corrupt_clean_state_at_step(
                            record,
                            start_step_id,
                            rng,
                            corruption_config,
                        )
                        init_points, init_slot_assignment = _state_ids_to_reset_args(
                            corruption_result.input_state_ids
                        )
                        env = _build_env(config)
                        env.reset(
                            init_points=init_points,
                            init_slot_assignment=init_slot_assignment,
                        )
                        partial_episodes.append(
                            _run_episode_loop(
                                model,
                                env,
                                start_step_id=start_step_id,
                                decoding_config=base_decoding_config,
                                config=config,
                                device=resolved_device,
                                mode="partial_state",
                                masked_ratio_actual=corruption_result.masked_ratio_actual,
                            )
                        )

                    partial_summary = _summarize_episodes(partial_episodes)
                    by_start_step: dict[int, dict[str, object]] = {}
                    for start_step_id in rollout_cfg.partial_start_step_ids:
                        grouped = [
                            episode
                            for episode in partial_episodes
                            if int(episode["start_step_id"]) == start_step_id
                        ]
                        by_start_step[start_step_id] = _summarize_episodes(grouped)
                    partial_summary["by_start_step"] = by_start_step
                    results["partial_state"] = partial_summary

            if "decode_ablation" in rollout_cfg.eval_types:
                ablation_results: dict[str, dict[str, object]] = {}
                for ablation_name, override in rollout_cfg.decode_ablation_configs.items():
                    decoding_config = _merge_decoding_config(base_decoding_config, override)
                    ablation_episodes: list[dict[str, object]] = []
                    for _ in range(rollout_cfg.num_episodes):
                        env = _build_env(config)
                        env.reset()
                        ablation_episodes.append(
                            _run_episode_loop(
                                model,
                                env,
                                start_step_id=config.corruption.total_steps - 1,
                                decoding_config=decoding_config,
                                config=config,
                                device=resolved_device,
                                mode="standard",
                            )
                        )
                    ablation_results[ablation_name] = _summarize_episodes(ablation_episodes)
                results["decode_ablation"] = {"ablations": ablation_results}
    finally:
        if was_training:
            model.train()
    return results
