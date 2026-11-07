"""Center-only supervised data construction helpers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from circle_pack.collect import collect_episode
from circle_pack.config import ProjectConfig
from circle_pack.env import CirclePackingEnv
from circle_pack.geometry import center_to_latent_approx, latent_to_center_simple, normalize_centers
from circle_pack.policies import make_policy


def _ensure_project_config(cfg: ProjectConfig | None) -> ProjectConfig:
    return cfg if cfg is not None else ProjectConfig()


def _as_center_array(layout: np.ndarray | torch.Tensor | Any) -> np.ndarray:
    """Convert a center or legacy circle layout to `[K, 2]` centers."""

    array = np.asarray(layout, dtype=np.float64)
    if array.ndim != 2 or array.shape[-1] not in {2, 3}:
        raise ValueError("layouts must have shape [K, 2] or legacy [K, 3].")
    return array[:, :2].copy()


def _to_torch_example(example: dict[str, np.ndarray | float | int | bool], cfg: ProjectConfig) -> dict[str, torch.Tensor]:
    tensor_example: dict[str, torch.Tensor] = {}
    for key, value in example.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                tensor_example[key] = torch.from_numpy(value.astype(np.bool_, copy=False))
            elif np.issubdtype(value.dtype, np.integer):
                tensor_example[key] = torch.from_numpy(value.astype(np.int64, copy=False))
            else:
                tensor_example[key] = torch.as_tensor(value, dtype=cfg.torch_dtype)
        else:
            if isinstance(value, (int, np.integer)):
                tensor_example[key] = torch.as_tensor(value, dtype=torch.int64)
            else:
                tensor_example[key] = torch.as_tensor(value, dtype=cfg.torch_dtype)
    return tensor_example


def _ddpm_alpha_bars(num_steps: int, beta_start: float, beta_end: float) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    return np.cumprod(1.0 - betas, axis=0)


def _sample_inner_time(num_steps: int, rng: np.random.Generator) -> tuple[int, float]:
    step_idx = int(rng.integers(0, num_steps))
    denom = max(num_steps - 1, 1)
    return step_idx, float(step_idx / denom)


def assign_slots_random(
    target_centers: np.ndarray | torch.Tensor | Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly permute a full teacher center layout."""

    target_array = _as_center_array(target_centers)
    order = rng.permutation(target_array.shape[0])
    return target_array[order], order


def assign_slots_xy(target_centers: np.ndarray | torch.Tensor | Any) -> tuple[np.ndarray, np.ndarray]:
    """Sort a full teacher layout by x, then y."""

    target_array = _as_center_array(target_centers)
    order = np.lexsort((target_array[:, 1], target_array[:, 0]))
    return target_array[order], order


def match_target_to_draft_greedy(
    draft_centers: np.ndarray | torch.Tensor | Any,
    target_centers: np.ndarray | torch.Tensor | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily match target centers to draft slots by physical center distance."""

    draft_array = _as_center_array(draft_centers)
    target_array = _as_center_array(target_centers)
    if draft_array.shape[0] != target_array.shape[0]:
        raise ValueError("draft_centers and target_centers must contain the same number of slots.")

    num_slots = draft_array.shape[0]
    remaining_slots = set(range(num_slots))
    remaining_targets = set(range(num_slots))
    assignment = np.full(num_slots, -1, dtype=np.int64)
    cost = np.linalg.norm(draft_array[:, None, :] - target_array[None, :, :], axis=-1)
    while remaining_slots:
        best_slot = -1
        best_target = -1
        best_cost = np.inf
        for slot_idx in remaining_slots:
            for target_idx in remaining_targets:
                candidate_cost = float(cost[slot_idx, target_idx])
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_slot = slot_idx
                    best_target = target_idx
        assignment[best_slot] = best_target
        remaining_slots.remove(best_slot)
        remaining_targets.remove(best_target)
    return target_array[assignment], assignment


def build_synthetic_draft(
    target_latent: np.ndarray | torch.Tensor | Any,
    target_centers: np.ndarray | torch.Tensor | Any,
    mode: str,
    corruption_scale: float,
    rng: np.random.Generator,
    cfg: ProjectConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a synthetic draft latent/center pair from a clean target."""

    config = _ensure_project_config(cfg)
    target_latent_np = np.asarray(target_latent, dtype=np.float64)
    target_centers_np = _as_center_array(target_centers)
    if mode == "mixed":
        mode = rng.choice(["gaussian_latent", "center_jitter"]).item()
    if mode == "radius_shrink_and_jitter":
        mode = "center_jitter"

    if mode == "gaussian_latent":
        draft_latent = target_latent_np + rng.normal(scale=corruption_scale, size=target_latent_np.shape)
        draft_centers = latent_to_center_simple(draft_latent)
        return draft_latent, draft_centers
    if mode != "center_jitter":
        raise ValueError("build_synthetic_draft mode must be 'gaussian_latent', 'center_jitter', or 'mixed'.")

    draft_centers = target_centers_np.copy()
    draft_centers += rng.normal(scale=0.5 * corruption_scale, size=draft_centers.shape)
    draft_centers = normalize_centers(draft_centers, config.safety)
    draft_latent = center_to_latent_approx(draft_centers)
    return draft_latent, draft_centers


def build_jittered_draft(
    target_centers: np.ndarray | torch.Tensor | Any,
    fixed_mask: np.ndarray | torch.Tensor | Any,
    rng: np.random.Generator,
    cfg: ProjectConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a draft by perturbing only free center slots."""

    config = _ensure_project_config(cfg)
    target_centers_np = _as_center_array(target_centers)
    fixed = np.asarray(fixed_mask, dtype=bool)
    if fixed.shape != (target_centers_np.shape[0],):
        raise ValueError("fixed_mask must have shape [K] matching target_centers.")

    draft_centers = target_centers_np.copy()
    free_mask = ~fixed
    if np.any(free_mask):
        jitter_std = float(config.train.draft_center_jitter_std)
        if jitter_std > 0.0:
            draft_centers[free_mask] += rng.normal(scale=jitter_std, size=draft_centers[free_mask].shape)
        draft_centers[free_mask] = normalize_centers(draft_centers[free_mask], config.safety)
    draft_centers[fixed] = target_centers_np[fixed]
    draft_latent = center_to_latent_approx(draft_centers)
    return draft_latent, draft_centers


def sample_fixed_mask(
    K: int,
    fixed_ratio_range: tuple[float, float],
    rng: np.random.Generator,
    occupied_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a boolean fixed-mask over occupied teacher slots."""

    fixed_min, fixed_max = fixed_ratio_range
    if not 0.0 <= fixed_min <= fixed_max <= 1.0:
        raise ValueError("fixed_ratio_range must lie within [0, 1] and satisfy min <= max.")
    fixed_mask = np.zeros(K, dtype=bool)
    eligible = np.arange(K) if occupied_mask is None else np.flatnonzero(np.asarray(occupied_mask, dtype=bool))
    if eligible.size == 0:
        return fixed_mask
    ratio = float(rng.uniform(fixed_min, fixed_max))
    num_fixed = int(np.round(ratio * eligible.size))
    if num_fixed <= 0:
        return fixed_mask
    chosen = rng.choice(eligible, size=min(num_fixed, eligible.size), replace=False)
    fixed_mask[chosen] = True
    return fixed_mask


def _assignment_mode_for_example(cfg: ProjectConfig, draft_centers: np.ndarray | None, assignment_mode: str | None) -> str:
    if assignment_mode is not None:
        return assignment_mode
    if draft_centers is None:
        return cfg.train.assignment_mode_no_draft
    return cfg.train.assignment_mode_with_draft


def _prepare_teacher_and_draft(
    teacher_centers: np.ndarray,
    draft_centers: np.ndarray | None,
    assignment_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if assignment_mode == "random":
        target_centers, order = assign_slots_random(teacher_centers, rng)
        return target_centers, draft_centers, order
    if assignment_mode == "xy":
        target_centers, order = assign_slots_xy(teacher_centers)
        return target_centers, draft_centers, order
    if assignment_mode == "draft_match":
        if draft_centers is None:
            raise ValueError("assignment_mode='draft_match' requires draft_centers.")
        target_centers, order = match_target_to_draft_greedy(draft_centers, teacher_centers)
        return target_centers, draft_centers, order
    raise ValueError("assignment_mode must be 'random', 'xy', or 'draft_match'.")


def _as_edit_mask(edit_mask: np.ndarray | torch.Tensor | Any, num_slots: int) -> np.ndarray:
    mask = np.asarray(edit_mask, dtype=bool)
    if mask.shape != (num_slots,):
        raise ValueError(f"edit_mask must have shape ({num_slots},).")
    if not np.any(mask):
        raise ValueError("edit_mask must select at least one edited/free slot.")
    return mask


def _match_edited_targets_to_draft(
    target_centers: np.ndarray,
    draft_centers: np.ndarray,
    edit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = np.asarray(target_centers, dtype=np.float64).copy()
    assignment = np.arange(prepared.shape[0], dtype=np.int64)
    edit_ids = np.flatnonzero(edit_mask)
    if edit_ids.size <= 1:
        return prepared, assignment
    matched_edited, local_assignment = match_target_to_draft_greedy(draft_centers[edit_ids], target_centers[edit_ids])
    prepared[edit_ids] = matched_edited
    assignment[edit_ids] = edit_ids[local_assignment]
    return prepared, assignment


def make_patternboost_supervised_example(
    teacher_centers: np.ndarray | torch.Tensor | Any,
    edit_mask: np.ndarray | torch.Tensor | Any,
    mode: str,
    *,
    draft_centers: np.ndarray | torch.Tensor | Any = None,
    use_draft_match: bool = True,
    cfg: ProjectConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Construct a PatternBoost supervised example with an explicit edit set."""

    config = _ensure_project_config(cfg)
    rng = np.random.default_rng(config.project.seed) if rng is None else rng
    teacher_array = _as_center_array(teacher_centers)
    edit = _as_edit_mask(edit_mask, teacher_array.shape[0])
    fixed_mask = ~edit

    draft_array = None if draft_centers is None else _as_center_array(draft_centers).copy()
    if draft_array is None:
        _, draft_array = build_jittered_draft(teacher_array, fixed_mask, rng, cfg=config)
    else:
        draft_array[fixed_mask] = teacher_array[fixed_mask]
        if np.any(edit):
            draft_array[edit] = normalize_centers(draft_array[edit], config.safety)

    if use_draft_match:
        target_centers, assignment = _match_edited_targets_to_draft(teacher_array, draft_array, edit)
    else:
        target_centers = teacher_array.copy()
        assignment = np.arange(target_centers.shape[0], dtype=np.int64)

    target_latent = center_to_latent_approx(target_centers)
    draft_latent = center_to_latent_approx(draft_array)
    example: dict[str, np.ndarray | float | int | bool] = {
        "target_latent": target_latent.astype(np.float32),
        "target_centers": target_centers.astype(np.float32),
        "fixed_mask": fixed_mask,
        "edit_mask": edit,
        "assignment": assignment.astype(np.int64),
    }

    draft_latent = draft_latent.astype(np.float32).copy()
    draft_array = draft_array.astype(np.float32).copy()
    draft_latent[fixed_mask] = target_latent[fixed_mask].astype(np.float32)
    draft_array[fixed_mask] = target_centers[fixed_mask].astype(np.float32)

    if mode == "ddpm":
        step_idx, inner_time = _sample_inner_time(config.ddpm.num_steps, rng)
        alpha_bar_t = float(_ddpm_alpha_bars(config.ddpm.num_steps, config.ddpm.beta_start, config.ddpm.beta_end)[step_idx])
        noise = rng.normal(size=target_latent.shape).astype(np.float32)
        target_latent_f32 = target_latent.astype(np.float32)
        slot_latent = target_latent_f32.copy()
        if config.ddpm.noising_center == "target":
            slot_latent[edit] = np.sqrt(alpha_bar_t) * target_latent_f32[edit] + np.sqrt(1.0 - alpha_bar_t) * noise[edit]
            noise_target = np.zeros_like(slot_latent, dtype=np.float32)
            noise_target[edit] = noise[edit]
        else:
            alpha_eff = float(config.ddpm.draft_strength) ** 2 * alpha_bar_t
            slot_latent[edit] = np.sqrt(alpha_eff) * draft_latent[edit] + np.sqrt(1.0 - alpha_eff) * noise[edit]
            denom = np.sqrt(1.0 - alpha_bar_t)
            noise_target = np.zeros_like(slot_latent, dtype=np.float32)
            noise_target[edit] = ((slot_latent[edit] - np.sqrt(alpha_bar_t) * target_latent_f32[edit]) / denom).astype(np.float32)
        slot_centers = latent_to_center_simple(slot_latent).astype(np.float32)
        slot_latent[fixed_mask] = target_latent_f32[fixed_mask]
        slot_centers[fixed_mask] = target_centers[fixed_mask].astype(np.float32)
        example.update(
            {
                "slot_latent": slot_latent,
                "slot_centers": slot_centers,
                "inner_time": np.float32(inner_time),
                "noise_target": noise_target,
                "draft_latent": draft_latent,
                "draft_centers": draft_array,
            }
        )
        if config.ddpm.use_draft_feature:
            example["extra_slot_features"] = np.concatenate((draft_latent, draft_array), axis=-1).astype(np.float32)
        return _to_torch_example(example, config)

    if mode != "fm":
        raise ValueError("mode must be 'ddpm' or 'fm'.")
    inner_time = np.float32(rng.uniform(0.0, 1.0))
    vel_target = (target_latent.astype(np.float32) - draft_latent).astype(np.float32)
    vel_target[fixed_mask] = 0.0
    slot_latent = (draft_latent + inner_time * vel_target).astype(np.float32)
    slot_latent[fixed_mask] = target_latent[fixed_mask].astype(np.float32)
    slot_centers = latent_to_center_simple(slot_latent).astype(np.float32)
    slot_centers[fixed_mask] = target_centers[fixed_mask].astype(np.float32)
    example.update(
        {
            "slot_latent": slot_latent,
            "slot_centers": slot_centers,
            "draft_latent": draft_latent,
            "draft_centers": draft_array,
            "inner_time": inner_time,
            "vel_target": vel_target,
        }
    )
    if config.fm.use_draft_feature:
        example["extra_slot_features"] = np.concatenate((draft_latent, draft_array), axis=-1).astype(np.float32)
    return _to_torch_example(example, config)


def make_supervised_example(
    teacher_layout: np.ndarray | torch.Tensor | Any,
    mode: str,
    assignment_mode: str | None = None,
    draft_centers: np.ndarray | torch.Tensor | Any = None,
    cfg: ProjectConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Construct one supervised DDPM or FM training example."""

    config = _ensure_project_config(cfg)
    rng = np.random.default_rng(config.project.seed) if rng is None else rng
    teacher_array = _as_center_array(teacher_layout)
    draft_array = None if draft_centers is None else _as_center_array(draft_centers)
    resolved_assignment = _assignment_mode_for_example(config, draft_array, assignment_mode)
    target_centers, draft_array, assignment = _prepare_teacher_and_draft(
        teacher_array,
        draft_array,
        resolved_assignment,
        rng,
    )
    target_latent = center_to_latent_approx(target_centers)
    fixed_mask = sample_fixed_mask(
        target_centers.shape[0],
        (config.train.fixed_ratio_min, config.train.fixed_ratio_max),
        rng,
        occupied_mask=np.ones(target_centers.shape[0], dtype=bool),
    )

    draft_latent = None if draft_array is None else center_to_latent_approx(draft_array)
    if mode == "ddpm":
        if draft_array is None or draft_latent is None:
            draft_latent, draft_array = build_jittered_draft(target_centers, fixed_mask, rng, cfg=config)
    elif mode == "fm":
        if draft_array is None or draft_latent is None:
            draft_latent, draft_array = build_synthetic_draft(
                target_latent,
                target_centers,
                config.train.synthetic_draft_mode,
                config.train.synthetic_draft_scale,
                rng,
                cfg=config,
            )
    else:
        raise ValueError("mode must be 'ddpm' or 'fm'.")

    example: dict[str, np.ndarray | float | int | bool] = {
        "target_latent": target_latent.astype(np.float32),
        "target_centers": target_centers.astype(np.float32),
        "fixed_mask": fixed_mask,
        "assignment": assignment.astype(np.int64),
    }

    assert draft_array is not None and draft_latent is not None
    draft_latent = draft_latent.astype(np.float32).copy()
    draft_array = draft_array.astype(np.float32).copy()
    draft_latent[fixed_mask] = target_latent[fixed_mask].astype(np.float32)
    draft_array[fixed_mask] = target_centers[fixed_mask].astype(np.float32)

    if mode == "ddpm":
        step_idx, inner_time = _sample_inner_time(config.ddpm.num_steps, rng)
        alpha_bar_t = float(_ddpm_alpha_bars(config.ddpm.num_steps, config.ddpm.beta_start, config.ddpm.beta_end)[step_idx])
        noise = rng.normal(size=target_latent.shape).astype(np.float32)
        target_latent_f32 = target_latent.astype(np.float32)
        slot_latent = target_latent_f32.copy()
        free_mask = ~fixed_mask
        if config.ddpm.noising_center == "target":
            slot_latent[free_mask] = np.sqrt(alpha_bar_t) * target_latent_f32[free_mask] + np.sqrt(1.0 - alpha_bar_t) * noise[free_mask]
            noise_target = np.zeros_like(slot_latent, dtype=np.float32)
            noise_target[free_mask] = noise[free_mask]
        else:
            alpha_eff = float(config.ddpm.draft_strength) ** 2 * alpha_bar_t
            slot_latent[free_mask] = np.sqrt(alpha_eff) * draft_latent[free_mask] + np.sqrt(1.0 - alpha_eff) * noise[free_mask]
            denom = np.sqrt(1.0 - alpha_bar_t)
            noise_target = np.zeros_like(slot_latent, dtype=np.float32)
            noise_target[free_mask] = ((slot_latent[free_mask] - np.sqrt(alpha_bar_t) * target_latent_f32[free_mask]) / denom).astype(np.float32)
        slot_centers = latent_to_center_simple(slot_latent).astype(np.float32)
        slot_latent[fixed_mask] = target_latent_f32[fixed_mask]
        slot_centers[fixed_mask] = target_centers[fixed_mask].astype(np.float32)
        example.update(
            {
                "slot_latent": slot_latent,
                "slot_centers": slot_centers,
                "inner_time": np.float32(inner_time),
                "noise_target": noise_target,
                "draft_latent": draft_latent,
                "draft_centers": draft_array,
            }
        )
        if config.ddpm.use_draft_feature:
            example["extra_slot_features"] = np.concatenate((draft_latent, draft_array), axis=-1).astype(np.float32)
        return _to_torch_example(example, config)

    inner_time = np.float32(rng.uniform(0.0, 1.0))
    vel_target = (target_latent.astype(np.float32) - draft_latent).astype(np.float32)
    vel_target[fixed_mask] = 0.0
    slot_latent = (draft_latent + inner_time * vel_target).astype(np.float32)
    slot_latent[fixed_mask] = target_latent[fixed_mask].astype(np.float32)
    slot_centers = latent_to_center_simple(slot_latent).astype(np.float32)
    slot_centers[fixed_mask] = target_centers[fixed_mask].astype(np.float32)
    example.update(
        {
            "slot_latent": slot_latent,
            "slot_centers": slot_centers,
            "draft_latent": draft_latent,
            "draft_centers": draft_array,
            "inner_time": inner_time,
            "vel_target": vel_target,
        }
    )
    if config.fm.use_draft_feature:
        example["extra_slot_features"] = np.concatenate((draft_latent, draft_array), axis=-1).astype(np.float32)
    return _to_torch_example(example, config)


def _validate_teacher_layout_bank(teacher_layouts: np.ndarray | torch.Tensor | Any) -> np.ndarray:
    """Validate a teacher-layout bank to shape `[M, K, 2]` or legacy `[M, K, 3]`."""

    array = np.asarray(teacher_layouts, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] not in {2, 3}:
        raise ValueError("teacher layout banks must have shape [num_layouts, K, 2] or legacy [num_layouts, K, 3].")
    return array


def _resolve_teacher_layout_format(path: str | Path, *, format_name: str = "auto") -> str:
    if format_name not in {"auto", "pt", "jsonl"}:
        raise ValueError("format_name must be one of {'auto', 'pt', 'jsonl'}.")
    if format_name != "auto":
        return format_name
    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        return "pt"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError("Teacher layout files must use the .pt or .jsonl suffix.")


def save_teacher_layouts(
    teacher_layouts: np.ndarray | torch.Tensor | Any,
    path: str | Path,
    *,
    format_name: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a teacher-layout bank to `.pt` or `.jsonl`."""

    teacher_path = Path(path)
    resolved_format = _resolve_teacher_layout_format(teacher_path, format_name=format_name)
    array = _validate_teacher_layout_bank(teacher_layouts)
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "pt":
        payload: dict[str, Any] = {"teacher_layouts": torch.from_numpy(array.copy())}
        if metadata:
            payload["metadata"] = dict(metadata)
        torch.save(payload, teacher_path)
        return teacher_path

    with teacher_path.open("w", encoding="utf-8") as handle:
        for layout_idx, layout in enumerate(array):
            record = {"layout_index": int(layout_idx)}
            if layout.shape[-1] == 2:
                record["centers"] = np.asarray(layout, dtype=np.float64).tolist()
            else:
                record["circles"] = np.asarray(layout, dtype=np.float64).tolist()
            if metadata:
                record.update(dict(metadata))
            handle.write(json.dumps(record))
            handle.write("\n")
    return teacher_path


def load_teacher_layouts(path: str | Path) -> np.ndarray:
    """Load teacher layouts from a `.pt` or `.jsonl` file."""

    teacher_path = Path(path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher layout file not found: {teacher_path}")
    resolved_format = _resolve_teacher_layout_format(teacher_path)
    if resolved_format == "pt":
        payload = torch.load(teacher_path, map_location="cpu")
        teacher_layouts = payload.get("teacher_layouts") if isinstance(payload, dict) else payload
        if teacher_layouts is None:
            raise ValueError(".pt teacher layout files must contain a 'teacher_layouts' entry.")
        if isinstance(teacher_layouts, torch.Tensor):
            teacher_layouts = teacher_layouts.detach().cpu().numpy()
        return _validate_teacher_layout_bank(teacher_layouts)

    layouts: list[np.ndarray] = []
    with teacher_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "centers" in record:
                layouts.append(_as_center_array(record["centers"]))
            elif "circles" in record:
                layouts.append(np.asarray(record["circles"], dtype=np.float64))
            else:
                raise ValueError(f"JSONL teacher layout record {line_idx} is missing centers/circles.")
    if not layouts:
        raise ValueError("JSONL teacher layout files must contain at least one non-empty record.")
    return _validate_teacher_layout_bank(np.stack(layouts, axis=0))


def generate_synthetic_teacher_layouts(
    config: ProjectConfig,
    count: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a small bank of center teacher layouts from environment rollouts."""

    base_seed = config.project.seed if seed is None else seed
    rollout_config = replace(config, project=replace(config.project, dtype="float64"))
    env = CirclePackingEnv(rollout_config)
    policy = make_policy(rollout_config.rollout.policy_name, rollout_config)
    layouts: list[np.ndarray] = []
    for layout_idx in range(count):
        collect_episode(env, policy, seed=base_seed + layout_idx, record_snapshots=False)
        if not bool(np.all(env.active_mask)):
            env.greedy_fill_masked_slots()
        final_obs = env.observe()
        layouts.append(np.asarray(final_obs["centers"], dtype=config.numpy_dtype).copy())
    return np.stack(layouts, axis=0)


class CirclePackingSupervisedDataset(Dataset):
    """Lightweight on-the-fly supervised dataset for DDPM or FM training."""

    def __init__(
        self,
        teacher_layouts: np.ndarray | torch.Tensor | Any,
        mode: str,
        cfg: ProjectConfig,
        *,
        draft_layouts: np.ndarray | torch.Tensor | Any = None,
        seed: int | None = None,
    ) -> None:
        self.teacher_layouts = _validate_teacher_layout_bank(teacher_layouts)
        self.mode = mode
        self.cfg = cfg
        self.seed = cfg.project.seed if seed is None else seed
        self.draft_layouts = None if draft_layouts is None else _validate_teacher_layout_bank(draft_layouts)
        if self.draft_layouts is not None and self.draft_layouts.shape[:2] != self.teacher_layouts.shape[:2]:
            raise ValueError("draft_layouts must have matching [num_layouts, K].")

    def __len__(self) -> int:
        return int(self.teacher_layouts.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(index))
        teacher_layout = self.teacher_layouts[index]
        draft_layout = None if self.draft_layouts is None else self.draft_layouts[index]
        assignment_mode = (
            self.cfg.train.assignment_mode_no_draft
            if draft_layout is None
            else self.cfg.train.assignment_mode_with_draft
        )
        return make_supervised_example(
            teacher_layout,
            mode=self.mode,
            assignment_mode=assignment_mode,
            draft_centers=draft_layout,
            cfg=self.cfg,
            rng=rng,
        )
