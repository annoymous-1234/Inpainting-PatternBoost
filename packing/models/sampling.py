"""Version-1 center proposal samplers for DDPM- and FM-style outer-loop inference.

These functions belong to the model-side pipeline wrapper layer. The model only
produces raw head outputs; this module turns those outputs into `N` latent and
center proposal samples that can then be reduced or ranked by controller-side
logic.
"""

from __future__ import annotations

from typing import Any

import torch

from circle_pack.config import ProjectConfig
from circle_pack.geometry import center_to_latent_approx
from models.interfaces import decode_latent_proposal_for_test
from pipelines.types import BatchedProposalBundle, ProposalBundle


def _as_single_batch_tensor(
    value: torch.Tensor | Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    is_mask: bool = False,
) -> torch.Tensor:
    """Normalize a sampler input to batched tensor form `[1, ...]`.

    Parameters
    ----------
    value:
        Input tensor-like object.
    device:
        Target device.
    dtype:
        Optional floating dtype for non-mask inputs.
    is_mask:
        When `True`, cast to boolean instead of a floating dtype.
    """

    tensor = torch.as_tensor(value, device=device)
    if is_mask:
        tensor = tensor.to(dtype=torch.bool)
    elif dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if is_mask:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError("Sampler fixed_mask inputs must have shape [K] or [1, K].")
    else:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError("Sampler slot inputs must have shape [K, 2] or [1, K, 2].")
        if tensor.shape[-1] != 2:
            raise ValueError("Sampler slot inputs must have trailing dimension 2.")
    if tensor.shape[0] != 1:
        raise ValueError("Version-1 proposal samplers expect inputs with leading batch size 1.")
    return tensor


def _as_batched_tensor(
    value: torch.Tensor | Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    is_mask: bool = False,
) -> torch.Tensor:
    """Normalize sampler inputs to `[B, K, 2]` or `[B, K]` tensor form."""

    tensor = torch.as_tensor(value, device=device)
    if is_mask:
        tensor = tensor.to(dtype=torch.bool)
    elif dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if is_mask:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError("Sampler fixed_mask inputs must have shape [K] or [B, K].")
    else:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError("Sampler slot inputs must have shape [K, 2] or [B, K, 2].")
        if tensor.shape[-1] != 2:
            raise ValueError("Sampler slot inputs must have trailing dimension 2.")
    return tensor


def build_ddpm_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Construct the simple linear DDPM schedule used in v1.

    Returns a dictionary containing tensors of shape `[T]` for:
    - `betas`
    - `alphas`
    - `alpha_bars`
    - `sqrt_alpha_bars`
    - `sqrt_one_minus_alpha_bars`
    """

    betas = torch.linspace(beta_start, beta_end, steps=num_steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }


def ddpm_time_to_step_index(inner_time: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Map normalized scalar DDPM times in `[0, 1]` to discrete step indices."""

    inner_time = torch.as_tensor(inner_time)
    if inner_time.ndim == 2 and inner_time.shape[-1] == 1:
        inner_time = inner_time.squeeze(-1)
    if inner_time.ndim != 1:
        raise ValueError("inner_time must have shape [B] or [B, 1].")
    if num_steps <= 1:
        return torch.zeros_like(inner_time, dtype=torch.long)
    scaled = inner_time.clamp(0.0, 1.0) * float(num_steps - 1)
    return torch.round(scaled).to(dtype=torch.long)


def ddpm_velocity_target(
    target_latent: torch.Tensor,
    noise_target: torch.Tensor,
    alpha_bars_t: torch.Tensor,
) -> torch.Tensor:
    """Compute the v-parameterization target for DDPM supervision.

    Parameters
    ----------
    target_latent:
        Clean latent tensor `x0` with shape `[B, K, 2]`.
    noise_target:
        Gaussian noise tensor `eps` with shape `[B, K, 2]`.
    alpha_bars_t:
        Per-example alpha-bar tensor with shape `[B]`.
    """

    alpha_bars_t = alpha_bars_t.view(-1, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bars_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bars_t)
    return sqrt_alpha_bar * noise_target - sqrt_one_minus * target_latent


def ddpm_prediction_to_x0_and_eps(
    model_prediction: torch.Tensor,
    x_t: torch.Tensor,
    alpha_bars_t: torch.Tensor,
    *,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert any supported DDPM parameterization to `(x0_pred, eps_pred)`.

    Parameters
    ----------
    model_prediction:
        Raw DDPM head output with shape `[B, K, 2]`.
    x_t:
        Current latent state with shape `[B, K, 2]`.
    alpha_bars_t:
        Per-example alpha-bar tensor with shape `[B]`.
    target_mode:
        One of `"eps"`, `"x0"`, or `"v"`.
    """

    alpha_bars_t = alpha_bars_t.view(-1, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bars_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bars_t)
    denom = sqrt_alpha_bar.clamp_min(torch.finfo(x_t.dtype).eps)
    noise_denom = sqrt_one_minus.clamp_min(torch.finfo(x_t.dtype).eps)

    if target_mode == "eps":
        eps_pred = model_prediction
        x0_pred = (x_t - sqrt_one_minus * eps_pred) / denom
        return x0_pred, eps_pred
    if target_mode == "x0":
        x0_pred = model_prediction
        eps_pred = (x_t - sqrt_alpha_bar * x0_pred) / noise_denom
        return x0_pred, eps_pred
    if target_mode == "v":
        v_pred = model_prediction
        x0_pred = sqrt_alpha_bar * x_t - sqrt_one_minus * v_pred
        eps_pred = sqrt_one_minus * x_t + sqrt_alpha_bar * v_pred
        return x0_pred, eps_pred
    raise ValueError("target_mode must be 'eps', 'x0', or 'v'.")


def _validate_draft_features(model: Any, config: ProjectConfig, *, owner: str = "ddpm") -> None:
    """Ensure draft-feature conditioning matches the configured model."""

    del model
    slot_cfg = config.model.slot_features
    if not slot_cfg.use_extra_slot_features or int(slot_cfg.extra_slot_feature_dim) != 4:
        raise ValueError(
            f"{owner}.use_draft_feature=True requires model.slot_features.use_extra_slot_features=True "
            "and extra_slot_feature_dim=4."
        )


def _build_draft_extra_features(
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    *,
    num_samples: int,
) -> torch.Tensor:
    """Repeat current draft latent/center state into `[N, K, 4]` extra features."""

    draft_latent = slot_latent.repeat(num_samples, 1, 1)
    draft_centers = slot_centers.repeat(num_samples, 1, 1)
    return torch.cat((draft_latent, draft_centers), dim=-1)


def _build_draft_extra_features_batched(
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    *,
    num_samples: int,
) -> torch.Tensor:
    """Repeat batched draft latent/center state into `[B * N, K, 4]` features."""

    batch_size, num_slots, _ = slot_latent.shape
    draft_latent = (
        slot_latent.unsqueeze(1)
        .expand(batch_size, int(num_samples), num_slots, 2)
        .reshape(batch_size * int(num_samples), num_slots, 2)
    )
    draft_centers = (
        slot_centers.unsqueeze(1)
        .expand(batch_size, int(num_samples), num_slots, 2)
        .reshape(batch_size * int(num_samples), num_slots, 2)
    )
    return torch.cat((draft_latent, draft_centers), dim=-1)


def _build_ddpm_extra_features(
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    *,
    num_samples: int,
) -> torch.Tensor:
    """Repeat current draft latent/center state into `[N, K, 4]` extra features."""

    return _build_draft_extra_features(slot_latent, slot_centers, num_samples=num_samples)


def _build_ddpm_extra_features_batched(
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    *,
    num_samples: int,
) -> torch.Tensor:
    """Repeat batched draft latent/center state into `[B * N, K, 4]` features."""

    return _build_draft_extra_features_batched(slot_latent, slot_centers, num_samples=num_samples)


def _decode_latent_samples(
    latent_samples: torch.Tensor,
    latent_to_circle_fn: Any,
) -> torch.Tensor:
    """Decode latent samples into center samples while preserving device and dtype."""

    return decode_latent_proposal_for_test(latent_samples, latent_to_circle_fn=latent_to_circle_fn)


def _ddpm_step_trace(
    *,
    start_step: int,
    cfg: ProjectConfig,
    num_steps_override: int | None = None,
) -> list[int]:
    """Return the reverse timestep trace for continuous or subsampled DDPM inference."""

    if not cfg.ddpm.use_timestep_subsampling:
        return list(range(int(start_step), -1, -1))

    requested_steps = int(cfg.ddpm.infer_num_steps if num_steps_override is None else num_steps_override)
    trace_len = min(max(requested_steps, 1), int(start_step) + 1)
    if trace_len == 1:
        return [int(start_step)]
    raw_steps = torch.linspace(float(start_step), 0.0, steps=trace_len)
    trace = [int(round(float(value))) for value in raw_steps]
    trace[0] = int(start_step)
    trace[-1] = 0
    for idx in range(1, len(trace)):
        if trace[idx] > trace[idx - 1]:
            trace[idx] = trace[idx - 1]
    return trace


@torch.no_grad()
def sample_ddpm_proposals_batched(
    model: Any,
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    fixed_mask: torch.Tensor,
    cfg: ProjectConfig,
    latent_to_circle_fn: Any,
    *,
    num_steps: int | None = None,
    tau_start: float | None = None,
) -> BatchedProposalBundle:
    """Generate DDPM proposal samples for a batch of independent candidate states.

    Parameters
    ----------
    model:
        Top-level circle packing model exposing a DDPM head.
    slot_latent:
        Current draft latent state with shape `[B, K, 2]`.
    slot_centers:
        Current draft physical state with shape `[B, K, 2]`.
    fixed_mask:
        Boolean mask with shape `[B, K]` where `True` marks fixed slots.
    cfg:
        Typed project config.
    latent_to_circle_fn:
        Convenience callback used only for proposal decoding in v1.
    num_steps:
        Optional inference-step override used by the segmented demo wrapper when
        timestep subsampling is enabled.
    tau_start:
        Optional normalized start-time override used by the segmented demo wrapper.
    """

    device = model.device
    dtype = model.dtype
    slot_latent = _as_batched_tensor(slot_latent, device=device, dtype=dtype)
    slot_centers = _as_batched_tensor(slot_centers, device=device, dtype=dtype)
    fixed_mask = _as_batched_tensor(fixed_mask, device=device, is_mask=True)
    if slot_latent.shape != slot_centers.shape:
        raise ValueError("slot_latent and slot_centers must have identical shape [B, K, 2].")
    if fixed_mask.shape != slot_latent.shape[:2]:
        raise ValueError("fixed_mask must have shape [B, K] matching slot inputs.")

    sample_count = int(cfg.ddpm.num_samples)
    batch_size, num_slots, _ = slot_latent.shape
    schedule_steps = int(cfg.ddpm.num_steps)
    start_tau = float(cfg.ddpm.tau_start_infer if tau_start is None else tau_start)
    start_tau = min(max(start_tau, 0.0), 1.0)
    schedule = build_ddpm_schedule(
        schedule_steps,
        cfg.ddpm.beta_start,
        cfg.ddpm.beta_end,
        device=device,
        dtype=dtype,
    )
    start_step = int(round(start_tau * max(schedule_steps - 1, 0)))

    draft_latent = (
        slot_latent.unsqueeze(1)
        .expand(batch_size, sample_count, num_slots, 2)
        .reshape(batch_size * sample_count, num_slots, 2)
        .contiguous()
    )
    draft_centers = (
        slot_centers.unsqueeze(1)
        .expand(batch_size, sample_count, num_slots, 2)
        .reshape(batch_size * sample_count, num_slots, 2)
        .contiguous()
    )
    repeated_fixed_mask = (
        fixed_mask.unsqueeze(1)
        .expand(batch_size, sample_count, num_slots)
        .reshape(batch_size * sample_count, num_slots)
        .contiguous()
    )

    if cfg.ddpm.use_draft_feature:
        _validate_draft_features(model, cfg)
        extra_slot_features = _build_ddpm_extra_features_batched(slot_latent, slot_centers, num_samples=sample_count)
    else:
        extra_slot_features = None

    draft_strength = float(cfg.ddpm.draft_strength if cfg.ddpm.use_draft_warmstart else 0.0)
    alpha_bar_start = schedule["alpha_bars"][start_step]
    alpha_eff = (draft_strength * draft_strength) * alpha_bar_start
    noise = torch.randn_like(draft_latent)
    state = torch.sqrt(alpha_eff) * draft_latent + torch.sqrt(1.0 - alpha_eff) * noise
    state = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_latent, state)

    step_trace = _ddpm_step_trace(
        start_step=start_step,
        cfg=cfg,
        num_steps_override=num_steps,
    )
    for trace_idx, step_idx in enumerate(step_trace):
        alpha_bar_t = schedule["alpha_bars"][step_idx].expand(batch_size * sample_count)
        inner_time = torch.full(
            (batch_size * sample_count,),
            float(step_idx / max(schedule_steps - 1, 1)),
            device=device,
            dtype=dtype,
        )
        state_centers = _decode_latent_samples(state, latent_to_circle_fn)
        state_centers = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_centers, state_centers)
        outputs = model(
            state,
            state_centers,
            repeated_fixed_mask,
            inner_time=inner_time,
            extra_slot_features=extra_slot_features,
            predict_ddpm=True,
        )
        ddpm_pred = outputs["ddpm_pred"]
        if ddpm_pred is None:
            raise AssertionError("sample_ddpm_proposals requires an enabled DDPM head.")

        x0_pred, eps_pred = ddpm_prediction_to_x0_and_eps(
            ddpm_pred,
            state,
            alpha_bar_t,
            target_mode=cfg.model.heads.ddpm_target,
        )

        if cfg.ddpm.use_timestep_subsampling:
            if trace_idx + 1 < len(step_trace):
                next_step = step_trace[trace_idx + 1]
                alpha_bar_next = schedule["alpha_bars"][next_step]
                state = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps_pred
            else:
                state = x0_pred
        else:
            alpha_t = schedule["alphas"][step_idx]
            beta_t = schedule["betas"][step_idx]
            alpha_bar_value = schedule["alpha_bars"][step_idx]
            mean = (state - (beta_t / torch.sqrt(1.0 - alpha_bar_value)) * eps_pred) / torch.sqrt(alpha_t)
            if step_idx > 0:
                state = mean + torch.sqrt(beta_t) * torch.randn_like(mean)
            else:
                state = mean

        state = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_latent, state)

    center_samples = _decode_latent_samples(state, latent_to_circle_fn)
    center_samples = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_centers, center_samples)
    aux = {
        "start_step": torch.tensor(start_step, device=device, dtype=torch.long),
        "step_trace": torch.tensor(step_trace, device=device, dtype=torch.long),
    }
    latent_samples = state.reshape(batch_size, sample_count, num_slots, 2)
    center_samples = center_samples.reshape(batch_size, sample_count, num_slots, 2)
    return BatchedProposalBundle(
        latent_samples=latent_samples,
        center_samples=center_samples,
        aux=aux,
    )


@torch.no_grad()
def sample_ddpm_proposals(
    model: Any,
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    fixed_mask: torch.Tensor,
    cfg: ProjectConfig,
    latent_to_circle_fn: Any,
    *,
    num_steps: int | None = None,
    tau_start: float | None = None,
) -> ProposalBundle:
    """Generate DDPM-style latent and center proposal samples for one state."""

    batched = sample_ddpm_proposals_batched(
        model,
        slot_latent,
        slot_centers,
        fixed_mask,
        cfg,
        latent_to_circle_fn,
        num_steps=num_steps,
        tau_start=tau_start,
    )
    if batched.latent_samples.shape[0] != 1:
        raise ValueError("sample_ddpm_proposals expects a single batch element.")
    return ProposalBundle(
        latent_samples=batched.latent_samples[0],
        center_samples=batched.center_samples[0],
        aux=batched.aux,
    )


@torch.no_grad()
def sample_fm_proposals_batched(
    model: Any,
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    fixed_mask: torch.Tensor,
    cfg: ProjectConfig,
    latent_to_circle_fn: Any,
    *,
    num_steps: int | None = None,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> BatchedProposalBundle:
    """Generate FM proposal samples for a batch of independent candidate states.

    Parameters
    ----------
    model:
        Top-level circle packing model exposing an FM head.
    slot_latent:
        Current draft latent state with shape `[B, K, 2]`.
    slot_centers:
        Current draft physical state with shape `[B, K, 2]`.
    fixed_mask:
        Boolean mask with shape `[B, K]` where `True` marks fixed slots.
    cfg:
        Typed project config.
    latent_to_circle_fn:
        Convenience callback used only for proposal decoding in v1.
    num_steps:
        Optional Euler-step override used by the segmented demo wrapper.
    t_start, t_end:
        Integration interval in normalized flow time.
    """

    if not 0.0 <= t_start <= t_end <= 1.0:
        raise ValueError("FM integration bounds must satisfy 0 <= t_start <= t_end <= 1.")

    device = model.device
    dtype = model.dtype
    slot_latent = _as_batched_tensor(slot_latent, device=device, dtype=dtype)
    slot_centers = _as_batched_tensor(slot_centers, device=device, dtype=dtype)
    fixed_mask = _as_batched_tensor(fixed_mask, device=device, is_mask=True)
    if slot_latent.shape != slot_centers.shape:
        raise ValueError("slot_latent and slot_centers must have identical shape [B, K, 2].")
    if fixed_mask.shape != slot_latent.shape[:2]:
        raise ValueError("fixed_mask must have shape [B, K] matching slot inputs.")

    sample_count = int(cfg.fm.num_samples)
    batch_size, num_slots, _ = slot_latent.shape
    integration_steps = max(int(cfg.fm.num_steps if num_steps is None else num_steps), 1)
    draft_latent = (
        slot_latent.unsqueeze(1)
        .expand(batch_size, sample_count, num_slots, 2)
        .reshape(batch_size * sample_count, num_slots, 2)
        .contiguous()
    )
    draft_centers = (
        slot_centers.unsqueeze(1)
        .expand(batch_size, sample_count, num_slots, 2)
        .reshape(batch_size * sample_count, num_slots, 2)
        .contiguous()
    )
    repeated_fixed_mask = (
        fixed_mask.unsqueeze(1)
        .expand(batch_size, sample_count, num_slots)
        .reshape(batch_size * sample_count, num_slots)
        .contiguous()
    )
    free_mask = ~repeated_fixed_mask
    if cfg.fm.use_draft_feature:
        _validate_draft_features(model, cfg, owner="fm")
        extra_slot_features = _build_draft_extra_features_batched(
            slot_latent,
            slot_centers,
            num_samples=sample_count,
        )
    elif cfg.model.slot_features.use_extra_slot_features:
        raise ValueError(
            "FM sampling with model.slot_features.use_extra_slot_features=True requires "
            "fm.use_draft_feature=True."
        )
    else:
        extra_slot_features = None

    state = draft_latent.clone()
    if cfg.fm.source_noise_std > 0.0:
        perturbation = cfg.fm.source_noise_std * torch.randn_like(state)
        state = torch.where(free_mask.unsqueeze(-1), state + perturbation, state)

    state = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_latent, state)
    times = torch.linspace(t_start, t_end, steps=integration_steps + 1, device=device, dtype=dtype)
    initial_state = state.clone()
    short_fraction = float(cfg.fm.short_displacement_fraction)
    short_displacement: torch.Tensor | None = None
    terminal_velocity_norm: torch.Tensor | None = None

    for step_idx in range(integration_steps):
        current_time = times[step_idx]
        next_time = times[step_idx + 1]
        dt = next_time - current_time
        inner_time = torch.full((batch_size * sample_count,), float(current_time), device=device, dtype=dtype)
        state_centers = _decode_latent_samples(state, latent_to_circle_fn)
        state_centers = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_centers, state_centers)
        outputs = model(
            state,
            state_centers,
            repeated_fixed_mask,
            inner_time=inner_time,
            extra_slot_features=extra_slot_features,
            predict_fm=True,
        )
        velocity = outputs["fm_pred"]
        if velocity is None:
            raise AssertionError("sample_fm_proposals_batched requires an enabled FM head.")
        velocity = torch.where(repeated_fixed_mask.unsqueeze(-1), torch.zeros_like(velocity), velocity)
        state = state + dt * velocity
        state = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_latent, state)

        if short_displacement is None and float(next_time) >= short_fraction:
            short_displacement = torch.linalg.norm(state - initial_state, dim=-1)
        if step_idx == integration_steps - 1:
            terminal_velocity_norm = torch.linalg.norm(velocity, dim=-1)

    center_samples = _decode_latent_samples(state, latent_to_circle_fn)
    center_samples = torch.where(repeated_fixed_mask.unsqueeze(-1), draft_centers, center_samples)
    aux: dict[str, torch.Tensor] = {}
    if cfg.fm.return_terminal_velocity_norm and terminal_velocity_norm is not None:
        aux["terminal_velocity_norm"] = terminal_velocity_norm.reshape(batch_size, sample_count, num_slots)
    if cfg.fm.return_short_displacement:
        if short_displacement is None:
            short_displacement = torch.linalg.norm(state - initial_state, dim=-1)
        aux["short_displacement"] = short_displacement.reshape(batch_size, sample_count, num_slots)
    latent_samples = state.reshape(batch_size, sample_count, num_slots, 2)
    center_samples = center_samples.reshape(batch_size, sample_count, num_slots, 2)
    return BatchedProposalBundle(
        latent_samples=latent_samples,
        center_samples=center_samples,
        aux=aux,
    )


@torch.no_grad()
def sample_fm_proposals(
    model: Any,
    slot_latent: torch.Tensor,
    slot_centers: torch.Tensor,
    fixed_mask: torch.Tensor,
    cfg: ProjectConfig,
    latent_to_circle_fn: Any,
    *,
    num_steps: int | None = None,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> ProposalBundle:
    """Generate FM-style latent and center proposal samples with Euler integration."""

    batched = sample_fm_proposals_batched(
        model,
        slot_latent,
        slot_centers,
        fixed_mask,
        cfg,
        latent_to_circle_fn,
        num_steps=num_steps,
        t_start=t_start,
        t_end=t_end,
    )
    if batched.latent_samples.shape[0] != 1:
        raise ValueError("sample_fm_proposals expects a single batch element.")
    aux: dict[str, torch.Tensor] = {}
    for key, value in batched.aux.items():
        aux[key] = value[0]
    return ProposalBundle(
        latent_samples=batched.latent_samples[0],
        center_samples=batched.center_samples[0],
        aux=aux,
    )


def _random_sample_count(cfg: ProjectConfig) -> int:
    """Return the proposal-sample count matching the configured learned head.

    The random operator must draw as many proposals per round as the learned
    sampler it replaces, so that dispersion-based priority scoring downstream
    sees an identically shaped sample bank.
    """

    if str(cfg.patternboost.training.mode) == "fm":
        return int(cfg.fm.num_samples)
    return int(cfg.ddpm.num_samples)


def sample_random_proposals_batched(
    slot_centers: torch.Tensor | Any,
    fixed_mask: torch.Tensor | Any,
    cfg: ProjectConfig,
) -> BatchedProposalBundle:
    """Generate model-free center proposals for a batch of candidate states.

    This is the random-operator counterpart of `sample_ddpm_proposals_batched`
    used by the completion-operator ablation. It takes no model and touches no
    model attribute: device and dtype come from `cfg`, so a run configured with
    `patternboost.generation.completion_operator != "learned"` never needs a
    network to be built, loaded, or evaluated.

    Parameters
    ----------
    slot_centers:
        Current physical state with shape `[B, K, 2]`; for free slots this is
        the draft that the learned reverse process would have started from.
    fixed_mask:
        Boolean mask with shape `[B, K]` where `True` marks active-core slots.
    cfg:
        Typed project config; `patternboost.generation.completion_operator`
        selects the operator and `random_center_jitter_std` sets the
        perturbation scale.

    Returns
    -------
    A `BatchedProposalBundle` with `[B, N, K, 2]` samples and empty `aux`.
    Fixed slots are returned bit-identical to their input values.
    """

    device = torch.device("cpu")
    dtype = cfg.torch_dtype
    centers = _as_batched_tensor(slot_centers, device=device, dtype=dtype)
    fixed = _as_batched_tensor(fixed_mask, device=device, is_mask=True)
    if fixed.shape != centers.shape[:2]:
        raise ValueError("fixed_mask must have shape [B, K] matching slot_centers.")

    operator = str(cfg.patternboost.generation.completion_operator)
    sample_count = _random_sample_count(cfg)
    batch_size, num_slots, _ = centers.shape
    draft = centers.unsqueeze(1).expand(batch_size, sample_count, num_slots, 2).contiguous()

    if operator == "random_perturb":
        sigma = float(cfg.patternboost.generation.random_center_jitter_std)
        proposals = draft + sigma * torch.randn_like(draft)
    elif operator == "random_scratch":
        proposals = torch.rand_like(draft)
    else:
        raise ValueError(
            "sample_random_proposals_batched requires completion_operator "
            "'random_perturb' or 'random_scratch'."
        )

    if bool(getattr(cfg.safety, "clip_positions", True)):
        proposals = proposals.clamp(0.0, 1.0)
    center_samples = torch.where(fixed.unsqueeze(1).unsqueeze(-1), draft, proposals)

    latent_samples = torch.as_tensor(
        center_to_latent_approx(center_samples.detach().cpu().numpy()),
        device=device,
        dtype=dtype,
    )
    return BatchedProposalBundle(
        latent_samples=latent_samples,
        center_samples=center_samples,
        aux={},
    )


def sample_random_proposals(
    slot_centers: torch.Tensor | Any,
    fixed_mask: torch.Tensor | Any,
    cfg: ProjectConfig,
) -> ProposalBundle:
    """Generate model-free center proposals for one candidate state."""

    batched = sample_random_proposals_batched(slot_centers, fixed_mask, cfg)
    if batched.latent_samples.shape[0] != 1:
        raise ValueError("sample_random_proposals expects a single batch element.")
    return ProposalBundle(
        latent_samples=batched.latent_samples[0],
        center_samples=batched.center_samples[0],
        aux={},
    )
