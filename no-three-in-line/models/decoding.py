from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from proposal_eval import CandidatePoint, ProposalSlot
from project_config import load_project_config


@dataclass(frozen=True)
class SelectionStrategyConfig:
    """Configure one decoding stage's ranking or sampling rule.

    ``strategy`` names the selection rule, ``top_k`` controls how many outputs
    are kept, and ``top_n`` optionally controls the candidate pool size for
    strategies such as ``topn_sample_mask``. The dataclass stores configuration
    only, returns no value on construction, and does not mutate external state.
    """

    strategy: str = "topk"
    top_k: int = 1
    top_n: int | None = None


@dataclass(frozen=True)
class DecodingConfig:
    """Bundle slot-selection and candidate-selection decoding configs.

    ``slot_selection`` controls which masked slots are proposed and
    ``candidate_selection`` controls how many row candidates are emitted for
    each chosen slot. The dataclass stores configuration only and does not
    mutate external state.
    """
    slot_selection: SelectionStrategyConfig = field(
        default_factory=lambda: SelectionStrategyConfig(strategy="topk", top_k=4)
    )
    candidate_selection: SelectionStrategyConfig = field(
        default_factory=lambda: SelectionStrategyConfig(strategy="topk", top_k=1)
    )


def _selection_strategy_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    default_top_k: int,
) -> SelectionStrategyConfig:
    """Build one strategy config from a YAML mapping or use defaults.

    ``mapping`` is expected to contain ``strategy``, ``top_k``, and optionally
    ``top_n`` keys when present. The function returns a validated
    ``SelectionStrategyConfig``, does not mutate the input mapping, and assumes
    the numeric keys can be converted to non-negative integers when provided.
    """
    defaults = SelectionStrategyConfig(top_k=default_top_k)
    if mapping is None:
        return defaults
    if not isinstance(mapping, Mapping):
        raise ValueError("decoding strategy config must be a mapping")

    strategy = str(mapping.get("strategy", defaults.strategy))
    top_k = int(mapping.get("top_k", defaults.top_k))
    top_n_value = mapping.get("top_n", defaults.top_n)
    top_n = None if top_n_value is None else int(top_n_value)
    if top_k < 0:
        raise ValueError("decoding top_k must be non-negative")
    if top_n is not None and top_n < 0:
        raise ValueError("decoding top_n must be non-negative")
    return SelectionStrategyConfig(strategy=strategy, top_k=top_k, top_n=top_n)


def load_decoding_config(path: str | Path | None = None) -> DecodingConfig:
    """Load the decoding section of the project configuration.

    ``path`` optionally overrides the root ``config.yaml`` location. The
    function returns a ``DecodingConfig``, does not mutate external state, and
    assumes the ``decoding`` section is either absent or a mapping with
    slot-selection and candidate-selection submappings.
    """
    loaded = load_project_config(path)
    decoding = loaded.get("decoding", {})
    if not isinstance(decoding, Mapping):
        raise ValueError("decoding config must be a mapping")

    return DecodingConfig(
        slot_selection=_selection_strategy_from_mapping(
            decoding.get("slot_selection"),
            default_top_k=4,
        ),
        candidate_selection=_selection_strategy_from_mapping(
            decoding.get("candidate_selection"),
            default_top_k=1,
        ),
    )


def _resolve_decoding_config(
    decoding_config: DecodingConfig | None,
    *,
    k: int | None,
    num_candidates_per_slot: int | None,
) -> DecodingConfig:
    """Combine explicit overrides with a base decoding configuration.

    ``decoding_config`` supplies the base strategy settings, while ``k`` and
    ``num_candidates_per_slot`` optionally override the corresponding ``top_k``
    values. The function returns a resolved ``DecodingConfig``, does not mutate
    the input config, and assumes any overrides are compatible with the active
    strategy names. Any configured ``top_n`` values are preserved.
    """
    resolved = load_decoding_config() if decoding_config is None else decoding_config
    return DecodingConfig(
        slot_selection=SelectionStrategyConfig(
            strategy=resolved.slot_selection.strategy,
            top_k=resolved.slot_selection.top_k if k is None else k,
            top_n=resolved.slot_selection.top_n,
        ),
        candidate_selection=SelectionStrategyConfig(
            strategy=resolved.candidate_selection.strategy,
            top_k=resolved.candidate_selection.top_k
            if num_candidates_per_slot is None
            else num_candidates_per_slot,
            top_n=resolved.candidate_selection.top_n,
        ),
    )


def _validate_masked_slot_selection_inputs(
    state_logits: Tensor,
    slot_state_ids: Tensor,
) -> None:
    """Validate tensor shapes for masked-slot selection helpers.

    ``state_logits`` must be a floating tensor of shape ``[B, 2n, n + 1]`` and
    ``slot_state_ids`` must be a ``LongTensor`` of shape ``[B, 2n]`` whose
    ``0`` entries mark masked slots. The function returns ``None``, does not
    mutate the inputs, and assumes strategy-specific ``k`` validation happens
    in the caller.
    """
    if state_logits.ndim != 3:
        raise ValueError("state_logits must have shape [B, 2n, n+1]")
    if slot_state_ids.ndim != 2 or slot_state_ids.shape[:2] != state_logits.shape[:2]:
        raise ValueError("slot_state_ids must have shape [B, 2n] and match state_logits")


def _mask_probabilities(state_logits: Tensor) -> Tensor:
    """Return ``p([MASK])`` for every slot token.

    ``state_logits`` is a floating tensor of shape ``[B, 2n, n + 1]``. The
    function returns a floating tensor of shape ``[B, 2n]`` on the same device,
    does not mutate the input tensor, and assumes index ``0`` corresponds to
    the ``[MASK]`` class.
    """
    probabilities = torch.softmax(state_logits, dim=-1)
    return probabilities[..., 0]


def _normalized_y_entropy(state_logits: Tensor) -> Tensor:
    """Return normalized entropy over the non-mask ``y`` distribution.

    ``state_logits`` is a floating tensor of shape ``[B, 2n, n + 1]``. The
    function returns a floating tensor of shape ``[B, 2n]`` on the same device
    with values in ``[0, 1]`` when ``n > 1``, does not mutate the input, and
    assumes the last-axis slice ``[..., 1:]`` contains the ``y`` logits only.
    """
    num_y = state_logits.shape[-1] - 1
    if num_y <= 1:
        return state_logits.new_zeros(state_logits.shape[:2])

    y_probabilities = torch.softmax(state_logits[..., 1:], dim=-1)
    tiny = torch.finfo(y_probabilities.dtype).tiny
    # Clamp before ``log`` so near-zero probabilities do not create ``-inf``.
    entropy = -(y_probabilities * y_probabilities.clamp_min(tiny).log()).sum(dim=-1)
    normalizer = torch.log(y_probabilities.new_tensor(float(num_y)))
    return entropy / normalizer


def _select_topk_masked_slots_from_scores(
    slot_state_ids: Tensor,
    masked_scores: Tensor,
    k: int,
) -> Tensor:
    """Select masked slots by descending per-slot score.

    ``slot_state_ids`` is a ``LongTensor`` of shape ``[B, 2n]`` and
    ``masked_scores`` is a floating tensor of shape ``[B, 2n]`` whose values
    are only consulted for masked slots. The function returns a padded
    ``LongTensor`` of shape ``[B, S]``, does not mutate the inputs, and assumes
    the caller has already validated tensor shapes.
    """
    if k <= 0:
        return slot_state_ids.new_empty((slot_state_ids.shape[0], 0))

    masked = slot_state_ids.eq(0)
    selected_rows: list[Tensor] = []
    max_selected = 0
    for batch_index in range(slot_state_ids.shape[0]):
        masked_indices = torch.nonzero(masked[batch_index], as_tuple=False).squeeze(-1)
        if masked_indices.numel() == 0:
            selected_rows.append(masked_indices)
            continue

        scores = masked_scores[batch_index, masked_indices]
        num_select = min(k, masked_indices.numel())
        topk_positions = torch.argsort(scores, descending=True)[:num_select]
        chosen_indices = masked_indices[topk_positions]
        selected_rows.append(chosen_indices)
        max_selected = max(max_selected, num_select)

    if max_selected == 0:
        return slot_state_ids.new_empty((slot_state_ids.shape[0], 0))

    output = slot_state_ids.new_full((slot_state_ids.shape[0], max_selected), -1)
    for batch_index, chosen_indices in enumerate(selected_rows):
        if chosen_indices.numel() > 0:
            output[batch_index, : chosen_indices.numel()] = chosen_indices

    return output


def masked_fill_strength(state_logits: Tensor) -> Tensor:
    """Return fill strengths ``1 - p([MASK])`` for each slot token.

    ``state_logits`` is a floating tensor of shape ``[B, 2n, n + 1]``. The
    function applies ``softmax`` over the last dimension and returns a floating
    tensor of shape ``[B, 2n]`` on the same device. It does not mutate the
    input tensor and assumes index ``0`` corresponds to ``[MASK]``.
    """
    return 1.0 - _mask_probabilities(state_logits)


def decode_y_argmax(state_logits: Tensor) -> Tensor:
    """Return the highest-logit row index for every slot token.

    ``state_logits`` is a floating tensor of shape ``[B, 2n, n + 1]``. The
    function ignores the ``[MASK]`` column at index ``0`` and returns a
    ``LongTensor`` of shape ``[B, 2n]`` with row ids in ``[0, n - 1]``. It does
    not mutate the input tensor.
    """
    return torch.argmax(state_logits[..., 1:], dim=-1)


def select_topk_masked_slots(
    state_logits: Tensor,
    slot_state_ids: Tensor,
    k: int,
) -> Tensor:
    """Select up to ``k`` masked slots per batch item by fill strength.

    ``state_logits`` must have shape ``[B, 2n, n + 1]`` and ``slot_state_ids``
    must be a ``LongTensor`` of shape ``[B, 2n]`` with ``0`` for masked slots.
    The function returns a padded ``LongTensor`` of shape ``[B, S]`` containing
    selected token ids and ``-1`` in unused positions. It does not mutate the
    inputs and assumes masked-slot eligibility is determined solely by
    ``slot_state_ids == 0``.
    """
    _validate_masked_slot_selection_inputs(state_logits, slot_state_ids)
    strengths = masked_fill_strength(state_logits)
    return _select_topk_masked_slots_from_scores(slot_state_ids, strengths, k)


def _select_topk_fill_entropy_masked_slots(
    state_logits: Tensor,
    slot_state_ids: Tensor,
    k: int,
) -> Tensor:
    """Select masked slots by ``p(fill) * (1 - H_norm)``.

    ``state_logits`` must have shape ``[B, 2n, n + 1]`` and ``slot_state_ids``
    must be a ``LongTensor`` of shape ``[B, 2n]``. The function returns a
    padded ``LongTensor`` of shape ``[B, S]``, does not mutate the inputs, and
    assumes ``H_norm`` is the normalized entropy over the non-mask ``y``
    distribution for each slot token.
    """
    _validate_masked_slot_selection_inputs(state_logits, slot_state_ids)
    fill_strength = masked_fill_strength(state_logits)
    normalized_entropy = _normalized_y_entropy(state_logits)
    weighted_scores = fill_strength * (1.0 - normalized_entropy)
    return _select_topk_masked_slots_from_scores(slot_state_ids, weighted_scores, k)


def _select_topn_sample_masked_slots(
    state_logits: Tensor,
    slot_state_ids: Tensor,
    k: int,
    top_n: int | None,
) -> Tensor:
    """Sample masked slots from the highest-``p([MASK])`` pool.

    ``state_logits`` must have shape ``[B, 2n, n + 1]`` and ``slot_state_ids``
    must be a ``LongTensor`` of shape ``[B, 2n]``. ``k`` controls how many
    slots to sample without replacement and ``top_n`` optionally expands the
    pool sampled from. The function returns a padded ``LongTensor`` of shape
    ``[B, S]``, does not mutate the inputs, and assumes masked-slot selection
    should depend only on ``p([MASK])`` among currently masked tokens.
    """
    _validate_masked_slot_selection_inputs(state_logits, slot_state_ids)
    if k <= 0:
        return slot_state_ids.new_empty((slot_state_ids.shape[0], 0))

    mask_probabilities = _mask_probabilities(state_logits)
    effective_top_n = k if top_n is None else top_n
    masked = slot_state_ids.eq(0)

    selected_rows: list[Tensor] = []
    max_selected = 0
    for batch_index in range(slot_state_ids.shape[0]):
        masked_indices = torch.nonzero(masked[batch_index], as_tuple=False).squeeze(-1)
        if masked_indices.numel() == 0:
            selected_rows.append(masked_indices)
            continue

        masked_scores = mask_probabilities[batch_index, masked_indices]
        pool_size = min(effective_top_n, masked_indices.numel())
        if pool_size <= 0:
            selected_rows.append(masked_indices[:0])
            continue

        topn_positions = torch.argsort(masked_scores, descending=True)[:pool_size]
        pool_indices = masked_indices[topn_positions]
        pool_weights = masked_scores[topn_positions]
        num_select = min(k, pool_size)
        sampled_pool_positions = torch.multinomial(pool_weights, num_samples=num_select, replacement=False)
        # Restore score order so downstream proposal ranking stays deterministic.
        sampled_pool_positions = torch.sort(sampled_pool_positions).values
        chosen_indices = pool_indices[sampled_pool_positions]
        selected_rows.append(chosen_indices)
        max_selected = max(max_selected, num_select)

    if max_selected == 0:
        return slot_state_ids.new_empty((slot_state_ids.shape[0], 0))

    output = slot_state_ids.new_full((slot_state_ids.shape[0], max_selected), -1)
    for batch_index, chosen_indices in enumerate(selected_rows):
        if chosen_indices.numel() > 0:
            output[batch_index, : chosen_indices.numel()] = chosen_indices

    return output


def select_topk_candidate_ys(
    state_logits: Tensor,
    selected_slot_indices: Tensor,
    k: int,
) -> Tensor:
    """Select up to ``k`` candidate row ids for each chosen slot token.

    ``state_logits`` has shape ``[B, 2n, n + 1]`` and ``selected_slot_indices``
    is a padded ``LongTensor`` of shape ``[B, S]`` containing token ids or
    ``-1``. The function returns a padded ``LongTensor`` of shape ``[B, S, K]``
    with row ids in ``[0, n - 1]`` and ``-1`` for unused positions. It does not
    mutate the inputs and assumes row candidates should be ranked by descending
    non-mask logit.
    """
    if state_logits.ndim != 3:
        raise ValueError("state_logits must have shape [B, 2n, n+1]")
    if selected_slot_indices.ndim != 2 or selected_slot_indices.shape[0] != state_logits.shape[0]:
        raise ValueError("selected_slot_indices must have shape [B, num_selected]")
    if k <= 0:
        return selected_slot_indices.new_empty((*selected_slot_indices.shape, 0))

    num_y = state_logits.shape[-1] - 1
    max_candidates = min(k, num_y)
    output = selected_slot_indices.new_full((*selected_slot_indices.shape, max_candidates), -1)

    for batch_index in range(state_logits.shape[0]):
        for selection_index, token_id_tensor in enumerate(selected_slot_indices[batch_index]):
            token_id = int(token_id_tensor.item())
            if token_id < 0:
                continue

            scores = state_logits[batch_index, token_id, 1:]
            chosen_y = torch.argsort(scores, descending=True)[:max_candidates]
            output[batch_index, selection_index, : chosen_y.numel()] = chosen_y

    return output


def select_masked_slots(
    state_logits: Tensor,
    slot_state_ids: Tensor,
    selection_config: SelectionStrategyConfig,
) -> Tensor:
    """Dispatch masked-slot selection according to ``selection_config``.

    The tensor shapes match :func:`select_topk_masked_slots`. The function
    returns a padded ``LongTensor`` of selected token ids, does not mutate the
    inputs, and assumes ``selection_config.strategy`` is one of ``topk``,
    ``topk_fill_entropy``, or ``topn_sample_mask``.
    """
    if selection_config.strategy == "topk":
        return select_topk_masked_slots(state_logits, slot_state_ids, selection_config.top_k)
    if selection_config.strategy == "topk_fill_entropy":
        return _select_topk_fill_entropy_masked_slots(state_logits, slot_state_ids, selection_config.top_k)
    if selection_config.strategy == "topn_sample_mask":
        return _select_topn_sample_masked_slots(
            state_logits,
            slot_state_ids,
            selection_config.top_k,
            selection_config.top_n,
        )
    raise ValueError(f"unsupported slot selection strategy: {selection_config.strategy}")


def select_candidate_ys(
    state_logits: Tensor,
    selected_slot_indices: Tensor,
    selection_config: SelectionStrategyConfig,
) -> Tensor:
    """Dispatch per-slot candidate-row selection according to ``selection_config``.

    The tensor shapes match :func:`select_topk_candidate_ys`. The function
    returns a padded ``LongTensor`` of candidate row ids, does not mutate the
    inputs, and assumes ``candidate_selection`` still uses ``topk`` only.
    """
    if selection_config.strategy != "topk":
        raise ValueError(f"unsupported candidate selection strategy: {selection_config.strategy}")
    return select_topk_candidate_ys(state_logits, selected_slot_indices, selection_config.top_k)


def logits_to_proposal_slots(
    state_logits: Tensor,
    slot_state_ids: Tensor,
    k: int | None = None,
    num_candidates_per_slot: int | None = None,
    decoding_config: DecodingConfig | None = None,
) -> list[list[ProposalSlot]]:
    """Decode logits into env-compatible ``ProposalSlot`` objects.

    ``state_logits`` is a floating tensor of shape ``[B, 2n, n + 1]`` and
    ``slot_state_ids`` is a ``LongTensor`` of shape ``[B, 2n]``. ``k`` and
    ``num_candidates_per_slot`` optionally override the config-driven ``top_k``
    values for slot selection and per-slot candidate selection. The function
    returns one ordered ``ProposalSlot`` list per batch item, does not mutate
    the inputs or model state, and assumes index ``0`` in the logits is the
    ``[MASK]`` class.
    """
    resolved_config = _resolve_decoding_config(
        decoding_config,
        k=k,
        num_candidates_per_slot=num_candidates_per_slot,
    )
    selected_indices = select_masked_slots(state_logits, slot_state_ids, resolved_config.slot_selection)
    selected_y = select_candidate_ys(state_logits, selected_indices, resolved_config.candidate_selection)

    batch_proposals: list[list[ProposalSlot]] = []
    for batch_index in range(state_logits.shape[0]):
        proposals: list[ProposalSlot] = []
        for selection_index, token_id_tensor in enumerate(selected_indices[batch_index]):
            token_id = int(token_id_tensor.item())
            if token_id < 0:
                continue

            x = token_id // 2
            candidates: list[CandidatePoint] = []
            for y_tensor in selected_y[batch_index, selection_index]:
                y = int(y_tensor.item())
                if y < 0:
                    continue
                score = float(state_logits[batch_index, token_id, y + 1].item())
                candidates.append(CandidatePoint(x=x, y=y, score=score))

            if candidates:
                proposals.append(ProposalSlot(slot_id=token_id, candidates=candidates))
        batch_proposals.append(proposals)

    return batch_proposals


def build_proposals_from_logits(
    state_logits: Tensor,
    slot_state_ids: Tensor,
    k: int | None = None,
    num_candidates_per_slot: int | None = None,
    decoding_config: DecodingConfig | None = None,
) -> list[list[tuple[int, int, int]]]:
    """Decode logits into flattened ``(slot_id, x, y)`` tuples.

    The inputs and override semantics match :func:`logits_to_proposal_slots`.
    The function returns one flattened proposal list per batch item, does not
    mutate the inputs, and assumes it is acceptable to lose the grouping of
    candidate rows by slot in the returned tuples.
    """
    proposal_slots = logits_to_proposal_slots(
        state_logits,
        slot_state_ids,
        k=k,
        num_candidates_per_slot=num_candidates_per_slot,
        decoding_config=decoding_config,
    )
    batch_proposals: list[list[tuple[int, int, int]]] = []
    for batch_item in proposal_slots:
        proposals: list[tuple[int, int, int]] = []
        for proposal_slot in batch_item:
            for candidate in proposal_slot.candidates:
                proposals.append((proposal_slot.slot_id, candidate.x, candidate.y))
        batch_proposals.append(proposals)
    return batch_proposals
