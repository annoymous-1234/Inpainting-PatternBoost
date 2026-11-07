from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from proposal_eval import CandidatePoint, ProposalSlot

from .config import DecodeConfig, SelectorConfig


def _legal_probabilities(slot_logits: Tensor, legal_rows: np.ndarray, temperature: float) -> Tensor:
    """Return a probability distribution over legal rows for one slot.

    ``slot_logits`` has shape ``[n + 1]`` and ``legal_rows`` is a boolean NumPy
    mask of shape ``[n]``. The function returns a CPU tensor of shape
    ``[num_legal]`` containing probabilities, does not mutate inputs, and
    assumes ``slot_logits[1:]`` contains row logits.
    """

    legal_indices = torch.as_tensor(np.flatnonzero(legal_rows), dtype=torch.long, device=slot_logits.device)
    if legal_indices.numel() == 0:
        return slot_logits.new_empty((0,))
    row_logits = slot_logits[1:].index_select(0, legal_indices) / float(temperature)
    return torch.softmax(row_logits, dim=0)


def confidence_score(
    slot_logits: Tensor,
    legal_rows: np.ndarray,
    metric: str,
    temperature: float,
) -> float:
    """Compute one slot's confidence over legal row candidates.

    ``slot_logits`` is a tensor of shape ``[n + 1]`` and ``legal_rows`` marks
    legal rows for the slot column. The function returns a scalar confidence,
    does not mutate tensors, and assumes illegal rows should not affect the
    model certainty used for slot selection.
    """

    probabilities = _legal_probabilities(slot_logits, legal_rows, temperature)
    if probabilities.numel() == 0:
        return float("-inf")
    sorted_probs = torch.sort(probabilities, descending=True).values
    if metric == "top_prob":
        return float(sorted_probs[0].detach().cpu().item())
    if metric == "margin":
        second = sorted_probs[1] if sorted_probs.numel() > 1 else sorted_probs.new_tensor(0.0)
        return float((sorted_probs[0] - second).detach().cpu().item())
    if metric == "negative_entropy":
        tiny = torch.finfo(probabilities.dtype).tiny
        entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum()
        return float((-entropy).detach().cpu().item())
    raise ValueError(f"unsupported confidence metric: {metric}")


def legal_choice_score(legal_count: int, n: int, metric: str) -> float:
    """Return a normalized legal-choice priority component for one slot.

    ``legal_count`` is the number of legal rows, ``n`` is the grid size, and
    ``metric`` selects MRV or easy-first. The function returns a scalar score,
    does not mutate inputs, and assumes zero legal rows are filtered upstream.
    """

    safe_count = max(int(legal_count), 1)
    if metric == "mrv":
        return 1.0 / float(safe_count)
    if metric == "easy_first":
        return float(safe_count) / float(max(n, 1))
    raise ValueError(f"unsupported legal choice metric: {metric}")


def select_slots_for_step(
    state_logits: Tensor,
    masked_slot_ids: list[int],
    legal_rows_by_column: dict[int, np.ndarray],
    legal_counts: dict[int, int],
    quota: int,
    selector_config: SelectorConfig,
    decode_config: DecodeConfig,
    rng: np.random.Generator,
) -> list[int]:
    """Select masked slots to propose in one decode step.

    ``state_logits`` has shape ``[2n, n + 1]`` on the model device. The slot
    inputs describe currently masked slots and legal-row masks. The function
    returns selected slot ids sorted by descending stochastic priority, does
    not mutate model tensors, and assumes ``quota`` has already been capped to
    the available masked-slot count.
    """

    if quota <= 0:
        return []
    n = state_logits.shape[-1] - 1
    scored_slots: list[tuple[float, int]] = []
    for slot_id in masked_slot_ids:
        legal_count = int(legal_counts.get(slot_id, 0))
        if legal_count <= 0:
            continue
        legal_rows = legal_rows_by_column[int(slot_id // 2)]
        confidence = confidence_score(
            state_logits[slot_id],
            legal_rows,
            selector_config.confidence_metric,
            decode_config.y_temperature,
        )
        legal_score = legal_choice_score(legal_count, n, selector_config.legal_choice_metric)
        noise = 0.0 if selector_config.noise_std == 0.0 else float(rng.normal(0.0, selector_config.noise_std))
        priority = (
            selector_config.confidence_weight * confidence
            + selector_config.legal_weight * legal_score
            + noise
        )
        scored_slots.append((priority, int(slot_id)))

    ranked = sorted(scored_slots, key=lambda item: (-item[0], item[1]))
    return [slot_id for _, slot_id in ranked[:quota]]


def sample_candidate_ys(
    slot_logits: Tensor,
    legal_rows: np.ndarray,
    num_candidates: int,
    temperature: float,
    rng: np.random.Generator,
) -> list[int]:
    """Sample candidate rows without replacement from legal logits.

    ``slot_logits`` has shape ``[n + 1]`` and ``legal_rows`` marks legal rows.
    The function returns sampled row ids in fallback order, does not mutate
    tensors, and uses temperature-softmax probabilities only over legal rows to
    avoid wasting proposal slots on immediately illegal points.
    """

    legal_indices = np.flatnonzero(legal_rows)
    if legal_indices.size == 0 or num_candidates <= 0:
        return []
    row_logits = slot_logits[1:].detach().cpu().to(dtype=torch.float64).numpy()
    legal_logits = row_logits[legal_indices] / float(temperature)
    stable_logits = legal_logits - np.max(legal_logits)
    weights = np.exp(stable_logits)
    probabilities = weights / np.sum(weights)
    count = min(int(num_candidates), legal_indices.size)
    sampled = rng.choice(legal_indices, size=count, replace=False, p=probabilities)
    return [int(row) for row in sampled.tolist()]


def build_proposal_slots_from_selected(
    state_logits: Tensor,
    selected_slot_ids: list[int],
    legal_rows_by_column: dict[int, np.ndarray],
    decode_config: DecodeConfig,
    rng: np.random.Generator,
) -> list[ProposalSlot]:
    """Build env-compatible proposal slots from selected slot ids.

    ``state_logits`` has shape ``[2n, n + 1]`` and ``selected_slot_ids`` names
    slots chosen for this step. The function returns ``ProposalSlot`` objects,
    does not mutate tensors, and assigns artificial descending scores so the
    environment fallback order follows the sampled order rather than raw logits.
    """

    proposals: list[ProposalSlot] = []
    for slot_id in selected_slot_ids:
        legal_rows = legal_rows_by_column[int(slot_id // 2)]
        sampled_ys = sample_candidate_ys(
            state_logits[slot_id],
            legal_rows,
            decode_config.num_candidates_per_slot,
            decode_config.y_temperature,
            rng,
        )
        candidates = [
            CandidatePoint(
                x=int(slot_id // 2),
                y=int(y),
                score=float(len(sampled_ys) - index),
                slot_id=int(slot_id),
            )
            for index, y in enumerate(sampled_ys)
        ]
        if candidates:
            proposals.append(ProposalSlot(slot_id=int(slot_id), candidates=candidates))
    return proposals
