from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from env import NoThreeInLineEnv
from models import current_point_count_from_slot_states, slot_states_to_state_ids
from saturation import greedy_saturate

from .config import PatternBoostConfig
from .legal import legal_counts_by_slot, legal_rows_for_masked_slots
from .remask import generation_remask
from .selector import build_proposal_slots_from_selected, select_slots_for_step
from .state import (
    canonicalize_state_ids,
    filled_count,
    state_ids_to_env_reset_args,
)
from .types import DecodeStepStats, GenerationResult, TopPoolEntry


@dataclass
class _CandidateState:
    """Track one in-flight PatternBoost candidate during batched decoding."""

    env: NoThreeInLineEnv
    seed_hash: str
    remask_mode: str
    mask_ratio: float
    seed_score: int = 0
    step_stats: list[DecodeStepStats] = field(default_factory=list)
    active: bool = True


class PatternBoostGenerator:
    """Generate PatternBoost candidates from TopPool seeds using a model."""

    def __init__(
        self,
        model: torch.nn.Module | None,
        config: PatternBoostConfig,
        device: torch.device,
        rng: np.random.Generator,
        mask_ratios: list[float] | None = None,
        remask_mode_weights: list[float] | None = None,
    ) -> None:
        """Initialize the generator with model, config, device, and RNG.

        ``model`` supplies state logits and may be ``None`` when
        ``config.generation.completion_operator`` is ``"random"``, in which case
        no forward pass ever runs. ``config`` controls remask and decode,
        ``device`` receives model tensors, and ``rng`` drives stochastic
        remask/selection/proposal sampling. Optional ``mask_ratios`` and
        ``remask_mode_weights`` override the static config for one generation.
        The constructor mutates only the generator instance and assumes the
        model is already on ``device``.
        """

        self.model = model
        self.config = config
        self.device = device
        self.rng = rng
        self.mask_ratios = list(config.mask_ratios if mask_ratios is None else mask_ratios)
        self.remask_mode_weights = remask_mode_weights

    def generate(self, seed_entry: TopPoolEntry) -> GenerationResult:
        """Generate one candidate state from a TopPool seed.

        ``seed_entry`` provides a canonical legal state. The method returns a
        ``GenerationResult``, mutates only a temporary environment, and assumes
        the environment remains responsible for duplicate resolution and
        no-three-in-line repair.
        """

        candidate = self._initialize_candidate(seed_entry)
        if self._uses_random_operator():
            return self._finalize_candidate(candidate)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for step_index, quota_ratio in enumerate(self.config.decode.quota_ratios):
                    stats = self._decode_step(candidate.env, step_index, quota_ratio)
                    if stats is None:
                        break
                    candidate.step_stats.append(stats)
        finally:
            if was_training:
                self.model.train()
        return self._finalize_candidate(candidate)

    def generate_batch(self, seed_entries: list[TopPoolEntry]) -> list[GenerationResult]:
        """Generate candidates from a batch of TopPool seeds.

        ``seed_entries`` is a batch of independent TopPool entries. Model
        forward inputs are ``slot_state_ids`` with shape ``[B_active, 2n]`` and
        dtype ``torch.long`` on ``self.device``, ``progress_ids`` with shape
        ``[B_active]`` and dtype ``torch.long``, and ``current_point_counts``
        with shape ``[B_active]`` and dtype ``torch.float32``. The method
        returns one ``GenerationResult`` per seed in input order, mutates only
        temporary per-candidate environments, and assumes the shared model is
        already on ``self.device``.
        """

        if not seed_entries:
            return []
        candidates = [self._initialize_candidate(seed_entry) for seed_entry in seed_entries]
        if self._uses_random_operator():
            return [self._finalize_candidate(candidate) for candidate in candidates]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for step_index, quota_ratio in enumerate(self.config.decode.quota_ratios):
                    active_indices = [index for index, candidate in enumerate(candidates) if candidate.active]
                    if not active_indices:
                        break
                    active_envs = [candidates[index].env for index in active_indices]
                    slot_state_ids = torch.stack(
                        [slot_states_to_state_ids(env.slot_states, env.n) for env in active_envs],
                        dim=0,
                    ).to(self.device)
                    point_counts = [current_point_count_from_slot_states(env.slot_states) for env in active_envs]
                    progress_ids = torch.tensor(point_counts, dtype=torch.long, device=self.device)
                    current_point_counts = torch.tensor(point_counts, dtype=torch.float32, device=self.device)
                    outputs = self.model(slot_state_ids, progress_ids, current_point_counts)
                    state_logits_batch = outputs["state_logits"]

                    # The model forward is batched, while legal proposal search
                    # and environment mutation remain per candidate.
                    for batch_offset, candidate_index in enumerate(active_indices):
                        candidate = candidates[candidate_index]
                        stats = self._decode_step_for_env(
                            candidate.env,
                            state_logits_batch[batch_offset],
                            step_index,
                            quota_ratio,
                        )
                        if stats is None:
                            candidate.active = False
                            continue
                        candidate.step_stats.append(stats)
        finally:
            if was_training:
                self.model.train()
        return [self._finalize_candidate(candidate) for candidate in candidates]

    def _uses_random_operator(self) -> bool:
        """Return whether completion should skip the model and only saturate.

        Under ``completion_operator="random"`` the remasked slots are filled by
        the environment's own ``greedy_saturate``, which already picks a legal
        row uniformly at random for the slot it selects. That makes the learned
        decoder the single removed ingredient, and it means the run needs no
        model at all: no forward pass and no training.
        """

        return self.config.generation.completion_operator == "random"

    def _initialize_candidate(self, seed_entry: TopPoolEntry) -> _CandidateState:
        """Create one remasked environment and metadata record for generation."""

        remask_mode = str(
            self.rng.choice(
                self.config.generation.remask_modes,
                p=self._remask_mode_probabilities(),
            )
        )
        mask_ratio = float(self.rng.choice(self.mask_ratios))
        partial_state, _ = generation_remask(
            seed_entry.slot_state_ids,
            self.config.n,
            mask_ratio,
            remask_mode,
            self.rng,
        )
        init_points, init_assignment = state_ids_to_env_reset_args(partial_state)
        env = NoThreeInLineEnv(
            n=self.config.n,
            num_slots=2 * self.config.n,
            proposal_search_config={
                "num_search_attempts": self.config.decode.env_num_search_attempts,
                "allow_existing_deletions": self.config.decode.env_allow_existing_deletions,
            },
        )
        env.reset(init_points=init_points, init_slot_assignment=init_assignment)
        return _CandidateState(
            env=env,
            seed_hash=seed_entry.canonical_hash,
            seed_score=int(seed_entry.score),
            remask_mode=remask_mode,
            mask_ratio=mask_ratio,
        )

    def _finalize_candidate(self, candidate: _CandidateState) -> GenerationResult:
        """Apply optional saturation and return one immutable generation result."""

        if self.config.saturation.enabled:
            greedy_saturate(
                candidate.env,
                self.rng,
                max_steps=self.config.saturation.max_steps,
                slot_order=self.config.saturation.slot_order,
                y_order=self.config.saturation.y_order,
            )

        final_state = slot_states_to_state_ids(candidate.env.slot_states, candidate.env.n).cpu().numpy()
        canonical_state = canonicalize_state_ids(final_state, self.config.n)
        return GenerationResult(
            seed_hash=candidate.seed_hash,
            seed_score=candidate.seed_score,
            remask_mode=candidate.remask_mode,
            mask_ratio=candidate.mask_ratio,
            slot_state_ids=canonical_state,
            score=filled_count(canonical_state),
            inserted=False,
            step_stats=candidate.step_stats,
        )

    def _remask_mode_probabilities(self) -> list[float] | None:
        """Return normalized generation remask probabilities if configured.

        The returned list has one probability per configured remask mode, or
        ``None`` to request NumPy's uniform choice. The method does not mutate
        state and assumes config validation has checked weight length and sum.
        """

        weights = self.config.generation.remask_mode_weights if self.remask_mode_weights is None else self.remask_mode_weights
        if weights is None:
            return None
        total = float(sum(weights))
        return [float(weight) / total for weight in weights]

    def _decode_step(
        self,
        env: NoThreeInLineEnv,
        step_index: int,
        quota_ratio: float,
    ) -> DecodeStepStats | None:
        """Run one model-forward, slot-selection, proposal, and env step."""

        slot_state_ids = slot_states_to_state_ids(env.slot_states, env.n).unsqueeze(0).to(self.device)
        current_point_count = current_point_count_from_slot_states(env.slot_states)
        progress_ids = torch.tensor([current_point_count], dtype=torch.long, device=self.device)
        current_point_counts = torch.tensor([float(current_point_count)], dtype=torch.float32, device=self.device)
        outputs = self.model(slot_state_ids, progress_ids, current_point_counts)
        state_logits = outputs["state_logits"].squeeze(0)
        return self._decode_step_for_env(env, state_logits, step_index, quota_ratio)

    def _decode_step_for_env(
        self,
        env: NoThreeInLineEnv,
        state_logits: torch.Tensor,
        step_index: int,
        quota_ratio: float,
    ) -> DecodeStepStats | None:
        """Apply one decoded logit slice to one environment.

        ``state_logits`` has shape ``[2n, n + 1]``, floating dtype, and lives on
        ``self.device``. The method returns step diagnostics or ``None`` when
        the candidate cannot progress, mutates only ``env``, and assumes logits
        were produced from the current environment state before this call.
        """

        slot_state_ids = slot_states_to_state_ids(env.slot_states, env.n)
        masked_slot_ids = [
            int(slot_id)
            for slot_id, state_id in enumerate(slot_state_ids.tolist())
            if state_id == 0
        ]
        if not masked_slot_ids:
            return None

        legal_rows_by_column = legal_rows_for_masked_slots(env, masked_slot_ids)
        legal_counts = legal_counts_by_slot(masked_slot_ids, legal_rows_by_column)
        feasible_slot_ids = [
            slot_id
            for slot_id in masked_slot_ids
            if legal_counts.get(slot_id, 0) > 0
        ]
        if not feasible_slot_ids:
            return None

        quota = max(1, int(round(float(quota_ratio) * len(masked_slot_ids))))
        quota = min(quota, len(feasible_slot_ids))
        selected_slot_ids = select_slots_for_step(
            state_logits,
            feasible_slot_ids,
            legal_rows_by_column,
            legal_counts,
            quota,
            self.config.selector,
            self.config.decode,
            self.rng,
        )
        proposal_slots = build_proposal_slots_from_selected(
            state_logits,
            selected_slot_ids,
            legal_rows_by_column,
            self.config.decode,
            self.rng,
        )
        if not proposal_slots:
            return None

        point_count_before = int(env.observe()["num_selected"])
        observation, _, _, info = env.step(proposal_slots)
        return DecodeStepStats(
            step_index=step_index,
            quota=quota,
            selected_slot_ids=list(selected_slot_ids),
            accepted_points=list(info["accepted_proposal_points"]),
            rejected_points=list(info["rejected_proposal_points"]),
            point_count_before=point_count_before,
            point_count_after=int(observation["num_selected"]),
        )
