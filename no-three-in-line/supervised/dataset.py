from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from .config import SupervisedConfig
from .corruption import corrupt_clean_state
from .losses import compute_token_weights
from .targets import build_full_target, build_partial_target, choose_order_id
from .types import EncodedSolutionRecord, SupervisedSample


def _stable_seed(*parts: object) -> int:
    """Return a deterministic integer seed derived from the given parts."""

    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _build_supervision_mask(
    input_state_ids: torch.Tensor,
    clean_state_ids: torch.Tensor,
    target_state_ids: torch.Tensor,
    task_type: str,
    config: SupervisedConfig,
) -> torch.Tensor:
    """Return the unbatched per-slot supervision mask for one sample.

    ``input_state_ids``, ``clean_state_ids``, and ``target_state_ids`` are
    ``LongTensor[num_slots]`` tensors on the same device and use ``0`` to mark
    masked or empty slots. ``task_type`` selects the full-versus-partial rule
    and ``config`` provides the configured supervision scope for each task. The
    function returns a ``FloatTensor[num_slots]`` with values in ``{0.0, 1.0}``,
    does not mutate its inputs, and assumes the target and clean tensors share
    the same slot ordering as the input tensor.
    """

    if task_type == "full":
        if config.loss.full_supervision_mode == "masked_filled_only":
            return torch.logical_and(input_state_ids.eq(0), target_state_ids.ne(0)).to(dtype=torch.float32)
        if config.loss.full_supervision_mode == "all_tokens":
            return torch.ones_like(target_state_ids, dtype=torch.float32)
        raise ValueError(f"unsupported full supervision mode: {config.loss.full_supervision_mode}")
    if task_type == "partial":
        if config.loss.partial_supervision_mode == "all_missing":
            return torch.logical_and(input_state_ids.eq(0), clean_state_ids.ne(0)).to(dtype=torch.float32)
        if config.loss.partial_supervision_mode == "restored_only":
            return torch.logical_and(input_state_ids.eq(0), target_state_ids.ne(0)).to(dtype=torch.float32)
        if config.loss.partial_supervision_mode == "all_tokens":
            return torch.ones_like(target_state_ids, dtype=torch.float32)
        raise ValueError(f"unsupported partial supervision mode: {config.loss.partial_supervision_mode}")
    raise ValueError(f"unsupported task type: {task_type}")


class SupervisedDenoisingDataset(Dataset[SupervisedSample]):
    """Dynamically generate supervised denoising samples from clean solutions."""

    def __init__(
        self,
        records: Sequence[EncodedSolutionRecord],
        config: SupervisedConfig,
        split_name: str,
        seed: int,
        mode: str | None = None,
        partial_prob: float | None = None,
    ) -> None:
        """Initialize the dataset from preprocessed clean solution records."""

        self.records = list(records)
        self.config = config
        self.split_name = split_name
        self.base_seed = seed
        self.mode = config.dataset.mode if mode is None else mode
        self.partial_prob = config.dataset.partial_prob if partial_prob is None else partial_prob
        self.epoch = 0

    def __len__(self) -> int:
        """Return the number of clean solution records in the dataset."""

        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index used in deterministic per-item RNG seeding."""

        self.epoch = epoch
        # self.epoch = 3  # ignore epoch for now to disable curriculum changes across epochs

    def _make_sample_rng(self, index: int) -> random.Random:
        """Return the deterministic RNG used for one dataset item at one epoch."""

        record = self.records[index]
        return random.Random(_stable_seed(self.base_seed, self.split_name, self.epoch, index, record.solution_id))

    def _sample_task_type(self, rng: random.Random) -> str:
        """Sample which supervised task type to build for the current item."""

        if self.mode == "full":
            return "full"
        if self.mode == "partial":
            return "partial"
        if self.mode != "mixed":
            raise ValueError(f"unsupported dataset mode: {self.mode}")
        return "partial" if rng.random() < self.partial_prob else "full"

    def _build_sample(self, record: EncodedSolutionRecord, rng: random.Random) -> SupervisedSample:
        """Build one dynamic supervised sample from a clean encoded solution."""

        corruption_result = corrupt_clean_state(record, rng, self.config.corruption)
        task_type = self._sample_task_type(rng)
        metadata: dict[str, object] = {
            "solution_id": record.solution_id,
            "task_type": task_type,
            "step_id": corruption_result.step_id,
            "current_point_count": corruption_result.current_point_count,
            "masked_ratio_actual": corruption_result.masked_ratio_actual,
        }

        if task_type == "full":
            target_state_ids = build_full_target(record)
        else:
            order_id = choose_order_id(record, rng, self.config.partial_target.use_multiple_orders)
            target_state_ids, partial_metadata = build_partial_target(
                record=record,
                corruption_result=corruption_result,
                order_id=order_id,
                config=self.config.partial_target,
                total_steps=self.config.corruption.total_steps,
            )
            metadata.update(partial_metadata)

        input_tensor = torch.tensor(corruption_result.input_state_ids, dtype=torch.long)
        clean_tensor = torch.tensor(record.clean_state_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_state_ids, dtype=torch.long)
        supervision_mask = _build_supervision_mask(
            input_state_ids=input_tensor,
            clean_state_ids=clean_tensor,
            target_state_ids=target_tensor,
            task_type=task_type,
            config=self.config,
        )
        return SupervisedSample(
            input_state_ids=input_tensor,
            target_state_ids=target_tensor,
            step_id=torch.tensor(corruption_result.step_id, dtype=torch.long),
            current_point_count=torch.tensor(corruption_result.current_point_count, dtype=torch.float32),
            token_weights=compute_token_weights(target_tensor, self.config.loss),
            supervision_mask=supervision_mask,
            task_type=task_type,
            metadata=metadata,
        )

    def __getitem__(self, index: int) -> SupervisedSample:
        """Return one dynamically generated supervised denoising sample."""

        rng = self._make_sample_rng(index)
        return self._build_sample(self.records[index], rng)


def collate_supervised_batch(samples: list[SupervisedSample]) -> dict[str, object]:
    """Collate unbatched supervised samples into one batch dictionary.

    ``samples`` contains per-example CPU tensors with a shared slot dimension.
    The returned dictionary stacks ``input_state_ids``, ``target_state_ids``,
    ``token_weights``, and ``supervision_mask`` into ``[B, S]`` tensors and
    stacks ``step_ids`` and ``current_point_counts`` into ``[B]`` tensors.
    ``task_types`` and ``metadata`` remain Python lists of length ``B``. The
    function returns a new batch mapping, does not mutate the input samples,
    and assumes every sample uses the same slot ordering and slot count.
    """

    return {
        "input_state_ids": torch.stack([sample.input_state_ids for sample in samples], dim=0),
        "target_state_ids": torch.stack([sample.target_state_ids for sample in samples], dim=0),
        "step_ids": torch.stack([sample.step_id for sample in samples], dim=0),
        "current_point_counts": torch.stack([sample.current_point_count for sample in samples], dim=0),
        "token_weights": torch.stack([sample.token_weights for sample in samples], dim=0),
        "supervision_mask": torch.stack([sample.supervision_mask for sample in samples], dim=0),
        "task_types": [sample.task_type for sample in samples],
        "metadata": [sample.metadata for sample in samples],
    }
