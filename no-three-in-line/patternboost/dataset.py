from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import Dataset

from supervised.types import SupervisedSample

from .config import PoolSamplingConfig
from .noise import build_supervised_sample_from_clean, sample_mask_ratio
from .top_pool import TopPool


def _stable_seed(*parts: object) -> int:
    """Return a deterministic integer seed for PatternBoost dataset sampling."""

    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class PatternBoostDenoisingDataset(Dataset[SupervisedSample]):
    """Generate supervised denoising samples from a live PatternBoost TopPool."""

    def __init__(
        self,
        top_pool: TopPool,
        mask_ratios: list[float],
        loss_config,
        num_samples: int,
        seed: int,
        pool_sampling_config: PoolSamplingConfig | None = None,
        split_name: str = "patternboost",
    ) -> None:
        """Initialize the dynamic TopPool-backed training dataset.

        ``top_pool`` provides clean canonical states, ``mask_ratios`` controls
        random training noise, ``loss_config`` supplies token weights,
        ``num_samples`` fixes epoch length, ``seed``/``split_name`` make the
        generated sample set deterministic across epochs, and
        ``pool_sampling_config`` controls which TopPool entries are favored.
        The constructor stores references but does not mutate the pool.
        """

        self.top_pool = top_pool
        self.mask_ratios = list(mask_ratios)
        self.loss_config = loss_config
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.pool_sampling_config = pool_sampling_config
        self.split_name = split_name
        self.epoch = 0

    def __len__(self) -> int:
        """Return the configured number of dynamic samples per epoch."""

        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch index while keeping the sample set fixed.

        ``epoch`` is accepted for compatibility with the supervised trainer.
        The method stores the value for diagnostics, returns ``None``, and
        deliberately does not affect sampling so all epochs reuse the same
        PatternBoost training set.
        """

        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> SupervisedSample:
        """Return one noised TopPool sample in supervised batch format."""

        rng = np.random.default_rng(_stable_seed(self.seed, self.split_name, index))
        entry = self.top_pool.sample_seed(rng, self.pool_sampling_config)
        ratio = sample_mask_ratio(self.mask_ratios, rng)
        return build_supervised_sample_from_clean(
            entry.slot_state_ids,
            ratio,
            rng,
            self.loss_config,
        )
