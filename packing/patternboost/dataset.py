"""TopPool-backed supervised data for PatternBoost."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from circle_pack.config import ProjectConfig
from patternboost.edit import sample_edit_set
from patternboost.top_pool import TopPool
from pipelines.data import build_jittered_draft, make_patternboost_supervised_example


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class PatternBoostSupervisedDataset(Dataset):
    """Generate FM/DDPM supervised samples from a live PatternBoost TopPool."""

    def __init__(
        self,
        top_pool: TopPool,
        cfg: ProjectConfig,
        *,
        mode: str,
        num_samples: int,
        seed: int,
        split_name: str = "patternboost",
    ) -> None:
        self.top_pool = top_pool
        self.cfg = cfg
        self.mode = mode
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.split_name = split_name
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if mode not in {"fm", "ddpm"}:
            raise ValueError("mode must be 'fm' or 'ddpm'.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(_stable_seed(self.seed, self.split_name, index))
        seed_entry = self.top_pool.sample_seed(rng, self.cfg.patternboost.pool.sampling)
        edit_set = sample_edit_set(seed_entry.centers, seed_entry.active_mask, self.cfg.patternboost.edit, rng)
        fixed_mask = ~edit_set.edit_mask
        _, draft_centers = build_jittered_draft(seed_entry.centers, fixed_mask, rng, cfg=self.cfg)
        example = make_patternboost_supervised_example(
            seed_entry.centers,
            edit_set.edit_mask,
            self.mode,
            draft_centers=draft_centers,
            use_draft_match=bool(self.cfg.patternboost.training.use_draft_match),
            cfg=self.cfg,
            rng=rng,
        )
        example["seed_score"] = torch.as_tensor(seed_entry.score, dtype=self.cfg.torch_dtype)
        return example
