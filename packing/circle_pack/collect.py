"""Rollout collection helpers for future supervised, RL, or search pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np

from circle_pack.types import Transition


def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy an observation dictionary containing numpy arrays."""

    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def collect_episode(
    env: Any,
    policy: Any,
    *,
    seed: int | None = None,
    record_snapshots: bool = False,
) -> tuple[list[Transition], dict[str, Any]]:
    """Collect one full rollout from `env` using `policy`."""

    observation = env.reset(seed=seed)
    policy.reset(env, seed=seed)

    transitions: list[Transition] = []
    done = False
    total_reward = 0.0

    while not done:
        proposal = np.asarray(policy.propose(observation), dtype=env.dtype)
        snapshot = env.snapshot() if record_snapshots else None
        next_observation, reward, done, info = env.step(proposal)
        info = dict(info)
        if snapshot is not None:
            info["snapshot_before"] = snapshot
        transitions.append(
            Transition(
                observation=_copy_observation(observation),
                proposal=proposal.copy(),
                reward=float(reward),
                done=bool(done),
                info=info,
            )
        )
        total_reward += reward
        observation = next_observation

    summary = {
        "num_steps": len(transitions),
        "total_reward": float(total_reward),
        "final_sum_active_r": float(observation["sum_active_r"]),
        "final_layout_score": None if observation.get("layout_score") is None else float(observation["layout_score"]),
        "num_active": int(np.sum(observation["active_mask"])),
        "done": bool(done),
    }
    return transitions, summary
