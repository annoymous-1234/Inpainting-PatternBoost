"""Baseline proposal policies for the circle packing environment."""

from circle_pack.policies.noisy_hint import NoisyHintPolicy
from circle_pack.policies.random_legal import RandomLegalPolicy


def make_policy(name, config):
    """Construct a baseline proposal policy by name."""

    if name == "random_legal":
        return RandomLegalPolicy(config)
    if name == "noisy_hint":
        return NoisyHintPolicy(config)
    raise ValueError(f"Unknown policy name: {name}")


__all__ = ["RandomLegalPolicy", "NoisyHintPolicy", "make_policy"]
