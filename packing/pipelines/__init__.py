"""Lightweight training and inference pipeline utilities.

Package-level exports stay intentionally minimal to avoid circular imports
between the sampler layer and the outer-loop controller layer.
"""

from pipelines.types import OuterLoopRecord, ProposalBundle

__all__ = ["OuterLoopRecord", "ProposalBundle"]
