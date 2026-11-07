from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from project_config import load_project_config

@dataclass(frozen=True)
class PointFeatureConfig:
    """Describe which raw point-feature blocks are enabled.

    Each field toggles one optional feature family or configures its
    hyperparameters. The dataclass stores configuration only, returns no value
    on construction, and does not mutate external state.
    """
    use_uv: bool = True
    use_uv_sumdiff: bool = True
    use_uv_product: bool = True
    use_fourier_u: bool = True
    use_fourier_v: bool = True
    use_fourier_sumdiff: bool = True
    fourier_frequencies: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    use_y_onehot: bool = False
    use_x_onehot: bool = False
    use_mod_features: bool = False
    mod_bases: tuple[int, ...] = ()
    use_mod_x: bool = True
    use_mod_y: bool = True
    use_mod_sum: bool = False
    use_mod_diff: bool = False


def point_feature_config_from_mapping(mapping: Mapping[str, Any] | None = None) -> PointFeatureConfig:
    """Build a :class:`PointFeatureConfig` from a mapping of YAML-style values.

    ``mapping`` may omit keys, in which case dataclass defaults are used. The
    function returns a validated ``PointFeatureConfig``, does not mutate the
    input mapping, and assumes only known point-feature keys are present.
    """
    defaults = PointFeatureConfig()
    config_fields = {field.name for field in fields(PointFeatureConfig)}
    mapping_dict = dict(mapping or {})
    unknown_keys = sorted(set(mapping_dict) - config_fields)
    if unknown_keys:
        raise ValueError(f"Unknown point feature config keys: {unknown_keys}")

    kwargs: dict[str, Any] = {}
    for name in config_fields:
        value = mapping_dict.get(name, getattr(defaults, name))
        if name in {"fourier_frequencies", "mod_bases"}:
            value = tuple(value)
        kwargs[name] = value
    return PointFeatureConfig(**kwargs)


def load_point_feature_config(path: str | Path | None = None) -> PointFeatureConfig:
    """Load the point-feature section of the project config.

    ``path`` optionally overrides the root ``config.yaml`` location. The
    function returns a ``PointFeatureConfig``, does not mutate external state,
    and assumes the ``point_features`` section is either absent or a mapping of
    valid point-feature keys.
    """
    loaded = load_project_config(path)
    point_features = loaded.get("point_features", loaded)
    if not isinstance(point_features, Mapping):
        raise ValueError("point_features config must be a mapping")
    return point_feature_config_from_mapping(point_features)


def build_fourier_features(z: Tensor, frequencies: tuple[float, ...]) -> Tensor:
    """Return sin/cos Fourier features for a scalar tensor input.

    ``z`` is any tensor shape on any device and is converted to ``float32`` for
    feature construction. ``frequencies`` is an ordered tuple of frequency
    multipliers. The function returns a tensor of shape ``z.shape +
    (2 * len(frequencies),)`` on the same device, does not mutate the inputs,
    and assumes each frequency is numeric.
    """
    if not frequencies:
        return z.new_zeros((*z.shape, 0), dtype=torch.float32)

    z_float = z.to(dtype=torch.float32)
    freq_tensor = z_float.new_tensor(frequencies)
    angles = (2.0 * math.pi) * z_float.unsqueeze(-1) * freq_tensor
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def get_point_raw_feature_dim(n: int, config: PointFeatureConfig) -> int:
    """Return the raw concatenated feature width implied by ``config``.

    ``n`` is the grid size used by one-hot and modular feature blocks, and
    ``config`` controls which blocks are active. The function returns an
    integer feature dimension, does not mutate state, and assumes ``n >= 1``.
    """
    fourier_dim = 2 * len(config.fourier_frequencies)
    dim = 0

    if config.use_uv:
        dim += 2
    if config.use_uv_sumdiff:
        dim += 2
    if config.use_uv_product:
        dim += 1
    if config.use_fourier_u:
        dim += fourier_dim
    if config.use_fourier_v:
        dim += fourier_dim
    if config.use_fourier_sumdiff:
        dim += 2 * fourier_dim
    if config.use_y_onehot:
        dim += n
    if config.use_x_onehot:
        dim += n
    if config.use_mod_features:
        for base in config.mod_bases:
            if config.use_mod_x:
                dim += base
            if config.use_mod_y:
                dim += base
            if config.use_mod_sum:
                dim += base
            if config.use_mod_diff:
                dim += base

    return dim


def build_point_raw_features(
    x: Tensor,
    y: Tensor,
    n: int,
    config: PointFeatureConfig,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build concatenated raw point features for point coordinates.

    ``x`` and ``y`` are integer tensors or tensor-like inputs with matching
    broadcasted shape, representing point coordinates on an ``n x n`` grid.
    The result is a ``float32`` tensor of shape ``[..., d_raw]`` on ``device``.
    The function does not mutate the inputs and assumes coordinates are meant
    for the same board size and feature configuration.
    """
    x_long = torch.as_tensor(x, dtype=torch.long, device=device)
    y_long = torch.as_tensor(y, dtype=torch.long, device=device)
    denom = float(max(n - 1, 1))
    u = x_long.to(dtype=torch.float32) / denom
    v = y_long.to(dtype=torch.float32) / denom

    blocks: list[Tensor] = []

    if config.use_uv:
        blocks.append(torch.stack((u, v), dim=-1))
    if config.use_uv_sumdiff:
        blocks.append(torch.stack((u + v, u - v), dim=-1))
    if config.use_uv_product:
        blocks.append((u * v).unsqueeze(-1))
    if config.use_fourier_u:
        blocks.append(build_fourier_features(u, config.fourier_frequencies))
    if config.use_fourier_v:
        blocks.append(build_fourier_features(v, config.fourier_frequencies))
    if config.use_fourier_sumdiff:
        blocks.append(build_fourier_features(u + v, config.fourier_frequencies))
        blocks.append(build_fourier_features(u - v, config.fourier_frequencies))
    if config.use_y_onehot:
        blocks.append(F.one_hot(y_long, num_classes=n).to(dtype=torch.float32))
    if config.use_x_onehot:
        blocks.append(F.one_hot(x_long, num_classes=n).to(dtype=torch.float32))
    if config.use_mod_features:
        for base in config.mod_bases:
            if config.use_mod_x:
                blocks.append(F.one_hot(torch.remainder(x_long, base), num_classes=base).to(dtype=torch.float32))
            if config.use_mod_y:
                blocks.append(F.one_hot(torch.remainder(y_long, base), num_classes=base).to(dtype=torch.float32))
            if config.use_mod_sum:
                blocks.append(
                    F.one_hot(torch.remainder(x_long + y_long, base), num_classes=base).to(dtype=torch.float32)
                )
            if config.use_mod_diff:
                blocks.append(
                    F.one_hot(torch.remainder(x_long - y_long, base), num_classes=base).to(dtype=torch.float32)
                )

    if not blocks:
        return u.new_zeros((*u.shape, 0))

    return torch.cat(blocks, dim=-1)


def build_point_feature_table(
    n: int,
    config: PointFeatureConfig,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build the full raw feature table for every grid point.

    ``n`` is the grid size and ``config`` selects the enabled feature blocks.
    The function returns a ``float32`` tensor of shape ``[n, n, d_raw]`` on the
    requested device, does not mutate state, and assumes ``n >= 1``.
    """
    coords = torch.arange(n, device=device, dtype=torch.long)
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing="ij")
    return build_point_raw_features(x_grid, y_grid, n, config, device=device)


def build_column_anchor_table(point_feature_table: Tensor) -> Tensor:
    """Mean-pool raw point features over rows within each column.

    ``point_feature_table`` is expected to have shape ``[n, n, d_raw]`` with
    dimensions ``[x, y, feature]``. The function returns a tensor of shape
    ``[n, d_raw]``, does not mutate the input tensor, and assumes the second
    dimension indexes candidate ``y`` positions for the same column.
    """
    return point_feature_table.mean(dim=1)
