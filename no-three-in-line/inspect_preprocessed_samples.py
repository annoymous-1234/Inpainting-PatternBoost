from __future__ import annotations

import argparse
import random
from itertools import combinations
from pathlib import Path

import torch

from geometry import Point
from render import print_board
from supervised.config import load_supervised_config
from supervised.preprocess import load_preprocessed_cache
from supervised.types import EncodedSolutionRecord


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preprocessed-sample inspection.

    The function reads CLI flags from ``sys.argv`` to configure which cache to
    inspect, how many records to print, and whether to take the first or a
    random subset of samples. It returns an ``argparse.Namespace``, does not
    mutate repository state, and assumes the caller provides arguments that fit
    the declared schema.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load K preprocessed supervised samples, report pairwise overlap and "
            "Hamming distance statistics, and render each board."
        )
    )
    parser.add_argument("-k", type=int, required=True, help="Number of cached records to inspect.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional .pt preprocessing cache path. Defaults to the configured split cache.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional project config path used when resolving the default cache path.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Configured cache split to use when --cache-path is omitted.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional board size override. Defaults to the cache payload n when available.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Sample K records uniformly without replacement instead of taking the first K.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --random-sample.",
    )
    return parser.parse_args()


def _resolve_cache_path(args: argparse.Namespace) -> Path:
    """Resolve which preprocessing cache file to load.

    ``args`` is the parsed CLI namespace produced by :func:`_parse_args`. The
    function returns a cache ``Path``, does not mutate repository files, and
    assumes either ``args.cache_path`` is set or the project config contains a
    valid cache path for the requested split.
    """

    if args.cache_path is not None:
        return args.cache_path

    config = load_supervised_config(args.config)
    if args.split == "train":
        return Path(config.data_paths.train_cache)

    if config.data_paths.val_cache is None:
        raise ValueError("val cache path is not configured; pass --cache-path explicitly")
    return Path(config.data_paths.val_cache)


def _load_board_size(cache_path: Path, cli_board_size: int | None) -> int:
    """Return the board size associated with one preprocessing cache.

    ``cache_path`` is the serialized preprocessing artifact and
    ``cli_board_size`` optionally overrides the stored board size. The function
    returns the positive integer board size, does not mutate cache contents,
    and assumes the cache payload matches the format produced by
    ``supervised.preprocess``.
    """

    if cli_board_size is not None:
        if cli_board_size < 1:
            raise ValueError("--n must be >= 1")
        return cli_board_size

    payload = torch.load(cache_path, weights_only=False)
    board_size = payload.get("n")
    if not isinstance(board_size, int) or board_size < 1:
        raise ValueError(f"cache payload is missing a valid board size: {cache_path}")
    return board_size


def _select_records(
    records: list[EncodedSolutionRecord],
    k: int,
    random_sample: bool,
    seed: int,
) -> list[EncodedSolutionRecord]:
    """Select the subset of cached records that will be analyzed.

    ``records`` is the full encoded cache list, ``k`` is the number of samples
    to inspect, and ``random_sample`` with ``seed`` controls deterministic
    random sampling without replacement. The function returns a new list of
    selected records, does not mutate the input list, and assumes ``records``
    contains valid ``EncodedSolutionRecord`` objects.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(records):
        raise ValueError(f"requested k={k}, but cache only contains {len(records)} records")

    if not random_sample:
        return records[:k]

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(records)), k)
    return [records[index] for index in sampled_indices]


def _overlap_count(points_a: set[Point], points_b: set[Point]) -> int:
    """Return the number of occupied points shared by two solutions.

    ``points_a`` and ``points_b`` are lattice-point sets for two boards. The
    function returns ``|A ∩ B|``, does not mutate either set, and assumes both
    sets describe positions on the same board.
    """

    return len(points_a & points_b)


def _hamming_distance(points_a: set[Point], points_b: set[Point]) -> int:
    """Return the occupancy Hamming distance between two solutions.

    ``points_a`` and ``points_b`` are lattice-point sets on the same board. The
    function treats each board as a binary occupancy vector over all grid
    cells, returns ``|A Δ B|``, does not mutate the inputs, and assumes both
    sets use the same coordinate system.
    """

    return len(points_a ^ points_b)


def _compute_pairwise_stats(records: list[EncodedSolutionRecord]) -> tuple[float | None, float | None, int]:
    """Compute mean pairwise overlap and Hamming distance for selected records.

    ``records`` is the sample subset to compare. The function returns
    ``(mean_overlap, mean_hamming, pair_count)``, uses ``None`` means when
    fewer than two records are available, does not mutate the records, and
    assumes each record's ``points`` field is a valid board occupancy list.
    """

    if len(records) < 2:
        return None, None, 0

    point_sets = [set(record.points) for record in records]
    total_overlap = 0
    total_hamming = 0
    pair_count = 0

    for points_a, points_b in combinations(point_sets, 2):
        total_overlap += _overlap_count(points_a, points_b)
        total_hamming += _hamming_distance(points_a, points_b)
        pair_count += 1

    return total_overlap / pair_count, total_hamming / pair_count, pair_count


def _render_records(records: list[EncodedSolutionRecord], board_size: int) -> None:
    """Print each selected board using the shared render helpers.

    ``records`` is the ordered sample subset and ``board_size`` is the common
    square-grid side length. The function writes human-readable board renders
    to standard output, returns ``None``, does not mutate cached records, and
    assumes every point lies within the ``board_size x board_size`` grid.
    """

    for index, record in enumerate(records):
        print()
        print(f"[sample {index}] solution_id={record.solution_id}, point_count={record.point_count}")
        print_board(board_size, set(record.points))


def main() -> None:
    """Run the preprocessed-sample inspection workflow from the command line.

    The function parses CLI arguments, loads a preprocessing cache, selects
    ``K`` records, reports pairwise overlap and Hamming statistics, and renders
    the selected boards. It returns ``None``, mutates no repository files, and
    assumes the local environment can import the project modules and read the
    target cache artifact.
    """

    args = _parse_args()
    cache_path = _resolve_cache_path(args)
    records = load_preprocessed_cache(cache_path)
    board_size = _load_board_size(cache_path, args.n)
    selected_records = _select_records(records, args.k, args.random_sample, args.seed)
    mean_overlap, mean_hamming, pair_count = _compute_pairwise_stats(selected_records)

    print(f"Cache path: {cache_path}")
    print(f"Loaded {len(records)} records, selected {len(selected_records)} on a {board_size}x{board_size} board.")
    if pair_count == 0:
        print("Need at least 2 records to compute pairwise statistics.")
    else:
        print(f"Pair count: {pair_count}")
        print(f"Average overlap count: {mean_overlap:.3f}")
        print(f"Average Hamming distance: {mean_hamming:.3f}")

    _render_records(selected_records, board_size)


if __name__ == "__main__":
    main()
