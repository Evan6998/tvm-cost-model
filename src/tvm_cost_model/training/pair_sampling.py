"""Utilities for constructing ranking pairs from measurement records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from enum import Enum

from typing import Dict, Iterable, List, Sequence, Tuple

from tvm_cost_model.data.dataset_builder import MeasurementRecord


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

PairKey = tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...], str | None, str]

@dataclass
class RankedPair:
    """Represents a pairwise comparison for ranking training."""

    better: MeasurementRecord
    worse: MeasurementRecord
    difficulty: Difficulty


def make_ranking_pairs(
    measurements: Sequence[MeasurementRecord],
    easy_frac: float = 0.3,
    hard_frac: float = 0.05,
) -> List[RankedPair]:
    """Create ranking pairs from measurements.

    Args:
        measurements: Records with runtime_ms (lower is better).
        easy_frac: Minimum relative runtime delta (fraction) for an "easy" pair (e.g., 0.3 = 30% slower).
        hard_frac: Maximum relative runtime delta (fraction) for a "hard" pair.
    
    Only pairs that share operator, workload_shape, target, and hardware_id are constructed.
    """

    pairs: List[RankedPair] = []

    for group in _group_measurements_by_key(measurements).values():
        if len(group) < 2:
            continue
        sorted_records = sorted(group, key=lambda m: m.runtime_ms)
        for i, better in enumerate(sorted_records):
            for worse in _subsequent_elements(sorted_records, start=i + 1):
                delta = worse.runtime_ms - better.runtime_ms
                if delta <= 0:
                    continue
                difficulty = _classify_delta(delta, better.runtime_ms, easy_frac, hard_frac)
                pairs.append(RankedPair(better=better, worse=worse, difficulty=difficulty))
    return pairs


def sample_ranking_pairs(
    measurements: Sequence[MeasurementRecord],
    num_pairs: int,
    easy_frac: float,
    hard_frac: float,
    seed: int | None = None,
    allowed_difficulties: set[Difficulty] | None = None,
) -> List[RankedPair]:
    """Randomly sample ranking pairs without enumerating all combinations.

    Only pairs that share operator, workload_shape, target, and hardware_id are constructed.
    """

    if num_pairs <= 0 or len(measurements) < 2:
        return []
    rng = random.Random(seed)
    pairs: List[RankedPair] = []

    grouped_by_key = _group_measurements_by_key(measurements)
    for key, group in grouped_by_key.items():
        operator, shape_key, target, hardware_id = key
        print(
            f"Pair sampling group size={len(group)} "
            f"operator={operator} "
            f"shape={_shape_key_to_str(shape_key)} "
            f"target={target} hardware_id={hardware_id}"
        )

    grouped = [group for group in grouped_by_key.values() if len(group) >= 2]
    if not grouped:
        return []

    max_attempts = num_pairs * 100
    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        group = rng.choices(grouped, weights=[len(g) for g in grouped], k=1)[0]
        a, b = rng.sample(group, 2)
        if a.runtime_ms == b.runtime_ms:
            continue
        better, worse = (a, b) if a.runtime_ms < b.runtime_ms else (b, a)
        rel_gap = (worse.runtime_ms - better.runtime_ms) / max(better.runtime_ms, 1e-9)
        if rel_gap < 0.01:  # discard only extremely small gaps
            continue
        difficulty = _classify_delta(
            worse.runtime_ms - better.runtime_ms,
            better.runtime_ms,
            easy_frac,
            hard_frac,
        )
        if allowed_difficulties is not None and difficulty not in allowed_difficulties:
            continue
        pairs.append(RankedPair(better=better, worse=worse, difficulty=difficulty))
    return pairs


def _classify_delta(delta: float, better_runtime: float, easy_frac: float, hard_frac: float) -> Difficulty:
    denom = max(better_runtime, 1e-9)
    rel_gap = delta / denom
    if rel_gap >= easy_frac:
        return Difficulty.EASY
    if rel_gap <= hard_frac:
        return Difficulty.HARD
    return Difficulty.MEDIUM


def _subsequent_elements(seq: Sequence[MeasurementRecord], start: int) -> Iterable[MeasurementRecord]:
    for idx in range(start, len(seq)):
        yield seq[idx]


def _workload_shape_key(workload_shape: Dict[str, tuple[int, ...]] | None) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """Create a hashable key for workload shapes."""
    if not workload_shape:
        return tuple()
    return tuple(sorted((name, tuple(dimensions)) for name, dimensions in workload_shape.items()))


def _shape_key_to_str(shape_key: Tuple[Tuple[str, Tuple[int, ...]], ...]) -> str:
    if not shape_key:
        return "{}"
    parts = [f"{name}:{'x'.join(str(d) for d in dims)}" for name, dims in shape_key]
    return "{" + ", ".join(parts) + "}"


def _pair_key(measurement: MeasurementRecord) -> PairKey:
    return (
        measurement.operator,
        _workload_shape_key(measurement.workload_shape),
        measurement.target,
        measurement.hardware_id,
    )


def _group_measurements_by_key(measurements: Sequence[MeasurementRecord]) -> dict[PairKey, list[MeasurementRecord]]:
    grouped: dict[PairKey, list[MeasurementRecord]] = defaultdict(list)
    for measurement in measurements:
        grouped[_pair_key(measurement)].append(measurement)
    return grouped
