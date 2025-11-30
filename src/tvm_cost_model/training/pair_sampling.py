"""Utilities for constructing ranking pairs from measurement records."""

from __future__ import annotations

from dataclasses import dataclass
import random
from enum import Enum

from typing import Iterable, List, Sequence

from tvm_cost_model.data.dataset_builder import MeasurementRecord


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

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
    """

    sorted_records = sorted(measurements, key=lambda m: m.runtime_ms)
    pairs: List[RankedPair] = []

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
    """Randomly sample ranking pairs without enumerating all combinations."""

    if num_pairs <= 0 or len(measurements) < 2:
        return []
    rng = random.Random(seed)
    pairs: List[RankedPair] = []
    max_attempts = num_pairs * 20
    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(measurements, 2)
        if a.runtime_ms == b.runtime_ms:
            continue
        better, worse = (a, b) if a.runtime_ms < b.runtime_ms else (b, a)
        rel_gap = (worse.runtime_ms - better.runtime_ms) / max(better.runtime_ms, 1e-9)
        if rel_gap < 0.05:  # discard near-ties below 5% relative difference
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
