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
    easy_gap: float = 0.5,
    hard_gap: float = 0.1,
) -> List[RankedPair]:
    """Create ranking pairs from measurements.

    Args:
        measurements: Records with runtime_ms (lower is better).
        easy_gap: Minimum runtime delta (ms) for an "easy" pair.
        hard_gap: Maximum runtime delta (ms) for a "hard" pair.
        curriculum: If True, include both easy and hard; otherwise return all pairs.
    """

    sorted_records = sorted(measurements, key=lambda m: m.runtime_ms)
    pairs: List[RankedPair] = []

    for i, better in enumerate(sorted_records):
        for worse in _subsequent_elements(sorted_records, start=i + 1):
            delta = worse.runtime_ms - better.runtime_ms
            if delta <= 0:
                continue
            difficulty = _classify_delta(delta, easy_gap, hard_gap)
            pairs.append(RankedPair(better=better, worse=worse, difficulty=difficulty))
    return pairs


def sample_ranking_pairs(
    measurements: Sequence[MeasurementRecord],
    num_pairs: int,
    easy_gap: float = 0.3,
    hard_gap: float = 0.1,
    seed: int | None = None,
    max_difficulty: Difficulty | None = None,
) -> List[RankedPair]:
    """Randomly sample ranking pairs without enumerating all combinations."""

    if num_pairs <= 0 or len(measurements) < 2:
        return []
    rng = random.Random(seed)
    pairs: List[RankedPair] = []
    while len(pairs) < num_pairs:
        a, b = rng.sample(measurements, 2)
        if a.runtime_ms == b.runtime_ms:
            continue
        better, worse = (a, b) if a.runtime_ms < b.runtime_ms else (b, a)
        difficulty = _classify_delta(worse.runtime_ms - better.runtime_ms, easy_gap, hard_gap)
        if max_difficulty is not None and difficulty.value > max_difficulty.value:
            continue
        pairs.append(RankedPair(better=better, worse=worse, difficulty=difficulty))
    return pairs


def _classify_delta(delta: float, easy_gap: float, hard_gap: float) -> Difficulty:
    if delta >= easy_gap:
        return Difficulty.EASY
    if delta <= hard_gap:
        return Difficulty.HARD
    return Difficulty.MEDIUM


def _subsequent_elements(seq: Sequence[MeasurementRecord], start: int) -> Iterable[MeasurementRecord]:
    for idx in range(start, len(seq)):
        yield seq[idx]
