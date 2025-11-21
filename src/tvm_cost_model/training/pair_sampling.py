"""Utilities for constructing ranking pairs from measurement records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from tvm_cost_model.data.dataset_builder import MeasurementRecord


@dataclass
class RankedPair:
    """Represents a pairwise comparison for ranking training."""

    better: MeasurementRecord
    worse: MeasurementRecord
    difficulty: str  # "easy" or "hard"


def make_ranking_pairs(
    measurements: Sequence[MeasurementRecord],
    easy_gap: float = 10.0,
    hard_gap: float = 2.0,
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


def _classify_delta(delta: float, easy_gap: float, hard_gap: float) -> str:
    if delta >= easy_gap:
        return "easy"
    if delta <= hard_gap:
        return "hard"
    return "medium"


def _subsequent_elements(seq: Sequence[MeasurementRecord], start: int) -> Iterable[MeasurementRecord]:
    for idx in range(start, len(seq)):
        yield seq[idx]
