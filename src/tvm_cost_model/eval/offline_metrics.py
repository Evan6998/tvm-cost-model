"""Offline evaluation helpers for cost models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class RankingMetrics:
    pairwise_accuracy: float
    spearman: float
    kendall: float
    topk_recall: dict[int, float]


def _coerce_arrays(
    latencies_ms: Sequence[float], predictions: Sequence[float]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert inputs to arrays and validate shape/emptiness."""

    lat_arr = np.asarray(latencies_ms)
    pred_arr = np.asarray(predictions)
    if lat_arr.size == 0 or pred_arr.size == 0:
        return None, None
    if lat_arr.shape[0] != pred_arr.shape[0]:
        return None, None
    return lat_arr, pred_arr


def _kendall_tau_fast(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            concordant += int((x[i] - x[j]) * (y[i] - y[j]) > 0)
            discordant += int((x[i] - x[j]) * (y[i] - y[j]) < 0)
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xranks = np.argsort(np.argsort(np.asarray(x)))
    yranks = np.argsort(np.argsort(np.asarray(y)))
    if xranks.std() == 0 or yranks.std() == 0:
        return float("nan")
    return float(np.corrcoef(xranks, yranks)[0, 1])


def pairwise_accuracy(
    latencies_ms: Sequence[float],
    predictions: Sequence[float],
    pred_higher_is_better: bool = True,
    delta_frac: float = 0.05,
    max_pairs: int = 5000,
) -> float:
    """Compute pairwise accuracy on sampled pairs.

    A pair is considered valid when the latency gap is at least delta_frac.
    """

    lat_arr, pred_arr = _coerce_arrays(latencies_ms, predictions)
    if lat_arr is None or pred_arr is None:
        return float("nan")

    n = len(lat_arr)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > max_pairs:
        rng = np.random.default_rng(seed=0)
        sampled_idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[int(i)] for i in sampled_idx]
    else:
        pairs = all_pairs

    correct = 0
    total = 0
    for i, j in pairs:
        true_i = float(lat_arr[i])
        true_j = float(lat_arr[j])
        if min(true_i, true_j) <= 0:
            continue
        gap = abs(true_i - true_j) / min(true_i, true_j)
        if gap < delta_frac:
            continue
        pred_i = float(pred_arr[i])
        pred_j = float(pred_arr[j])
        pred_cmp = pred_i > pred_j if pred_higher_is_better else pred_i < pred_j
        true_cmp = true_i < true_j
        correct += int(pred_cmp == true_cmp)
        total += 1

    if total == 0:
        return float("nan")
    return correct / total


def topk_recall(
    latencies_ms: Sequence[float],
    predictions: Sequence[float],
    k_values: Iterable[int],
    true_top_k: int = 20,
    pred_higher_is_better: bool = True,
) -> dict[int, float]:
    """Compute recall@k against the set of top true schedules."""

    lat_arr, pred_arr = _coerce_arrays(latencies_ms, predictions)
    if lat_arr is None or pred_arr is None:
        return {k: float("nan") for k in k_values}

    k_values = list(sorted(set(k_values)))
    true_order = np.argsort(lat_arr)  # lower latency is better
    true_top = set(true_order[: min(true_top_k, len(true_order))])

    pred_order = (
        np.argsort(-pred_arr) if pred_higher_is_better else np.argsort(pred_arr)
    )

    recalls: dict[int, float] = {}
    for k in k_values:
        if k <= 0:
            recalls[k] = float("nan")
            continue
        pred_top = set(pred_order[: min(k, len(pred_order))])
        recalls[k] = len(true_top & pred_top) / max(len(true_top), 1)
    return recalls


def correlation_metrics(
    latencies_ms: Sequence[float],
    predictions: Sequence[float],
    pred_higher_is_better: bool = True,
) -> tuple[float, float]:
    """Return (spearman, kendall) correlations with true latencies."""

    lat_arr, pred_arr = _coerce_arrays(latencies_ms, predictions)
    if lat_arr is None or pred_arr is None:
        return float("nan"), float("nan")
    adjusted_pred = -pred_arr if pred_higher_is_better else pred_arr
    return _spearman_corr(lat_arr, adjusted_pred), _kendall_tau_fast(
        lat_arr, adjusted_pred
    )


def compute_ranking_metrics(
    latencies_ms: Sequence[float],
    predictions: Sequence[float],
    pred_higher_is_better: bool = True,
    k_values: Iterable[int] = (1, 5, 10, 20),
    delta_frac: float = 0.05,
) -> RankingMetrics:
    pair_acc = pairwise_accuracy(
        latencies_ms,
        predictions,
        pred_higher_is_better=pred_higher_is_better,
        delta_frac=delta_frac,
    )
    topk = topk_recall(
        latencies_ms,
        predictions,
        k_values=k_values,
        pred_higher_is_better=pred_higher_is_better,
    )
    spearman, kendall = correlation_metrics(
        latencies_ms, predictions, pred_higher_is_better=pred_higher_is_better
    )
    return RankingMetrics(pair_acc, spearman, kendall, topk)
