"""Helpers for bridging TVM MetaSchedule objects to the internal training pipeline."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from tvm.ir import IRModule  # type: ignore[import]
from tvm.meta_schedule import MeasureCandidate  # type: ignore[import]
from tvm.meta_schedule.runner import RunnerResult  # type: ignore[import]
from tvm.tir import PrimFunc, Schedule  # type: ignore[import]

from tvm_cost_model.data.dataset_builder import MeasurementRecord


def measurement_to_score(latency_ms: float) -> float:
    """Convert a latency measurement (ms) to a model-friendly score (higher=better)."""

    return -math.log(latency_ms + 1e-6)


def runner_result_to_latency_ms(result: RunnerResult) -> float | None:
    """Extract a representative latency (ms) from a RunnerResult.

    Returns None when the result is unusable (errors or missing runtimes).
    """

    if getattr(result, "error_msg", None):
        return None
    run_secs = getattr(result, "run_secs", None)
    if not run_secs:
        return None

    values: list[float] = []
    for entry in run_secs:
        if entry is None:
            continue
        try:
            value = getattr(entry, "value", entry)
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    median_secs = statistics.median(values)
    return median_secs * 1000.0


def candidate_to_tir(candidate: MeasureCandidate) -> Any:
    """Return the scheduled TIR (IRModule/PrimFunc/Schedule) for a MeasureCandidate."""

    sch = getattr(candidate, "sch", None)
    if sch is not None:
        mod = getattr(sch, "mod", None)
        if mod is not None:
            return mod
        return sch

    mod = getattr(candidate, "mod", None)
    if mod is not None:
        return mod

    func = getattr(candidate, "func", None)
    if func is not None:
        return func

    return candidate


def _to_ir_module(obj: Any) -> IRModule | None:
    if isinstance(obj, IRModule):
        return obj
    if isinstance(obj, Schedule):
        return getattr(obj, "mod", None)
    if isinstance(obj, PrimFunc):
        return IRModule({"main": obj})
    return None


def _to_script(obj: Any) -> str:
    mod = _to_ir_module(obj)
    if mod is not None and hasattr(mod, "script"):
        try:
            return str(mod.script())
        except Exception:
            ...
    if hasattr(obj, "script"):
        try:
            return str(obj.script())
        except Exception:
            ...
    return str(obj)


def pack_measurements(
    candidates: Iterable[MeasureCandidate],
    results: Iterable[RunnerResult],
    context: Any | None = None,
) -> list[MeasurementRecord]:
    """Bundle MetaSchedule candidates/results into MeasurementRecords."""

    measurements: list[MeasurementRecord] = []
    for candidate, result in zip(candidates, results):
        latency_ms = runner_result_to_latency_ms(result)
        if latency_ms is None:
            continue

        scheduled_tir_obj = candidate_to_tir(candidate)
        original_tir_obj = getattr(context, "mod", None) or getattr(candidate, "mod", None) or scheduled_tir_obj

        operator = getattr(context, "workload_name", "") or getattr(context, "task_name", "") or ""
        target = getattr(context, "target", None)
        workload_key = getattr(context, "workload_key", None)
        workload_shape: dict[str, tuple[int, ...]] = getattr(context, "workload_shape", {}) or {}
        hardware_params = getattr(context, "hardware_params", None)

        measurements.append(
            MeasurementRecord(
                operator=str(operator),
                schedule_json="",
                original_tir=_to_script(original_tir_obj),
                scheduled_tir=_to_script(scheduled_tir_obj),
                workload_shape=workload_shape,
                runtime_ms=latency_ms,
                hardware_id=str(hardware_params) if hardware_params is not None else "",
                target=str(target) if target is not None else None,
                workload_key=workload_key,
                hardware_features=None,
            )
        )

    return measurements
