"""MetaSchedule-backed sampling/measurement hooks.

Requires TVM and compatible hardware/drivers. Sampling uses MetaSchedule design
spaces; measurement uses `tvm.tir.build` + local time_evaluator for pragmatic
local runs. Extend as needed for full runner/builder integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence, Any
import json
import numpy as np

import tvm  # type: ignore[import]
from tvm import meta_schedule as ms  # type: ignore[import]
from tvm.meta_schedule import space_generator # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]
from tvm.tir.schedule import Trace # type: ignore[import]
from tvm import tir # type: ignore[import]

from tvm_cost_model.data.dataset_builder import MeasurementRecord, ScheduleSample, ScheduleSampler, RuntimeEvaluator


class MetaScheduleSampler(ScheduleSampler):
    """Samples schedules via MetaSchedule using default design space rules."""

    def __init__(
        self,
        target: str,
        module_supplier: Callable[[str], "tvm.IRModule"],
        work_dir: Path,
        workload_shape_fn: Callable[[str], dict[str, Any]],
    ) -> None:
        self.target = tvm.target.Target(target)
        self.module_supplier = module_supplier
        self.work_dir = Path(work_dir)
        self._workload_shape_fn = workload_shape_fn

    def sample(self, operator: str, batch: int) -> Iterable[ScheduleSample]:
        mod = self.module_supplier(operator)

        ctx = ms.TuneContext(
            mod=mod,
            target=self.target,
            space_generator=space_generator.PostOrderApply(),  # 或者 "post-order-apply"
            task_name=operator,
        )

        design_spaces = ctx.generate_design_space()

        samples: list[ScheduleSample] = []
        for i in range(batch):
            sch = design_spaces[i % len(design_spaces)]
            j = sch.trace.as_json() # type: ignore[union-attr]
            trace_json = json.dumps(j, default=tvm_default_encoder)
            tir_script = sch.mod.script()
            tir_text = str(tir_script)
            samples.append(
                ScheduleSample(
                    operator=operator,
                    schedule_json=trace_json,
                    tir=tir_text,
                    workload_shape=self._workload_shape_fn(operator),
                )
            )
        return samples
    


def tvm_default_encoder(obj: Any) -> Any:
    # TVM specific JSON encoder for objects not serializable by default json module

    # IntImm / FloatImm
    if isinstance(obj, tvm.tir.IntImm):
        return int(obj.value)
    if isinstance(obj, tvm.tir.FloatImm):
        return float(obj.value)
    # default fallback
    return str(obj)


def apply_trace_to_module(mod: "tvm.IRModule", schedule_json: str | Any) -> "tvm.IRModule":
    """Apply a serialized trace (JSON object or JSON string) to an IRModule and return the new module."""
    sch = tir.Schedule(mod)

    if isinstance(schedule_json, str):
        if not schedule_json:
            raise ValueError("Empty schedule_json string provided.")
        json_obj = json.loads(schedule_json)
    else:
        json_obj = schedule_json

    Trace.apply_json_to_schedule(json_obj, sch)
    return sch.mod


def _normalize_shape(shape_spec: Any) -> tuple[int, ...]:
    """Coerce a shape description into a tuple of ints."""
    if isinstance(shape_spec, int):
        return (int(shape_spec),)
    if isinstance(shape_spec, (list, tuple)):
        return tuple(int(dim) for dim in shape_spec) # type: ignore
    raise TypeError(f"Unsupported workload shape type: {type(shape_spec)!r}")


def generate_inputs_from_workload(
    sample: ScheduleSample, device: "tvm.runtime.Device", dtype: str = "float32"
) -> Sequence["tvm.runtime.Tensor"]:
    """Create NDArray inputs from `workload_shape` hints.

    - If workload_shape is empty, returns an empty list.
    - If values are ints, treat each as a 1D buffer length.
    - If values are iterables, treat each as the full buffer shape.
    """
    if not sample.workload_shape:
        raise ValueError("workload_shape is empty; cannot generate inputs.")
    arrays: list["tvm.runtime.Tensor"] = []
    for shape_spec in sample.workload_shape.values():
        shape = _normalize_shape(shape_spec)
        data = np.ones(shape, dtype=dtype)
        arrays.append(tvm.runtime.tensor(data, device=device))  # type: ignore
    return arrays


def measure_schedules(
    schedules: Sequence[ScheduleSample],
    target: str,
    hardware_id: str,
    input_generator: Callable[[ScheduleSample, "tvm.runtime.Device"], Sequence["tvm.runtime.Tensor"]],
    number: int = 5,
    repeat: int = 1,
    device: "tvm.runtime.Device | None" = None,
    runner: Callable[
        ["tvm.runtime.Module", Sequence["tvm.runtime.Tensor"], "tvm.runtime.Device"], float
    ] | None = None,
):
    """Measure schedules using tvm.tir.build plus a pluggable runner.

    Defaults to local time_evaluator runs; callers can provide an input generator
    (e.g., to synthesize NDArrays from workload_shape) and a custom runner for
    RPC/remote execution.
    """

    tvm_target = tvm.target.Target(target)
    dev = device or tvm.device(str(tvm_target.kind.name), 0) # type: ignore[union-attr]
    for sample in schedules:
        mod = from_source(sample.tir)
        if sample.schedule_json:
            mod = apply_trace_to_module(mod, sample.schedule_json)
        built = tir.build(mod, target=tvm_target)
        inputs = input_generator(sample, dev)  # type: ignore
        if runner:
            result_ms = runner(built, inputs, dev)  # type: ignore
        else:
            time_eval = built.time_evaluator( # type: ignore[union-attr]
                built.entry_name, 
                dev, 
                number=number, 
                repeat=repeat
            )
            result_ms = float(time_eval(*inputs).mean) * 1000.0  # sec -> ms
        yield MeasurementRecord(
            operator=sample.operator,
            schedule_json=sample.schedule_json,
            tir=sample.tir,
            workload_shape=sample.workload_shape,
            runtime_ms=float(result_ms),
            hardware_id=hardware_id,
        )


class MetaScheduleRuntimeEvaluator(RuntimeEvaluator):
    """RuntimeEvaluator wrapper that delegates to measure_schedules."""

    def __init__(
        self,
        target: str,
        input_generator: Callable[
            [ScheduleSample, "tvm.runtime.Device"], Sequence["tvm.runtime.Tensor"]
        ] = generate_inputs_from_workload,
        hardware_id: str = "unknown",
        number: int = 5,
        repeat: int = 1,
        device: "tvm.runtime.Device | None" = None,
        runner: Callable[
            ["tvm.runtime.Module", Sequence["tvm.runtime.Tensor"], "tvm.runtime.Device"], float
        ] | None = None,
    ) -> None:
        self.target = target
        self.hardware_id = hardware_id
        self.number = number
        self.repeat = repeat
        self.device = device
        self.input_generator = input_generator
        self.runner = runner

    def evaluate(self, sample: ScheduleSample, hardware_id: str | None = None) -> float:
        record = next(
            measure_schedules(
                [sample],
                target=self.target,
                hardware_id=hardware_id or self.hardware_id,
                input_generator=self.input_generator,
                number=self.number,
                repeat=self.repeat,
                device=self.device,
                runner=self.runner,
            )
        )
        return record.runtime_ms
