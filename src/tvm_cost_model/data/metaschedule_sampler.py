"""MetaSchedule-backed sampling/measurement hooks.

Requires TVM and compatible hardware/drivers. Sampling uses MetaSchedule design
spaces; measurement uses `tvm.tir.build` + local time_evaluator for pragmatic
local runs. Extend as needed for full runner/builder integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence, Any
import json

import tvm  # type: ignore[import]
from tvm import meta_schedule as ms  # type: ignore[import]
from tvm.meta_schedule import space_generator # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]
from tvm.tir.schedule import Trace # type: ignore[import]
from tvm import tir # type: ignore[import]

from tvm_cost_model.data.dataset_builder import MeasurementRecord, ScheduleSample, ScheduleSampler


class MetaScheduleSampler(ScheduleSampler):
    """Samples schedules via MetaSchedule using default design space rules."""

    def __init__(
        self,
        target: str,
        module_supplier: Callable[[str], "tvm.IRModule"],
        work_dir: Path,
    ) -> None:
        self.target = tvm.target.Target(target)
        self.module_supplier = module_supplier
        self.work_dir = Path(work_dir)

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
                    workload_shape={},
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


def measure_schedules(
    schedules: Sequence[ScheduleSample],
    target: str,
    hardware_id: str,
    number: int = 5,
    repeat: int = 1,
) -> Iterable[MeasurementRecord]:
    """Measure schedules locally using tvm.tir.build + time_evaluator.

    Assumes the built function takes no inputs. If inputs are required, extend
    this to generate NDArrays from workload_shape metadata.
    """

    tvm_target = tvm.target.Target(target)
    dev = tvm.device(str(tvm_target.kind.name), 0) # type: ignore[union-attr]
    for sample in schedules:
        mod = from_source(sample.tir)
        if sample.schedule_json:
            mod = apply_trace_to_module(mod, sample.schedule_json)
        built = tir.build(mod, target=tvm_target)
        time_eval = built.time_evaluator( # type: ignore[union-attr]
            built.entry_name, 
            dev, 
            number=number, 
            repeat=repeat
        )
        result = time_eval().mean * 1000.0  # ms
        print(f"Measured runtime for operator {sample.operator}: {result} ms")
        yield MeasurementRecord(
            operator=sample.operator,
            schedule_json=sample.schedule_json,
            tir=sample.tir,
            workload_shape=sample.workload_shape,
            runtime_ms=float(result),
            hardware_id=hardware_id,
        )
