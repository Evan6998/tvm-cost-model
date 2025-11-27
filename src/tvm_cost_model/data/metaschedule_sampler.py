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
        original_tir = str(mod.script())
        workload_key = str(tvm.ir.structural_hash(mod)) # type: ignore[union-attr]

        ctx = ms.TuneContext(
            mod=mod,
            target=self.target,
            space_generator=space_generator.PostOrderApply(
                sch_rules="from-target",
                postprocs="from-target",
                mutator_probs="from-target",
            ),
            task_name=operator,
            num_threads=8,
            search_strategy="evolutionary",
        )
        print("Generating design spaces...")
        design_spaces = ctx.generate_design_space()
        print(f"Generated {len(design_spaces)} design spaces for operator {operator!r}.")

        print("Initializing tuning context...")
        # Initialize the search strategy state
        ctx.pre_tuning(
            max_trials=batch,               # or something larger if you want more than `batch`
            num_trials_per_iter=batch,      # or 64, etc.
            design_spaces=design_spaces,
            # database=None, cost_model=None -> TVM will create MemoryDatabase + RandomModel
        )
        print(f"Initialized tuning context for operator {operator!r}.")

        print("Generating measure candidates...")
        measure_candidates: list[ms.MeasureCandidate] = []
        while len(measure_candidates) < batch:
            cands = ctx.generate_measure_candidates()
            if not cands:  # search finished
                break
            print(f"Generated {len(cands)} new measure candidates for operator {operator!r}.")
            measure_candidates.extend(cands)
        measure_candidates = measure_candidates[:batch]

        assert measure_candidates is not None

        samples: list[ScheduleSample] = []
        # Always include the original TIR as a baseline sample
        samples.append(
            ScheduleSample(
                operator=operator,
                schedule_json="",
                original_tir=original_tir,
                scheduled_tir=original_tir,
                workload_shape=self._normalize_workload_shape(self._workload_shape_fn(operator)),
                target=str(self.target),
                workload_key=workload_key,
            )
        )
        if not design_spaces:
            raise RuntimeError(f"No design spaces generated for operator {operator!r}")

        scheduled_count = 0
        for cand in measure_candidates:
            sch = cand.sch
            trace = sch.trace.as_json() # type: ignore[union-attr]
            scheduled_script = str(sch.mod.script())
            trace_json = json.dumps(trace, default=tvm_default_encoder)

            if scheduled_script == original_tir:
                print(f"Skipping empty trace for operator {operator!r}.")
                continue

            scheduled_count += 1
            samples.append(
                ScheduleSample(
                    operator=operator,
                    schedule_json=trace_json,
                    original_tir=original_tir,
                    scheduled_tir=scheduled_script,
                    workload_shape=self._normalize_workload_shape(self._workload_shape_fn(operator)),
                    target=str(self.target),
                    workload_key=workload_key,
                )
            )

        if scheduled_count == 0:
            raise RuntimeError(
                f"Design spaces for operator {operator!r} on target {self.target} produced only empty traces."
                " No schedule rules fired; ensure the workload has schedulable blocks or supply custom schedule rules."
            )
        return samples
    
    @staticmethod
    def _normalize_workload_shape(workload: dict[str, Any]) -> dict[str, tuple[int, ...]]:
        normalized: dict[str, tuple[int, ...]] = {}
        for name, shape in workload.items():
            normalized[name] = _normalize_shape(shape)
        return normalized


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
    device: "tvm.runtime.Device | None",
    number: int = 5,
    repeat: int = 1,
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
    exec_device = device or tvm.device(str(tvm_target.kind.name), 0)  # type: ignore[union-attr]
    for sample in schedules:
        # mod_src = sample.original_tir
        # mod = from_source(mod_src)
        # if sample.schedule_json and sample.original_tir:
        #     mod = apply_trace_to_module(mod, sample.schedule_json)
        mod = from_source(sample.scheduled_tir)
        built = tir.build(mod, target=tvm_target)
        inputs = input_generator(sample, exec_device)  # type: ignore
        if runner:
            result_ms = runner(built, inputs, exec_device)  # type: ignore
        else:
            time_eval = built.time_evaluator( # type: ignore[union-attr]
                built.entry_name,
                exec_device,
                number=number,
                repeat=repeat
            )
            result_ms = float(time_eval(*inputs).mean) * 1000.0  # sec -> ms
        yield MeasurementRecord(
            operator=sample.operator,
            schedule_json=sample.schedule_json,
            scheduled_tir=sample.scheduled_tir,
            workload_shape=sample.workload_shape,
            runtime_ms=float(result_ms),
            hardware_id=hardware_id,
            target=str(tvm_target),
            workload_key=sample.workload_key,
            original_tir=sample.original_tir,
        )


class MetaScheduleRuntimeEvaluator(RuntimeEvaluator):
    """RuntimeEvaluator wrapper that delegates to measure_schedules."""

    def __init__(
        self,
        target: str,
        hardware_id: str = "unknown",
        number: int = 5,
        repeat: int = 1,
        device: "tvm.runtime.Device | None" = None,
        input_generator: Callable[
            [ScheduleSample, "tvm.runtime.Device"], Sequence["tvm.runtime.Tensor"]
        ] = generate_inputs_from_workload,
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
