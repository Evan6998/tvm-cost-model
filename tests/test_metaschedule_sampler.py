from pathlib import Path

import tvm # type: ignore[import]
from tvm import tir as tvm_tir  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]

T = tvm.script.tir  # type: ignore[import]

from tvm_cost_model.data.dataset_builder import ScheduleSample
from tvm_cost_model.data.metaschedule_sampler import (
    MetaScheduleSampler,
    MetaScheduleRuntimeEvaluator,
    apply_trace_to_module,
    generate_inputs_from_workload,
    measure_schedules,
)


def _module_supplier(_: str) -> tvm.IRModule: 
    @tvm.script.ir_module  # type: ignore[annotation-unchecked]
    class Module:
        @T.prim_func  # type: ignore[annotation-unchecked]
        def main(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})  # type: ignore[call-arg]
            # Simple elementwise block to expose design space
            for i in T.serial(0, 8): # type: ignore[call-arg]
                with T.block("C"): # type: ignore[call-arg]
                    vi = T.axis.spatial(8, i) # type: ignore[call-arg]
                    B[vi] = A[vi] + 1.0  # type: ignore[index]
    return Module  # type: ignore[return-value]


def test_metaschedule_sampler_emits_samples(tmp_path: Path):
    target = "llvm -num-cores 4"
    sampler = MetaScheduleSampler(target=target, module_supplier=_module_supplier, work_dir=tmp_path, workload_shape_fn=lambda _: {})
    try:
        samples = list(sampler.sample("main", batch=2))
    except RuntimeError as err:
        assert "empty traces" in str(err).lower()
        return
    assert len(samples) >= 2  # includes unscheduled baseline + at least one scheduled
    assert all(isinstance(s.schedule_json, str) for s in samples)
    assert all(isinstance(s.scheduled_tir, str) and s.scheduled_tir for s in samples)


def test_metaschedule_sampler_populates_workload_shapes(tmp_path: Path):
    workload = {"A": (4,)}
    sampler = MetaScheduleSampler(
        target="llvm -num-cores 4",
        module_supplier=_module_supplier,
        work_dir=tmp_path,
        workload_shape_fn=lambda _: workload,
    )
    try:
        sample = next(iter(sampler.sample("main", batch=1)))
    except RuntimeError as err:
        assert "empty traces" in str(err).lower()
        return
    assert sample.workload_shape == workload


def test_measure_schedules_generates_inputs():
    tir_script = """
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(0, 4):
            A[i] = A[i] + 1.0
"""
    sample = ScheduleSample(
        operator="main",
        schedule_json="",
        original_tir=tir_script,
        scheduled_tir=tir_script,
        workload_shape={"A": (4,)},
    )
    records = list(
        measure_schedules(
            [sample],
            target="llvm",
            hardware_id="cpu",
            device=tvm.cpu(0), # type: ignore
            input_generator=generate_inputs_from_workload,
        )
    )
    assert len(records) == 1
    assert records[0].runtime_ms != float("inf")


def test_apply_trace_to_module_round_trips():
    mod = _module_supplier("main")
    # Build a schedule to get a trace
    sch = tvm_tir.Schedule(mod)
    loop = sch.get_loops(sch.get_block("C"))[0]
    sch.split(loop, factors=[2, 4])
    tr = sch.trace.as_json()  # type: ignore[union-attr]

    new_mod = apply_trace_to_module(mod, tr)
    assert isinstance(new_mod, tvm.IRModule)
    # Ensure the new module is buildable
    built = tvm_tir.build(new_mod, target="llvm")
    assert hasattr(built, "entry_name")


def test_metaschedule_runtime_evaluator_wraps_measurement():
    tir_script = """
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(0, 4):
            A[i] = A[i] + 1.0
"""
    sample = ScheduleSample(
        operator="main",
        schedule_json="",
        original_tir=tir_script,
        scheduled_tir=tir_script,
        workload_shape={"A": (4,)},
    )
    evaluator = MetaScheduleRuntimeEvaluator(
        target="llvm",
        hardware_id="cpu",
        device=tvm.cpu(0), # type: ignore
        input_generator=generate_inputs_from_workload,
    )
    runtime_ms = evaluator.evaluate(sample)
    assert runtime_ms != float("inf")
