from pathlib import Path

import tvm # type: ignore[import]
from tvm import tir as tvm_tir  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]

T = tvm.script.tir  # type: ignore[import]

from tvm_cost_model.data.dataset_builder import ScheduleSample
from tvm_cost_model.data.metaschedule_sampler import (
    MetaScheduleSampler,
    apply_trace_to_module,
    measure_schedules,
)


def _module_supplier(_: str) -> tvm.IRModule: 
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # Simple loop with a named block to expose design space
            for i in T.serial(0, 8):
                with T.block("C"):
                    vi = T.axis.spatial(8, i)
                    T.evaluate(vi)
    return Module


def test_metaschedule_sampler_emits_samples(tmp_path: Path):
    target = "llvm -num-cores 4"
    sampler = MetaScheduleSampler(target=target, module_supplier=_module_supplier, work_dir=tmp_path)
    samples = list(sampler.sample("main", batch=2))
    assert len(samples) == 2
    assert all(isinstance(s.schedule_json, str) for s in samples)
    assert all(isinstance(s.tir, str) and s.tir for s in samples)


def test_measure_schedules_runs_time_evaluator():
    # Use a trivial program with no inputs
    tir_script = """
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(0, 4):
            T.evaluate(i)
"""
    sample = ScheduleSample(
        operator="main",
        schedule_json="",
        tir=tir_script,
        workload_shape={},
    )
    records = list(measure_schedules([sample], target="llvm -num-cores 4", hardware_id="cpu"))
    assert len(records) == 1
    assert records[0].runtime_ms != float("inf")
    assert records[0].operator == "main"


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
