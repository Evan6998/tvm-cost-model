#!/usr/bin/env python3
"""Run a minimal MetaSchedule tuning loop with real measurements.

This exercises TuneContext.generate_design_space -> pre_tuning -> generate_measure_candidates
and feeds the measured RunnerResults back via notify_runner_results. XGBoost is used by
default, but you can swap in the GraphPyCostModel adapter (and load a saved model) with
--cost-model graph --graph-model-path model.pth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple
import typing
import uuid

import tvm  # type: ignore[import]
from tvm import meta_schedule as ms  # type: ignore[import]
from tvm.meta_schedule import cost_model as ms_cost_model  # type: ignore[import]
from tvm.meta_schedule.builder import BuilderInput, BuilderResult  # type: ignore[import]
from tvm.meta_schedule.runner import EvaluatorConfig, RunnerInput, RunnerResult  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]

from tvm_cost_model.integration.metaschedule_adapter import GraphPyCostModel
from tvm_cost_model.integration.utils import runner_result_to_latency_ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight MetaSchedule tuning loop.")
    parser.add_argument(
        "--operator",
        default="gemm",
        choices=[
            "vecadd",
            "gemm",
            "conv2d_nchw",
            "depthwise_conv2d",
            "bmm",
            "layernorm",
            "softmax",
        ],
        help="Builtin operator to tune (ignored if --tir is provided).",
    )
    parser.add_argument(
        "--shape",
        type=str,
        required=False,
        default='{"m":1024,"n":1024,"k":1024}',
        help='JSON dict for builtin operator shapes (e.g., \'{"m":1024,"n":1024,"k":1024}\').',
    )
    parser.add_argument(
        "--tir",
        type=str,
        default="",
        help="Optional path to a TVMScript TIR file to tune instead of a builtin.",
    )
    parser.add_argument("--target", default="llvm -num-cores=8", help="TVM target string.")
    parser.add_argument("--task-name", default="", help="Task name to report to TuneContext.")
    parser.add_argument("--num-threads", type=int, default=8, help="Threads used by search.")
    parser.add_argument("--max-trials", type=int, default=32, help="Total candidates to measure.")
    parser.add_argument(
        "--trials-per-iter", type=int, default=16, help="Batch size for generate_measure_candidates."
    )
    parser.add_argument(
        "--search-strategy",
        default="evolutionary",
        help="Search strategy for TuneContext (string alias or object).",
    )
    parser.add_argument(
        "--cost-model",
        choices=["xgb", "graph"],
        default="graph",
        help="Cost model to plug into pre_tuning.",
    )
    parser.add_argument(
        "--graph-model-path",
        default="./model.pth",
        help="Path to a saved GraphCostModel to load when --cost-model graph is set.",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/tuning",
        help="Directory for any artifacts emitted by TVM builders.",
    )
    parser.add_argument("--dtype", default="float32", help="DType for builtin modules.")
    parser.add_argument(
        "--device-type",
        default="",
        help="Device string for RunnerInput (defaults to target.kind, maps llvm->cpu).",
    )
    parser.add_argument(
        "--build-timeout",
        type=float,
        default=30.0,
        help="Timeout (sec) for LocalBuilder.",
    )
    parser.add_argument(
        "--runner-timeout",
        type=float,
        default=10.0,
        help="Timeout (sec) for LocalRunner.",
    )
    parser.add_argument("--number", type=int, default=5, help="time_evaluator number.")
    parser.add_argument("--repeat", type=int, default=1, help="time_evaluator repeat.")
    parser.add_argument(
        "--min-repeat-ms",
        type=int,
        default=0,
        help="time_evaluator min_repeat_ms (0 to disable).",
    )
    parser.add_argument(
        "--cpu-cache-flush",
        action="store_true",
        help="Enable CPU cache flush in EvaluatorConfig.",
    )
    return parser


def load_module_and_shape(args: argparse.Namespace) -> Tuple["tvm.IRModule", dict[str, tuple[int, ...]]]:
    """Return the module to tune and the workload_shape hint."""
    if args.tir:
        mod_src = Path(args.tir).read_text()
        return from_source(mod_src), {}
    shape = parse_shape_arg(args.shape)
    supplier, workload_shape = build_builtin_module_supplier(
        operator=args.operator,
        shape=shape,
        dtype=args.dtype,
        global_symbol=args.operator,
    )
    return supplier(args.operator), workload_shape


def make_cost_model(args: argparse.Namespace):
    if args.cost_model == "graph":
        model = GraphPyCostModel()
        model_path = Path(args.graph_model_path)
        if model_path.exists():
            print(f"Loading GraphPyCostModel weights from {model_path}...")
            model.load(args.graph_model_path)
        else:
            print(f"GraphPyCostModel selected but model file not found at {model_path}; using fresh weights.")
        return model
    return ms_cost_model.XGBModel()


def _infer_device_type(target: "tvm.target.Target", override: str) -> str:
    if override:
        return override
    kind = str(target.kind.name) # type: ignore[attr-defined]
    if kind == "llvm":
        return "cpu"
    return kind


def _build_and_run(
    candidates: Sequence[ms.MeasureCandidate],
    builder: ms.builder.Builder,
    runner: ms.runner.Runner,
    target: "tvm.target.Target",
    device_type: str,
) -> Tuple[list[ms.MeasureCandidate], list[RunnerResult]]:
    """Build MeasureCandidates, run them, and return filtered candidates/results."""
    builder_inputs = [BuilderInput(cand.sch.mod, target) for cand in candidates]
    build_results: List[BuilderResult] = builder.build(builder_inputs)

    runner_inputs: list[RunnerInput] = []
    runnable_candidates: list[ms.MeasureCandidate] = []
    artifact_paths: list[Path] = []
    for cand, build_res in zip(candidates, build_results):
        if build_res.error_msg or not build_res.artifact_path:
            print(f"Build failed for candidate: {build_res.error_msg}")
            continue
        args_info = cand.args_info
        if len(args_info) == 0:
            args_info = ms.arg_info.TensorInfo.from_prim_func(cand.sch.mod["main"]) # type: ignore[union-attr]
        runner_inputs.append(
            RunnerInput(
                artifact_path=build_res.artifact_path,
                device_type=device_type,
                args_info=args_info,
            )
        )
        runnable_candidates.append(cand)
        artifact_paths.append(Path(build_res.artifact_path))

    if not runner_inputs:
        return [], []

    futures = runner.run(runner_inputs)
    results = [future.result() for future in futures]
    for path in artifact_paths:
        try:
            path.unlink()
        except OSError:
            pass
    return runnable_candidates, results


def make_local_builder(work_dir: Path, timeout_sec: float):
    work_dir.mkdir(parents=True, exist_ok=True)

    def _export_func(mod: "tvm.runtime.Module") -> str:
        path = work_dir / f"artifact_{uuid.uuid4().hex}.so"
        mod.export_library(path)  # type: ignore[attr-defined]
        return str(path)

    return ms.builder.LocalBuilder(timeout_sec=timeout_sec, max_workers=1, f_export=_export_func)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mod, workload_shape = load_module_and_shape(args)
    target = tvm.target.Target(args.target)
    device_type = _infer_device_type(target, args.device_type)

    cost_model = make_cost_model(args)
    database = ms.database.MemoryDatabase()  # defaults to memory DB to keep API happy

    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules="from-target",
            postprocs="from-target",
            mutator_probs="from-target",
        ),
        search_strategy=args.search_strategy,
        task_name=args.task_name or args.operator,
        num_threads=args.num_threads,
    )

    design_spaces = ctx.generate_design_space()
    print(f"Generated {len(design_spaces)} design spaces.")
    ctx.pre_tuning(
        max_trials=args.max_trials,
        num_trials_per_iter=args.trials_per_iter,
        design_spaces=design_spaces,
        database=database,
        cost_model=cost_model, # type: ignore[arg-type]
    )

    builder = make_local_builder(Path(args.work_dir), args.build_timeout)
    runner = ms.runner.LocalRunner(
        timeout_sec=args.runner_timeout,
        evaluator_config=EvaluatorConfig(
            number=args.number,
            repeat=args.repeat,
            min_repeat_ms=args.min_repeat_ms,
            enable_cpu_cache_flush=args.cpu_cache_flush,
        ),
    )

    measured = 0
    remaining = args.max_trials
    iteration = 0

    while remaining > 0:
        iteration += 1
        measure_candidates = ctx.generate_measure_candidates()
        if not measure_candidates:
            print("Search strategy returned no more candidates.")
            break
        measure_candidates = measure_candidates[:remaining]
        print(f"[iter {iteration}] Measuring {len(measure_candidates)} candidates (remaining={remaining})...")

        runnable_candidates, runner_results = _build_and_run(
            measure_candidates, builder, runner, target, device_type
        )
        if not runnable_candidates:
            print("No runnable candidates in this batch; stopping early.")
            break

        ctx.notify_runner_results(runnable_candidates, runner_results)

        for _, result in zip(runnable_candidates, runner_results):
            latency = runner_result_to_latency_ms(result)
            if latency is None:
                print("  - candidate had invalid runner result, skipping log")
                continue
            print(f"  - candidate runtime: {latency:.3f} ms")

        measured += len(runnable_candidates)
        remaining = max(args.max_trials - measured, 0)

    ctx.post_tuning()
    print(f"Finished tuning. Measured {measured} candidates. Workload shape: {workload_shape}")



def build_builtin_module_supplier(
    operator: str, shape: dict[str, int], dtype: str, global_symbol: str
) -> Tuple[typing.Callable[[str], "tvm.IRModule"], dict[str, Tuple[int, ...]]]:
    """Construct simple TVMScript IRModules for common operators."""
    op = operator.lower()
    if op == "vecadd":
        n = shape["n"]
        tir_script = f"""
@T.prim_func
def main(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
    A = T.match_buffer(a, ({n},), "{dtype}")
    B = T.match_buffer(b, ({n},), "{dtype}")
    C = T.match_buffer(c, ({n},), "{dtype}")

    for i in T.grid({n}):
        with T.block("vecadd"):
            vi = T.axis.spatial({n}, i)
            C[vi] = A[vi] + B[vi]
"""
        workload_shape = {"A": (n,), "B": (n,), "C": (n,)}
    elif op == "gemm":
        m = shape["m"]
        n = shape["n"]
        k = shape["k"]
        tir_script = f"""
@T.prim_func
def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, ({m}, {k}))
    B = T.match_buffer(b, ({k}, {n}))
    C = T.match_buffer(c, ({m}, {n}))

    for i, j, k in T.grid({m}, {n}, {k}):
        with T.block():
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] += A[vi, vk] * B[vj, vk]
"""
        workload_shape = {"A": (m, k), "B": (k, n), "C": (m, n)}
    elif op == "bmm":
        b = shape.get("batch", 8)
        m = shape.get("m", 128)
        n = shape.get("n", 128)
        k = shape.get("k", 128)
        tir_script = f"""
@T.prim_func
def main(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
    A = T.match_buffer(a, ({b}, {m}, {k}), "{dtype}")
    B = T.match_buffer(b, ({b}, {k}, {n}), "{dtype}")
    C = T.match_buffer(c, ({b}, {m}, {n}), "{dtype}")

    for batch, i, j, kk in T.grid({b}, {m}, {n}, {k}):
        with T.block("bmm"):
            vb, vi, vj, vkk = T.axis.remap("SSSR", [batch, i, j, kk])
            with T.init():
                C[vb, vi, vj] = T.cast(0, "{dtype}")
            C[vb, vi, vj] = C[vb, vi, vj] + A[vb, vi, vkk] * B[vb, vkk, vj]
"""
        workload_shape = {"A": (b, m, k), "B": (b, k, n), "C": (b, m, n)}
    elif op == "conv2d_nchw":
        n = shape.get("n", 1)
        ci = shape.get("ci", 64)
        co = shape.get("co", 64)
        h = shape.get("h", 56)
        w = shape.get("w", 56)
        kh = shape.get("kh", 3)
        kw = shape.get("kw", 3)
        stride = shape.get("stride", 1)
        padding = shape.get("padding", 0)
        if padding != 0:
            raise ValueError("Builtin conv2d_nchw does not implement padding; set padding=0.")
        ho = (h - kh) // stride + 1
        wo = (w - kw) // stride + 1
        tir_script = f"""
@T.prim_func
def main(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
    A = T.match_buffer(a, ({n}, {ci}, {h}, {w}), "{dtype}")
    B = T.match_buffer(b, ({co}, {ci}, {kh}, {kw}), "{dtype}")
    C = T.match_buffer(c, ({n}, {co}, {ho}, {wo}), "{dtype}")

    for nn, ff, yy, xx, rc, ry, rx in T.grid({n}, {co}, {ho}, {wo}, {ci}, {kh}, {kw}):
        with T.block("conv2d_nchw"):
            vnn, vff, vyy, vxx, vrc, vry, vrx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            with T.init():
                C[vnn, vff, vyy, vxx] = T.cast(0, "{dtype}")
            C[vnn, vff, vyy, vxx] = C[vnn, vff, vyy, vxx] + A[vnn, vrc, vyy * {stride} + vry, vxx * {stride} + vrx] * B[vff, vrc, vry, vrx]
"""
        workload_shape = {
            "A": (n, ci, h, w),
            "B": (co, ci, kh, kw),
            "C": (n, co, ho, wo),
        }
    elif op == "depthwise_conv2d":
        n = shape.get("n", 1)
        ci = shape.get("ci", 64)
        h = shape.get("h", 56)
        w = shape.get("w", 56)
        kh = shape.get("kh", 3)
        kw = shape.get("kw", 3)
        stride = shape.get("stride", 1)
        padding = shape.get("padding", 0)
        if padding != 0:
            raise ValueError("Builtin depthwise_conv2d does not implement padding; set padding=0.")
        ho = (h - kh) // stride + 1
        wo = (w - kw) // stride + 1
        tir_script = f"""
@T.prim_func
def main(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
    A = T.match_buffer(a, ({n}, {ci}, {h}, {w}), "{dtype}")
    B = T.match_buffer(b, ({ci}, 1, {kh}, {kw}), "{dtype}")
    C = T.match_buffer(c, ({n}, {ci}, {ho}, {wo}), "{dtype}")

    for nn, cc, yy, xx, ry, rx in T.grid({n}, {ci}, {ho}, {wo}, {kh}, {kw}):
        with T.block("depthwise_conv2d"):
            vnn, vcc, vyy, vxx, vry, vrx = T.axis.remap("SSSSRR", [nn, cc, yy, xx, ry, rx])
            with T.init():
                C[vnn, vcc, vyy, vxx] = T.cast(0, "{dtype}")
            C[vnn, vcc, vyy, vxx] = C[vnn, vcc, vyy, vxx] + A[vnn, vcc, vyy * {stride} + vry, vxx * {stride} + vrx] * B[vcc, 0, vry, vrx]
"""
        workload_shape = {
            "A": (n, ci, h, w),
            "B": (ci, 1, kh, kw),
            "C": (n, ci, ho, wo),
        }
    elif op == "layernorm":
        n = shape.get("n", 64)
        hidden = shape.get("hidden", 256)
        tir_script = f"""
@T.prim_func
def main(x: T.handle, gamma: T.handle, beta: T.handle, y: T.handle) -> None:
    T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
    X = T.match_buffer(x, ({n}, {hidden}), "{dtype}")
    Gamma = T.match_buffer(gamma, ({hidden},), "{dtype}")
    Beta = T.match_buffer(beta, ({hidden},), "{dtype}")
    Y = T.match_buffer(y, ({n}, {hidden}), "{dtype}")

    for i, j in T.grid({n}, {hidden}):
        with T.block("layernorm"):
            vi, vj = T.axis.remap("SS", [i, j])
            Y[vi, vj] = X[vi, vj] * Gamma[vj] + Beta[vj]
"""
        workload_shape: dict[str, tuple[int, ...]] = {
            "X": (n, hidden),
            "Gamma": (hidden,),
            "Beta": (hidden,),
            "Y": (n, hidden),
        }
    elif op == "softmax":
        n = shape.get("n", 64)
        k = shape.get("k", 256)
        tir_script = f"""
@T.prim_func
def main(x: T.handle, y: T.handle) -> None:
    T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
    X = T.match_buffer(x, ({n}, {k}), "{dtype}")
    Y = T.match_buffer(y, ({n}, {k}), "{dtype}")

    for i, j in T.grid({n}, {k}):
        with T.block("softmax"):
            vi, vj = T.axis.remap("SS", [i, j])
            Y[vi, vj] = X[vi, vj]
"""
        workload_shape = {"X": (n, k), "Y": (n, k)}
    else:
        raise ValueError(f"Unsupported operator '{operator}'")

    module = from_source(tir_script)

    def supplier(_: str) -> tvm.IRModule:
        return module

    return supplier, workload_shape


def parse_shape_arg(shape_arg: str) -> dict[str, int]:
    if not shape_arg:
        raise ValueError("--shape argument is required")
    try:
        parsed: dict[str, int] = json.loads(shape_arg)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse --shape JSON: {err}") from err
    try:
        return {k: int(v) for k, v in parsed.items()}
    except Exception as err:
        raise ValueError("--shape values must be convertible to int") from err


if __name__ == "__main__":
    main()
