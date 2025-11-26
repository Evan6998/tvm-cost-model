"""CLI entry point for dataset bootstrapping."""

from __future__ import annotations

import argparse
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import tvm  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]

from tvm_cost_model.data.metaschedule_sampler import (
    MetaScheduleRuntimeEvaluator,
    MetaScheduleSampler,
    RuntimeEvaluator,
    ScheduleSampler,
    generate_inputs_from_workload,
)
from tvm_cost_model.data.dataset_builder import (
    DatasetBuilder,
    SyntheticRuntimeEvaluator,
    SyntheticScheduleSampler,
)

DEFAULT_SHAPES: Dict[str, Dict[str, int]] = {
    "vecadd": {"n": 1024},
    "gemm": {"m": 128, "n": 128, "k": 128},
    "bmm": {"batch": 8, "m": 128, "n": 128, "k": 128},
    "conv2d_nchw": {"n": 1, "ci": 64, "co": 64, "h": 56, "w": 56, "kh": 3, "kw": 3, "stride": 1, "padding": 0},
    "depthwise_conv2d": {"n": 1, "ci": 64, "h": 56, "w": 56, "kh": 3, "kw": 3, "stride": 1, "padding": 0},
    "layernorm": {"n": 64, "hidden": 256},
    "softmax": {"n": 64, "k": 256},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap measurement data")
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
        help="Operator to sample",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="",
        help='JSON dict overriding default shape dims (e.g., \'{"m":128,"n":128,"k":128}\')',
    )
    parser.add_argument("--batches", type=int, default=1, help="Number of sampler batches")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per batch")
    parser.add_argument("--hardware", default="cpu", help="Hardware identifier")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "metaschedule"],
        default="metaschedule",
        help="Sampling/evaluation mode",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/datasets",
        help="Directory to place dataset artifacts",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    # MetaSchedule-specific options
    parser.add_argument("--target", default="llvm -num-cores 8", help="TVM target string")
    parser.add_argument(
        "--work-dir",
        default="artifacts/metaschedule",
        help="Working directory for MetaSchedule runs",
    )
    parser.add_argument(
        "--vector-len",
        type=int,
        default=1024,
        help="Vector length for the builtin elementwise add module",
    )
    parser.add_argument("--dtype", default="float32", help="DType for builtin module buffers")
    parser.add_argument("--device-kind", default="llvm", help="Device kind for measurement")
    parser.add_argument("--device-idx", type=int, default=0, help="Device index for measurement")
    parser.add_argument("--number", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats")
    parser.add_argument("--rpc-host", default="", help="RPC host for remote measurement")
    parser.add_argument("--rpc-port", type=int, default=9090, help="RPC port for remote measurement")
    parser.add_argument("--rpc-key", default="", help="RPC key for tracker if used")
    return parser


def parse_shape_arg(shape_arg: str) -> Dict[str, int]:
    if not shape_arg:
        return {}
    try:
        parsed: Dict[str, int] = json.loads(shape_arg)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse --shape JSON: {err}") from err
    try:
        return {k: int(v) for k, v in parsed.items()}
    except Exception as err:
        raise ValueError("--shape values must be convertible to int") from err


def build_device_and_runner(args: argparse.Namespace) -> Tuple["tvm.runtime.Device", Callable[["tvm.runtime.Module", Sequence["tvm.runtime.Tensor"], "tvm.runtime.Device"], float]]:
    """Return a device handle and a runner (local or RPC-backed)."""
    if args.rpc_host:

        remote = tvm.rpc.connect(args.rpc_host, args.rpc_port, key=args.rpc_key or None) # type: ignore
        dev = remote.device(args.device_kind, args.device_idx) # type: ignore

        def rpc_runner(module: "tvm.runtime.Module", inputs: Sequence["tvm.runtime.Tensor"], _dev: "tvm.runtime.Device") -> float:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "mod.so"
                module.export_library(path) # type: ignore
                remote.upload(str(path)) # type: ignore
                remote_mod = remote.load_module(path.name) # type: ignore
                time_eval = remote_mod.time_evaluator( # type: ignore[attr-defined]
                    remote_mod.entry_name, dev, number=args.number, repeat=args.repeat # type: ignore[attr-defined]
                )
                return float(time_eval(*inputs).mean) * 1000.0  # type: ignore

        return dev, rpc_runner # type: ignore

    dev = tvm.runtime.device(args.device_kind, args.device_idx) # type: ignore

    def local_runner(module: "tvm.runtime.Module", inputs: Sequence["tvm.runtime.Tensor"], _dev: tvm.runtime.Device) -> float:
        # fallback: use module's time_evaluator on the provided device
        time_eval = module.time_evaluator(  # type: ignore[attr-defined]
            module.entry_name, dev, number=args.number, repeat=args.repeat # type: ignore[attr-defined]
        )
        return float(time_eval(*inputs).mean) * 1000.0

    return dev, local_runner # type: ignore


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    shape_override = parse_shape_arg(args.shape)
    base_shape = dict(DEFAULT_SHAPES.get(args.operator, {}))
    if args.operator == "vecadd" and "n" not in shape_override:
        base_shape["n"] = args.vector_len
    shape = {**base_shape, **shape_override}

    output_dir = Path(args.output_dir)
    if args.mode == "synthetic":
        sampler: ScheduleSampler = SyntheticScheduleSampler(seed=args.seed)
        evaluator: RuntimeEvaluator = SyntheticRuntimeEvaluator(seed=args.seed)
        artifact_name = f"{args.mode}_{args.operator}_{args.hardware}".lower()
    elif args.mode == "metaschedule":
        module_supplier, workload_shape = build_builtin_module_supplier(
            operator=args.operator,
            shape=shape,
            dtype=args.dtype,
            global_symbol=args.operator,
        )
        workload_shape_fn = lambda _op, ws=workload_shape: ws  # type: ignore
        sampler = MetaScheduleSampler(
            target=args.target,
            module_supplier=module_supplier,
            work_dir=Path(args.work_dir),
            workload_shape_fn=workload_shape_fn,  # type: ignore
        )
        device, runner = build_device_and_runner(args)
        evaluator = MetaScheduleRuntimeEvaluator(
            target=args.target,
            hardware_id=args.hardware,
            number=args.number,
            repeat=args.repeat,
            device=device,
            runner=runner,
            input_generator=partial(generate_inputs_from_workload, dtype=args.dtype),
        )
        artifact_name = f"{args.mode}_{args.operator}_{args.hardware}_{args.target}".lower()
    else:
        raise ValueError(f"Unsupported mode '{args.mode}'")

    artifact_name = artifact_name.replace(" ", "_")
    builder = DatasetBuilder(sampler, evaluator, output_dir)
    measurements = builder.collect(
        operator=args.operator,
        batches=args.batches,
        batch_size=args.batch_size,
        hardware_id=args.hardware,
    )
    artifact = builder.export(measurements, artifact_name=artifact_name)
    print(f"Wrote dataset to {artifact}")


def build_builtin_module_supplier(
    operator: str, shape: Dict[str, int], dtype: str, global_symbol: str
) -> Tuple[Callable[[str], "tvm.IRModule"], Dict[str, Tuple[int, ...]]]:
    """Construct simple TVMScript IRModules for common operators."""
    op = operator.lower()
    if op == "vecadd":
        n = shape.get("n", DEFAULT_SHAPES["vecadd"]["n"])
        tir_script = f"""
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer(({n},), "{dtype}"),
        B: T.Buffer(({n},), "{dtype}"),
        C: T.Buffer(({n},), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for i in T.serial(0, {n}):
            C[i] = A[i] + B[i]
"""
        workload_shape = {"A": (n,), "B": (n,), "C": (n,)}
    elif op == "gemm":
        m = shape.get("m", 128)
        n = shape.get("n", 128)
        k = shape.get("k", 128)
        tir_script = f"""
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer(({m}, {k}), "{dtype}"),
        B: T.Buffer(({k}, {n}), "{dtype}"),
        C: T.Buffer(({m}, {n}), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for i in T.serial(0, {m}):
            for j in T.serial(0, {n}):
                C[i, j] = T.cast(0, "{dtype}")
                for kk in T.serial(0, {k}):
                    C[i, j] = C[i, j] + A[i, kk] * B[kk, j]
"""
        workload_shape = {"A": (m, k), "B": (k, n), "C": (m, n)}
    elif op == "bmm":
        b = shape.get("batch", 8)
        m = shape.get("m", 128)
        n = shape.get("n", 128)
        k = shape.get("k", 128)
        tir_script = f"""
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer(({b}, {m}, {k}), "{dtype}"),
        B: T.Buffer(({b}, {k}, {n}), "{dtype}"),
        C: T.Buffer(({b}, {m}, {n}), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for batch in T.serial(0, {b}):
            for i in T.serial(0, {m}):
                for j in T.serial(0, {n}):
                    C[batch, i, j] = T.cast(0, "{dtype}")
                    for kk in T.serial(0, {k}):
                        C[batch, i, j] = C[batch, i, j] + A[batch, i, kk] * B[batch, kk, j]
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
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer(({n}, {ci}, {h}, {w}), "{dtype}"),
        B: T.Buffer(({co}, {ci}, {kh}, {kw}), "{dtype}"),
        C: T.Buffer(({n}, {co}, {ho}, {wo}), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for nn in T.serial(0, {n}):
            for ff in T.serial(0, {co}):
                for yy in T.serial(0, {ho}):
                    for xx in T.serial(0, {wo}):
                        C[nn, ff, yy, xx] = T.cast(0, "{dtype}")
                        for rc in T.serial(0, {ci}):
                            for ry in T.serial(0, {kh}):
                                for rx in T.serial(0, {kw}):
                                    C[nn, ff, yy, xx] = C[nn, ff, yy, xx] + A[nn, rc, yy*{stride}+ry, xx*{stride}+rx] * B[ff, rc, ry, rx]
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
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer(({n}, {ci}, {h}, {w}), "{dtype}"),
        B: T.Buffer(({ci}, 1, {kh}, {kw}), "{dtype}"),
        C: T.Buffer(({n}, {ci}, {ho}, {wo}), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for nn in T.serial(0, {n}):
            for cc in T.serial(0, {ci}):
                for yy in T.serial(0, {ho}):
                    for xx in T.serial(0, {wo}):
                        C[nn, cc, yy, xx] = T.cast(0, "{dtype}")
                        for ry in T.serial(0, {kh}):
                            for rx in T.serial(0, {kw}):
                                C[nn, cc, yy, xx] = C[nn, cc, yy, xx] + A[nn, cc, yy*{stride}+ry, xx*{stride}+rx] * B[cc, 0, ry, rx]
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
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        X: T.Buffer(({n}, {hidden}), "{dtype}"),
        Gamma: T.Buffer(({hidden},), "{dtype}"),
        Beta: T.Buffer(({hidden},), "{dtype}"),
        Y: T.Buffer(({n}, {hidden}), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for i in T.serial(0, {n}):
            for j in T.serial(0, {hidden}):
                Y[i, j] = X[i, j] * Gamma[j] + Beta[j]
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
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        X: T.Buffer(({n}, {k}), "{dtype}"),
        Y: T.Buffer(({n}, {k}), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for i in T.serial(0, {n}):
            for j in T.serial(0, {k}):
                Y[i, j] = X[i, j]
"""
        workload_shape = {"X": (n, k), "Y": (n, k)}
    else:
        raise ValueError(f"Unsupported operator '{operator}'")

    module = from_source(tir_script)

    def supplier(_: str) -> tvm.IRModule:
        return module

    return supplier, workload_shape


if __name__ == "__main__":
    main()
