"""CLI entry point for dataset bootstrapping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import tvm  # type: ignore[import]
from tvm import script  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]

from tvm_cost_model.data.metaschedule_sampler import (
    MetaScheduleRuntimeEvaluator,
    MetaScheduleSampler,
    ScheduleSampler,
    RuntimeEvaluator,
)
from tvm_cost_model.data.dataset_builder import (
    DatasetBuilder,
    SyntheticRuntimeEvaluator,
    SyntheticScheduleSampler,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap measurement data")
    parser.add_argument("--operator", default="gemm", help="Operator to sample")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.mode == "synthetic":
        sampler: ScheduleSampler = SyntheticScheduleSampler(seed=args.seed)
        evaluator: RuntimeEvaluator = SyntheticRuntimeEvaluator(seed=args.seed)
        artifact_name = f"{args.mode}_{args.operator}_{args.hardware}".lower()
    else:
        module_supplier = build_builtin_module_supplier(
            vector_len=args.vector_len, dtype=args.dtype, global_symbol=args.operator
        )
        workload_shape_fn = lambda _op: {  # type: ignore
            "A": (args.vector_len,),
            "B": (args.vector_len,),
            "C": (args.vector_len,),
        }
        sampler = MetaScheduleSampler(
            target=args.target,
            module_supplier=module_supplier,
            work_dir=Path(args.work_dir),
            workload_shape_fn=workload_shape_fn,  # type: ignore
        )
        device = tvm.runtime.device(args.device_kind, args.device_idx)
        evaluator = MetaScheduleRuntimeEvaluator(
            target=args.target,
            hardware_id=args.hardware,
            number=args.number,
            repeat=args.repeat,
            device=device, # type: ignore
        )
        artifact_name = f"{args.mode}_{args.operator}_{args.hardware}_{args.target}".lower()

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
    vector_len: int, dtype: str, global_symbol: str
) -> Callable[[str], "tvm.IRModule"]:
    """Construct a simple elementwise add module for MetaSchedule runs."""

    tir_script = f"""
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer(({vector_len},), "{dtype}"),
        B: T.Buffer(({vector_len},), "{dtype}"),
        C: T.Buffer(({vector_len},), "{dtype}"),
    ):
        T.func_attr({{"global_symbol": "{global_symbol}", "tir.noalias": True}})
        for i in T.serial(0, {vector_len}):
            C[i] = A[i] + B[i]
"""

    module = from_source(tir_script)

    def supplier(_: str) -> tvm.IRModule:
        return module

    return supplier


if __name__ == "__main__":
    main()
