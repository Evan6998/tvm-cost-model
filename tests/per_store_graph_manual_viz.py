"""Manual helper: build a per-store graph and dump DOT.

Usage (from tvm-cost-model repo root):

    PYTHONPATH=src \\
    python tests/per_store_graph_manual_viz.py --mod all --out store_graph.dot

Then render with graphviz, e.g.:

    dot -Tpng store_graph_simple.dot -o store_graph_simple.png
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import tvm
from tvm.script import tir as T
from tvm.tir.schedule import Schedule

from tvm.meta_schedule.feature_extractor import FeatureExtractor  # type: ignore[import]
from tvm.meta_schedule.search_strategy import MeasureCandidate  # type: ignore[import]
from tvm.meta_schedule.tune_context import TuneContext  # type: ignore[import]

from tvm_cost_model.features.per_store_graph import (
    build_per_store_graph,
    enumerate_buffer_stores,
    expand_per_buffer_features_to_per_store,
)
from tvm_cost_model.features.per_store_viz import (
    per_store_graph_to_dot,
    print_node_features,
)


@tvm.script.ir_module
class SimpleModule:  # pylint: disable=too-few-public-methods
    """Single block, two stores, simple dependency B -> C."""

    @T.prim_func
    def main(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
        C: T.Buffer((16,), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(16):
            with T.block("compute"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi] + T.float32(1)
                C[vi] = B[vi] * T.float32(2)


@tvm.script.ir_module
class TwoBlocksModule:  # pylint: disable=too-few-public-methods
    """Two separate blocks: A -> B in first, B -> C in second."""

    @T.prim_func
    def main(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
        C: T.Buffer((16,), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(16):
            with T.block("produce_B"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi] + T.float32(1)
        for i in T.serial(16):
            with T.block("produce_C"):
                vi = T.axis.spatial(16, i)
                C[vi] = B[vi] * T.float32(2)


@tvm.script.ir_module
class NestedModule:  # pylint: disable=too-few-public-methods
    """2D nested loops, one store uses (i,j), another only i."""

    @T.prim_func
    def main(
        A: T.Buffer((8, 8), "float32"),
        B: T.Buffer((8, 8), "float32"),
        D: T.Buffer((8,), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j in T.grid(8, 8):
            with T.block("body"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)
                D[vi] = D[vi] + B[vi, vj]


@tvm.script.ir_module
class Chain3Module:  # pylint: disable=too-few-public-methods
    """Single block, three-step chain: A -> B -> C -> D."""

    @T.prim_func
    def main(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
        C: T.Buffer((16,), "float32"),
        D: T.Buffer((16,), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(16):
            with T.block("compute"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi] + T.float32(1)
                C[vi] = B[vi] * T.float32(2)
                D[vi] = C[vi] - T.float32(3)


@tvm.script.ir_module
class FanOutInModule:  # pylint: disable=too-few-public-methods
    """Fan-out from A to B/C, then fan-in to D/E."""

    @T.prim_func
    def main(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
        C: T.Buffer((16,), "float32"),
        D: T.Buffer((16,), "float32"),
        E: T.Buffer((16,), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.serial(16):
            with T.block("produce_BC"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi]
                C[vi] = A[vi] * T.float32(2)
        for i in T.serial(16):
            with T.block("produce_D"):
                vi = T.axis.spatial(16, i)
                D[vi] = B[vi] + C[vi]
        for i in T.serial(16):
            with T.block("produce_E"):
                vi = T.axis.spatial(16, i)
                E[vi] = D[vi] * T.float32(3)


WORKLOADS: Dict[str, tvm.IRModule] = {
    "simple": SimpleModule,
    "two_blocks": TwoBlocksModule,
    "nested": NestedModule,
    "chain3": Chain3Module,
    "fan_out_in": FanOutInModule,
}


def _extract_per_store_features(
    ctx: TuneContext,
    cand: MeasureCandidate,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """Wrapper around TVM's per-store feature extractor."""

    tvm_arr = extractor.extract_from(ctx, [cand])[0]
    feat = tvm_arr.numpy().astype("float32", copy=False)
    return feat


def _run_one(mod_name: str, mod: tvm.IRModule, out_path: str) -> None:
    print("=" * 80)
    print(f"Workload: {mod_name}")
    print("=" * 80)

    sch = Schedule(mod)
    cand = MeasureCandidate(sch, [])
    ctx = TuneContext(mod=mod, target="llvm", num_threads=1)

    extractor = FeatureExtractor.create("per-store-feature")
    per_buffer_feat = _extract_per_store_features(ctx, cand, extractor)

    graph = build_per_store_graph(mod)
    stores = enumerate_buffer_stores(mod)
    per_store_feat = expand_per_buffer_features_to_per_store(stores, per_buffer_feat)

    print(f"#nodes={per_store_feat.shape[0]}, #edges={graph.edge_index.shape[1]}")
    print("edge types:", graph.type_vocab)

    print("\nPer-node features:")
    print_node_features(per_store_feat, stores=stores)

    per_store_graph_to_dot(per_store_feat, graph, out=out_path, stores=stores)
    print(f"DOT graph written to: {out_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump per-store graphs as DOT")
    parser.add_argument(
        "--mod",
        type=str,
        default="all",
        choices=list(WORKLOADS.keys()) + ["all"],
        help="Which TIR workload to use (or 'all').",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="store_graph.dot",
        help="Output DOT file (or prefix when --mod=all)",
    )
    args = parser.parse_args()

    if args.mod == "all":
        prefix, dot_suffix = args.out, ""
        if "." in args.out:
            prefix, dot_suffix = args.out.rsplit(".", maxsplit=1)
            dot_suffix = "." + dot_suffix
        for name, mod in WORKLOADS.items():
            out_path = f"{prefix}_{name}{dot_suffix or '.dot'}"
            _run_one(name, mod, out_path)
    else:
        mod = WORKLOADS[args.mod]
        _run_one(args.mod, mod, args.out)

    print(f"Example: dot -Tpng {args.out} -o store_graph.png")


if __name__ == "__main__":
    main()
