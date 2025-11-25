"""TVM-backed TIR to ProgramGraph extraction using Python visitors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import tvm  # type: ignore[import]
from tvm import tir  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]
from tvm.tir import stmt_functor  # type: ignore[import]

from tvm_cost_model.features.graph_builder import GraphBuilder, GraphNode, LoopInfo, ProgramGraph


@dataclass
class _LoopCapture:
    var: str
    extent: int


def _to_int(expr: tir.PrimExpr) -> int:
    """Best-effort conversion of a TIR extent to int, defaulting to -1."""

    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    try:
        return int(expr) # type: ignore[arg-type]
    except TypeError:
        return -1


class _TIRVisitor:
    """Collects loops and buffer names via post-order traversal."""

    def __init__(self) -> None:
        self.loops: List[_LoopCapture] = []
        self.buffers: Set[str] = set()

    def visit(self, stmt: tir.Stmt) -> None:
        def _callback(node: tir.Stmt) -> None:
            if isinstance(node, tir.For):
                self._record_loop(node)
            elif isinstance(node, tir.BufferStore):
                self._record_buffer(node.buffer)
            elif isinstance(node, tir.BufferLoad):
                self._record_buffer(node.buffer)
            elif isinstance(node, tir.Block):
                for region in list(node.reads) + list(node.writes):
                    self._record_buffer(region.buffer)

        stmt_functor.post_order_visit(stmt, _callback)  # type: ignore[arg-type]

    def _record_loop(self, stmt: tir.For) -> None:
        self.loops.append(_LoopCapture(var=str(stmt.loop_var), extent=_to_int(stmt.extent)))

    def _record_buffer(self, buffer: tir.Buffer) -> None:
        self.buffers.add(str(buffer.name))  # type: ignore[arg-type]


class TVMGraphBuilder(GraphBuilder):
    """Concrete graph builder using TVM's TIR parser and visitors."""

    def __init__(self) -> None:
        super().__init__()

    def build(self, tir_script: str) -> ProgramGraph:  # type: ignore[override]
        mod = from_source(tir_script)
        func = next(iter(mod.functions.values()))
        visitor = _TIRVisitor()
        visitor.visit(func.body)

        loops = [LoopInfo(var=l.var, extent=l.extent) for l in visitor.loops]
        buffers = sorted(visitor.buffers)

        nodes: List[GraphNode] = []
        edges: List[tuple[int, int, str]] = []

        loop_nodes = [GraphNode(name=f"loop:{l.var}", attrs={"extent": l.extent}) for l in loops]
        buffer_nodes = [GraphNode(name=f"buffer:{name}", attrs={"name_len": len(name)}) for name in buffers]
        nodes.extend(loop_nodes)
        nodes.extend(buffer_nodes)

        compute_node = GraphNode(
            name="compute",
            attrs={
                "loop_depth": len(loop_nodes),
                "buffer_count": len(buffer_nodes),
                "total_extent": sum(node.attrs.get("extent", 0) for node in loop_nodes),
            },
        )
        compute_idx = len(nodes)
        nodes.append(compute_node)

        for idx in range(len(loop_nodes)):
            edges.append((compute_idx, idx, "iterates"))
        for offset, _ in enumerate(buffer_nodes):
            edges.append((compute_idx, len(loop_nodes) + offset, "accesses"))

        return ProgramGraph(nodes=nodes, edges=edges)
