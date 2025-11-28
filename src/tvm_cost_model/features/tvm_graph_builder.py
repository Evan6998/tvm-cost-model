"""TVM-backed TIR to ProgramGraph extraction using Python visitors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from tvm import tir  # type: ignore[import]
from tvm.runtime import DataType  # type: ignore[import]
from tvm.script import from_source  # type: ignore[import]
from tvm.tir import stmt_functor  # type: ignore[import]

from tvm_cost_model.features.graph_builder import GraphBuilder, GraphNode, ProgramGraph


@dataclass
class _LoopCapture:
    var: str
    extent: int
    depth: int
    for_kind: int
    parent: int | None


@dataclass
class _BufferStats:
    scope_code: int = -1
    elem_bytes: int = -1
    total_bytes: int = -1
    read_count: int = 0
    write_count: int = 0


def _to_int(expr: tir.PrimExpr) -> int:
    """Best-effort conversion of a TIR extent to int, defaulting to -1."""

    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    try:
        return int(expr)  # type: ignore[arg-type]
    except TypeError:
        return -1


def _scope_to_code(scope: str) -> int:
    scope_lut = {
        "global": 0,
        "": 0,
        "shared": 1,
        "local": 2,
        "warp": 3,
    }
    return scope_lut.get(scope, -1)


class _TIRVisitor:
    """Collects loop, buffer, and flop statistics with nesting awareness."""

    def __init__(self) -> None:
        self.loops: List[_LoopCapture] = []
        self.buffers: Dict[str, _BufferStats] = {}
        self.loop_stack: List[int] = []
        self.loop_buffer_accesses: Set[Tuple[int, str]] = set()
        self.total_flops: int = 0
        self.reduction_loop_vars: Set[str] = set()
        self.unrecognized_stmts: Set[str] = set()
        self.unrecognized_exprs: Set[str] = set()

    def visit(self, stmt: tir.Stmt) -> None:
        self._visit_stmt(stmt)
        if self.unrecognized_stmts or self.unrecognized_exprs:
            print(
                "Unrecognized TIR nodes | stmts:",
                sorted(self.unrecognized_stmts),
                "| exprs:",
                sorted(self.unrecognized_exprs),
            )

    def _visit_stmt(self, stmt: tir.Stmt) -> None:
        if isinstance(stmt, tir.For):
            self._enter_loop(stmt)
            self._visit_stmt(stmt.body)
            self.loop_stack.pop()
            return

        if isinstance(stmt, tir.SeqStmt):
            for seq in stmt.seq:
                self._visit_stmt(seq)
            return

        if isinstance(stmt, tir.BlockRealize):
            self._record_block(stmt.block, list(stmt.iter_values))
            if stmt.block.init is not None:
                self._visit_stmt(stmt.block.init)
            self._visit_stmt(stmt.block.body)
            return

        if isinstance(stmt, tir.Block):
            self._record_block(stmt)
            if stmt.init is not None:
                self._visit_stmt(stmt.init)
            self._visit_stmt(stmt.body)
            return

        if isinstance(stmt, tir.BufferStore):
            self._record_write(stmt.buffer)
            for idx in stmt.indices:
                self._visit_expr(idx)
            self._visit_expr(stmt.value)
            return

        if isinstance(stmt, tir.Evaluate):
            self._visit_expr(stmt.value)
            return

        if isinstance(stmt, tir.IfThenElse):
            self._visit_expr(stmt.condition)
            self._visit_stmt(stmt.then_case)
            if stmt.else_case:
                self._visit_stmt(stmt.else_case)
            return

        if isinstance(stmt, tir.AssertStmt):
            self._visit_expr(stmt.condition)
            self._visit_stmt(stmt.body)
            return

        if isinstance(stmt, tir.AttrStmt):
            self._visit_stmt(stmt.body)
            return

        if isinstance(stmt, tir.Allocate):
            for extent in stmt.extents:
                self._visit_expr(extent)
            self._visit_expr(stmt.condition)
            self._visit_stmt(stmt.body)
            return

        if isinstance(stmt, tir.AllocateConst):
            self._visit_stmt(stmt.body)
            return

        # Fallback: still attempt to collect flops and buffer touches.
        self.unrecognized_stmts.add(type(stmt).__name__)
        stmt_functor.post_order_visit(stmt, self._visit_expr)  # type: ignore[arg-type]

    def _visit_expr(self, expr: tir.PrimExpr) -> None:
        if isinstance(expr, tir.BufferLoad):
            self._record_read(expr.buffer)
            for idx in expr.indices:
                self._visit_expr(idx)
            return

        if isinstance(expr, (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.FloorDiv, tir.FloorMod, tir.Mod, tir.Max, tir.Min)):
            self.total_flops += 1
            self._visit_expr(expr.a)  # type: ignore[arg-type]
            self._visit_expr(expr.b)  # type: ignore[arg-type]
            return

        if isinstance(expr, tir.Select):
            self._visit_expr(expr.condition)
            self._visit_expr(expr.true_value)
            self._visit_expr(expr.false_value)
            return

        if isinstance(expr, tir.Call):
            self.total_flops += 1
            for arg in expr.args:
                self._visit_expr(arg)
            return

        if isinstance(expr, tir.Cast):
            self._visit_expr(expr.value)
            return

        if isinstance(expr, tir.Let):
            self._visit_expr(expr.value)
            self._visit_expr(expr.body)
            return

        # Leaf expressions (constants, vars) are ignored.
        self.unrecognized_exprs.add(type(expr).__name__)

    def _enter_loop(self, stmt: tir.For) -> None:
        parent_idx = self.loop_stack[-1] if self.loop_stack else None
        loop_idx = len(self.loops)
        capture = _LoopCapture(
            var=str(stmt.loop_var),
            extent=_to_int(stmt.extent),
            depth=len(self.loop_stack),
            for_kind=int(stmt.kind),
            parent=parent_idx,
        )
        self.loops.append(capture)
        self.loop_stack.append(loop_idx)

    def _record_block(self, block: tir.Block, iter_values: List[tir.PrimExpr] | None = None) -> None:
        for region in list(block.reads) + list(block.writes):
            self._ensure_buffer(region.buffer)

        if iter_values is not None and block.iter_vars:
            for iter_var, value in zip(block.iter_vars, iter_values):
                if iter_var.iter_type == tir.IterVar.CommReduce and isinstance(value, tir.Var):
                    self.reduction_loop_vars.add(str(value.name))  # type: ignore[attr-defined]

    def _record_loop_access(self, buffer: tir.Buffer) -> None:
        buffer_name = str(buffer.name)  # type: ignore[attr-defined]
        for loop_idx in self.loop_stack:
            self.loop_buffer_accesses.add((loop_idx, buffer_name))

    def _ensure_buffer(self, buffer: tir.Buffer) -> _BufferStats:
        name = str(buffer.name) # type: ignore[attr-defined] 
        stats = self.buffers.setdefault(name, _BufferStats())
        scope_code = _scope_to_code(buffer.scope())  # type: ignore[attr-defined]
        if stats.scope_code == -1 and scope_code != -1:
            stats.scope_code = scope_code

        elem_bytes = _elem_bytes(buffer)
        if stats.elem_bytes == -1 and elem_bytes != -1:
            stats.elem_bytes = elem_bytes

        total_bytes = _buffer_total_bytes(buffer, elem_bytes)
        if stats.total_bytes == -1 and total_bytes != -1:
            stats.total_bytes = total_bytes
        return stats

    def _record_read(self, buffer: tir.Buffer) -> None:
        stats = self._ensure_buffer(buffer)
        stats.read_count += 1
        self._record_loop_access(buffer)

    def _record_write(self, buffer: tir.Buffer) -> None:
        stats = self._ensure_buffer(buffer)
        stats.write_count += 1
        self._record_loop_access(buffer)


def _elem_bytes(buffer: tir.Buffer) -> int:
    dtype = DataType(buffer.dtype)  # type: ignore[arg-type]
    lanes = max(dtype.lanes, 1)
    return (dtype.bits // 8) * lanes


def _buffer_total_bytes(buffer: tir.Buffer, elem_bytes: int | None = None) -> int:
    if elem_bytes is None or elem_bytes == -1:
        elem_bytes = _elem_bytes(buffer)
    if elem_bytes == -1:
        return -1

    total_elems = 1
    for dim in buffer.shape: # type: ignore[attr-defined]
        dim_int = _to_int(dim)  # type: ignore[arg-type]
        if dim_int == -1:
            return -1
        total_elems *= dim_int
    return total_elems * elem_bytes


class TVMGraphBuilder(GraphBuilder):
    """Concrete graph builder using TVM's TIR parser and visitors."""

    def __init__(self) -> None:
        super().__init__()

    def build(self, tir_script: str) -> ProgramGraph:  # type: ignore[override]
        mod = from_source(tir_script)
        func = next(iter(mod.functions.values()))
        visitor = _TIRVisitor()
        visitor.visit(func.body)

        buffers = sorted(visitor.buffers.keys())

        nodes: List[GraphNode] = []
        edges: List[tuple[int, int, str]] = []

        loop_nodes: List[GraphNode] = []
        for loop in visitor.loops:
            loop_nodes.append(
                GraphNode(
                    name=f"loop:{loop.var}",
                    attrs={
                        "extent": loop.extent,
                        "depth": loop.depth,
                        "is_parallel": int(loop.for_kind == int(tir.ForKind.PARALLEL)),
                        "is_vectorized": int(loop.for_kind == int(tir.ForKind.VECTORIZED)),
                        "is_unrolled": int(loop.for_kind == int(tir.ForKind.UNROLLED)),
                        "is_thread_bound": int(loop.for_kind == int(tir.ForKind.THREAD_BINDING)),
                        "is_reduction": int(loop.var in visitor.reduction_loop_vars),
                    },
                )
            )

        buffer_nodes: List[GraphNode] = []
        for name in buffers:
            stats = visitor.buffers[name]
            buffer_nodes.append(
                GraphNode(
                    name=f"buffer:{name}",
                    attrs={
                        "name_len": len(name),
                        "scope_code": stats.scope_code,
                        "elem_bytes": stats.elem_bytes,
                        "total_bytes": stats.total_bytes,
                        "read_count": stats.read_count,
                        "write_count": stats.write_count,
                    },
                )
            )

        nodes.extend(loop_nodes)
        nodes.extend(buffer_nodes)

        global_bytes = 0
        shared_bytes = 0
        local_bytes = 0
        other_bytes = 0
        for stats in visitor.buffers.values():
            elem_bytes = stats.elem_bytes if stats.elem_bytes != -1 else 0
            buffer_bytes = (stats.read_count + stats.write_count) * elem_bytes
            if stats.scope_code == 0:
                global_bytes += buffer_bytes
            elif stats.scope_code == 1:
                shared_bytes += buffer_bytes
            elif stats.scope_code == 2:
                local_bytes += buffer_bytes
            else:
                other_bytes += buffer_bytes

        compute_node = GraphNode(
            name="compute",
            attrs={
                "loop_depth": len(loop_nodes),
                "buffer_count": len(buffer_nodes),
                "total_extent": sum(node.attrs.get("extent", 0) for node in loop_nodes),
                "total_flops": visitor.total_flops,
                "global_bytes": global_bytes,
                "shared_bytes": shared_bytes,
                "local_bytes": local_bytes,
                "other_bytes": other_bytes,
                "arith_intensity": visitor.total_flops / float(global_bytes if global_bytes > 0 else 1),
            },
        )
        compute_idx = len(nodes)
        nodes.append(compute_node)

        for idx in range(len(loop_nodes)):
            edges.append((compute_idx, idx, "iterates"))
        for offset, _ in enumerate(buffer_nodes):
            edges.append((compute_idx, len(loop_nodes) + offset, "accesses"))

        for idx, loop in enumerate(visitor.loops):
            if loop.parent is not None:
                edges.append((loop.parent, idx, "loop_child"))

        buffer_indices = {name: len(loop_nodes) + offset for offset, name in enumerate(buffers)}
        for loop_idx, buffer_name in visitor.loop_buffer_accesses:
            buffer_idx = buffer_indices.get(buffer_name)
            if buffer_idx is not None:
                edges.append((loop_idx, buffer_idx, "loop_accesses"))

        return ProgramGraph(nodes=nodes, edges=edges)
