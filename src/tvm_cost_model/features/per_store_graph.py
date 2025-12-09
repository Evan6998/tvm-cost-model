"""Build per-store graphs from TIR and adapt them to GraphEncoding.

This module mirrors the per-store graph prototype that was experimented with
inside TVM's MetaSchedule, but is self-contained so that graph construction
and GNN logic can live entirely in this repository.

The high‑level split is:

- :func:`enumerate_buffer_stores` walks TIR and collects contextual metadata
  for every ``BufferStore`` site.
- :func:`build_per_store_graph` creates typed edges between stores via
  data/control/structural relations.
- :func:`encode_per_store_graph` converts a per-store graph plus a precomputed
  per-store feature matrix into a :class:`GraphEncoding` that can be consumed
  by the GNN ranker.

Node features are intentionally kept external so they can come from TVM's
PerStoreFeature extractor or any other source; this module only requires a
``(num_nodes, feature_dim)`` numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tvm
from tvm.ir import IRModule  # type: ignore[import]
from tvm.tir import (  # type: ignore[import]
    AttrStmt,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    Evaluate,
    FloatImm,
    For,
    IntImm,
    PrimFunc,
    SeqStmt,
    Stmt,
    Var,
)

from tvm_cost_model.features.graph_encoder import GraphEncoding


@dataclass(frozen=True)
class StoreInfo:
    """Information collected for a ``BufferStore`` site.

    Parameters
    ----------
    id : int
        Consecutive id aligned with per-store feature row order.
    buffer : tvm.tir.Buffer
        The buffer written by this store.
    block_id : int
        The id of the enclosing block that contains this store.
    loop_vars : Tuple[tvm.tir.Var, ...]
        The surrounding loop variables from outermost to innermost.
    reads : Tuple[tvm.tir.Buffer, ...]
        Set of buffers read by the enclosing block (unique, order not guaranteed).
    value_reads : Tuple[tvm.tir.Buffer, ...]
        Set of buffers read on the right-hand side value of this store.
    is_const_value : bool
        Whether the store value is a scalar IntImm/FloatImm constant. These
        stores are ignored by PerStoreFeature on the TVM side.
    """

    id: int
    buffer: Buffer
    block_id: int
    loop_vars: Tuple[Var, ...]
    reads: Tuple[Buffer, ...]
    value_reads: Tuple[Buffer, ...]
    is_const_value: bool


@dataclass(frozen=True)
class PerStoreGraph:
    """Graph structure over ``BufferStore`` sites without node features.

    Parameters
    ----------
    edge_index : np.ndarray
        Shape ``(2, E)``, dtype int64. COO directed edges (src -> dst).
    edge_type : np.ndarray
        Shape ``(E,)``, dtype int64. Edge type id per edge.
    edge_attr : Optional[np.ndarray]
        Optional shape ``(E, F)`` continuous attributes. Currently ``F=1`` for
        share_loop depth (0 for non-share_loop edges).
    type_vocab : Dict[str, int]
        Mapping from relation name to type id.
    stores : List[StoreInfo]
        Metadata for each node in the same order as edges/node features.
    """

    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_attr: Optional[np.ndarray]
    type_vocab: Dict[str, int]
    stores: List[StoreInfo]


_TIRObject = Union[IRModule, PrimFunc, Stmt]


def _get_main_prim_func(mod: IRModule) -> PrimFunc:
    """Return the main PrimFunc from an IRModule.

    Prefers ``"main"`` if present, otherwise returns the first PrimFunc.
    """

    if isinstance(mod, IRModule):
        if hasattr(mod, "get_global_vars"):
            if "main" in mod.get_global_vars():  # type: ignore[operator]
                return mod["main"]  # type: ignore[index]
            for gv in mod.get_global_vars():  # type: ignore[assignment]
                func = mod[gv]
                if isinstance(func, PrimFunc):
                    return func
    raise ValueError("IRModule does not contain a PrimFunc")


def _to_prim_func(obj: _TIRObject) -> PrimFunc:
    if isinstance(obj, IRModule):
        return _get_main_prim_func(obj)
    if isinstance(obj, PrimFunc):
        return obj
    # Stmt: wrap into a dummy PrimFunc
    return PrimFunc(params=[], body=obj)  # type: ignore[arg-type]


def _collect_block_reads(block: Block) -> Tuple[Buffer, ...]:
    """Use TIR analysis to find read regions, then take buffers."""

    from tvm.tir import analysis as tir_analysis  # local import to keep surface small

    reads_regions, _writes_regions = tir_analysis.get_block_read_write_region(block, {})
    reads = tuple({br.buffer for br in reads_regions})
    return reads


def _collect_value_reads(value) -> Tuple[Buffer, ...]:
    """Collect buffers read on the RHS expression of a BufferStore."""

    bufs: List[Buffer] = []

    def fvisit(node):  # pylint: disable=unused-argument
        if isinstance(node, tvm.tir.BufferLoad):
            bufs.append(node.buffer)

    tvm.tir.stmt_functor.post_order_visit(value, fvisit)
    return tuple({b for b in bufs})


def enumerate_buffer_stores(obj: _TIRObject) -> List[StoreInfo]:
    """Enumerate ``BufferStore`` statements in pre-order and collect context.

    The enumeration order aims to match TVM's ``PerStoreFeature`` row order,
    so that external feature matrices can be aligned directly with the result.
    """

    prim = _to_prim_func(obj)
    body: Optional[Stmt] = prim.body
    stores: List[StoreInfo] = []
    block_id_counter = 0

    # Cache of block -> (block_id, reads set)
    block_meta: Dict[Block, Tuple[int, Tuple[Buffer, ...]]] = {}

    def rec(stmt: Stmt, loop_stack: List[Var], current_block: Optional[Block]) -> None:
        nonlocal block_id_counter

        if isinstance(stmt, For):
            loop_stack.append(stmt.loop_var)
            rec(stmt.body, loop_stack, current_block)
            loop_stack.pop()
            return

        if isinstance(stmt, BlockRealize):
            blk = stmt.block
            if blk not in block_meta:
                block_meta[blk] = (block_id_counter, _collect_block_reads(blk))
                block_id_counter += 1
            rec(blk.body, loop_stack, blk)
            if blk.init is not None:
                rec(blk.init, loop_stack, blk)
            return

        if isinstance(stmt, Block):
            if stmt not in block_meta:
                block_meta[stmt] = (block_id_counter, _collect_block_reads(stmt))
                block_id_counter += 1
            rec(stmt.body, loop_stack, stmt)
            if stmt.init is not None:
                rec(stmt.init, loop_stack, stmt)
            return

        if isinstance(stmt, BufferStore):
            if current_block is None:
                # Top-level store (rare), synthesize a block id.
                dummy_block = Block(
                    iter_vars=[],
                    reads=[],
                    writes=[],
                    name_hint="synth_block",
                    body=Evaluate(tvm.tir.IntImm("int32", 0)),
                )
                block_meta[dummy_block] = (block_id_counter, tuple())
                current_block = dummy_block
                block_id_counter += 1
            bid, reads = block_meta[current_block]
            value_reads = _collect_value_reads(stmt.value)
            is_const = isinstance(stmt.value, (IntImm, FloatImm))
            stores.append(
                StoreInfo(
                    id=len(stores),
                    buffer=stmt.buffer,
                    block_id=bid,
                    loop_vars=tuple(loop_stack),
                    reads=reads,
                    value_reads=value_reads,
                    is_const_value=is_const,
                )
            )
            return

        # Handle common container statements explicitly.
        if isinstance(stmt, SeqStmt):
            for s in stmt:
                rec(s, loop_stack, current_block)
            return
        if isinstance(stmt, tvm.tir.IfThenElse):
            rec(stmt.then_case, loop_stack, current_block)
            if stmt.else_case is not None:
                rec(stmt.else_case, loop_stack, current_block)
            return
        if isinstance(stmt, AttrStmt):
            rec(stmt.body, loop_stack, current_block)
            return
        if isinstance(stmt, tvm.tir.stmt.DeclBuffer):
            rec(stmt.body, loop_stack, current_block)
            return
        if isinstance(stmt, Evaluate):
            return

    if body is not None:
        rec(body, [], None)
    return stores


def _common_loop_depth(a: Sequence[Var], b: Sequence[Var]) -> int:
    """Return the number of shared outer loop variables between two stores."""

    depth = 0
    for va, vb in zip(a, b):
        if va == vb:
            depth += 1
        else:
            break
    return depth


def _build_edges(
    stores: List[StoreInfo],
    *,
    use_produce_consume: bool = True,
    use_control_order: bool = True,
    use_same_block: bool = True,
    use_share_loop: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, int]]:
    """Build typed edges over stores.

    Returns
    -------
    edge_index, edge_type, edge_attr, type_vocab
        Edge attributes use a single column for shared loop depth, with 0 for
        non-``share_loop`` edges.
    """

    type_vocab: Dict[str, int] = {
        "produce_consume": 0,
        "control_order": 1,
        "same_block": 2,
        "share_loop": 3,
    }

    edges: List[Tuple[int, int, int, float]] = []  # (src, dst, type_id, attr0)
    n = len(stores)

    if use_produce_consume:
        for i in range(n):
            wi = stores[i].buffer
            for j in range(n):
                if i == j:
                    continue
                same_blk = stores[i].block_id == stores[j].block_id
                reads_j = stores[j].reads
                value_reads_j = stores[j].value_reads
                if same_blk:
                    # Within the same block, rely on block-level reads and store order.
                    if j > i and (
                        any(getattr(r, "data", None) is not None and r.data.same_as(wi.data) for r in stores[i].reads)
                        or any(
                            getattr(r, "data", None) is not None and r.data.same_as(wi.data) for r in value_reads_j
                        )
                    ):
                        edges.append((i, j, type_vocab["produce_consume"], 0.0))
                else:
                    # Cross-block: if j's block reads i's buffer.
                    if any(
                        getattr(r, "data", None) is not None and r.data.same_as(wi.data) for r in reads_j
                    ) or any(
                        getattr(r, "data", None) is not None and r.data.same_as(wi.data) for r in value_reads_j
                    ):
                        edges.append((i, j, type_vocab["produce_consume"], 0.0))

    if use_control_order and n >= 2:
        for i in range(n - 1):
            edges.append((i, i + 1, type_vocab["control_order"], 0.0))

    if use_same_block:
        for i in range(n):
            for j in range(i + 1, n):
                if stores[i].block_id == stores[j].block_id:
                    edges.append((i, j, type_vocab["same_block"], 0.0))
                    edges.append((j, i, type_vocab["same_block"], 0.0))

    if use_share_loop:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = _common_loop_depth(stores[i].loop_vars, stores[j].loop_vars)
                if d > 0:
                    edges.append((i, j, type_vocab["share_loop"], float(d)))

    if not edges:
        edge_index = np.zeros((2, 0), dtype="int64")
        edge_type = np.zeros((0,), dtype="int64")
        edge_attr: Optional[np.ndarray] = np.zeros((0, 1), dtype="float32")
        return edge_index, edge_type, edge_attr, type_vocab

    src = np.array([e[0] for e in edges], dtype="int64")
    dst = np.array([e[1] for e in edges], dtype="int64")
    edge_index = np.stack([src, dst], axis=0)
    edge_type = np.array([e[2] for e in edges], dtype="int64")
    edge_attr = np.array([[e[3]] for e in edges], dtype="float32")
    return edge_index, edge_type, edge_attr, type_vocab


def build_per_store_graph(
    obj: _TIRObject,
    *,
    edge_cfg: Optional[Dict[str, bool]] = None,
) -> PerStoreGraph:
    """Construct a :class:`PerStoreGraph` from TIR."""

    stores = enumerate_buffer_stores(obj)
    cfg = {
        "use_produce_consume": True,
        "use_control_order": True,
        "use_same_block": True,
        "use_share_loop": True,
    }
    if edge_cfg is not None:
        cfg.update(edge_cfg)

    edge_index, edge_type, edge_attr, type_vocab = _build_edges(
        stores,
        use_produce_consume=cfg["use_produce_consume"],
        use_control_order=cfg["use_control_order"],
        use_same_block=cfg["use_same_block"],
        use_share_loop=cfg["use_share_loop"],
    )
    return PerStoreGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        type_vocab=type_vocab,
        stores=stores,
    )


def encode_per_store_graph(node_feat: np.ndarray, graph: PerStoreGraph) -> GraphEncoding:
    """Convert per-store node features plus graph structure into GraphEncoding.

    Parameters
    ----------
    node_feat : np.ndarray
        Shape ``(N, D)`` float array of node features, typically produced by
        TVM's PerStoreFeature extractor. ``N`` must equal ``len(graph.stores)``.
    graph : PerStoreGraph
        Graph structure over ``N`` BufferStore sites.
    """

    if node_feat.ndim != 2:
        raise ValueError("Per-store features must be a 2D array of shape (N, D).")
    n, D = node_feat.shape  # noqa: N806
    if n != len(graph.stores):
        raise ValueError(f"Number of nodes {len(graph.stores)} mismatches feature rows {n}.")

    node_features = node_feat.astype("float32", copy=False).tolist()
    node_types = [0] * n

    edge_index_list: List[Tuple[int, int]] = []
    if graph.edge_index.size > 0:
        src = graph.edge_index[0, :].tolist()
        dst = graph.edge_index[1, :].tolist()
        edge_index_list = list(zip(src, dst))

    edge_types_list: List[int] = []
    if graph.edge_type.size > 0:
        edge_types_list = graph.edge_type.astype("int64", copy=False).tolist()

    feature_names = [f"f{i}" for i in range(D)]
    return GraphEncoding(
        node_features=node_features,
        node_types=node_types,
        edge_index=edge_index_list,
        edge_types=edge_types_list,
        feature_names=feature_names,
    )


def expand_per_buffer_features_to_per_store(
    stores: Sequence[StoreInfo],
    per_buffer_feat: np.ndarray,
) -> np.ndarray:
    """Expand per-buffer features from TVM's PerStoreFeature to per-store features.

    TVM's PerStoreFeature groups features by buffer: each buffer that has at
    least one non-constant BufferStore contributes a single row. Our graph
    uses one node per BufferStore statement. To align them, we replicate the
    per-buffer feature row for every store of that buffer.

    Constant stores (value is IntImm/FloatImm) do not produce a per-buffer
    feature row on the TVM side; for such stores we keep a zero feature
    vector.
    """

    if per_buffer_feat.ndim != 2:
        raise ValueError("Per-buffer features must be a 2D array of shape (M, D).")
    num_buffers, feat_dim = per_buffer_feat.shape

    # Assign an order to buffers based on first non-constant store, matching
    # the logic in PerStoreFeatureCollector (first-seen non-constant store).
    buffer_order: Dict[Buffer, int] = {}
    for store in stores:
        if store.is_const_value:
            continue
        buf = store.buffer
        if buf not in buffer_order:
            buffer_order[buf] = len(buffer_order)

    if num_buffers != len(buffer_order):
        # Best-effort safeguard: when counts do not match, we still try to
        # align by clamping to the smaller size.
        # This should be rare; a warning could be added here if desired.
        pass

    per_store_feat = np.zeros((len(stores), feat_dim), dtype=per_buffer_feat.dtype)
    for store in stores:
        order = buffer_order.get(store.buffer)
        if order is None or order >= num_buffers:
            # Constant-only buffer or mismatch; leave zeros.
            continue
        per_store_feat[store.id, :] = per_buffer_feat[order, :]
    return per_store_feat
