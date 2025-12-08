import numpy as np
import tvm
from tvm.script import tir as T

from tvm_cost_model.features.per_store_graph import (
    PerStoreGraph,
    build_per_store_graph,
    encode_per_store_graph,
    enumerate_buffer_stores,
)


@tvm.script.ir_module
class TwoStores:
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
                # two BufferStore in the same block
                B[vi] = A[vi] + T.float32(1)
                C[vi] = B[vi] * T.float32(2)


@tvm.script.ir_module
class NoStoreModule:
    @T.prim_func
    def main() -> None:
        T.func_attr({"global_symbol": "main"})
        T.evaluate(0)


def test_enumerate_buffer_stores_two_stores():
    stores = enumerate_buffer_stores(TwoStores)
    assert len(stores) == 2
    buf_names = {
        getattr(s.buffer, "name", None) or getattr(getattr(s.buffer, "data", None), "name", None) for s in stores
    }
    assert {"B", "C"}.issubset(buf_names)


def test_build_per_store_graph_edges_and_types():
    g: PerStoreGraph = build_per_store_graph(TwoStores)
    assert len(g.stores) == 2

    num_edges = g.edge_index.shape[1]
    assert g.edge_index.shape == (2, num_edges)
    assert g.edge_type.shape == (num_edges,)
    assert g.edge_attr.shape == (num_edges, 1)

    t = g.type_vocab
    et = g.edge_type
    # control_order (0 -> 1)
    assert (et == t["control_order"]).any()
    # same_block (two stores in same block)
    assert (et == t["same_block"]).any()
    # share_loop edges with positive depth
    share_mask = et == t["share_loop"]
    assert share_mask.any()
    assert (g.edge_attr[share_mask, 0] > 0).any()
    # produce_consume: B -> C
    assert (et == t["produce_consume"]).any()


def test_build_per_store_graph_edge_toggles():
    g = build_per_store_graph(
        TwoStores,
        edge_cfg={"use_produce_consume": False},
    )
    t = g.type_vocab
    assert not (g.edge_type == t["produce_consume"]).any()


def test_build_per_store_graph_empty_module():
    stores = enumerate_buffer_stores(NoStoreModule)
    assert len(stores) == 0

    g = build_per_store_graph(NoStoreModule)
    assert len(g.stores) == 0
    assert g.edge_index.shape == (2, 0)
    assert g.edge_type.shape == (0,)
    assert g.edge_attr.shape == (0, 1)


def test_encode_per_store_graph_alignment():
    g = build_per_store_graph(TwoStores)
    n = len(g.stores)
    feat_dim = 4
    node_feat = np.arange(n * feat_dim, dtype="float32").reshape(n, feat_dim)

    enc = encode_per_store_graph(node_feat, g)
    assert len(enc.node_features) == n
    assert all(len(row) == feat_dim for row in enc.node_features)
    assert enc.node_types == [0] * n
    assert len(enc.edge_index) == g.edge_index.shape[1]
    assert len(enc.edge_types) == g.edge_type.shape[0]
    assert enc.feature_names == [f"f{i}" for i in range(feat_dim)]


def test_encode_per_store_graph_empty_features():
    g = build_per_store_graph(NoStoreModule)
    node_feat = np.zeros((0, 3), dtype="float32")

    enc = encode_per_store_graph(node_feat, g)
    assert enc.node_features == []
    assert enc.node_types == []
    assert enc.edge_index == []
    assert enc.edge_types == []

