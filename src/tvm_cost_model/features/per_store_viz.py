"""Small utilities for inspecting and visualizing per-store graphs."""

from __future__ import annotations

from typing import Optional, Sequence, TextIO, Dict

import sys
import numpy as np

from tvm_cost_model.features.per_store_graph import PerStoreGraph, StoreInfo


def per_store_graph_to_dot(
    node_feat: np.ndarray,
    graph: PerStoreGraph,
    *,
    out: str,
    stores: Optional[Sequence[StoreInfo]] = None,
) -> None:
    """Dump a per-store graph plus node features to a DOT file."""

    type_names: Dict[int, str] = {tid: name for name, tid in graph.type_vocab.items()}

    lines: list[str] = []
    lines.append("digraph StoreGraph {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=rounded];")

    num_nodes = int(node_feat.shape[0])
    for i in range(num_nodes):
        label_parts: list[str] = [f"#{i}"]
        if stores is not None and i < len(stores):
            buf = stores[i].buffer
            buf_name = getattr(buf, "name", None)
            if buf_name is None and getattr(buf, "data", None) is not None:
                buf_name = getattr(buf.data, "name", None)
            if buf_name is not None:
                label_parts.append(str(buf_name))
            label_parts.append(f"block={stores[i].block_id}")
        label = " ".join(label_parts)
        lines.append(f'  n{i} [label="{label}"];')

    num_edges = int(graph.edge_index.shape[1])
    for k in range(num_edges):
        src = int(graph.edge_index[0, k])
        dst = int(graph.edge_index[1, k])
        t_id = int(graph.edge_type[k])
        t_name = type_names.get(t_id, str(t_id))
        edge_label = t_name
        if t_name == "share_loop" and graph.edge_attr is not None:
            depth = float(graph.edge_attr[k, 0])
            edge_label = f"{t_name}\\n{int(depth)}"
        lines.append(f'  n{src} -> n{dst} [label="{edge_label}"];')

    lines.append("}")
    with open(out, "w", encoding="utf-8") as f:  # noqa: PTH123
        f.write("\n".join(lines))


def print_node_features(
    node_feat: np.ndarray,
    *,
    stores: Optional[Sequence[StoreInfo]] = None,
    file: TextIO = sys.stdout,
) -> None:
    """Print per-node features for manual inspection."""

    num_nodes, feat_dim = node_feat.shape
    print(f"node_feat shape = ({num_nodes}, {feat_dim})", file=file)
    for i in range(num_nodes):
        buf_name = None
        block_id = None
        if stores is not None and i < len(stores):
            buf = stores[i].buffer
            buf_name = getattr(buf, "name", None)
            if buf_name is None and getattr(buf, "data", None) is not None:
                buf_name = getattr(buf.data, "name", None)
            block_id = stores[i].block_id
        header: list[str] = [f"node {i}"]
        if buf_name is not None:
            header.append(f"buf={buf_name}")
        if block_id is not None:
            header.append(f"block={block_id}")
        print("[" + ", ".join(header) + "]", file=file)
        print(node_feat[i], file=file)
        print(file=file)

