"""Utility to convert ProgramGraph into numeric encodings for ML models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from tvm_cost_model.features.graph_builder import ProgramGraph


@dataclass
class GraphEncoding:
    node_features: list[list[float]]
    node_types: list[int]
    edge_index: list[tuple[int, int]]
    edge_types: list[int]
    feature_names: list[str] = field(default_factory=list[str])


class GraphEncoder:
    """Encodes ProgramGraph nodes/edges into integer labels and dense features.

    This encoder keeps expanding vocabularies as new node/edge types appear to
    maintain consistent IDs across calls.
    """

    _LOG1P_FIELDS = {
        "extent",
        "total_extent",
        "total_bytes",
        "global_bytes",
        "shared_bytes",
        "local_bytes",
        "other_bytes",
        "total_flops",
        "read_count",
        "write_count",
    }
    _COMBO_FEATURES = {
        "loop:depth_x_log_extent",
        "loop:parallel_log_extent",
        "buffer:traffic_bytes",
        "buffer:traffic_ratio",
    }
    _STRUCTURAL_FEATURES = {"deg_in", "deg_out", "deg_total"}

    def __init__(self) -> None:
        self.node_type_to_id: dict[str, int] = {}
        self.edge_type_to_id: dict[str, int] = {}
        self._feature_names: list[str] = []

    def encode(self, graph: ProgramGraph) -> GraphEncoding:
        if not graph.nodes:
            return GraphEncoding([], [], [], [], [])

        feature_names = self._collect_feature_names(graph)
        node_features: list[list[float]] = []
        node_types: list[int] = []
        edge_types_in_graph = self._edge_types_in_graph(graph)
        num_nodes = len(graph.nodes)
        deg_in = [0] * num_nodes
        deg_out = [0] * num_nodes
        deg_in_by_type = {etype: [0] * num_nodes for etype in edge_types_in_graph}
        deg_out_by_type = {etype: [0] * num_nodes for etype in edge_types_in_graph}
        for src, dst, etype in graph.edges:
            deg_out[src] += 1
            deg_in[dst] += 1
            deg_out_by_type.setdefault(etype, [0] * num_nodes)[src] += 1
            deg_in_by_type.setdefault(etype, [0] * num_nodes)[dst] += 1
        deg_total = [i + o for i, o in zip(deg_in, deg_out)]

        for node in graph.nodes:
            node_type = node.name.split(":", maxsplit=1)[0]
            node_types.append(self._node_type_id(node_type))
            node_features.append(
                [
                    self._feature_value(
                        name=name,
                        attrs=node.attrs,
                        node_type=node_type,
                        deg_in=deg_in,
                        deg_out=deg_out,
                        deg_total=deg_total,
                        deg_in_by_type=deg_in_by_type,
                        deg_out_by_type=deg_out_by_type,
                        node_idx=len(node_features),
                    )
                    for name in feature_names
                ]
            )

        edge_index: list[tuple[int, int]] = []
        edge_types: list[int] = []
        for src, dst, label in graph.edges:
            edge_index.append((src, dst))
            edge_types.append(self._edge_type_id(label))

        return GraphEncoding(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            feature_names=feature_names,
        )

    def _node_type_id(self, node_type: str) -> int:
        if node_type not in self.node_type_to_id:
            self.node_type_to_id[node_type] = len(self.node_type_to_id)
        return self.node_type_to_id[node_type]

    def _edge_type_id(self, edge_type: str) -> int:
        if edge_type not in self.edge_type_to_id:
            self.edge_type_to_id[edge_type] = len(self.edge_type_to_id)
        return self.edge_type_to_id[edge_type]

    def _collect_feature_names(self, graph: ProgramGraph) -> list[str]:
        feature_names = set(self._feature_names)
        edge_types = self._edge_types_in_graph(graph)
        for node in graph.nodes:
            feature_names.update(node.attrs.keys())
        feature_names.update(self._derived_feature_names(edge_types))
        sorted_names = sorted(feature_names)
        self._feature_names = sorted_names
        return sorted_names

    def prime_feature_names(self, graphs: Sequence[ProgramGraph]) -> list[str]:
        """Pre-compute the feature union across graphs to keep encodings aligned."""

        feature_names = set(self._feature_names)
        edge_types = set(self.edge_type_to_id.keys())
        for graph in graphs:
            for node in graph.nodes:
                feature_names.update(node.attrs.keys())
            edge_types.update(label for _, _, label in graph.edges)
        feature_names.update(self._derived_feature_names(edge_types))
        self._feature_names = sorted(feature_names)
        return self._feature_names

    def _edge_types_in_graph(self, graph: ProgramGraph) -> set[str]:
        edge_types = set(self.edge_type_to_id.keys())
        edge_types.update(label for _, _, label in graph.edges)
        return edge_types

    def _derived_feature_names(self, edge_types: Iterable[str]) -> set[str]:
        names = set(self._STRUCTURAL_FEATURES)
        names.update(f"log1p:{field}" for field in self._LOG1P_FIELDS)
        names.update(self._COMBO_FEATURES)
        for etype in edge_types:
            names.add(f"deg_in:{etype}")
            names.add(f"deg_out:{etype}")
        return names

    def _feature_value(
        self,
        name: str,
        attrs: dict[str, float | int],
        node_type: str,
        deg_in: list[int],
        deg_out: list[int],
        deg_total: list[int],
        deg_in_by_type: dict[str, list[int]],
        deg_out_by_type: dict[str, list[int]],
        node_idx: int,
    ) -> float:
        if name.startswith("log1p:"):
            raw_name = name.split("log1p:", maxsplit=1)[1]
            raw_value = float(attrs.get(raw_name, 0.0))
            return math.log1p(max(raw_value, 0.0))
        if name in self._COMBO_FEATURES:
            return self._combo_feature_value(name, attrs, node_type)
        if name == "deg_in":
            return float(deg_in[node_idx])
        if name == "deg_out":
            return float(deg_out[node_idx])
        if name == "deg_total":
            return float(deg_total[node_idx])
        if name.startswith("deg_in:"):
            etype = name.split("deg_in:", maxsplit=1)[1]
            return float(deg_in_by_type.get(etype, [0.0] * len(deg_in))[node_idx])
        if name.startswith("deg_out:"):
            etype = name.split("deg_out:", maxsplit=1)[1]
            return float(deg_out_by_type.get(etype, [0.0] * len(deg_out))[node_idx])
        return float(attrs.get(name, 0.0))

    def _combo_feature_value(
        self, name: str, attrs: dict[str, float | int], node_type: str
    ) -> float:
        if name == "loop:depth_x_log_extent" and node_type == "loop":
            depth = float(attrs.get("depth", 0.0))
            extent = float(attrs.get("extent", 0.0))
            return depth * math.log1p(max(extent, 0.0))
        if name == "loop:parallel_log_extent" and node_type == "loop":
            is_parallel = float(attrs.get("is_parallel", 0.0))
            extent = float(attrs.get("extent", 0.0))
            return is_parallel * math.log1p(max(extent, 0.0))
        if name == "buffer:traffic_bytes" and node_type == "buffer":
            reads = float(attrs.get("read_count", 0.0))
            writes = float(attrs.get("write_count", 0.0))
            elem_bytes = float(attrs.get("elem_bytes", 0.0))
            return (reads + writes) * elem_bytes
        if name == "buffer:traffic_ratio" and node_type == "buffer":
            traffic_bytes = self._combo_feature_value("buffer:traffic_bytes", attrs, node_type)
            total_bytes = float(attrs.get("total_bytes", 0.0))
            denom = max(total_bytes, 1.0)
            ratio = traffic_bytes / denom
            return min(max(ratio, 0.0), 1.0)
        return 0.0
