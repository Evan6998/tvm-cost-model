"""Utility to convert ProgramGraph into numeric encodings for ML models."""

from __future__ import annotations

from dataclasses import dataclass, field

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
        for node in graph.nodes:
            node_type = node.name.split(":", maxsplit=1)[0]
            node_types.append(self._node_type_id(node_type))
            node_features.append([float(node.attrs.get(name, 0.0)) for name in feature_names])

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
        for node in graph.nodes:
            feature_names.update(node.attrs.keys())
        sorted_names = sorted(feature_names)
        self._feature_names = sorted_names
        return sorted_names
