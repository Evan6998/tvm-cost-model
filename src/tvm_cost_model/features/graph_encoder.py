"""Utility to convert ProgramGraph into numeric encodings for ML models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from tvm_cost_model.features.graph_builder import ProgramGraph


@dataclass
class GraphEncoding:
    node_features: list[list[float]]
    node_types: list[int]
    edge_index: list[tuple[int, int]]
    edge_types: list[int]
    feature_names: list[str] = field(default_factory=list[str])


@dataclass
class TensorGraphEncoding:
    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    feature_names: list[str]


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

    def __init__(self) -> None:
        self.node_type_to_id: dict[str, int] = {}
        self.edge_type_to_id: dict[str, int] = {}
        self.feature_names: list[str] = []

    def encode(self, graph: ProgramGraph) -> GraphEncoding:
        if not graph.nodes:
            return GraphEncoding([], [], [], [], [])

        feature_names = self._collect_feature_names(graph)
        node_features: list[list[float]] = []
        node_types: list[int] = []

        for node in graph.nodes:
            node_type = node.name.split(":", maxsplit=1)[0]
            node_types.append(self._node_type_id(node_type))
            node_features.append(
                [
                    math.log1p(node.attrs.get(fname, 0.0)) if fname in self._LOG1P_FIELDS 
                    else float(node.attrs.get(fname, 0.0)) 
                    for fname in feature_names
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
        feature_names = set(self.feature_names)
        for node in graph.nodes:
            feature_names.update(node.attrs.keys())
        sorted_names = sorted(feature_names)
        self.feature_names = sorted_names
        return sorted_names

    def prime_feature_names(self, graphs: Sequence[ProgramGraph]) -> list[str]:
        """Pre-compute the feature union across graphs to keep encodings aligned."""

        feature_names = set(self.feature_names)
        for graph in graphs:
            for node in graph.nodes:
                feature_names.update(node.attrs.keys())
        self.feature_names = sorted(feature_names)
        return self.feature_names

    def to_tensor_encoding(self, encoding: GraphEncoding, device: torch.device) -> TensorGraphEncoding:
        """Convert a GraphEncoding into tensor form on the given device."""

        node_features = torch.tensor(encoding.node_features, dtype=torch.float32, device=device)
        node_types = torch.tensor(encoding.node_types, dtype=torch.long, device=device)
        edge_index = (
            torch.tensor(encoding.edge_index, dtype=torch.long, device=device)
            if encoding.edge_index
            else torch.empty((0, 2), dtype=torch.long, device=device)
        )
        edge_types = (
            torch.tensor(encoding.edge_types, dtype=torch.long, device=device)
            if encoding.edge_types
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        return TensorGraphEncoding(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            feature_names=encoding.feature_names,
        )
