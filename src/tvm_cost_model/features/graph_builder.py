"""Graph extraction and canonicalization stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GraphNode:
    """Generic node in the heterogeneous TIR graph."""

    name: str
    attrs: Dict[str, float]


@dataclass
class ProgramGraph:
    """Container for graph nodes/edges."""

    nodes: List[GraphNode]
    edges: List[tuple[int, int, str]]


class GraphBuilder:
    """Turns TIR text into a canonical ProgramGraph."""

    def build(self, tir_module: str) -> ProgramGraph:
        # TODO: Replace with actual TIR parsing and canonicalization.
        loop_node = GraphNode(name="loop", attrs={"extent": 1.0})
        memory_node = GraphNode(name="shared_mem", attrs={"bytes": 0.0})
        return ProgramGraph(nodes=[loop_node, memory_node], edges=[(0, 1, "uses")])
