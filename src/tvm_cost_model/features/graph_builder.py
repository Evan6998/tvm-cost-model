"""Graph extraction and canonicalization stubs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GraphNode:
    """Generic node in the heterogeneous TIR graph."""

    name: str
    attrs: Dict[str, float | int]


@dataclass
class ProgramGraph:
    """Container for graph nodes/edges."""

    nodes: List[GraphNode]
    edges: List[tuple[int, int, str]]


@dataclass
class LoopInfo:
    """Loop metadata extracted from a simplified TIR string."""

    var: str
    extent: int


class GraphBuilder:
    """Turns TIR-like text into a canonical ProgramGraph.

    This is a lightweight, TVM-free parser meant to exercise the pipeline until
    we swap in TVM's TIR parser. It enforces invariance by sorting loops by
    (extent, var) and deduplicating buffer references.
    """

    _loop_pattern = re.compile(r"for\s+(\w+)\s+in\s+range\((\d+)\)")
    _buffer_pattern = re.compile(r"(\w+)\[")

    def build(self, tir_module: str) -> ProgramGraph:
        loops = self._parse_loops(tir_module)
        buffers = sorted(self._parse_buffers(tir_module))

        nodes: List[GraphNode] = []
        edges: List[tuple[int, int, str]] = []

        # Canonicalize loops by (extent, var) so semantically similar nests match.
        loop_nodes: list[GraphNode] = []
        for loop in sorted(loops, key=lambda l: (l.extent, l.var)):
            loop_nodes.append(GraphNode(name=f"loop:{loop.var}", attrs={"extent": loop.extent}))

        buffer_nodes = [GraphNode(name=f"buffer:{name}", attrs={"name_len": len(name)}) for name in buffers]

        nodes.extend(loop_nodes)
        nodes.extend(buffer_nodes)

        # Compute node aggregates loop extents and number of buffers for quick features.
        compute_node = GraphNode(
            name="compute",
            attrs={
                "loop_depth": len(loop_nodes),
                "buffer_count": len(buffer_nodes),
                "total_extent": sum(loop.attrs["extent"] for loop in loop_nodes) if loop_nodes else 0,
            },
        )
        compute_idx = len(nodes)
        nodes.append(compute_node)

        # Edges from compute -> loops and compute -> buffers capture dependencies.
        for idx in range(len(loop_nodes)):
            edges.append((compute_idx, idx, "iterates"))
        for offset, _ in enumerate(buffer_nodes):
            edges.append((compute_idx, len(loop_nodes) + offset, "accesses"))

        return ProgramGraph(nodes=nodes, edges=edges)

    def _parse_loops(self, text: str) -> List[LoopInfo]:
        return [LoopInfo(var=var, extent=int(extent)) for var, extent in self._loop_pattern.findall(text)]

    def _parse_buffers(self, text: str) -> List[str]:
        return list({match for match in self._buffer_pattern.findall(text)})
