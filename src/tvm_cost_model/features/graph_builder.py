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

    def __repr__(self) -> str:
        return f"GraphNode(name={self.name}, attrs={self.attrs})"
    
    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class ProgramGraph:
    """Container for graph nodes/edges."""

    nodes: List[GraphNode]
    edges: List[tuple[int, int, str]]

    def summary(self) -> str:
        """Return a one-line summary of the graph (for logging/debug)."""
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        edge_types: Dict[str, int] = {}
        for _, _, etype in self.edges:
            edge_types[etype] = edge_types.get(etype, 0) + 1
        edge_type_str = ", ".join(f"{k}={v}" for k, v in sorted(edge_types.items())) or "-"
        return f"ProgramGraph(nodes={num_nodes}, edges={num_edges}, edge_types=[{edge_type_str}])"

    def format(self, max_nodes: int | None = None, max_edges: int | None = None) -> str:
        """Return a detailed, nicely formatted multi-line string for terminal display.

        Args:
            max_nodes: Optional limit on the number of nodes to display. If None, show all.
            max_edges: Optional limit on the number of edges to display. If None, show all.
        """
        lines: List[str] = []

        # Header
        lines.append("=" * 72)
        lines.append("ProgramGraph")
        lines.append("=" * 72)

        # Summary line
        lines.append(self.summary())
        lines.append("")

        # Nodes section
        total_nodes = len(self.nodes)
        lines.append(f"Nodes (total={total_nodes}):")
        lines.append("-" * 72)

        node_limit = total_nodes if max_nodes is None else min(total_nodes, max_nodes)
        for idx in range(node_limit):
            node = self.nodes[idx]
            lines.append(f"[{idx}] {node.name}")
            if node.attrs:
                for k, v in sorted(node.attrs.items()):
                    lines.append(f"    {k:16}: {v}")
            else:
                lines.append("    (no attributes)")
            lines.append("")

        if node_limit < total_nodes:
            lines.append(f"... ({total_nodes - node_limit} more nodes not shown)")
            lines.append("")

        # Edges section
        total_edges = len(self.edges)
        lines.append(f"Edges (total={total_edges}):")
        lines.append("-" * 72)

        edge_limit = total_edges if max_edges is None else min(total_edges, max_edges)
        for idx in range(edge_limit):
            src, dst, etype = self.edges[idx]
            src_name = self.nodes[src].name if 0 <= src < total_nodes else "<out-of-range>"
            dst_name = self.nodes[dst].name if 0 <= dst < total_nodes else "<out-of-range>"
            lines.append(f"[{idx}] {src:3} -> {dst:3}  ({etype})  # {src_name} -> {dst_name}")

        if edge_limit < total_edges:
            lines.append(f"... ({total_edges - edge_limit} more edges not shown)")

        return "\n".join(lines)

    def pretty_print(self, max_nodes: int | None = None, max_edges: int | None = None) -> None:
        """Print the formatted graph directly to stdout.

        This is a convenience wrapper around :meth:`format`.
        """
        print(self.format(max_nodes=max_nodes, max_edges=max_edges))


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
