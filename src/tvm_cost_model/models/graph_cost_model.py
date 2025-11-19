"""Placeholder for the R-GAT based runtime predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tvm_cost_model.features.graph_builder import ProgramGraph


@dataclass
class Prediction:
    runtime_ms: float
    attribution: dict[str, float]


class GraphCostModel:
    """Skeleton cost model implementing predict/update APIs."""

    def __init__(self) -> None:
        self._is_trained = False

    def predict(self, graph: ProgramGraph) -> Prediction:
        """Return dummy runtime and uniform attribution."""

        attribution = {node.name: 1.0 / len(graph.nodes) for node in graph.nodes}
        return Prediction(runtime_ms=0.0, attribution=attribution)

    def update(self, graphs: Sequence[ProgramGraph], runtimes_ms: Sequence[float]) -> None:
        """Mock training routine to flip the trained flag."""

        if graphs and runtimes_ms:
            self._is_trained = True
