"""Placeholder for the R-GAT based ranking predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tvm_cost_model.features.graph_builder import ProgramGraph


@dataclass
class Prediction:
    score: float
    attribution: dict[str, float]


class GraphCostModel:
    """Skeleton cost model implementing predict/update APIs."""

    def __init__(self) -> None:
        self._is_trained = False

    def predict(self, graph: ProgramGraph) -> Prediction:
        """Return dummy score (higher is better) and uniform attribution."""

        if not graph.nodes:
            return Prediction(score=0.0, attribution={})
        attribution = {node.name: 1.0 / len(graph.nodes) for node in graph.nodes}
        return Prediction(score=0.0, attribution=attribution)

    def update(self, graphs: Sequence[ProgramGraph], scores: Sequence[float]) -> None:
        """Mock training routine to flip the trained flag."""

        if graphs and scores:
            self._is_trained = True
