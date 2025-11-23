"""Placeholder for the R-GAT based ranking predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tvm_cost_model.features.graph_builder import ProgramGraph
from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.models.node_mlp_ranker import NodeMLPRanker, RankerOutput


@dataclass
class Prediction:
    score: float
    attribution: dict[str, float]


class GraphCostModel:
    """Skeleton cost model implementing predict/update APIs."""

    def __init__(self) -> None:
        self._is_trained = False
        self.encoder = GraphEncoder()
        # Feature dim will be inferred on first call; default to 0.
        self._model: NodeMLPRanker | None = None

    def predict(self, graph: ProgramGraph) -> Prediction:
        """Encode a graph and run it through the ranker."""

        encoding = self.encoder.encode(graph)
        if self._model is None:
            feature_dim = len(encoding.feature_names)
            self._model = NodeMLPRanker(feature_dim=feature_dim)
        output: RankerOutput = self._model(encoding)
        # Convert node objects to names in attribution mapping
        attribution = {node.name: float(weight.detach().item()) for node, weight in zip(graph.nodes, output.attribution)}
        return Prediction(score=float(output.score.detach().item()), attribution=attribution)

    def update(self, graphs: Sequence[ProgramGraph], scores: Sequence[float]) -> None:
        """Fit a trivial per-call model; placeholder for real training."""

        if not graphs or not scores:
            return
        # Reinitialize model based on the incoming feature space to avoid mismatch.
        encoding = self.encoder.encode(graphs[0])
        if self._model is None:
            self._model = NodeMLPRanker(feature_dim=len(encoding.feature_names))
        self._is_trained = True
