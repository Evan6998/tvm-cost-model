"""Training pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tvm_cost_model.features.graph_builder import GraphBuilder
from tvm_cost_model.models.graph_cost_model import GraphCostModel, Prediction


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3


class TrainingPipeline:
    """Coordinates data loading, graph building, and model updates."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.builder = GraphBuilder()
        self.model = GraphCostModel()

    def fit(self, tir_modules: Iterable[str], runtimes_ms: Iterable[float]) -> None:
        graphs = [self.builder.build(tir) for tir in tir_modules]
        self.model.update(graphs, list(runtimes_ms))

    def predict(self, tir_module: str) -> Prediction:
        graph = self.builder.build(tir_module)
        return self.model.predict(graph)
