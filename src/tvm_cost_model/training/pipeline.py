"""Training pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.models.graph_cost_model import GraphCostModel, Prediction
from tvm_cost_model.training.pair_sampling import sample_ranking_pairs
from tvm_cost_model.training.ranking_dataset import EncodedPair
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder
from tvm_cost_model.features.graph_builder import GraphBuilder, ProgramGraph


def _default_builder() -> GraphBuilder:
    return TVMGraphBuilder()


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_pairs: int = 2048
    easy_gap: float = 10.0
    hard_gap: float = 2.0
    margin: float = 1.0
    weight_decay: float = 1e-4
    pair_seed: int = 0


class TrainingPipeline:
    """Coordinates data loading, graph building, and model updates."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.builder = _default_builder()
        self.model = GraphCostModel(
            learning_rate=self.config.learning_rate,
            margin=self.config.margin,
            weight_decay=self.config.weight_decay,
        )

    def fit(self, tir_modules: Iterable[str], scores: Iterable[float]) -> None:
        graphs = [self._build_graph(tir) for tir in tir_modules]
        self.model.update(graphs, list(scores))

    def predict(self, tir_module: str) -> Prediction:
        graph = self._build_graph(tir_module)
        return self.model.predict(graph)

    def fit_measurements(self, measurements: Sequence[MeasurementRecord]) -> int:
        """Train on MeasurementRecords using pairwise ranking."""

        if not measurements:
            return 0
        
        print(f"Building graphs for {len(measurements)-1=} scheduled TIR modules...")
        graphs = [self._build_graph(m.scheduled_tir or m.original_tir) for m in measurements[1:]]
        # graphs[5].pretty_print()
        self.model.encoder.prime_feature_names(graphs)
        print("Encoding graphs...")
        encodings = {id(m): self.model.encoder.encode(g) for m, g in zip(measurements, graphs)}

        max_pairs = self.config.max_pairs
        if max_pairs <= 0:
            max_pairs = len(measurements)
        print(f"Sampling up to {max_pairs} ranking pairs...")
        pairs = sample_ranking_pairs(
            measurements,
            num_pairs=max_pairs,
            easy_gap=self.config.easy_gap,
            hard_gap=self.config.hard_gap,
            seed=self.config.pair_seed,
        )
        print(f"Sampled {len(pairs)} ranking pairs.")
        encoded_pairs: list[EncodedPair] = []
        for pair in pairs:
            better_enc = encodings.get(id(pair.better))
            worse_enc = encodings.get(id(pair.worse))
            if better_enc is None or worse_enc is None:
                continue
            encoded_pairs.append(
                EncodedPair(better=better_enc, worse=worse_enc, difficulty=pair.difficulty)
            )
        if not encoded_pairs:
            return 0
        print(f"Training on {len(encoded_pairs)} encoded ranking pairs...")
        self.model.train_on_pairs(
            encoded_pairs,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
        )
        return len(encoded_pairs)

    def _build_graph(self, tir_module: str) -> ProgramGraph:
        return self.builder.build(tir_module)
