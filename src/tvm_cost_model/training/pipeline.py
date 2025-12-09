"""Training pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm

from tvm.script import from_source  # type: ignore[import]
from tvm.tir.schedule import Schedule  # type: ignore[import]
from tvm.meta_schedule.feature_extractor import FeatureExtractor  # type: ignore[import]
from tvm.meta_schedule.search_strategy import MeasureCandidate  # type: ignore[import]
from tvm.meta_schedule.tune_context import TuneContext  # type: ignore[import]

from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.features.graph_builder import GraphBuilder, ProgramGraph
from tvm_cost_model.features.per_store_graph import (
    build_per_store_graph,
    encode_per_store_graph,
    expand_per_buffer_features_to_per_store,
)
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder
from tvm_cost_model.integration.utils import measurement_to_score
from tvm_cost_model.models.graph_cost_model import GraphCostModel, Prediction
from tvm_cost_model.training.pair_sampling import Difficulty, RankedPair, sample_ranking_pairs
from tvm_cost_model.training.ranking_dataset import EncodedPair


def _default_builder() -> GraphBuilder:
    return TVMGraphBuilder()


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_pairs: int = 2048
    easy_frac: float = 0.3  # 30% slower = easy
    hard_frac: float = 0.1  # <=10% slower = hard (close calls) boundary
    margin: float = 1.0
    weight_decay: float = 1e-4
    pair_seed: int = 0
    show_progress: bool = True
    curriculum: bool = True
    curriculum_early_frac: float = 0.2  # ~first 20% epochs on easy pairs
    curriculum_mid_frac: float = 0.4    # ~middle 40% on medium pairs (gap between easy/hard)
    feature_mode: str = "tvm-graph"  # or "per-store"
    metrics_path: str | None = None


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
        self.feature_mode = self.config.feature_mode
        if self.model.device.type != "cpu":
            print(f"Using device: {self.model.device}")

    def fit(self, tir_modules: Iterable[Any], scores: Iterable[float]) -> None:
        graphs = [self._build_graph(tir) for tir in tir_modules]
        self.model.update(graphs, list(scores))

    def predict(self, tir_module: Any) -> Prediction:
        graph = self._build_graph(tir_module)
        return self.model.predict(graph)

    def save_model(self, path: str | Path) -> None:
        """Persist the underlying cost model to disk."""

        self.model.save(path)

    def fit_measurements(self, measurements: list[MeasurementRecord]) -> int:
        """Train on MeasurementRecords using pairwise ranking."""

        if not measurements:
            return 0

        import random

        random.shuffle(measurements)

        tensor_encodings: dict[int, Any] = {}
        if self.feature_mode == "per-store":
            print(f"Building per-store encodings for {len(measurements)=} scheduled TIR modules...")
            extractor = FeatureExtractor.create("per-store-feature")
            with tqdm(total=len(measurements), desc="Building per-store encodings", unit="graph") as pbar:
                for m in measurements:
                    tir_str = m.scheduled_tir or m.original_tir
                    try:
                        mod = from_source(tir_str)
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"Skipping measurement (failed to parse TIR): {exc}")
                        pbar.update(1)
                        continue

                    try:
                        sch = Schedule(mod)
                        cand = MeasureCandidate(sch, [])
                        target = getattr(m, "target", None) or "llvm"
                        ctx = TuneContext(mod=mod, target=target, num_threads=1)
                        tvm_arr = extractor.extract_from(ctx, [cand])[0]
                        per_buffer_feat = tvm_arr.numpy().astype("float32", copy=False)
                        graph = build_per_store_graph(mod)
                        per_store_feat = expand_per_buffer_features_to_per_store(graph.stores, per_buffer_feat)
                        enc = encode_per_store_graph(per_store_feat, graph)
                        tensor_encodings[id(m)] = self.model.encoder.to_tensor_encoding(
                            enc, device=self.model.device
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"Skipping measurement (failed to build per-store graph): {exc}")
                    finally:
                        pbar.update(1)
        else:
            print(f"Building graphs for {len(measurements)=} scheduled TIR modules...")
            graphs: list[ProgramGraph] = []
            with tqdm(total=len(measurements), desc="Building graphs", unit="graph") as pbar:
                for m in measurements:
                    graph = self._build_graph(m.scheduled_tir or m.original_tir)
                    graphs.append(graph)
                    pbar.update(1)
            self.model.encoder.prime_feature_names(graphs)
            print("Encoding graphs...")
            encodings = {id(m): self.model.encoder.encode(g) for m, g in zip(measurements, graphs)}
            tensor_encodings = {
                key: self.model.encoder.to_tensor_encoding(enc, device=self.model.device)
                for key, enc in encodings.items()
            }

        max_pairs = self.config.max_pairs
        if max_pairs <= 0:
            max_pairs = len(measurements)
        print(f"Sampling up to {max_pairs} ranking pairs...")

        def _encode_pairs(pairs: Sequence[RankedPair]) -> list[EncodedPair]:
            encoded_pairs: list[EncodedPair] = []
            for pair in pairs:
                better_enc = tensor_encodings.get(id(pair.better))
                worse_enc = tensor_encodings.get(id(pair.worse))
                if better_enc is None or worse_enc is None:
                    continue
                encoded_pairs.append(
                    EncodedPair(better=better_enc, worse=worse_enc, difficulty=pair.difficulty.name)
                )
            return encoded_pairs

        total_epochs = self.config.epochs
        if not self.config.curriculum:
            pairs = sample_ranking_pairs(
                measurements,
                num_pairs=max_pairs,
                easy_frac=self.config.easy_frac,
                hard_frac=self.config.hard_frac,
                seed=self.config.pair_seed,
            )
            encoded_pairs = _encode_pairs(pairs)
            if not encoded_pairs:
                return 0
            print(f"Training on {len(encoded_pairs)} encoded ranking pairs (no curriculum)...")
            self.model.train_on_pairs(
                encoded_pairs,
                epochs=total_epochs,
                batch_size=self.config.batch_size,
                show_progress=self.config.show_progress,
                stage_name="all",
                metrics_log_path=self.config.metrics_path,
            )
            return len(encoded_pairs)

        # Curriculum: early easy, mid medium, late hard.
        early_epochs = max(1, int(total_epochs * self.config.curriculum_early_frac))
        mid_epochs = max(1, int(total_epochs * self.config.curriculum_mid_frac))
        late_epochs = max(total_epochs - early_epochs - mid_epochs, 1)
        if early_epochs + mid_epochs + late_epochs > total_epochs:
            # Adjust to ensure total epochs budget is respected.
            late_epochs = max(total_epochs - early_epochs - mid_epochs, 0)
            if late_epochs == 0:
                mid_epochs = max(total_epochs - early_epochs, 0)
        stages = [
            ("easy", {Difficulty.EASY}, early_epochs, 0),
            ("medium", {Difficulty.MEDIUM}, mid_epochs, 1),
            ("hard", {Difficulty.HARD}, late_epochs, 2),
        ]

        total_trained_pairs = 0
        for stage_name, allowed, epochs, seed_offset in stages:
            if epochs <= 0:
                continue
            pairs = sample_ranking_pairs(
                measurements,
                num_pairs=max_pairs,
                easy_frac=self.config.easy_frac,
                hard_frac=self.config.hard_frac,
                seed=self.config.pair_seed + seed_offset,
                allowed_difficulties=allowed,
            )
            encoded_pairs = _encode_pairs(pairs)
            if not encoded_pairs:
                print(f"Skipping stage '{stage_name}' (no pairs for difficulties {allowed}).")
                continue
            print(
                f"Stage '{stage_name}': training on {len(encoded_pairs)} pairs for {epochs} epochs "
                f"(difficulties={[d.name for d in allowed]}), "
                f"{self.config.easy_frac=}, {self.config.hard_frac=}, "
                f"min rel gap: {min((pair.worse.runtime_ms - pair.better.runtime_ms) / max(pair.better.runtime_ms, 1e-9) for pair in pairs):.3f}, "
                f"max rel gap: {max((pair.worse.runtime_ms - pair.better.runtime_ms) / max(pair.better.runtime_ms, 1e-9) for pair in pairs):.3f}."
            )
            self.model.train_on_pairs(
                encoded_pairs,
                epochs=epochs,
                batch_size=self.config.batch_size,
                show_progress=self.config.show_progress,
                stage_name=stage_name,
                metrics_log_path=self.config.metrics_path,
            )
            total_trained_pairs += len(encoded_pairs)

        return total_trained_pairs

    def _build_graph(self, tir_module: Any) -> ProgramGraph:
        return self.builder.build(tir_module)

    def fit_pointwise_measurements(self, measurements: Sequence[MeasurementRecord]) -> int:
        """Pointwise regression training from MeasurementRecords."""

        if not measurements:
            return 0
        graphs = [self._build_graph(m.scheduled_tir or m.original_tir) for m in measurements]
        scores = [measurement_to_score(m.runtime_ms) for m in measurements]
        self.model.encoder.prime_feature_names(graphs)
        self.model.update(graphs, scores)
        return len(graphs)
