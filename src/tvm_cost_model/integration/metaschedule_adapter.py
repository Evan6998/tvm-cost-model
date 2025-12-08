"""Adapters to plug the learned model into TVM MetaSchedule."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from tvm.meta_schedule import MeasureCandidate  # type: ignore[import]
from tvm.meta_schedule.cost_model import PyCostModel  # type: ignore[import]
from tvm.meta_schedule.runner import RunnerResult  # type: ignore[import]
from tvm.meta_schedule import utils as ms_utils  # type: ignore[import]

from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.integration.utils import candidate_to_tir, pack_measurements
from tvm_cost_model.training.pipeline import TrainingConfig, TrainingPipeline


@ms_utils.derived_object
class GraphPyCostModel(PyCostModel):
    """PyCostModel-compatible wrapper around the GraphCostModel pipeline."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        buffer_size: int = 256,
        pointwise_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.pipeline = TrainingPipeline(config)
        self._pending_measurements: list[MeasurementRecord] = []
        self._buffer_size = max(1, buffer_size)
        self._pointwise_fallback = pointwise_fallback

    def predict(self, context: Any, candidates: list[MeasureCandidate]) -> np.ndarray:  # type: ignore[override]
        """Return scores (higher is better) for a batch of measure candidates."""

        if not candidates:
            return np.zeros((0,), dtype="float64")

        graphs = [self.pipeline._build_graph(candidate_to_tir(candidate)) for candidate in candidates]  # type: ignore[call-arg]
        self.pipeline.model.encoder.prime_feature_names(graphs)
        scores = [self.pipeline.model.predict(graph).score for graph in graphs]
        return np.asarray(scores, dtype="float64")

    def update(  # type: ignore[override]
        self,
        context: Any,
        candidates: list[MeasureCandidate],
        results: list[RunnerResult],
    ) -> None:
        """Accumulate measurements and trigger pairwise training."""

        if not candidates or not results:
            return

        new_measurements = pack_measurements(candidates, results, context)
        if not new_measurements:
            return
        self._pending_measurements.extend(new_measurements)
        if len(self._pending_measurements) >= self._buffer_size:
            self._train_on_pending()

    def save(self, path: str) -> None:  # type: ignore[override]
        if self._pending_measurements:
            self._train_on_pending()
        self.pipeline.save_model(path)

    def load(self, path: str) -> None:  # type: ignore[override]
        self.pipeline.model.load(path)

    def flush_pending(self) -> None:
        """Force training on any buffered measurements."""

        self._train_on_pending()

    def _train_on_pending(self) -> None:
        if not self._pending_measurements:
            return
        trained_pairs = self.pipeline.fit_measurements(self._pending_measurements)
        if trained_pairs == 0 and self._pointwise_fallback:
            self.pipeline.fit_pointwise_measurements(self._pending_measurements)
        self._pending_measurements.clear()


def MetaScheduleAdapter(
    config: TrainingConfig | None = None,
    buffer_size: int = 256,
    pointwise_fallback: bool = True,
):
    """Deprecated alias; use GraphPyCostModel directly."""

    warnings.warn(
        "MetaScheduleAdapter is deprecated; use GraphPyCostModel which matches PyCostModel APIs.",
        DeprecationWarning,
        stacklevel=2,
    )
    return GraphPyCostModel(config=config, buffer_size=buffer_size, pointwise_fallback=pointwise_fallback)
