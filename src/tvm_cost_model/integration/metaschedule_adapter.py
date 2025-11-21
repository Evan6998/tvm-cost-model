"""Adapters to plug the learned model into TVM MetaSchedule."""

from __future__ import annotations

from typing import Any

from tvm_cost_model.training.pipeline import TrainingPipeline


class MetaScheduleAdapter:
    """Minimal interface mirroring PyCostModel expectations."""

    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()

    def predict(self, context: Any) -> float:
        """Return a dummy score (higher is better) for a MetaSchedule trace."""

        tir_module = getattr(context, "tir", "")
        prediction = self.pipeline.predict(tir_module)
        return prediction.score

    def update(self, context: Any, measured_cost: float) -> None:
        """Placeholder update hook converting runtime to a ranking score."""

        score = -measured_cost  # lower runtime => higher score
        self.pipeline.fit([getattr(context, "tir", "")], [score])
