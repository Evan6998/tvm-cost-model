"""CLI entry point for running the training pipeline."""

from __future__ import annotations

from tvm_cost_model.training.pipeline import TrainingPipeline


def main() -> None:
    pipeline = TrainingPipeline()
    pipeline.fit(["tir_module"], [0.0])
    prediction = pipeline.predict("tir_module")
    print(f"Dummy runtime: {prediction.runtime_ms}ms")


if __name__ == "__main__":
    main()
