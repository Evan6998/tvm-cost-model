"""CLI entry point for running the ranking training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from tvm_cost_model.data.dataset_builder import load_measurement_records
from tvm_cost_model.training.pipeline import TrainingConfig, TrainingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the graph-based cost model.")
    parser.add_argument("--dataset", type=str, default="", help="Path to a MeasurementRecord Parquet file")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Pairs per optimization step")
    parser.add_argument("--max-pairs", type=int, default=2048, help="Number of ranking pairs to sample")
    parser.add_argument("--easy-gap", type=float, default=0.5, help="Easy pair runtime delta (ms)")
    parser.add_argument("--hard-gap", type=float, default=0.1, help="Hard pair runtime delta (ms)")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for ranking loss")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--pair-seed", type=int, default=0, help="Seed for pair sampling")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        easy_gap=args.easy_gap,
        hard_gap=args.hard_gap,
        margin=args.margin,
        weight_decay=args.weight_decay,
        pair_seed=args.pair_seed,
    )
    pipeline = TrainingPipeline(config=config)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        print(f"Loading measurements from {dataset_path}...")
        measurements = load_measurement_records(dataset_path)
        print(f"Loaded {len(measurements)} measurement records.")

        print("Training cost model...")
        pair_count = pipeline.fit_measurements(measurements)
        print(
            f"Trained on {len(measurements)} measurements "
            f"using {pair_count} ranking pairs."
        )
    else:
        pipeline.fit(["tir_module"], [0.0])
        prediction = pipeline.predict("tir_module")
        print(f"Dummy score (no dataset provided): {prediction.score}")


if __name__ == "__main__":
    main()
