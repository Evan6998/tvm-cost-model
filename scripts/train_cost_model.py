"""CLI entry point for running the ranking training pipeline."""

from __future__ import annotations

import argparse
import pickle
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
    parser.add_argument("--easy-frac", type=float, default=0.3, help="Easy pair relative gap (fraction, e.g., 0.3 = 30%)")
    parser.add_argument("--hard-frac", type=float, default=0.1, help="Hard pair relative gap (fraction, e.g., 0.1 = 10%)")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for ranking loss")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--pair-seed", type=int, default=0, help="Seed for pair sampling")
    parser.add_argument("--output", type=str, default="", help="Path to save the trained model (torch format)")
    parser.add_argument("--graph-cache", type=str, default=None, help="Path to pre-computed graph cache (speeds up training)")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning (use shuffled training)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        easy_frac=args.easy_frac,
        hard_frac=args.hard_frac,
        margin=args.margin,
        weight_decay=args.weight_decay,
        pair_seed=args.pair_seed,
        curriculum=not args.no_curriculum,  # Disable curriculum if flag set
        show_progress=True,
    )
    pipeline = TrainingPipeline(config=config)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        print(f"Loading measurements from {dataset_path}...")
        measurements = load_measurement_records(dataset_path)
        print(f"Loaded {len(measurements)} measurement records.")
        
        # Load cached graphs if available
        cached_graphs = None
        if args.graph_cache:
            cache_path = Path(args.graph_cache)
            if cache_path.exists():
                print(f"Loading pre-computed graphs from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    cached_graphs = cache_data['graphs']
                print(f"✓ Loaded {len(cached_graphs)} cached graphs")
            else:
                print(f"Warning: Graph cache {cache_path} not found, will build graphs from scratch")

        print("Training cost model...")
        pair_count = pipeline.fit_measurements(measurements, cached_graphs=cached_graphs)
        print(
            f"Trained on {len(measurements)} measurements "
            f"using {pair_count} ranking pairs."
        )
    else:
        pipeline.fit(["tir_module"], [0.0])
        prediction = pipeline.predict("tir_module")
        print(f"Dummy score (no dataset provided): {prediction.score}")

    if args.output:
        output_path = Path(args.output)
        pipeline.save_model(output_path)
        print(f"Saved trained model to {output_path.resolve()}")


if __name__ == "__main__":
    main()
