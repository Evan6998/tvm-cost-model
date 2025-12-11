#!/usr/bin/env python3
"""Quick debug script to test improvements on HARD pairs only."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tvm_cost_model.data.dataset_builder import load_measurement_records
from tvm_cost_model.training.pipeline import TrainingConfig, TrainingPipeline


def main():
    # Load dataset
    dataset_path = Path("artifacts/sweeps/sweep_merged.parquet")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run data collection first.")
        sys.exit(1)
    
    print(f"Loading dataset from {dataset_path}...")
    measurements = load_measurement_records(dataset_path)
    print(f"Loaded {len(measurements)} measurements")
    
    # Configuration: FOCUS ON HARD PAIRS ONLY
    config = TrainingConfig(
        epochs=50,                    # Fewer epochs for debug
        learning_rate=5e-4,           # Lower LR for stability
        batch_size=128,               # Smaller batch for debug
        max_pairs=5000,               # Fewer pairs for quick test
        easy_frac=0.05,               # Almost no easy pairs (only 5%)
        hard_frac=0.20,               # LOTS of hard pairs (20% = hard boundary)
        margin=0.05,                  # Small base margin
        weight_decay=1e-4,
        curriculum=False,             # NO curriculum - train on hard immediately!
        show_progress=True,
    )
    
    print("\n" + "="*70)
    print("HARD PAIRS FOCUSED TRAINING")
    print("="*70)
    print(f"Config:")
    print(f"  - Easy fraction: {config.easy_frac:.1%} (minimal)")
    print(f"  - Hard fraction: {config.hard_frac:.1%} (high threshold)")
    print(f"  - Most pairs will be MEDIUM-to-HARD difficulty")
    print(f"  - NO curriculum learning - direct hard training")
    print(f"  - ListNet loss: ENABLED")
    print(f"  - Adaptive margins: ENABLED")
    print(f"  - Hard pair reweighting: ENABLED")
    print("="*70)
    print()
    
    # Create pipeline with improvements
    pipeline = TrainingPipeline(config)
    
    # Train on hard pairs
    print("Training on hard pairs...")
    total_trained = pipeline.fit_measurements(measurements)
    
    print(f"\nTrained on {total_trained} pairs")
    print("="*70)
    print("Hard pairs training complete!")
    print("Check the validation accuracy on HARD pairs above.")
    print("If val_acc > 65% on hard stage, the improvements are working!")
    print("="*70)


if __name__ == "__main__":
    main()

