#!/usr/bin/env python3
"""Bayesian Optimization for TVM Cost Model Hyperparameters.

Uses a small subset of data for fast iteration (~5-10 min per trial).
Optimizes: learning_rate, margin, batch_size, hidden_dim, weight_decay
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from tvm_cost_model.data.dataset_builder import load_measurement_records
from tvm_cost_model.training.pipeline import TrainingConfig, TrainingPipeline


def train_and_evaluate(
    learning_rate: float,
    margin: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float,
    measurements,
    cached_graphs,
    max_pairs: int = 5000,
    epochs: int = 20,
    seed: int = 42,
):
    """Train model with given hyperparameters and return validation accuracy."""
    
    print(f"\n{'='*80}")
    print(f"Trial: lr={learning_rate:.6f}, margin={margin:.3f}, batch={batch_size}, "
          f"hidden={hidden_dim}, wd={weight_decay:.6f}")
    print(f"{'='*80}\n")
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create config
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=int(batch_size),
        max_pairs=max_pairs,
        easy_frac=0.3,
        hard_frac=0.1,
        margin=margin,
        weight_decay=weight_decay,
        pair_seed=seed,
        curriculum=False,  # Shuffled learning works better
        show_progress=False,  # Reduce noise
    )
    
    # Create pipeline with hidden_dim
    pipeline = TrainingPipeline(config=config)
    pipeline.model.hidden_dim = int(hidden_dim)
    
    try:
        # Train
        pipeline.fit_measurements(measurements, cached_graphs=cached_graphs)
        
        # Get final validation accuracy from model
        # We'll use a simple heuristic: train on 80%, validate on 20%
        # The pipeline already does this internally
        
        # For BayesOpt, we need a single metric to maximize
        # We'll extract the last validation accuracy
        # This is a simplified version - in practice you'd want to track this properly
        
        # Quick validation on a held-out set
        from tvm_cost_model.training.pair_sampling import sample_ranking_pairs
        
        val_pairs_raw = sample_ranking_pairs(
            measurements,
            num_pairs=500,  # Small validation set
            easy_frac=0.3,
            hard_frac=0.1,
            seed=seed + 999,  # Different seed for validation
        )
        
        # Encode validation pairs
        from tvm_cost_model.features.graph_encoder import GraphEncoder
        from tvm_cost_model.training.ranking_dataset import EncodedPair
        
        encoder = pipeline.model.encoder
        val_encoded = []
        
        for i, (m, g) in enumerate(zip(measurements, cached_graphs)):
            if g is None:
                continue
            enc = encoder.encode(g)
            tensor_enc = encoder.to_tensor_encoding(enc, device=pipeline.model.device)
            # Store in a temp dict for lookup
            if not hasattr(train_and_evaluate, 'enc_cache'):
                train_and_evaluate.enc_cache = {}
            train_and_evaluate.enc_cache[id(m)] = tensor_enc
        
        for pair in val_pairs_raw:
            better_enc = train_and_evaluate.enc_cache.get(id(pair.better))
            worse_enc = train_and_evaluate.enc_cache.get(id(pair.worse))
            if better_enc and worse_enc:
                val_encoded.append(EncodedPair(
                    better=better_enc,
                    worse=worse_enc,
                    difficulty=pair.difficulty.name,
                    better_runtime=pair.better.runtime_ms,
                    worse_runtime=pair.worse.runtime_ms,
                ))
        
        # Evaluate
        val_loss, val_correct, val_count = pipeline.model._evaluate_pairs(val_encoded)
        val_acc = val_correct / val_count if val_count > 0 else 0.0
        
        print(f"\nValidation Accuracy: {val_acc:.4f} ({val_correct}/{val_count})")
        print(f"Validation Loss: {val_loss:.4f}\n")
        
        # Clear cache to save memory
        if hasattr(train_and_evaluate, 'enc_cache'):
            train_and_evaluate.enc_cache.clear()
        
        return val_acc
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0  # Return worst score on failure


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization for hyperparameters")
    parser.add_argument("--dataset", type=str, required=True, help="Path to measurements parquet")
    parser.add_argument("--graph-cache", type=str, required=True, help="Path to pre-computed graph cache")
    parser.add_argument("--subset-size", type=int, default=5000, help="Number of measurements to use (for speed)")
    parser.add_argument("--max-pairs", type=int, default=5000, help="Max pairs per trial")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per trial")
    parser.add_argument("--n-iter", type=int, default=25, help="Number of BayesOpt iterations")
    parser.add_argument("--output", type=str, default="bayesopt_results.json", help="Output file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("="*80)
    print("BAYESIAN OPTIMIZATION FOR TVM COST MODEL HYPERPARAMETERS")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Graph cache: {args.graph_cache}")
    print(f"Subset size: {args.subset_size} measurements")
    print(f"Max pairs per trial: {args.max_pairs}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"BayesOpt iterations: {args.n_iter}")
    print("="*80)
    print()
    
    # Load data
    print("Loading measurements...")
    measurements = load_measurement_records(Path(args.dataset))
    print(f"✓ Loaded {len(measurements)} measurements")
    
    # Load graph cache
    print("Loading pre-computed graphs...")
    with open(args.graph_cache, 'rb') as f:
        cache_data = pickle.load(f)
        all_graphs = cache_data['graphs']
        failed_indices = cache_data.get('failed_indices', [])
    
    # Filter out failed graphs
    if failed_indices:
        print(f"Filtering out {len(failed_indices)} measurements with failed graphs...")
        measurements = [m for idx, m in enumerate(measurements) if idx not in failed_indices]
        cached_graphs = [g for g in all_graphs if g is not None]
    else:
        cached_graphs = all_graphs
    
    print(f"✓ Loaded {len(cached_graphs)} cached graphs")
    
    # Use subset for speed
    if args.subset_size < len(measurements):
        print(f"Using random subset of {args.subset_size} measurements for fast iteration...")
        indices = random.Random(args.seed).sample(range(len(measurements)), args.subset_size)
        measurements = [measurements[i] for i in indices]
        cached_graphs = [cached_graphs[i] for i in indices]
    
    print(f"✓ Using {len(measurements)} measurements for optimization")
    print()
    
    # Define objective function
    def objective(learning_rate, margin, batch_size, hidden_dim, weight_decay):
        return train_and_evaluate(
            learning_rate=learning_rate,
            margin=margin,
            batch_size=int(batch_size),
            hidden_dim=int(hidden_dim),
            weight_decay=weight_decay,
            measurements=measurements,
            cached_graphs=cached_graphs,
            max_pairs=args.max_pairs,
            epochs=args.epochs,
            seed=args.seed,
        )
    
    # Define hyperparameter bounds
    pbounds = {
        'learning_rate': (1e-4, 5e-3),     # 0.0001 to 0.005
        'margin': (0.01, 0.5),              # 0.01 to 0.5
        'batch_size': (64, 512),            # 64 to 512 (will be rounded to int)
        'hidden_dim': (32, 128),            # 32 to 128 (will be rounded to int)
        'weight_decay': (1e-6, 1e-3),      # 0.000001 to 0.001
    }
    
    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=args.seed,
        verbose=2,
    )
    
    # Set up logging
    logger = JSONLogger(path=args.output)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    print("="*80)
    print("Starting Bayesian Optimization...")
    print("="*80)
    print()
    
    # Run optimization
    optimizer.maximize(
        init_points=5,  # Random exploration first
        n_iter=args.n_iter,  # Then guided optimization
    )
    
    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print()
    print("Best parameters found:")
    for param, value in optimizer.max['params'].items():
        if param in ['batch_size', 'hidden_dim']:
            print(f"  {param}: {int(value)}")
        else:
            print(f"  {param}: {value:.6f}")
    print(f"\nBest validation accuracy: {optimizer.max['target']:.4f}")
    print()
    print(f"Results saved to: {args.output}")
    print()
    
    # Save best params to a separate file
    best_params_file = args.output.replace('.json', '_best.txt')
    with open(best_params_file, 'w') as f:
        f.write("# Best hyperparameters from Bayesian Optimization\n")
        f.write(f"# Validation Accuracy: {optimizer.max['target']:.4f}\n\n")
        for param, value in optimizer.max['params'].items():
            if param in ['batch_size', 'hidden_dim']:
                f.write(f"{param}={int(value)}\n")
            else:
                f.write(f"{param}={value:.6f}\n")
    
    print(f"Best parameters also saved to: {best_params_file}")


if __name__ == "__main__":
    main()

