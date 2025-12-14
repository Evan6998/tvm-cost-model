#!/usr/bin/env python3
"""Validate Top Configs from BayesOpt on Full Dataset.

Strategy:
1. BayesOpt finds promising configs on subset (5K measurements, fast)
2. This script takes top N configs and validates on full dataset (30K measurements)
3. Launch N jobs in parallel (typically N=5-10)
"""

import argparse
import json
import subprocess
from pathlib import Path


def load_bayesopt_results(results_file):
    """Load BayesOpt results and return sorted by target."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Get all iterations with their scores
    iterations = data.get('all_iterations', [])
    
    # Sort by target (validation accuracy) descending
    iterations.sort(key=lambda x: x['target'], reverse=True)
    
    return iterations


def create_validation_job(config, job_id, output_dir, dataset, graph_cache, epochs=50, max_pairs=30000):
    """Create SLURM script for full-dataset validation of a config."""
    
    script_path = output_dir / f"validate_{job_id}.sh"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    params = config['params']
    lr = params['learning_rate']
    margin = params['margin']
    batch_size = int(params['batch_size'])
    hidden_dim = int(params['hidden_dim'])
    weight_decay = params['weight_decay']
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=val_{job_id}
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={results_dir}/validate_{job_id}.out
#SBATCH --error={results_dir}/validate_{job_id}.err

set -e
set -u

echo "========================================================================"
echo "Full Dataset Validation - Config {job_id}"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""
echo "BayesOpt subset accuracy: {config['target']:.4f}"
echo ""
echo "Configuration:"
echo "  learning_rate: {lr}"
echo "  margin: {margin}"
echo "  batch_size: {batch_size}"
echo "  hidden_dim: {hidden_dim}"
echo "  weight_decay: {weight_decay}"
echo ""
echo "Training on FULL dataset:"
echo "  measurements: 30,000 pairs"
echo "  epochs: {epochs}"
echo "========================================================================"
echo ""

# Load modules and activate conda
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.9 || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

cd /home/hrangara/tvm-cost-model

# Run training on full dataset
python -u scripts/train_cost_model.py \\
    --dataset {dataset} \\
    --graph-cache {graph_cache} \\
    --epochs {epochs} \\
    --max-pairs {max_pairs} \\
    --learning-rate {lr} \\
    --margin {margin} \\
    --batch-size {batch_size} \\
    --hidden-dim {hidden_dim} \\
    --weight-decay {weight_decay} \\
    --no-curriculum \\
    --output {results_dir}/model_validated_{job_id}.pth

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Validation {job_id} completed with exit code: $EXIT_CODE"
echo "End: $(date)"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================================================"

exit $EXIT_CODE
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    return script_path


def extract_final_accuracy(log_file):
    """Extract final validation accuracy from log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in reversed(lines):
            if 'val_acc=' in line:
                parts = line.split('val_acc=')
                if len(parts) > 1:
                    acc_str = parts[1].split()[0]
                    return float(acc_str)
    except Exception as e:
        print(f"Warning: Could not parse {log_file}: {e}")
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Validate top BayesOpt configs on full dataset"
    )
    parser.add_argument(
        "--bayesopt-results",
        type=str,
        default="artifacts/bayesopt/bayesopt_results_latest.json",
        help="Path to BayesOpt results JSON (defaults to latest symlink)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top configs to validate (default: 5)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="artifacts/sweeps/sweep_merged.parquet",
        help="Path to dataset"
    )
    parser.add_argument(
        "--graph-cache",
        type=str,
        default="artifacts/sweeps/sweep_merged_graphs.pkl",
        help="Path to graph cache"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for validation"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=30000,
        help="Number of pairs (full dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/full_validation",
        help="Directory for validation jobs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts but don't submit"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze validation results"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.analyze:
        # Analyze validation results
        results_dir = output_dir / "results"
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return
        
        print("="*80)
        print("FULL DATASET VALIDATION RESULTS")
        print("="*80)
        print()
        
        results = []
        for log_file in sorted(results_dir.glob("validate_*.out")):
            job_id = log_file.stem.split('_')[1]
            
            # Load config
            config_file = output_dir / f"config_{job_id}.json"
            if not config_file.exists():
                continue
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Extract accuracy
            val_acc = extract_final_accuracy(log_file)
            
            results.append({
                'job_id': job_id,
                'subset_acc': config.get('target', 0.0),
                'full_acc': val_acc,
                'params': config['params']
            })
        
        if not results:
            print("No results found. Jobs may still be running.")
            return
        
        # Sort by full dataset accuracy
        results.sort(key=lambda x: x['full_acc'], reverse=True)
        
        print(f"Validated {len(results)} configurations on full dataset")
        print()
        print(f"{'Rank':<6} {'Job':<6} {'Subset Acc':<12} {'Full Acc':<12} {'Config'}")
        print("-"*80)
        
        for i, result in enumerate(results, 1):
            params = result['params']
            config_str = f"lr={params['learning_rate']:.1e} margin={params['margin']:.2f} batch={int(params['batch_size'])}"
            print(f"{i:<6} {result['job_id']:<6} {result['subset_acc']:.4f}      {result['full_acc']:.4f}      {config_str}")
        
        # Save best config
        best = results[0]
        best_config_file = output_dir / "best_validated_config.json"
        with open(best_config_file, 'w') as f:
            json.dump(best['params'], f, indent=2)
        
        best_params_file = output_dir / "best_validated_params.txt"
        with open(best_params_file, 'w') as f:
            f.write(f"# Best Hyperparameters (Validated on Full Dataset)\n")
            f.write(f"# Full Dataset Validation Accuracy: {best['full_acc']:.4f}\n")
            f.write(f"# Subset Validation Accuracy: {best['subset_acc']:.4f}\n")
            f.write(f"# Job ID: {best['job_id']}\n\n")
            for key, value in best['params'].items():
                f.write(f"{key}={value}\n")
        
        print()
        print("="*80)
        print("WINNER: Best configuration on full dataset")
        print(f"  Validation Accuracy: {best['full_acc']:.4f}")
        print(f"  Saved to:")
        print(f"    - {best_config_file}")
        print(f"    - {best_params_file}")
        print("="*80)
        
        return
    
    # Load BayesOpt results
    results_file = Path(args.bayesopt_results)
    if not results_file.exists():
        print(f"Error: BayesOpt results not found: {results_file}")
        print("Run BayesOpt first: sbatch slurm_bayesopt.sh")
        return
    
    print("="*80)
    print("FULL DATASET VALIDATION OF TOP BAYESOPT CONFIGS")
    print("="*80)
    print(f"Loading results from: {results_file}")
    
    iterations = load_bayesopt_results(results_file)
    top_configs = iterations[:args.top_n]
    
    print(f"Found {len(iterations)} BayesOpt trials")
    print(f"Selecting top {len(top_configs)} for full dataset validation")
    print("="*80)
    print()
    
    print("Top configurations from BayesOpt (subset accuracy):")
    print("-"*80)
    for i, config in enumerate(top_configs, 1):
        params = config['params']
        print(f"{i}. Acc={config['target']:.4f} | lr={params['learning_rate']:.1e} margin={params['margin']:.2f} batch={int(params['batch_size'])} hidden={int(params['hidden_dim'])}")
    print()
    
    if args.top_n > 10:
        print(f"Warning: {args.top_n} jobs is a lot. Consider using --top-n 5")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    # Create validation jobs
    job_scripts = []
    for i, config in enumerate(top_configs):
        # Save config
        config_file = output_dir / f"config_{i}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create SLURM script
        script_path = create_validation_job(
            config, i, output_dir,
            args.dataset, args.graph_cache,
            args.epochs, args.max_pairs
        )
        job_scripts.append(script_path)
    
    print(f"Generated {len(job_scripts)} validation job scripts")
    print()
    
    if args.dry_run:
        print("Dry run - scripts generated but not submitted")
        print()
        print("Scripts saved in:", output_dir)
        print("To submit manually:")
        print(f"  cd {output_dir}")
        print("  for script in validate_*.sh; do sbatch $script; done")
        return
    
    # Submit jobs
    print("Submitting validation jobs...")
    submitted_jobs = []
    
    for script in job_scripts:
        try:
            result = subprocess.run(
                ['sbatch', str(script)],
                capture_output=True,
                text=True,
                check=True
            )
            job_id = result.stdout.strip().split()[-1]
            submitted_jobs.append(job_id)
            print(f"  ✓ {script.name}: SLURM Job ID {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {script.name}: {e}")
    
    print()
    print("="*80)
    print(f"Successfully submitted {len(submitted_jobs)}/{len(job_scripts)} validation jobs")
    print(f"Each job runs on FULL dataset ({args.max_pairs} pairs, {args.epochs} epochs)")
    print(f"Expected time per job: ~1-1.5 hours")
    print("="*80)
    print()
    print("Monitor progress:")
    print("  squeue -u $USER")
    print()
    print("Analyze results when complete:")
    print(f"  python {__file__} --analyze")
    print()


if __name__ == "__main__":
    main()

