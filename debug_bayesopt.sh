#!/bin/bash
# Quick debug script for Bayesian optimization
# Runs on compute node with minimal iterations to test

set -e

echo "========================================================================"
echo "DEBUG: Bayesian Optimization Test"
echo "========================================================================"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Load CUDA
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.9

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

cd /home/hrangara/tvm-cost-model

# Install bayes_opt if needed
python -c "import bayes_opt" 2>/dev/null || {
    echo "Installing bayesian-optimization..."
    pip install bayesian-optimization
}

echo ""
echo "Running debug BayesOpt (2 iterations only)..."
echo ""

mkdir -p artifacts/bayesopt

python -u scripts/bayesopt_hyperparameters.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --graph-cache artifacts/sweeps/sweep_merged_graphs.pkl \
    --subset-size 1000 \
    --max-pairs 1000 \
    --epochs 5 \
    --n-iter 2 \
    --output artifacts/bayesopt/debug_results.json \
    --seed 42

echo ""
echo "========================================================================"
echo "Debug completed at $(date)"
echo "========================================================================"

