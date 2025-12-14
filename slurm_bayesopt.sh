#!/bin/bash
#SBATCH --job-name=tvm_bayesopt
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bayesopt_%j.out
#SBATCH --error=logs/bayesopt_%j.err

# ============================================================================
# Bayesian Optimization for Hyperparameter Tuning
# ============================================================================
# Runs ~25-30 trials to find optimal hyperparameters
# Each trial: ~5-10 min on subset (5000 measurements, 5000 pairs, 20 epochs)
# Total time: ~2-5 hours
# ============================================================================

set -e
set -u

echo "========================================================================"
echo "BAYESIAN OPTIMIZATION FOR HYPERPARAMETER TUNING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================================================"
echo ""

# Create logs directory
mkdir -p logs artifacts/bayesopt

# Initialize module system
source /etc/profile.d/modules.sh 2>/dev/null || true

# Load CUDA
echo "Loading CUDA..."
module load cuda-12.9 || echo "Warning: Could not load cuda-12.9"
echo ""

# Activate conda environment
echo "Activating medCalcEnv..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv
echo "Python: $(which python)"
echo ""

# Verify environment
echo "Checking environment..."
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo ""

# Check for bayes_opt package
python -c "import bayes_opt" 2>/dev/null || {
    echo "Installing bayesian-optimization..."
    pip install bayesian-optimization
}
echo ""

# Set working directory
cd /home/hrangara/tvm-cost-model
echo "Working directory: $(pwd)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Verify graph cache exists
GRAPH_CACHE="artifacts/sweeps/sweep_merged_graphs.pkl"
if [ ! -f "$GRAPH_CACHE" ]; then
    echo "❌ Error: Graph cache not found at $GRAPH_CACHE"
    echo "Please run: python scripts/precompute_graphs.py first"
    exit 1
fi

echo "✓ Graph cache found: $(ls -lh $GRAPH_CACHE | awk '{print $5}')"
echo ""

# ============================================================================
# Run Bayesian Optimization
# ============================================================================
echo "========================================================================"
echo "Starting Bayesian Optimization..."
echo "  - Subset size: 10,000 measurements"
echo "  - Max pairs per trial: 10,000"
echo "  - Epochs per trial: 25"
echo "  - Number of trials: 15"
echo "  - Shuffled learning (no curriculum)"
echo ""
echo "Optimizing hyperparameters:"
echo "  - learning_rate: [5e-4, 5e-3]"
echo "  - margin: [0.1, 1.5] (includes your previous 1.0)"
echo "  - batch_size: [128, 512]"
echo "  - hidden_dim: [32, 128]"
echo "  - weight_decay: [1e-5, 1e-3]"
echo "========================================================================"
echo ""

python -u scripts/bayesopt_hyperparameters.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --graph-cache "$GRAPH_CACHE" \
    --subset-size 10000 \
    --max-pairs 10000 \
    --epochs 25 \
    --n-iter 15 \
    --output artifacts/bayesopt/bayesopt_results_${SLURM_JOB_ID}.json \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Bayesian Optimization completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Optimization completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  - artifacts/bayesopt/bayesopt_results_${SLURM_JOB_ID}.json"
    echo "  - artifacts/bayesopt/bayesopt_results_${SLURM_JOB_ID}_best.txt"
    echo ""
    echo "Best parameters:"
    cat artifacts/bayesopt/bayesopt_results_${SLURM_JOB_ID}_best.txt
    echo ""
    
    # Also create a symlink to latest for easy access
    ln -sf bayesopt_results_${SLURM_JOB_ID}.json artifacts/bayesopt/bayesopt_results_latest.json
    ln -sf bayesopt_results_${SLURM_JOB_ID}_best.txt artifacts/bayesopt/bayesopt_results_latest_best.txt
    echo "Symlinks created:"
    echo "  - artifacts/bayesopt/bayesopt_results_latest.json -> results_${SLURM_JOB_ID}.json"
    echo "  - artifacts/bayesopt/bayesopt_results_latest_best.txt -> results_${SLURM_JOB_ID}_best.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Review best parameters above"
    echo "  2. Validate top configs on full dataset:"
    echo "     python scripts/validate_top_configs.py --top-n 5 --bayesopt-results artifacts/bayesopt/bayesopt_results_${SLURM_JOB_ID}.json"
else
    echo ""
    echo "✗ Optimization failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/bayesopt_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

