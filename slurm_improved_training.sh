#!/bin/bash
#SBATCH --job-name=tvm_improved_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/improved_train_%j.out
#SBATCH --error=logs/improved_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hrangara@cs.cmu.edu

# ============================================================================
# IMPROVED TRAINING with ListNet Loss + Adaptive Margins
# ============================================================================
# This script runs training with TWO KEY IMPROVEMENTS for hard pairs:
# 1. ListNet loss - listwise ranking instead of pairwise
# 2. Adaptive margins + hard pair reweighting
#
# Expected: +15-25% validation accuracy on hard pairs (56% → 70-80%)
# ============================================================================

set -e
set -u

echo "========================================================================"
echo "IMPROVED TRAINING - ListNet + Adaptive Margins"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================================================"
echo ""

# Create logs directory
mkdir -p logs

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
echo "Python version: $(python --version)"
echo ""

# Verify environment
echo "Checking environment..."
python -c "import tvm; print('TVM version:', tvm.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
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

# ============================================================================
# MAIN TRAINING - Improved Configuration
# ============================================================================
echo "========================================================================"
echo "Starting IMPROVED training with:"
echo "  ✓ ListNet loss (listwise ranking)"
echo "  ✓ Adaptive margins (performance-gap aware)"
echo "  ✓ Hard pair reweighting (3x weight for similar schedules)"
echo ""
echo "Expected improvements on HARD pairs:"
echo "  - Baseline: ~56% accuracy"
echo "  - Target:   ~70-80% accuracy (+15-25%)"
echo "========================================================================"
echo ""

python scripts/train_cost_model.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --epochs 200 \
    --max-pairs 30000 \
    --batch-size 256 \
    --learning-rate 5e-4 \
    --margin 0.05 \
    --output model_improved.pth

# ============================================================================
# Job completion
# ============================================================================
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model saved to: model_improved.pth"
    echo ""
    echo "Key metrics to check:"
    echo "  1. Hard stage validation accuracy (should be > 65%)"
    echo "  2. Compare to baseline (was ~56%)"
    echo "  3. Look for 'hard' stage in logs above"
    echo ""
    echo "Log files:"
    echo "  stdout: logs/improved_train_${SLURM_JOB_ID}.out"
    echo "  stderr: logs/improved_train_${SLURM_JOB_ID}.err"
else
    echo ""
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/improved_train_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

