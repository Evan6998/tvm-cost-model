#!/bin/bash
#SBATCH --job-name=tvm_test_30ep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_30epochs_%j.out
#SBATCH --error=logs/test_30epochs_%j.err

# ============================================================================
# QUICK TEST: 30 Epochs with Full Dataset (30K pairs)
# ============================================================================
# Tests ListNet + Adaptive Margins + Hard Pair Reweighting
# Expected time: ~30-40 minutes
# ============================================================================

set -e
set -u

echo "========================================================================"
echo "QUICK TEST: 30 Epochs with ListNet + Adaptive Margins"
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
# STEP 1: Verify graph cache exists
# ============================================================================
GRAPH_CACHE="artifacts/sweeps/sweep_merged_graphs.pkl"

echo "========================================================================"
echo "Checking for pre-computed graphs..."
echo "========================================================================"

if [ ! -f "$GRAPH_CACHE" ]; then
    echo "❌ Graph cache not found. Building graphs..."
    echo "   This will take ~30-60 minutes but only needs to be done once."
    echo ""
    python scripts/precompute_graphs.py \
        --dataset artifacts/sweeps/sweep_merged.parquet \
        --output "$GRAPH_CACHE"
    echo ""
    echo "✓ Graph cache created: $GRAPH_CACHE"
else
    echo "✓ Using existing graph cache: $GRAPH_CACHE"
    ls -lh "$GRAPH_CACHE"
fi
echo ""

# ============================================================================
# STEP 2: Run 30-epoch training
# ============================================================================
echo "========================================================================"
echo "Starting 30-epoch training with:"
echo "  ✓ 30,000 ranking pairs (full dataset)"
echo "  ✓ 30 epochs (quick test)"
echo "  ✓ Shuffled training (no curriculum)"
echo "  ✓ ListNet loss (listwise ranking)"
echo "  ✓ Adaptive margins (performance-gap aware)"
echo "  ✓ Hard pair reweighting (3x)"
echo ""
echo "Expected improvements on HARD pairs:"
echo "  - Baseline: ~56% accuracy"
echo "  - Target:   ~70-80% accuracy (+15-25%)"
echo "========================================================================"
echo ""

python -u scripts/train_cost_model.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --epochs 30 \
    --max-pairs 30000 \
    --batch-size 256 \
    --learning-rate 5e-4 \
    --margin 0.05 \
    --weight-decay 1e-4 \
    --output model_test_30epochs.pth \
    --graph-cache "$GRAPH_CACHE" \
    --no-curriculum

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
    echo "Model saved to: model_test_30epochs.pth"
    echo ""
    echo "Next steps:"
    echo "  1. Check validation accuracy trends"
    echo "  2. If looks good, run full 200-epoch training"
    echo "  3. Command: sbatch slurm_improved_training.sh"
    echo ""
    echo "Log files:"
    echo "  stdout: logs/test_30epochs_${SLURM_JOB_ID}.out"
    echo "  stderr: logs/test_30epochs_${SLURM_JOB_ID}.err"
else
    echo ""
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/test_30epochs_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

