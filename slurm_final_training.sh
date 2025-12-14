#!/bin/bash
#SBATCH --job-name=tvm_final_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/final_train_%j.out
#SBATCH --error=logs/final_train_%j.err

# ============================================================================
# Final Training with Optimized Hyperparameters
# ============================================================================
# Uses best hyperparameters from Bayesian optimization
# Full dataset: 30,000 pairs, 50 epochs
# Shuffled learning (no curriculum)
# Expected time: ~1-1.5 hours
# ============================================================================

set -e
set -u

echo "========================================================================"
echo "FINAL TRAINING WITH OPTIMIZED HYPERPARAMETERS"
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
# Load Best Hyperparameters from BayesOpt
# ============================================================================
BEST_PARAMS="artifacts/bayesopt/bayesopt_results_best.txt"

if [ -f "$BEST_PARAMS" ]; then
    echo "Loading best hyperparameters from: $BEST_PARAMS"
    echo "========================================================================"
    cat "$BEST_PARAMS"
    echo "========================================================================"
    echo ""
    
    # Parse parameters
    LEARNING_RATE=$(grep "learning_rate" "$BEST_PARAMS" | cut -d'=' -f2)
    MARGIN=$(grep "margin" "$BEST_PARAMS" | cut -d'=' -f2)
    BATCH_SIZE=$(grep "batch_size" "$BEST_PARAMS" | cut -d'=' -f2)
    HIDDEN_DIM=$(grep "hidden_dim" "$BEST_PARAMS" | cut -d'=' -f2)
    WEIGHT_DECAY=$(grep "weight_decay" "$BEST_PARAMS" | cut -d'=' -f2)
else
    echo "⚠️  Best parameters file not found, using reasonable defaults..."
    LEARNING_RATE="0.001"
    MARGIN="0.1"
    BATCH_SIZE="256"
    HIDDEN_DIM="64"
    WEIGHT_DECAY="0.0001"
fi

echo "Using hyperparameters:"
echo "  learning_rate: $LEARNING_RATE"
echo "  margin: $MARGIN"
echo "  batch_size: $BATCH_SIZE"
echo "  hidden_dim: $HIDDEN_DIM"  
echo "  weight_decay: $WEIGHT_DECAY"
echo ""

# ============================================================================
# Run Final Training
# ============================================================================
echo "========================================================================"
echo "Starting final training with:"
echo "  ✓ Full dataset: 30,000 pairs"
echo "  ✓ 50 epochs (faster iteration)"
echo "  ✓ Shuffled learning (no curriculum)"
echo "  ✓ Optimized hyperparameters from BayesOpt"
echo "========================================================================"
echo ""

python -u scripts/train_cost_model.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --epochs 50 \
    --max-pairs 30000 \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --margin "$MARGIN" \
    --weight-decay "$WEIGHT_DECAY" \
    --output model_final_optimized.pth \
    --graph-cache "$GRAPH_CACHE" \
    --no-curriculum

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Final training completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Duration: $((SECONDS / 60)) minutes"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model saved to: model_final_optimized.pth"
    echo ""
    echo "Log files:"
    echo "  stdout: logs/final_train_${SLURM_JOB_ID}.out"
    echo "  stderr: logs/final_train_${SLURM_JOB_ID}.err"
    echo ""
    echo "To use the model:"
    echo "  python -c \"from tvm_cost_model.models.graph_cost_model import GraphCostModel; model = GraphCostModel(); model.load('model_final_optimized.pth')\""
else
    echo ""
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Check logs: logs/final_train_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

