#!/bin/bash
#SBATCH --job-name=tvm_cost_model_training
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hrangara@cs.cmu.edu

# ============================================================================
# SLURM Training Script for TVM Cost Model
# ============================================================================
# This script trains the graph-based ranking model on TVM schedule data
# Duration: 6 hours
# Resources: 1 GPU, 8 CPUs, 64GB RAM
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Print job information
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize module system
source /etc/profile.d/modules.sh 2>/dev/null || true

# Load required modules
echo "Loading modules..."
module load cuda-12.9 || echo "Warning: Could not load cuda-12.9 module"
module list 2>&1 || true
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Verify TVM installation
echo "Checking TVM installation..."
python -c "import tvm; print('TVM version:', tvm.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo ""

# Set working directory
cd /home/hrangara/tvm-cost-model
echo "Working directory: $(pwd)"
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# ============================================================================
# Main Training Command
# ============================================================================
echo "========================================================================"
echo "Starting model training..."
echo "========================================================================"
echo ""

# Run training with specified configuration
python scripts/train_cost_model.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --epochs 200 \
    --max-pairs 10000 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --margin 1 \
    --output model.pth

# ============================================================================
# Job completion
# ============================================================================
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

# If training succeeded, print model artifacts
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training artifacts:"
    ls -lh artifacts/models/ 2>/dev/null || echo "No artifacts found"
fi

exit $EXIT_CODE

