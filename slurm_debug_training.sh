#!/bin/bash
#SBATCH --job-name=tvm_debug_train
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/debug_train_%j.out
#SBATCH --error=logs/debug_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hrangara@cs.cmu.edu

# ============================================================================
# SLURM Debug Training Script for TVM Cost Model
# ============================================================================
# This runs the debug/toy training pipeline for faster experimentation
# Duration: 6 hours (should complete much sooner)
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

# Verify installations
echo "Checking environment..."
python -c "import tvm; print('TVM version:', tvm.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import pyarrow; print('PyArrow version:', pyarrow.__version__)"
echo ""

# Set working directory
cd /home/hrangara/tvm-cost-model
echo "Working directory: $(pwd)"
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ============================================================================
# Main Training Command - Debug/Toy Training
# ============================================================================
echo "========================================================================"
echo "Starting debug training (curriculum learning on sample dataset)..."
echo "========================================================================"
echo ""

python scripts/debug_toy_training.py

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

# Print final status
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Log files:"
    echo "  stdout: logs/debug_train_${SLURM_JOB_ID}.out"
    echo "  stderr: logs/debug_train_${SLURM_JOB_ID}.err"
else
    echo ""
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Check error log: logs/debug_train_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

