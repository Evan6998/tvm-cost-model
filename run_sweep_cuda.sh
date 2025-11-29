#!/bin/bash
# Comprehensive workload sweep on CUDA GPU
# Run this on the compute node with GPU allocation

set -e

# Load CUDA module
module load cuda-12.9

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

# Set working directory
cd /home/hrangara/tvm-cost-model

# Run the sweep with CUDA target
echo "Starting comprehensive workload sweep on RTX A6000..."
echo "This will run 16 different workloads with varying shapes"
echo "Expected time: ~15-20 minutes"
echo ""

python scripts/sweep_workloads.py \
  --target "cuda -arch=sm_86 -max_threads_per_block=1024 -thread_warp_size=32 -max_shared_memory_per_block=49152 -registers_per_block=65536" \
  --device-kind cuda \
  --hardware rtx_a6000 \
  --batches 1 \
  --batch-size 5 \
  --output-dir artifacts/sweeps/comprehensive_cuda \
  --number 10 \
  --repeat 1

echo ""
echo "✓ Sweep complete!"
echo "Output: artifacts/sweeps/comprehensive_cuda/sweep_merged.parquet"

