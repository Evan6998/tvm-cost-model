#!/bin/bash
# Run this on the compute node (babel-y9-16)

set -e

echo "Loading CUDA module..."
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.9

echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

echo "Current location: $(hostname)"
echo "Python: $(which python)"
echo ""

cd /home/hrangara/tvm-cost-model

echo "Running debug script..."
python debug_graph_cache.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Debug passed! Ready to rebuild cache."
    echo ""
    echo "Next steps:"
    echo "1. rm artifacts/sweeps/sweep_merged_graphs.pkl"
    echo "2. python scripts/precompute_graphs.py --dataset artifacts/sweeps/sweep_merged.parquet --output artifacts/sweeps/sweep_merged_graphs.pkl"
else
    echo ""
    echo "✗ Debug found issues. Fix before rebuilding cache."
fi

exit $EXIT_CODE

