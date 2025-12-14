#!/bin/bash
# Quick 30-epoch test with ListNet + improvements
# Run this on the compute node (babel-y9-16)

set -e
set -u

echo "========================================================================"
echo "QUICK TEST: 30 Epochs with ListNet + Adaptive Margins"
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

echo "Step 1: Testing graph building (5 samples)..."
python -c "
from pathlib import Path
from tvm_cost_model.data.dataset_builder import load_measurement_records
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder

measurements = load_measurement_records(Path('artifacts/sweeps/sweep_merged.parquet'))
builder = TVMGraphBuilder()

print(f'Testing graph building on 5 samples...')
for i in range(5):
    try:
        graph = builder.build(measurements[i].scheduled_tir or measurements[i].original_tir)
        print(f'  [{i}] ✓ {len(graph.nodes)} nodes')
    except Exception as e:
        print(f'  [{i}] ✗ {e}')
        exit(1)
print('✓ Graph building works!')
"

if [ $? -ne 0 ]; then
    echo "✗ Graph building test failed!"
    exit 1
fi

echo ""
echo "Step 2: Building graph cache (if needed)..."
GRAPH_CACHE="artifacts/sweeps/sweep_merged_graphs.pkl"

if [ ! -f "$GRAPH_CACHE" ]; then
    echo "Cache not found, building..."
    python scripts/precompute_graphs.py \
        --dataset artifacts/sweeps/sweep_merged.parquet \
        --output "$GRAPH_CACHE"
else
    echo "✓ Using existing cache: $GRAPH_CACHE"
    ls -lh "$GRAPH_CACHE"
fi

echo ""
echo "Step 3: Running 30-epoch training with improvements..."
echo "  - Shuffled training (no curriculum)"
echo "  - ListNet loss (listwise ranking)"
echo "  - Adaptive margins (performance-gap aware)"
echo "  - Hard pair reweighting (3x)"
echo ""
echo "DEBUG: About to start Python script at $(date)"
echo "DEBUG: Python: $(which python)"
echo ""

python -u scripts/train_cost_model.py \
    --dataset artifacts/sweeps/sweep_merged.parquet \
    --epochs 30 \
    --max-pairs 5000 \
    --batch-size 128 \
    --learning-rate 5e-4 \
    --margin 0.05 \
    --output model_test_30epochs.pth \
    --graph-cache "$GRAPH_CACHE" \
    --no-curriculum 2>&1 | tee logs/test_train_30epochs_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Test completed with exit code: $EXIT_CODE"
echo "End: $(date)"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model saved to: model_test_30epochs.pth"
    echo "Check logs/ for detailed output"
else
    echo ""
    echo "✗ Training failed!"
fi

exit $EXIT_CODE

