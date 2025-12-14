#!/usr/bin/env python3
"""Debug script to test graph building and caching."""

import sys
from pathlib import Path
from tvm_cost_model.data.dataset_builder import load_measurement_records
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder

print("=" * 80)
print("GRAPH CACHE DEBUG SCRIPT")
print("=" * 80)

# Step 1: Load measurements
print("\n[1/4] Loading measurements...")
measurements = load_measurement_records(Path('artifacts/sweeps/sweep_merged.parquet'))
print(f"      ✓ Loaded {len(measurements)} measurements")

# Step 2: Test graph building on sample
print("\n[2/4] Testing graph building on 20 samples...")
builder = TVMGraphBuilder()
success_count = 0
none_count = 0
error_count = 0
errors = []

for i in range(min(20, len(measurements))):
    m = measurements[i]
    try:
        graph = builder.build(m.scheduled_tir or m.original_tir)
        if graph is None:
            none_count += 1
            print(f"      [{i:2d}] ⚠️  Graph is None")
        elif not hasattr(graph, 'nodes'):
            none_count += 1
            print(f"      [{i:2d}] ⚠️  Graph missing 'nodes' attribute")
        else:
            success_count += 1
            print(f"      [{i:2d}] ✓  {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    except Exception as e:
        error_count += 1
        error_msg = str(e)[:80]
        errors.append((i, error_msg))
        print(f"      [{i:2d}] ✗  {error_msg}")

print(f"\n      Results: {success_count} success, {none_count} None, {error_count} errors")

# Step 3: Check if existing cache has None values
print("\n[3/4] Checking existing cache...")
cache_path = Path('artifacts/sweeps/sweep_merged_graphs.pkl')
if cache_path.exists():
    import pickle
    print(f"      ✓ Cache exists: {cache_path}")
    print(f"      Size: {cache_path.stat().st_size / 1024:.1f} KB")
    
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    graphs = cache_data.get('graphs', [])
    none_indices = [i for i, g in enumerate(graphs) if g is None]
    
    print(f"      Total cached graphs: {len(graphs)}")
    print(f"      None graphs: {len(none_indices)}")
    
    if none_indices:
        print(f"      First 10 None indices: {none_indices[:10]}")
        print(f"\n      ⚠️  PROBLEM: Cache contains {len(none_indices)} None values!")
        print(f"      This will cause 'NoneType has no attribute nodes' error")
else:
    print(f"      ℹ️  No cache exists yet")

# Step 4: Recommendations
print("\n[4/4] Recommendations:")
print("=" * 80)

if none_count > 0 or error_count > 0:
    print("⚠️  ISSUES FOUND:")
    print(f"   - {none_count} graphs returned None")
    print(f"   - {error_count} graphs raised errors")
    print("\n   Root cause: Some TIR modules cannot be converted to graphs")
    print("   Solution: Filter out measurements with failed graphs")
    
    if cache_path.exists() and none_indices:
        print(f"\n   Current cache has {len(none_indices)} None values")
        print("   Action needed:")
        print("   1. Delete corrupted cache:")
        print("      rm artifacts/sweeps/sweep_merged_graphs.pkl")
        print("   2. Rebuild with fixed precompute_graphs.py")
        print("   3. Training script will auto-filter failed measurements")
else:
    print("✓ All test graphs built successfully!")
    print("  Cache should work fine")

print("=" * 80)

# Exit with error code if issues found
sys.exit(1 if (none_count > 0 or error_count > 0) else 0)

