#!/usr/bin/env python3
"""Pre-compute and cache ProgramGraph objects from measurement records."""

import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

from tvm_cost_model.data.dataset_builder import load_measurement_records
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder
from tvm_cost_model.features.graph_builder import ProgramGraph


def main():
    parser = argparse.ArgumentParser(description="Pre-compute graphs from measurements")
    parser.add_argument("--dataset", type=str, required=True, help="Path to measurements parquet")
    parser.add_argument("--output", type=str, required=True, help="Output path for cached graphs")
    args = parser.parse_args()

    print(f"Loading measurements from {args.dataset}...")
    measurements = load_measurement_records(Path(args.dataset))
    print(f"Loaded {len(measurements)} measurements")

    builder = TVMGraphBuilder()
    graphs = []
    failed_indices = []
    
    print("Building graphs...")
    with tqdm(total=len(measurements), desc="Building graphs", unit="graph") as pbar:
        for idx, m in enumerate(measurements):
            try:
                graph = builder.build(m.scheduled_tir or m.original_tir)
                graphs.append(graph)
            except Exception as e:
                print(f"\nWarning: Failed to build graph for measurement {idx}: {e}")
                graphs.append(None)
                failed_indices.append(idx)
            pbar.update(1)
    
    # Filter out None graphs
    valid_graphs = [g for g in graphs if g is not None]
    print(f"\nSuccessfully built {len(valid_graphs)}/{len(measurements)} graphs")
    if failed_indices:
        print(f"Failed to build {len(failed_indices)} graphs at indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
    
    # Save graphs with their corresponding measurement indices
    cache_data = {
        'graphs': graphs,  # Keep None values to maintain alignment with measurements
        'valid_count': len(valid_graphs),
        'failed_indices': failed_indices,
        'measurement_count': len(measurements),
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(graphs)} graphs to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Cached graphs saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

