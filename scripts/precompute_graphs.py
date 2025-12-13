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
    
    print("Building graphs...")
    with tqdm(total=len(measurements), desc="Building graphs", unit="graph") as pbar:
        for m in measurements:
            try:
                graph = builder.build_from_tir(m.scheduled_tir or m.original_tir)
                graphs.append(graph)
            except Exception as e:
                print(f"\nWarning: Failed to build graph: {e}")
                graphs.append(None)
            pbar.update(1)
    
    # Save graphs with their corresponding measurement indices
    cache_data = {
        'graphs': graphs,
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

