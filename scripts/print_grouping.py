"""Utility script to summarize measurement groupings in a Parquet dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tvm_cost_model.data.dataset_builder import load_measurement_records


ShapeKey = Tuple[Tuple[str, Tuple[int, ...]], ...]


def _workload_shape_key(workload_shape: Dict[str, tuple[int, ...]] | None) -> ShapeKey:
    if not workload_shape:
        return tuple()
    return tuple(sorted((name, tuple(dimensions)) for name, dimensions in workload_shape.items()))


def _shape_key_to_str(shape_key: ShapeKey) -> str:
    if not shape_key:
        return "{}"
    parts = [f"{name}:{'x'.join(str(d) for d in dims)}" for name, dims in shape_key]
    return "{" + ", ".join(parts) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print grouping info for measurement Parquet file.")
    parser.add_argument("dataset", type=str, help="Path to MeasurementRecord Parquet file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of groups to print (sorted by size, descending).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    measurements = load_measurement_records(dataset_path)
    grouped: dict[tuple[str, ShapeKey, str | None, str], list] = defaultdict(list)
    for m in measurements:
        key = (m.operator, _workload_shape_key(m.workload_shape), m.target, m.hardware_id)
        grouped[key].append(m)

    print(f"Loaded {len(measurements)} measurements from {dataset_path}")
    print(f"Found {len(grouped)} groups keyed by operator/workload_shape/target/hardware_id")

    sorted_groups = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    if args.limit is not None:
        sorted_groups = sorted_groups[: args.limit]

    for (operator, shape_key, target, hardware_id), group in sorted_groups:
        print(
            f"size={len(group):4d}  operator={operator:<16} "
            f"shape={_shape_key_to_str(shape_key)} target={target} hardware_id={hardware_id}"
        )


if __name__ == "__main__":
    main()
