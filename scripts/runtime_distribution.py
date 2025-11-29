"""Quick utility to inspect runtime_ms distribution in a measurement dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a standalone script.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tvm_cost_model.data.dataset_builder import load_measurement_records  # noqa: E402


def describe(values: list[float], bins: int) -> None:
    arr = np.array(values, dtype=np.float64)
    print(f"count={arr.size}")
    print(
        "min={:.6f} max={:.6f} mean={:.6f} std={:.6f}".format(
            float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())
        )
    )
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        print(f"p{int(q*100):02d}={np.quantile(arr, q):.6f}", end="  ")
    print("\n")

    hist, edges = np.histogram(arr, bins=bins)
    print("Histogram (bin_start -> count):")
    for left, count in zip(edges[:-1], hist):
        print(f"{left:.6f} -> {int(count)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show runtime_ms distribution for a measurement dataset.")
    parser.add_argument("dataset", type=Path, help="Path to measurement parquet file.")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins (default: 20).")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    records = load_measurement_records(args.dataset)
    if not records:
        print("No records found.")
        return

    runtimes = [float(r.runtime_ms) for r in records]
    describe(runtimes, bins=args.bins)


if __name__ == "__main__":
    main()
