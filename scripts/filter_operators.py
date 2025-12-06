"""Filter out specific operators from a MeasurementRecord Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tvm_cost_model.data.dataset_builder import MeasurementRecord, load_measurement_records


DEFAULT_OPERATORS = ("vecadd", "layernorm", "softmax")


def _default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_filtered{path.suffix}")


def _write_records(records: Iterable[MeasurementRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([m.as_dict() for m in records])
    pq.write_table(table, output)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove unwanted operators from a MeasurementRecord Parquet file."
    )
    parser.add_argument("dataset", type=str, help="Path to the input Parquet file.")
    parser.add_argument(
        "--operators",
        type=str,
        default=",".join(DEFAULT_OPERATORS),
        help=f"Comma-separated list of operator names to drop (default: {','.join(DEFAULT_OPERATORS)}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Parquet path. Defaults to <dataset> with '_filtered' suffix. Use --inplace to overwrite input.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of writing to a new one.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    remove_ops = {op.strip().lower() for op in args.operators.split(",") if op.strip()}
    if not remove_ops:
        print("No operators specified for removal; nothing to do.")
        return

    measurements = load_measurement_records(dataset_path)
    kept = [m for m in measurements if m.operator.lower() not in remove_ops]
    removed_count = len(measurements) - len(kept)

    output_path = dataset_path if args.inplace else Path(args.output or _default_output_path(dataset_path))
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    _write_records(kept, output_path)

    print(
        f"Loaded {len(measurements)} records from {dataset_path}. "
        f"Removed {removed_count} matching operators: {sorted(remove_ops)}. "
        f"Wrote {len(kept)} records to {output_path}."
    )


if __name__ == "__main__":
    main()
