"""Remove the first and last rows from a Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop the first and last record from a Parquet file."
    )
    parser.add_argument("input", type=Path, help="Path to the source Parquet file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to '<input>_trimmed.parquet'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    table = pq.read_table(args.input)  # type: ignore
    if table.num_rows < 2:
        raise ValueError("Input must contain at least two rows to drop first and last.")

    trimmed = table.slice(1, table.num_rows - 2)
    output = args.output or args.input.with_name(f"{args.input.stem}_trimmed{args.input.suffix}")

    pq.write_table(trimmed, output)  # type: ignore
    print(f"Wrote {trimmed.num_rows} rows to {output}")


if __name__ == "__main__":
    main()
