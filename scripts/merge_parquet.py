"""Merge all Parquet files in a directory into a single file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all .parquet files in a directory into a single file."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing .parquet files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Parquet path. Defaults to '<directory>/merged.parquet'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include .parquet files in nested subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the output file if it already exists.",
    )
    return parser.parse_args()


def collect_parquet_files(directory: Path, recursive: bool) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {directory}")

    pattern = "**/*.parquet" if recursive else "*.parquet"
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def merge_parquet_files(parquet_files: list[Path], output: Path) -> int:
    first_file = pq.ParquetFile(parquet_files[0])
    base_schema = first_file.schema_arrow.remove_metadata()

    output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with pq.ParquetWriter(output, base_schema) as writer:
        for path in parquet_files:
            try:
                parquet_file = pq.ParquetFile(path)
            except Exception as e:
                print(f"Skipping invalid Parquet file {path}: {e}")
                continue
            current_schema = parquet_file.schema_arrow.remove_metadata()

            if not current_schema.equals(base_schema, check_metadata=False):
                raise ValueError(
                    f"Schema mismatch in {path}. All files must share the same columns and types."
                )

            for batch in parquet_file.iter_batches():
                writer.write_batch(batch.cast(base_schema))

            total_rows += parquet_file.metadata.num_rows

    return total_rows


def main() -> None:
    args = parse_args()

    output_path = args.output or args.directory / "merged.parquet"
    parquet_files = collect_parquet_files(args.directory, args.recursive)
    parquet_files = [path for path in parquet_files if path.resolve() != output_path.resolve()]

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {args.directory}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    total_rows = merge_parquet_files(parquet_files, output_path)
    print(f"Merged {len(parquet_files)} files into {output_path} ({total_rows} rows).")


if __name__ == "__main__":
    main()
