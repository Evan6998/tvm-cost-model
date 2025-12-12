"""Normalize tuning logs into split datasets for offline training."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


def _shape_to_suffix(shape: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(shape):
        value = shape[key]
        if isinstance(value, (list, tuple)):
            value_str = "x".join(str(v) for v in value)
        else:
            value_str = str(value)
        parts.append(f"{key}{value_str}")
    return "_".join(parts) if parts else "nospec"


def _workload_id(operator: str, shape: dict[str, Any], target: str) -> str:
    return f"{operator}_{_shape_to_suffix(shape)}_{str(target).replace(' ', '')}"


def _canonical_shape(shape: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in shape.items():
        if isinstance(value, list):
            normalized[key] = tuple(value)
        elif isinstance(value, tuple):
            normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def _canonical_shape_str(shape: dict[str, Any]) -> str:
    return _shape_to_suffix(_canonical_shape(shape))


def _load_logs(paths: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pattern in paths:
        matched = glob.glob(pattern)
        if not matched and Path(pattern).exists():
            matched = [pattern]
        for path_str in matched:
            path = Path(path_str)
            if not path.exists():
                continue
            with path.open() as f:
                print(f"Loading log: {path}")
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect measurement logs into split datasets.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="JSONL log paths or glob patterns produced by run_metaschedule_tuning.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/benchmarks/offline",
        help="Directory to place split datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_workloads.yaml",
        help="Benchmark workload config to honor declared train/id_test/ood_test splits.",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train fraction for train workloads.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction for train workloads.")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test fraction for train workloads.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for splitting.")
    parser.add_argument("--limit-per-workload", type=int, default=0, help="Optional cap on records per workload.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def build_split_lookup(cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (by_name, by_signature) mappings from workload -> split."""

    by_name: dict[str, str] = {}
    by_signature: dict[str, str] = {}
    for workload in cfg.get("workloads", []):
        name = workload.get("name")
        split = workload.get("split", "train")
        shape = workload.get("shape", {})
        op = workload.get("operator")
        if name:
            by_name[str(name)] = split
        if op and shape:
            by_signature[f"{op}:{_canonical_shape_str(shape)}"] = split
    return by_name, by_signature


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    name_split, sig_split = build_split_lookup(cfg)
    print(f"Loaded {len(name_split)} named splits and {len(sig_split)} signature splits from config.")
    print("name_split: " + json.dumps(name_split, indent=2))
    print("sig_split: " + json.dumps(sig_split, indent=2))

    print(f"{args.logs=}")
    raw_records = _load_logs(args.logs)
    print(f"Loaded {len(raw_records)} raw records from logs.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in raw_records:
        latency = rec.get("latency_ms")
        if latency is None:
            continue
        shape = _canonical_shape(rec.get("shape", {}))
        workload_id = rec.get("workload_id") or _workload_id(
            rec.get("operator", "unknown"), shape, rec.get("target", "")
        )
        rec["workload_id"] = workload_id
        grouped[workload_id].append(rec)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[str, Any] = {"workloads": {}}
    rng = random.Random(args.seed)

    print(f"{grouped.keys()=}")
    for workload_id, records in grouped.items():
        records.sort(key=lambda r: r.get("measure_idx", 0))
        if args.limit_per_workload > 0:
            records = records[: args.limit_per_workload]
        _write_jsonl(output_dir / "raw" / f"{workload_id}.jsonl", records)

        first = records[0]
        name = first.get("task_name") or workload_id
        sig_key = f"{first.get('operator')}:{_canonical_shape_str(first.get('shape', {}))}"
        split = name_split.get(name) or sig_split.get(sig_key) or "train"

        metadata["workloads"][workload_id] = {
            "count": len(records),
            "split": split,
            "operator": first.get("operator"),
            "shape": first.get("shape"),
            "target": first.get("target"),
        }

        if split in {"id_test", "ood_test"}:
            split_buckets[split].extend(records)
            continue

        rng.shuffle(records)
        n = len(records)
        train_cut = int(n * args.train_frac)
        val_cut = train_cut + int(n * args.val_frac)
        split_buckets["train"].extend(records[:train_cut])
        split_buckets["val"].extend(records[train_cut:val_cut])
        split_buckets["test"].extend(records[val_cut:])

    for split, rows in split_buckets.items():
        if not rows:
            continue
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
        metadata[split] = len(rows)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"Wrote dataset splits to {output_dir}. "
        f"Counts: train={len(split_buckets.get('train', []))}, "
        f"val={len(split_buckets.get('val', []))}, "
        f"test={len(split_buckets.get('test', []))}, "
        f"id_test={len(split_buckets.get('id_test', []))}, "
        f"ood_test={len(split_buckets.get('ood_test', []))}"
    )


if __name__ == "__main__":
    main()
