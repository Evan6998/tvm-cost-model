"""Sweep multiple operators/shapes/targets and merge Parquet outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

from pydantic import BaseModel, TypeAdapter

DEFAULT_WORKLOADS = [
    ("vecadd", {"n": 1024}),
    ("gemm", {"m": 256, "n": 256, "k": 256}),
    ("bmm", {"batch": 8, "m": 128, "n": 128, "k": 128}),
    ("conv2d_nchw", {"n": 1, "ci": 64, "co": 64, "h": 56, "w": 56, "kh": 3, "kw": 3}),
    ("depthwise_conv2d", {"n": 1, "ci": 64, "h": 56, "w": 56, "kh": 3, "kw": 3}),
    ("layernorm", {"n": 64, "hidden": 256}),
    ("softmax", {"n": 64, "k": 256}),
]

class WorkloadSpec(BaseModel):
    op: str
    shape: Dict[str, int]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep workloads via bootstrap_dataset.py")
    p.add_argument("--workloads", type=str, default="", help="JSON list of (op, shape) dicts")
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target", type=str, default="llvm -num-cores 8")
    p.add_argument("--hardware", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="artifacts/sweeps")
    p.add_argument("--rpc-host", type=str, default="")
    p.add_argument("--rpc-port", type=int, default=9090)
    p.add_argument("--rpc-key", type=str, default="")
    p.add_argument("--device-kind", type=str, default="llvm")
    p.add_argument("--device-idx", type=int, default=0)
    p.add_argument("--number", type=int, default=5)
    p.add_argument("--repeat", type=int, default=1)
    return p


def parse_workloads(arg: str) -> List[tuple[str, Dict[str, int]]]:
    if not arg:
        return DEFAULT_WORKLOADS
    adapter = TypeAdapter(List[WorkloadSpec])
    parsed = adapter.validate_json(arg)
    return [(item.op, item.shape) for item in parsed]


def run_bootstrap(op: str, shape: Dict[str, int], args: argparse.Namespace, output_root: Path) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        tmp.write(json.dumps(shape))
        tmp_path = tmp.name
    cmd: list[str] = [
        "python",
        "scripts/bootstrap_dataset.py",
        "--mode",
        "metaschedule",
        "--operator",
        op,
        "--shape",
        Path(tmp_path).read_text(),
        "--batches",
        str(args.batches),
        "--batch-size",
        str(args.batch_size),
        "--hardware",
        args.hardware,
        "--target",
        args.target,
        "--dtype",
        args.dtype,
        "--output-dir",
        str(output_root),
        "--device-kind",
        args.device_kind,
        "--device-idx",
        str(args.device_idx),
        "--number",
        str(args.number),
        "--repeat",
        str(args.repeat),
    ]
    if args.rpc_host:
        cmd += ["--rpc-host", args.rpc_host, "--rpc-port", str(args.rpc_port)]
        if args.rpc_key:
            cmd += ["--rpc-key", args.rpc_key]
    subprocess.run(cmd, check=True)
    artifacts = sorted(output_root.glob(f"*{op}*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return artifacts[0]


def merge_parquet(paths: Iterable[Path], output: Path) -> None:
    import pyarrow.parquet as pq
    import pyarrow as pa

    tables = [pq.read_table(p) for p in paths] # type: ignore
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output) # type: ignore


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    workloads = parse_workloads(args.workloads)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shards: List[Path] = []
    for op, shape in workloads:
        shards.append(run_bootstrap(op, shape, args, output_root))
    merged = output_root / "sweep_merged.parquet"
    merge_parquet(shards, merged)
    print(f"Merged {len(shards)} shards into {merged}")


if __name__ == "__main__":
    main()
