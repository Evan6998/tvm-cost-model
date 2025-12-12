"""Cache GraphPy encodings for dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tvm.script import from_source  # type: ignore[import]

from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute GraphPy features for dataset splits.")
    parser.add_argument("--dataset-root", type=str, default="artifacts/benchmarks/offline", help="Path with split JSONL files.")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks/features", help="Output directory for feature caches.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test,id_test,ood_test",
        help="Comma-separated list of splits to encode.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap per split for quick smoke tests.")
    return parser.parse_args()


def load_split(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def serialize_encoding(enc) -> dict[str, Any]:
    return {
        "node_features": enc.node_features.cpu(),
        "node_types": enc.node_types.cpu(),
        "edge_index": enc.edge_index.cpu(),
        "edge_types": enc.edge_types.cpu(),
        "feature_names": enc.feature_names,
    }


def build_graph(record: dict[str, Any], builder: TVMGraphBuilder):
    tir_src = record.get("scheduled_tir") or record.get("original_tir")
    if not tir_src:
        return None
    try:
        mod = from_source(tir_src)
        return builder.build(mod)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s for s in args.splits.split(",") if s]
    builder = TVMGraphBuilder()
    encoder = GraphEncoder()

    # First pass: dedupe candidate encodings across all splits.
    all_records: list[dict[str, Any]] = []
    for split in splits:
        split_path = dataset_root / f"{split}.jsonl"
        rows = load_split(split_path, args.max_records or None)
        all_records.extend(rows)

    candidate_graphs: dict[str, Any] = {}
    for rec in all_records:
        cand_id = rec.get("candidate_id") or rec.get("schedule_trace")
        if cand_id in candidate_graphs:
            continue
        graph = build_graph(rec, builder)
        if graph is None:
            continue
        candidate_graphs[str(cand_id)] = graph

    encoder.prime_feature_names(candidate_graphs.values())
    encoding_map: dict[str, Any] = {}
    for cand_id, graph in candidate_graphs.items():
        try:
            enc = encoder.encode(graph)
            tensor_enc = encoder.to_tensor_encoding(enc, device=torch.device("cpu"))
            encoding_map[cand_id] = serialize_encoding(tensor_enc)
        except Exception:
            continue

    for split in splits:
        split_path = dataset_root / f"{split}.jsonl"
        rows = load_split(split_path, args.max_records or None)
        if not rows:
            continue
        cached: list[dict[str, Any]] = []
        for rec in rows:
            cand_id = str(rec.get("candidate_id") or rec.get("schedule_trace"))
            enc = encoding_map.get(cand_id)
            if enc is None:
                continue
            cached.append(
                {
                    "candidate_id": cand_id,
                    "encoding": enc,
                    "latency_ms": rec.get("latency_ms"),
                    "workload_id": rec.get("workload_id"),
                    "operator": rec.get("operator"),
                    "shape": rec.get("shape"),
                    "target": rec.get("target"),
                }
            )
        if not cached:
            continue
        out_path = output_dir / f"{split}.pt"
        torch.save({"feature_names": encoder.feature_names, "records": cached}, out_path)
        print(f"Wrote {len(cached)} encodings to {out_path}")


if __name__ == "__main__":
    main()
