"""Train/evaluate cost models on collected datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import xgboost  # type: ignore
import torch
from tvm.script import from_source  # type: ignore[import]

from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.eval.offline_metrics import compute_ranking_metrics
from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder
from tvm_cost_model.training.pipeline import TrainingConfig, TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline benchmark for GraphPy vs XGB baselines.")
    parser.add_argument("--dataset-root", type=str, default="artifacts/benchmarks/offline", help="Directory from collect_dataset.py")
    parser.add_argument("--output", type=str, default="artifacts/benchmarks/offline/metrics.json", help="Path to write metrics JSON.")
    parser.add_argument("--model", type=str, default="both", choices=["graph", "xgb", "both"], help="Which model(s) to evaluate.")
    parser.add_argument("--max-pairs", type=int, default=2048, help="Ranking pairs for GraphPy training.")
    parser.add_argument("--epochs", type=int, default=10, help="GraphPy training epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap on train records.")
    return parser.parse_args()


def configure_torch_threads(num_threads: int = 1) -> None:
    """Limit torch thread pools to avoid rare segfaults in CPU reductions."""
    import sys
    if sys.platform != "darwin":
        return
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
    except Exception as err:
        print(f"Warning: unable to set torch threads: {err}")


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


def to_measurements(rows: Iterable[dict[str, Any]]) -> list[MeasurementRecord]:
    measurements: list[MeasurementRecord] = []
    for rec in rows:
        latency = rec.get("latency_ms")
        if latency is None:
            continue
        shape = rec.get("shape", {})
        normalized_shape: dict[str, tuple[int, ...]] = {}
        for key, value in shape.items():
            if isinstance(value, list):
                normalized_shape[key] = tuple(int(v) for v in value)
            elif isinstance(value, tuple):
                normalized_shape[key] = tuple(int(v) for v in value)
            else:
                normalized_shape[key] = (int(value),)
        measurements.append(
            MeasurementRecord(
                operator=rec.get("operator", ""),
                schedule_json=json.dumps(rec.get("schedule_trace", "")),
                original_tir=rec.get("original_tir", ""),
                scheduled_tir=rec.get("scheduled_tir", rec.get("original_tir", "")),
                workload_shape=normalized_shape,
                runtime_ms=float(latency),
                hardware_id=rec.get("device_type", ""),
                target=rec.get("target"),
                workload_key=rec.get("task_name"),
                hardware_features=None,
            )
        )
    return measurements


def build_graphs(rows: Iterable[dict[str, Any]], builder: TVMGraphBuilder) -> dict[str, Any]:
    graphs: dict[str, Any] = {}
    for rec in rows:
        cid = str(rec.get("candidate_id") or rec.get("schedule_trace"))
        if cid in graphs:
            continue
        tir_src = rec.get("scheduled_tir") or rec.get("original_tir")
        if not tir_src:
            continue
        try:
            mod = from_source(tir_src)
            graphs[cid] = builder.build(mod)
        except Exception:
            continue
    return graphs


def aggregate_features(graphs: dict[str, Any], encoder: GraphEncoder) -> dict[str, list[float]]:
    encoder.prime_feature_names(graphs.values())
    features: dict[str, list[float]] = {}
    for cid, graph in graphs.items():
        try:
            enc = encoder.encode(graph)
            tensor_enc = encoder.to_tensor_encoding(enc, device=torch.device("cpu"))
            if tensor_enc.node_features.numel() == 0:
                feats = torch.zeros(len(tensor_enc.feature_names) * 2, dtype=torch.float32)
            else:
                mean_feats = tensor_enc.node_features.mean(dim=0)
                max_feats = tensor_enc.node_features.max(dim=0).values
                feats = torch.cat([mean_feats, max_feats])
            features[cid] = feats.tolist()
        except Exception:
            continue
    return features


def evaluate_graphpy(
    splits: dict[str, list[dict[str, Any]]],
    max_pairs: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    train_rows = splits.get("train", [])
    measurements = to_measurements(train_rows)
    if not measurements:
        return {}

    cfg = TrainingConfig(
        max_pairs=max_pairs,
        epochs=epochs,
        pair_seed=seed,
        show_progress=False,
    )
    pipeline = TrainingPipeline(config=cfg)
    pair_count = pipeline.fit_measurements(measurements)
    metrics: dict[str, Any] = {"trained_pairs": pair_count}

    for split_name, rows in splits.items():
        if not rows:
            continue
        latencies = []
        scores = []
        for rec in rows:
            latency = rec.get("latency_ms")
            if latency is None:
                continue
            latencies.append(float(latency))
            tir_src = rec.get("scheduled_tir") or rec.get("original_tir")
            if not tir_src:
                continue
            try:
                graph = pipeline._build_graph(from_source(tir_src))
                scores.append(pipeline.model.predict(graph).score)
            except Exception:
                latencies.pop()
                continue
        if not latencies or not scores or len(latencies) != len(scores):
            continue
        metrics[split_name] = compute_ranking_metrics(latencies, scores, pred_higher_is_better=True).__dict__
    return metrics


def evaluate_xgb(splits: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    configure_torch_threads()
    builder = TVMGraphBuilder()
    encoder = GraphEncoder()
    all_rows: list[dict[str, Any]] = []
    for rows in splits.values():
        all_rows.extend(rows)
    graphs = build_graphs(all_rows, builder)
    features = aggregate_features(graphs, encoder)

    def rows_to_xy(rows: list[dict[str, Any]]):
        X: list[list[float]] = []
        y: list[float] = []
        for rec in rows:
            cid = str(rec.get("candidate_id") or rec.get("schedule_trace"))
            feats = features.get(cid)
            if feats is None:
                continue
            latency = rec.get("latency_ms")
            if latency is None:
                continue
            X.append(feats)
            y.append(float(latency))
        return X, y

    train_X, train_y = rows_to_xy(splits.get("train", []))
    if not train_X:
        return {}
    model = xgboost.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    model.fit(train_X, train_y)
    metrics: dict[str, Any] = {}
    for split_name, rows in splits.items():
        X, y_true = rows_to_xy(rows)
        if not X or not y_true:
            continue
        preds = model.predict(X)
        metrics[split_name] = compute_ranking_metrics(
            y_true, preds, pred_higher_is_better=False
        ).__dict__
    return metrics


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    splits: dict[str, list[dict[str, Any]]] = {
        name: load_split(root / f"{name}.jsonl", args.max_train if name == "train" and args.max_train else None)
        for name in ["train", "val", "test", "id_test", "ood_test"]
    }

    results: dict[str, Any] = {"config": vars(args)}
    if args.model in {"graph", "both"}:
        results["graph"] = evaluate_graphpy(splits, max_pairs=args.max_pairs, epochs=args.epochs, seed=args.seed)
    if args.model in {"xgb", "both"}:
        results["xgb"] = evaluate_xgb(splits, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote offline metrics to {out_path}")


if __name__ == "__main__":
    main()
