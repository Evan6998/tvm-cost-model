"""Aggregate offline benchmark metrics into Markdown/CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize offline metrics.")
    parser.add_argument("--metrics", nargs="+", default=["artifacts/benchmarks/offline/metrics.json"], help="Metrics JSON files.")
    parser.add_argument("--markdown", type=str, default="artifacts/benchmarks/offline/summary.md", help="Output Markdown path.")
    parser.add_argument("--csv", type=str, default="artifacts/benchmarks/offline/summary.csv", help="Output CSV path.")
    return parser.parse_args()


def load_metrics(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    loaded: list[tuple[str, dict[str, Any]]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        loaded.append((path.stem, data))
    return loaded


def flatten_metrics(model: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, vals in metrics.items():
        if split in {"config", "trained_pairs"}:
            continue
        if not isinstance(vals, dict):
            continue
        topk = vals.get("topk_recall", {})
        rows.append(
            {
                "model": model,
                "split": split,
                "pairwise_accuracy": vals.get("pairwise_accuracy"),
                "spearman": vals.get("spearman"),
                "kendall": vals.get("kendall"),
                "recall@1": topk.get(1),
                "recall@5": topk.get(5),
                "recall@10": topk.get(10),
                "recall@20": topk.get(20),
            }
        )
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Split", "PairwiseAcc", "Spearman", "Kendall", "R@1", "R@5", "R@10", "R@20"]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("model")),
                        str(row.get("split")),
                        f"{row.get('pairwise_accuracy', float('nan')):.3f}" if row.get("pairwise_accuracy") is not None else "nan",
                        f"{row.get('spearman', float('nan')):.3f}" if row.get("spearman") is not None else "nan",
                        f"{row.get('kendall', float('nan')):.3f}" if row.get("kendall") is not None else "nan",
                        f"{row.get('recall@1', float('nan')):.3f}" if row.get("recall@1") is not None else "nan",
                        f"{row.get('recall@5', float('nan')):.3f}" if row.get("recall@5") is not None else "nan",
                        f"{row.get('recall@10', float('nan')):.3f}" if row.get("recall@10") is not None else "nan",
                        f"{row.get('recall@20', float('nan')):.3f}" if row.get("recall@20") is not None else "nan",
                    ]
                )
                + " |\n"
            )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model", "split", "pairwise_accuracy", "spearman", "kendall", "recall@1", "recall@5", "recall@10", "recall@20"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    loaded = load_metrics(args.metrics)
    rows: list[dict[str, Any]] = []
    for name, metrics in loaded:
        if "graph" in metrics:
            rows.extend(flatten_metrics("graph", metrics["graph"]))
        if "xgb" in metrics:
            rows.extend(flatten_metrics("xgb", metrics["xgb"]))
    md_path = Path(args.markdown)
    csv_path = Path(args.csv)
    write_markdown(rows, md_path)
    write_csv(rows, csv_path)
    print(f"Wrote summaries to {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
