"""Analyze online tuning logs and compute convergence metrics."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize online tuning logs.")
    parser.add_argument("--logs", nargs="+", required=True, help="JSONL logs or glob patterns.")
    parser.add_argument("--output", type=str, default="artifacts/benchmarks/online/summary.json", help="Output JSON path.")
    parser.add_argument(
        "--latency-targets",
        type=str,
        default="1.05,1.1",
        help="Comma-separated multipliers for measurement targets (e.g., 1.05=95%% of optimum).",
    )
    parser.add_argument("--milestones", type=str, default="16,32,64,128,256,512", help="Comma-separated measurement counts to sample curves at.")
    return parser.parse_args()


def load_logs(patterns: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if not matched and Path(pattern).exists():
            matched = [pattern]
        for path_str in matched:
            path = Path(path_str)
            if not path.exists():
                continue
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def summarize_group(rows: list[dict[str, Any]], milestones: list[int]) -> dict[str, Any]:
    rows = [r for r in rows if r.get("latency_ms") is not None]
    rows.sort(key=lambda r: r.get("measure_idx", 0))
    best = float("inf")
    curve = []
    latency_at_n: dict[int, float] = {}
    for r in rows:
        latency = float(r["latency_ms"])
        best = min(best, latency)
        curve.append({"measure_idx": r.get("measure_idx"), "best_latency_ms": best, "elapsed_sec": r.get("elapsed_sec")})
        if r.get("measure_idx") in milestones:
            latency_at_n[int(r["measure_idx"])] = best
    return {"curve": curve, "best_latency_ms": best, "latency_at_n": latency_at_n}


def main() -> None:
    args = parse_args()
    milestones = [int(x) for x in args.milestones.split(",") if x]
    targets = [float(x) for x in args.latency_targets.split(",") if x]
    logs = load_logs(args.logs)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in logs:
        workload = rec.get("workload_id") or rec.get("task_name") or rec.get("operator")
        cost_model = rec.get("cost_model", "unknown")
        grouped[(workload, cost_model)].append(rec)

    summaries: dict[str, Any] = {"groups": {}, "targets": targets, "milestones": milestones}
    workload_best: dict[str, float] = {}
    for (workload, cost_model), rows in grouped.items():
        summary = summarize_group(rows, milestones)
        summaries["groups"].setdefault(workload, {})[cost_model] = summary
        workload_best[workload] = min(workload_best.get(workload, float("inf")), summary["best_latency_ms"])

    # Measurement savings to target fractions of best overall per workload
    for workload, models in summaries["groups"].items():
        optimum = workload_best.get(workload, float("inf"))
        for cost_model, summary in models.items():
            curve = summary["curve"]
            measurements = {}
            for target_mult in targets:
                threshold = optimum * target_mult
                reached = next((pt["measure_idx"] for pt in curve if pt["best_latency_ms"] <= threshold), None)
                measurements[str(target_mult)] = reached
            summary["measurements_to_target"] = measurements

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote online summary to {out_path}")


if __name__ == "__main__":
    main()
