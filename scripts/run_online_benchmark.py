"""Grid driver to run MetaSchedule tuning across workloads/targets/cost models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online benchmark grid.")
    parser.add_argument("--config", type=str, default="configs/benchmark_workloads.yaml", help="Benchmark config YAML.")
    parser.add_argument("--workloads", type=str, default="", help="Comma-separated workload names to include.")
    parser.add_argument("--splits", type=str, default="train,id_test,ood_test", help="Comma-separated splits to include.")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated target names to include.")
    parser.add_argument("--cost-models", type=str, default="", help="Comma-separated cost models to include (graph,xgb).")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds to override config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--max-trials", type=int, default=0, help="Override max_trials from config.")
    parser.add_argument("--trials-per-iter", type=int, default=0, help="Override trials_per_iter from config.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    included_workloads = set([w for w in args.workloads.split(",") if w])
    included_targets = set([t for t in args.targets.split(",") if t])
    included_splits = set([s for s in args.splits.split(",") if s])
    cost_models = [m for m in (args.cost_models.split(",") if args.cost_models else cfg.get("online", {}).get("cost_models", [])) if m]
    seeds = [int(s) for s in args.seeds.split(",") if s] or cfg.get("online", {}).get("seeds", [0])

    online_cfg = cfg.get("online", {})
    max_trials = args.max_trials or online_cfg.get("max_trials", 256)
    trials_per_iter = args.trials_per_iter or online_cfg.get("trials_per_iter", 64)
    number = online_cfg.get("number", 5)
    repeat = online_cfg.get("repeat", 1)
    build_timeout = online_cfg.get("build_timeout", 30.0)
    runner_timeout = online_cfg.get("runner_timeout", 10.0)
    log_dir = Path(online_cfg.get("log_dir", "artifacts/benchmarks/online"))
    log_dir.mkdir(parents=True, exist_ok=True)

    targets_cfg = cfg.get("targets", {})
    workloads_cfg = cfg.get("workloads", [])
    for workload in workloads_cfg:
        if included_workloads and workload.get("name") not in included_workloads:
            continue
        if included_splits and workload.get("split", "train") not in included_splits:
            continue
        shape_json = json.dumps(workload.get("shape", {}))
        dtype = workload.get("dtype", "float32")
        for target_name, target_conf in targets_cfg.items():
            if included_targets and target_name not in included_targets:
                continue
            for cost_model in cost_models:
                for seed in seeds:
                    log_path = (
                        log_dir
                        / workload.get("name", workload.get("operator", ""))
                        / target_name
                        / f"{cost_model}_seed{seed}.jsonl"
                    )
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        "scripts/run_metaschedule_tuning.py",
                        "--operator",
                        workload["operator"],
                        "--shape",
                        shape_json,
                        "--target",
                        target_conf["target"],
                        "--task-name",
                        workload.get("name", workload["operator"]),
                        "--max-trials",
                        str(max_trials),
                        "--trials-per-iter",
                        str(trials_per_iter),
                        "--number",
                        str(number),
                        "--repeat",
                        str(repeat),
                        "--build-timeout",
                        str(build_timeout),
                        "--runner-timeout",
                        str(runner_timeout),
                        "--cost-model",
                        cost_model,
                        "--dtype",
                        dtype,
                        "--seed",
                        str(seed),
                        "--log-json-path",
                        str(log_path),
                        "--work-dir",
                        target_conf.get("work_dir", "artifacts/tuning"),
                    ]
                    if target_conf.get("device_type"):
                        cmd += ["--device-type", target_conf["device_type"]]
                    if args.dry_run:
                        print(" ".join(cmd))
                    else:
                        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
