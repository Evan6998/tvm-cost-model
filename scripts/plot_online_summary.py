"""Visualize online tuning summary curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot online tuning summaries.")
    parser.add_argument(
        "--summary",
        type=str,
        default="artifacts/benchmarks/online/summary.json",
        help="Path to the summary JSON produced by analyze_online_tuning.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/benchmarks/online/figures",
        help="Directory to write plot PNGs.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name)
    return safe.strip("_") or "workload"


def _points_from_curve(curve: Iterable[dict[str, Any]]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for pt in curve:
        idx = pt.get("measure_idx")
        latency = pt.get("best_latency_ms")
        if idx is None or latency is None:
            continue
        points.append((int(idx), float(latency)))
    return points


def plot_workload(
    workload: str,
    models: dict[str, Any],
    targets: list[float],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    if not models:
        return None

    best_overall = min(
        (m.get("best_latency_ms", float("inf")) for m in models.values()),
        default=float("inf"),
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    target_labels_added = False
    wrote_any = False

    for cost_model, summary in sorted(models.items()):
        points = _points_from_curve(summary.get("curve", []))
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=cost_model)
        wrote_any = True

        mt = summary.get("measurements_to_target", {}) or {}
        for tgt in targets:
            meas = mt.get(str(tgt))
            if meas is None:
                continue
            y_at_meas = next((lat for idx, lat in points if idx == meas), None)
            y_mark = y_at_meas if y_at_meas is not None else min(ys)
            ax.scatter([meas], [y_mark], s=30, zorder=5)
            ax.text(meas, y_mark, f"{cost_model}@{tgt}x", fontsize=8, ha="left", va="bottom")

    if best_overall != float("inf"):
        for tgt in targets:
            ax.axhline(best_overall * tgt, color="gray", linestyle="--", linewidth=1, label=None if target_labels_added else "target band")
            target_labels_added = True

    if not wrote_any:
        plt.close(fig)
        return None

    ax.set_title(workload)
    ax.set_xlabel("Measurements")
    ax.set_ylabel("Best latency (ms)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sanitize_filename(workload)}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = load_summary(summary_path)
    groups = summary.get("groups", {})
    targets = summary.get("targets", [])

    output_dir = Path(args.output_dir)
    written: list[Path] = []
    for workload, models in groups.items():
        out_path = plot_workload(workload, models, targets, output_dir, dpi=args.dpi)
        if out_path:
            written.append(out_path)

    if written:
        print(f"Wrote {len(written)} figure(s) to {output_dir}")
    else:
        print("No plots were generated; check that the summary contains curve data.")


if __name__ == "__main__":
    main()
