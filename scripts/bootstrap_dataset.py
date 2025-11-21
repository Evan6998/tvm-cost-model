"""CLI entry point for dataset bootstrapping."""

from __future__ import annotations

import argparse
from pathlib import Path

from tvm_cost_model.data.dataset_builder import (
    DatasetBuilder,
    SyntheticRuntimeEvaluator,
    SyntheticScheduleSampler,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap synthetic measurement data")
    parser.add_argument("--operator", default="gemm", help="Operator to sample")
    parser.add_argument("--batches", type=int, default=1, help="Number of sampler batches")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per batch")
    parser.add_argument("--hardware", default="ampere_a100", help="Hardware identifier")
    parser.add_argument(
        "--output-dir",
        default="artifacts/datasets",
        help="Directory to place dataset artifacts",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sampler = SyntheticScheduleSampler(seed=args.seed)
    evaluator = SyntheticRuntimeEvaluator(seed=args.seed)
    builder = DatasetBuilder(sampler, evaluator, output_dir)
    measurements = builder.collect(
        operator=args.operator,
        batches=args.batches,
        batch_size=args.batch_size,
        hardware_id=args.hardware,
    )
    artifact_name = f"{args.operator}_{args.hardware}".lower()
    artifact = builder.export(measurements, artifact_name=artifact_name)
    print(f"Wrote placeholder dataset to {artifact}")


if __name__ == "__main__":
    main()
