"""CLI entry point for dataset bootstrapping."""

from __future__ import annotations

from pathlib import Path

from tvm_cost_model.data.dataset_builder import DatasetBuilder, ScheduleMeasurement


class DummySampler:
    def sample(self, operator: str, batch: int):
        yield from (f"{operator}_schedule_{i}" for i in range(batch))


def main() -> None:
    output_dir = Path("artifacts/datasets")
    builder = DatasetBuilder(DummySampler(), output_dir)
    measurements = builder.collect(operator="gemm", batches=1)
    artifact = builder.export(measurements)
    print(f"Wrote placeholder dataset to {artifact}")


if __name__ == "__main__":
    main()
