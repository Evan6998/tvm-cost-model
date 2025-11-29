from pathlib import Path

from tvm_cost_model.data.dataset_builder import (
    DatasetBuilder,
    SyntheticRuntimeEvaluator,
    SyntheticScheduleSampler,
    load_measurement_records,
)


def test_dataset_builder_writes_parquet(tmp_path: Path) -> None:
    builder = DatasetBuilder(
        sampler=SyntheticScheduleSampler(seed=1),
        evaluator=SyntheticRuntimeEvaluator(seed=1),
        output_dir=tmp_path,
    )
    records = builder.collect(operator="gemm", batches=1, batch_size=4, hardware_id="ada")
    artifact = builder.export(records, artifact_name="test")
    assert artifact.exists()
    assert artifact.suffix == ".parquet"


def test_load_measurement_records_round_trips(tmp_path: Path) -> None:
    builder = DatasetBuilder(
        sampler=SyntheticScheduleSampler(seed=2),
        evaluator=SyntheticRuntimeEvaluator(seed=2),
        output_dir=tmp_path,
    )
    records = builder.collect(operator="gemm", batches=1, batch_size=3, hardware_id="ada")
    artifact = builder.export(records, artifact_name="load_test")
    loaded = load_measurement_records(artifact)
    assert len(loaded) == len(records)
    assert loaded[0].operator == "gemm"
    assert loaded[0].workload_shape
