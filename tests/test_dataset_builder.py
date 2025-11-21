from pathlib import Path

from tvm_cost_model.data.dataset_builder import (
    DatasetBuilder,
    SyntheticRuntimeEvaluator,
    SyntheticScheduleSampler,
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
