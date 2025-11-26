from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.features.graph_builder import GraphBuilder
from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.training.ranking_dataset import build_encoded_pairs


def _mk(records: list[float]):
    return [
        MeasurementRecord(
            operator="gemm",
            schedule_json="{}",
            original_tir=f"for i in range({extent}): c[i] = a[i]\n",
            scheduled_tir=f"for i in range({extent}): c[i] = a[i]\n",
            workload_shape={},
            runtime_ms=extent,
            hardware_id="hw",
        )
        for extent in records
    ]


def test_build_encoded_pairs_matches_pair_count():
    measurements = _mk([1.0, 3.0, 6.0])
    pairs = build_encoded_pairs(measurements, builder=GraphBuilder(), encoder=GraphEncoder())
    assert len(pairs) == 3  # all pairwise combinations respecting order
    assert all(p.better.node_features for p in pairs)
    assert all(p.worse.node_features for p in pairs)
