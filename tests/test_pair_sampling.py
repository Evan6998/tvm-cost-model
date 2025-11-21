from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.training.pair_sampling import _classify_delta, make_ranking_pairs # type: ignore


def _mk(measurements: list[float]):
    return [MeasurementRecord(operator="gemm", schedule_json="{}", tir="", workload_shape={}, runtime_ms=rt, hardware_id="hw") for rt in measurements]


def test_classify_delta_labels_easy_medium_hard():
    assert _classify_delta(15.0, easy_gap=10.0, hard_gap=2.0) == "easy"
    assert _classify_delta(1.0, easy_gap=10.0, hard_gap=2.0) == "hard"
    assert _classify_delta(5.0, easy_gap=10.0, hard_gap=2.0) == "medium"


def test_make_ranking_pairs_orders_by_runtime():
    records = _mk([1.0, 3.0, 12.0])
    pairs = make_ranking_pairs(records, easy_gap=10.0, hard_gap=2.0)

    assert len(pairs) == 3  # (1,3), (1,12), (3,12)
    # Ensure better/worse ordering is correct
    assert all(pair.better.runtime_ms < pair.worse.runtime_ms for pair in pairs)
    # Difficulty labels should include all difficulty classes
    assert {pair.difficulty for pair in pairs} == {"hard", "easy", "medium"}


def test_make_ranking_pairs_includes_medium_pairs():
    records = _mk([1.0, 3.0, 4.0])
    pairs = make_ranking_pairs(records, easy_gap=10.0, hard_gap=2.0)
    assert any(pair.difficulty == "medium" for pair in pairs)
