from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.training.pair_sampling import _classify_delta, make_ranking_pairs, sample_ranking_pairs, Difficulty # type: ignore


def _mk(measurements: list[float]):
    return [MeasurementRecord(operator="gemm", schedule_json="{}", original_tir="", scheduled_tir="", workload_shape={}, runtime_ms=rt, hardware_id="hw") for rt in measurements]


def test_classify_delta_labels_easy_medium_hard():
    assert _classify_delta(5.0, better_runtime=10.0, easy_frac=0.3, hard_frac=0.05) == Difficulty.EASY  # 50% slower
    assert _classify_delta(0.2, better_runtime=10.0, easy_frac=0.3, hard_frac=0.05) == Difficulty.HARD  # 2% slower
    assert _classify_delta(2.0, better_runtime=10.0, easy_frac=0.3, hard_frac=0.05) == Difficulty.MEDIUM  # 20% slower


def test_make_ranking_pairs_orders_by_runtime():
    records = _mk([10.0, 10.3, 13.0])
    pairs = make_ranking_pairs(records, easy_frac=0.3, hard_frac=0.05)

    assert len(pairs) == 3  # all pairwise combinations
    # Ensure better/worse ordering is correct
    assert all(pair.better.runtime_ms < pair.worse.runtime_ms for pair in pairs)
    # Difficulty labels should include all difficulty classes
    assert {pair.difficulty for pair in pairs} == {Difficulty.HARD, Difficulty.EASY, Difficulty.MEDIUM}


def test_make_ranking_pairs_includes_medium_pairs():
    records = _mk([10.0, 11.5, 15.0])
    pairs = make_ranking_pairs(records, easy_frac=0.3, hard_frac=0.05)
    assert any(pair.difficulty == Difficulty.MEDIUM for pair in pairs)


def test_sample_ranking_pairs_limits_size():
    records = _mk([1.0, 2.0, 3.0, 4.0])
    pairs = sample_ranking_pairs(records, num_pairs=3, easy_frac=0.3, hard_frac=0.05, seed=0)
    assert len(pairs) == 3
    assert all(pair.better.runtime_ms < pair.worse.runtime_ms for pair in pairs)


def test_sample_ranking_pairs_discards_near_ties():
    # 4% gap should be discarded by the sampler's hardcoded 5% filter
    records = _mk([100.0, 104.0, 120.0])
    pairs = sample_ranking_pairs(records, num_pairs=10, easy_frac=0.3, hard_frac=0.05, seed=1)
    assert all((p.worse.runtime_ms - p.better.runtime_ms) / p.better.runtime_ms >= 0.01 for p in pairs)
