from tvm_cost_model.data.dataset_builder import MeasurementRecord
from tvm_cost_model.training.pair_sampling import _classify_delta, make_ranking_pairs, sample_ranking_pairs, Difficulty # type: ignore


def _mk(measurements: list[float]):
    return [MeasurementRecord(operator="gemm", schedule_json="{}", original_tir="", scheduled_tir="", workload_shape={}, runtime_ms=rt, hardware_id="hw") for rt in measurements]


def _record(
    runtime: float,
    operator: str = "gemm",
    workload_shape: dict[str, tuple[int, ...]] | None = None,
    target: str | None = "cuda",
    hardware_id: str = "hw",
) -> MeasurementRecord:
    return MeasurementRecord(
        operator=operator,
        schedule_json="{}",
        original_tir="",
        scheduled_tir="",
        workload_shape=workload_shape or {},
        runtime_ms=runtime,
        hardware_id=hardware_id,
        target=target,
    )


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


def test_make_ranking_pairs_only_within_same_group():
    group_a = [
        _record(1.0, operator="op_a", workload_shape={"n": (1,)}, target="cuda", hardware_id="hw0"),
        _record(2.0, operator="op_a", workload_shape={"n": (1,)}, target="cuda", hardware_id="hw0"),
    ]
    # Different shape and hardware_id; should not be paired with group_a.
    group_b = [
        _record(0.5, operator="op_a", workload_shape={"n": (2,)}, target="cuda", hardware_id="hw1"),
    ]
    pairs = make_ranking_pairs(group_a + group_b, easy_frac=0.3, hard_frac=0.05)
    assert all(
        pair.better.operator == pair.worse.operator
        and pair.better.workload_shape == pair.worse.workload_shape
        and pair.better.target == pair.worse.target
        and pair.better.hardware_id == pair.worse.hardware_id
        for pair in pairs
    )
    # Only the two records in group_a should form pairs.
    assert len(pairs) == 1


def test_sample_ranking_pairs_respects_operator_shape_target_and_hardware():
    group_a = [
        _record(1.0, operator="conv2d", workload_shape={"n": (1,), "c": (64,)}, target="cuda", hardware_id="gpu0"),
        _record(3.0, operator="conv2d", workload_shape={"n": (1,), "c": (64,)}, target="cuda", hardware_id="gpu0"),
    ]
    group_b = [
        _record(2.0, operator="conv2d", workload_shape={"n": (1,), "c": (64,)}, target="metal", hardware_id="gpu1"),
        _record(4.0, operator="conv2d", workload_shape={"n": (1,), "c": (64,)}, target="metal", hardware_id="gpu1"),
    ]
    pairs = sample_ranking_pairs(group_a + group_b, num_pairs=10, easy_frac=0.3, hard_frac=0.05, seed=42)
    assert pairs  # Should still produce pairs
    assert all(
        pair.better.operator == pair.worse.operator
        and pair.better.workload_shape == pair.worse.workload_shape
        and pair.better.target == pair.worse.target
        and pair.better.hardware_id == pair.worse.hardware_id
        for pair in pairs
    )
