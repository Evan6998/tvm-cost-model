import math

from tvm_cost_model.features.graph_builder import GraphNode, ProgramGraph
from tvm_cost_model.features.graph_encoder import GraphEncoder


def test_graph_encoder_assigns_stable_node_and_edge_ids():
    graph = ProgramGraph(
        nodes=[
            GraphNode(name="loop:i", attrs={"extent": 8}),
            GraphNode(name="buffer:a", attrs={"name_len": 1}),
        ],
        edges=[(1, 0, "accesses")],
    )
    encoder = GraphEncoder()
    encoding_a = encoder.encode(graph)
    encoding_b = encoder.encode(graph)

    assert encoding_a.node_types == encoding_b.node_types
    assert encoding_a.edge_types == encoding_b.edge_types
    assert encoding_a.feature_names == encoding_b.feature_names


def test_graph_encoder_aligns_features_by_name():
    graph = ProgramGraph(
        nodes=[
            GraphNode(name="loop:j", attrs={"extent": 4, "other": 1.0}),
            GraphNode(name="compute", attrs={"loop_depth": 2}),
        ],
        edges=[],
    )
    encoding = GraphEncoder().encode(graph)

    # Ensure every node feature vector has the same length.
    feature_lengths = {len(vec) for vec in encoding.node_features}
    assert feature_lengths == {len(encoding.feature_names)}


def test_graph_encoder_emits_derived_features_and_values():
    graph = ProgramGraph(
        nodes=[
            GraphNode(name="loop:i", attrs={"extent": 9, "depth": 2, "is_parallel": 1}),
            GraphNode(
                name="buffer:a",
                attrs={"elem_bytes": 4, "total_bytes": 400, "read_count": 3, "write_count": 1},
            ),
            GraphNode(name="compute", attrs={"loop_depth": 2, "total_extent": 18, "total_flops": 100}),
        ],
        edges=[(2, 0, "iterates"), (2, 1, "accesses"), (0, 1, "writes")],
    )

    encoding = GraphEncoder().encode(graph)
    feature_names = encoding.feature_names
    for name in [
        "extent",
        "total_bytes",
        "loop_depth",
        "log1p:extent",
        "log1p:total_bytes",
        "log1p:total_extent",
        "log1p:total_flops",
        "loop:depth_x_log_extent",
        "loop:parallel_log_extent",
        "buffer:traffic_bytes",
        "buffer:traffic_ratio",
        "deg_in",
        "deg_out",
        "deg_total",
        "deg_in:iterates",
        "deg_out:iterates",
        "deg_in:accesses",
        "deg_out:writes",
    ]:
        assert name in feature_names

    def value(node_idx: int, feature: str) -> float:
        return encoding.node_features[node_idx][feature_names.index(feature)]

    log_extent = math.log1p(9)
    assert math.isclose(value(0, "log1p:extent"), log_extent)
    assert math.isclose(value(0, "loop:depth_x_log_extent"), 2 * log_extent)
    assert math.isclose(value(0, "loop:parallel_log_extent"), log_extent)

    buffer_traffic_bytes = 16.0
    assert math.isclose(value(1, "buffer:traffic_bytes"), buffer_traffic_bytes)
    assert math.isclose(value(1, "buffer:traffic_ratio"), buffer_traffic_bytes / 400.0)
    assert math.isclose(value(1, "log1p:total_bytes"), math.log1p(400))

    # Degree features
    assert value(0, "deg_in") == 1.0
    assert value(0, "deg_out") == 1.0
    assert value(0, "deg_total") == 2.0
    assert value(0, "deg_in:iterates") == 1.0
    assert value(0, "deg_out:writes") == 1.0
    assert value(2, "deg_out:iterates") == 1.0
    assert value(2, "deg_out:accesses") == 1.0
