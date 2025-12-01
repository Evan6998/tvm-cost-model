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
    expected_names = sorted(
        [
            "depth",
            "elem_bytes",
            "extent",
            "is_parallel",
            "loop_depth",
            "read_count",
            "total_bytes",
            "total_extent",
            "total_flops",
            "write_count",
        ]
    )
    assert encoding.feature_names == expected_names

    def value(node_idx: int, feature: str) -> float:
        idx = encoding.feature_names.index(feature)
        return encoding.node_features[node_idx][idx]

    # Log-scaled fields
    assert math.isclose(value(0, "extent"), math.log1p(9))
    assert math.isclose(value(1, "total_bytes"), math.log1p(400))
    assert math.isclose(value(1, "read_count"), math.log1p(3))
    assert math.isclose(value(1, "write_count"), math.log1p(1))
    assert math.isclose(value(2, "total_extent"), math.log1p(18))
    assert math.isclose(value(2, "total_flops"), math.log1p(100))

    # Raw fields
    assert value(0, "depth") == 2.0
    assert value(0, "is_parallel") == 1.0
    assert value(1, "elem_bytes") == 4.0
    assert value(2, "loop_depth") == 2.0

    # Missing attributes fall back to zero
    assert value(0, "total_flops") == 0.0
    assert value(2, "extent") == 0.0
