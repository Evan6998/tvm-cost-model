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
