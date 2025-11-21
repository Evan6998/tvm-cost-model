from tvm_cost_model.features.graph_builder import GraphBuilder


def test_graph_builder_is_order_invariant_for_loops():
    builder = GraphBuilder()
    tir_a = """
for i in range(128):
    for j in range(64):
        c[i, j] = a[i, j]
"""
    tir_b = """
for j in range(64):
    for i in range(128):
        c[i, j] = a[i, j]
"""
    graph_a = builder.build(tir_a)
    graph_b = builder.build(tir_b)

    loop_extents_a = sorted([node.attrs["extent"] for node in graph_a.nodes if node.name.startswith("loop:")])
    loop_extents_b = sorted([node.attrs["extent"] for node in graph_b.nodes if node.name.startswith("loop:")])

    assert loop_extents_a == loop_extents_b
    assert len(graph_a.edges) == len(graph_b.edges)


def test_graph_builder_extracts_buffers_and_compute_node():
    builder = GraphBuilder()
    tir = """
for i in range(8):
    for k in range(4):
        c[i] = a[i] + b[k]
"""
    graph = builder.build(tir)
    buffer_nodes = [node for node in graph.nodes if node.name.startswith("buffer:")]
    compute_nodes = [node for node in graph.nodes if node.name == "compute"]

    assert {node.name for node in buffer_nodes} == {"buffer:a", "buffer:b", "buffer:c"}
    assert compute_nodes, "compute node should be present"
    assert any(label == "accesses" for _, _, label in graph.edges)
