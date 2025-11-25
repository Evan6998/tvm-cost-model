from tvm_cost_model.features.tvm_graph_builder import TVMGraphBuilder


def test_tvm_graph_builder_extracts_loops_and_buffers():
    tir_script = """
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j in T.grid(16, 16):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]
"""
    builder = TVMGraphBuilder()
    graph = builder.build(tir_script)

    loop_nodes = [n for n in graph.nodes if n.name.startswith("loop:")]
    buffer_nodes = [n for n in graph.nodes if n.name.startswith("buffer:")]

    assert len(loop_nodes) == 2
    assert {n.name for n in buffer_nodes} == {"buffer:A", "buffer:B", "buffer:C"}


def test_tvm_graph_builder_handles_empty_module():
    builder = TVMGraphBuilder()
    graph = builder.build(
        """
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main"})
        T.evaluate(0)
"""
    )
    assert graph.nodes, "Graph should have at least compute node"
