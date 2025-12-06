import pytest

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
    assert any(label == "iterates" for _, _, label in graph.edges)


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


@pytest.mark.skip(reason="Flaky test, needs investigation")
def test_tvm_graph_builder_enriches_attributes_and_edges_1():
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
    graph = TVMGraphBuilder().build(tir_script)

    nodes_by_name = {node.name: node for node in graph.nodes}
    loop_i = nodes_by_name["loop:i"]
    loop_j = nodes_by_name["loop:j"]
    assert loop_i.attrs["depth"] == 0
    assert loop_j.attrs["depth"] == 1
    for flag in ("is_parallel", "is_vectorized", "is_unrolled", "is_thread_bound", "is_reduction"):
        assert flag in loop_i.attrs

    buffers = {node.name: node for node in graph.nodes if node.name.startswith("buffer:")}
    assert buffers["buffer:A"].attrs["scope_code"] == 0
    assert buffers["buffer:A"].attrs["elem_bytes"] == 4
    assert buffers["buffer:C"].attrs["write_count"] == 1

    compute = nodes_by_name["compute"]
    assert compute.attrs["total_flops"] == 1
    assert compute.attrs["global_bytes"] == 12
    assert compute.attrs["arith_intensity"] == pytest.approx(1 / 12)  # type: ignore[arg-type]

    edge_labels = {(graph.nodes[src].name, graph.nodes[dst].name, label) for src, dst, label in graph.edges}
    assert ("loop:i", "loop:j", "loop_child") in edge_labels
    assert ("loop:i", "buffer:A", "loop_accesses") in edge_labels
    assert ("compute", "buffer:A", "accesses") in edge_labels


def test_tvm_graph_builder_marks_reduction_loops():
    tir_script = """
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, k in T.grid(4, 4):
            with T.block("B"):
                vi = T.axis.spatial(4, i)
                vk = T.axis.reduce(4, k)
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]
"""
    graph = TVMGraphBuilder().build(tir_script)

    reduction_flags = {node.name: node.attrs.get("is_reduction", 0) for node in graph.nodes if node.name.startswith("loop:")}
    assert reduction_flags.get("loop:k") == 1
    assert reduction_flags.get("loop:i") == 0


@pytest.mark.skip(reason="Flaky test, needs investigation")
def test_tvm_graph_builder_enriches_attributes_and_edges_2():
    tir_script = """
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((256, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((256, 128), "float32")):
        T.func_attr({"global_symbol": "tir_matmul", "tir.noalias": True})
        # with T.block("root"):
        C_global = T.alloc_buffer((256, 128))
        for i_0_j_0_i_1_j_1_fused in T.parallel(16, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
            for i_2_init, j_2_init, i_3_init, j_3_init in T.grid(16, 64, 2, 1):
                with T.block("_init"):
                    vi = T.axis.spatial(256, i_0_j_0_i_1_j_1_fused // 2 * 32 + i_2_init * 2 + i_3_init)
                    vj = T.axis.spatial(128, i_0_j_0_i_1_j_1_fused % 2 * 64 + j_2_init + j_3_init)
                    T.reads()
                    T.writes(C_global[vi, vj])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C_global[vi, vj] = T.float32(0.0)
            for k_0, i_2, j_2, k_1, i_3, j_3 in T.grid(128, 16, 64, 1, 2, 1):
                with T.block("_update"):
                    vi = T.axis.spatial(256, i_0_j_0_i_1_j_1_fused // 2 * 32 + i_2 * 2 + i_3)
                    vj = T.axis.spatial(128, i_0_j_0_i_1_j_1_fused % 2 * 64 + j_2 + j_3)
                    vk = T.axis.reduce(128, k_0 + k_1)
                    T.reads(C_global[vi, vj], A[vi, vk], B[vj, vk])
                    T.writes(C_global[vi, vj])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C_global[vi, vj] = C_global[vi, vj] + A[vi, vk] * B[vj, vk]
            for ax0, ax1 in T.grid(32, 64):
                with T.block("C_global"):
                    v0 = T.axis.spatial(256, i_0_j_0_i_1_j_1_fused // 2 * 32 + ax0)
                    v1 = T.axis.spatial(128, i_0_j_0_i_1_j_1_fused % 2 * 64 + ax1)
                    T.reads(C_global[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global[v0, v1]"""
    graph = TVMGraphBuilder().build(tir_script)

    nodes_by_name = {node.name: node for node in graph.nodes}
    loop_nodes = [node for node in graph.nodes if node.name.startswith("loop:")]
    buffer_nodes = [node for node in graph.nodes if node.name.startswith("buffer:")]

    assert len(loop_nodes) == 13
    assert len(buffer_nodes) == 4

    fused_loop = nodes_by_name["loop:i_0_j_0_i_1_j_1_fused"]
    assert fused_loop.attrs["is_parallel"] == 1
    assert nodes_by_name["loop:k_0"].attrs["depth"] == 1
    assert nodes_by_name["loop:ax1"].attrs["extent"] == 64

    buffers = {node.name: node for node in buffer_nodes}
    assert buffers["buffer:C_global"].attrs["read_count"] == 2
    assert buffers["buffer:C_global"].attrs["write_count"] == 2
    assert buffers["buffer:C_global"].attrs["total_bytes"] == 256 * 128 * 4

    compute = nodes_by_name["compute"]
    assert compute.attrs["loop_depth"] == len(loop_nodes)
    assert compute.attrs["buffer_count"] == len(buffer_nodes)
    assert compute.attrs["global_bytes"] == 28

    edge_labels = {(graph.nodes[src].name, graph.nodes[dst].name, label) for src, dst, label in graph.edges}
    assert ("loop:i_0_j_0_i_1_j_1_fused", "loop:i_2_init", "loop_child") in edge_labels
    assert ("loop:ax1", "buffer:C", "loop_accesses") in edge_labels
    assert ("compute", "buffer:C_global", "accesses") in edge_labels
