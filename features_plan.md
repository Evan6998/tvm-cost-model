# TVM-Based Graph Feature Extraction Roadmap

This document describes the next stages for evolving our feature extraction pipeline from TIR to graph-based encodings suitable for GNN cost models.

The roadmap has **two main phases**:

1. **Enhance `TVMGraphBuilder` while preserving the current public API**  
2. **Upgrade the graph structure from a star-shaped topology to a more faithful loop/dataflow graph**

The plan is written for an AI agent (or future contributor) implementing these steps. All tasks must respect the existing `ProgramGraph` and `GraphEncoding` interfaces.

---

## 0. Current State (Baseline)

### 0.1. Key Types

Located in `src/tvm_cost_model/features/graph_builder.py`:

- `GraphNode`
  - Fields:
    - `name: str` (e.g., `"loop:i"`, `"buffer:A"`, `"compute"`)
    - `attrs: Dict[str, float | int]` (numeric attributes only)
- `ProgramGraph`
  - Fields:
    - `nodes: List[GraphNode]`
    - `edges: List[tuple[int, int, str]]` where each edge is `(src_idx, dst_idx, label)`
- `LoopInfo`
  - Fields:
    - `var: str`
    - `extent: int`

Located in `src/tvm_cost_model/features/graph_encoder.py`:

- `GraphEncoding`
  - `node_features: list[list[float]]`
  - `node_types: list[int]` (integer IDs derived from node name prefix, e.g., `"loop"`, `"buffer"`, `"compute"`)
  - `edge_index: list[tuple[int, int]]`
  - `edge_types: list[int]` (integer IDs for edge labels)
  - `feature_names: list[str]`
- `GraphEncoder`
  - Maintains vocabularies:
    - `node_type_to_id: dict[str, int]`
    - `edge_type_to_id: dict[str, int]`
  - Collects the union of all attribute keys across nodes as `feature_names`
  - Encodes each `GraphNode.attrs` as a dense numeric feature vector in a fixed feature order

### 0.2. Current TVM-backed Graph Builder (updated)

Located in `src/tvm_cost_model/features/tvm_graph_builder.py`:

- `_TIRVisitor`
  - Recursive walk that keeps a loop stack and captures `_LoopCapture(var, extent, depth, for_kind, parent)`.
  - Tracks `loop_buffer_accesses`, `reduction_loop_vars`, and `total_flops` (counts `Add/Sub/Mul/Div`/`Call`/`Min`/`Max` nodes).
  - Collects per-buffer stats: `scope_code`, `elem_bytes`, `total_bytes`, `read_count`, `write_count`.
- `TVMGraphBuilder.build(tir_script: str) -> ProgramGraph`
  - Parses TIR script via `tvm.script.from_source` and runs `_TIRVisitor`.
  - Nodes:
    - Loop nodes: `extent`, `depth`, `is_parallel`, `is_vectorized`, `is_unrolled`, `is_thread_bound`, `is_reduction`.
    - Buffer nodes: `name_len`, `scope_code`, `elem_bytes`, `total_bytes`, `read_count`, `write_count`.
    - Compute node: existing aggregates plus `total_flops`, `global_bytes`, `shared_bytes`, `local_bytes`, `other_bytes`, `arith_intensity`.
  - Edges:
    - `compute -> loop` (`"iterates"`)
    - `compute -> buffer` (`"accesses"`)
    - `loop -> loop` (`"loop_child"`)
    - `loop -> buffer` (`"loop_accesses"`)

**Important constraint:**  
The `ProgramGraph` and `GraphEncoder` APIs **must not break**. We will only extend attributes and add graph structure in a backward-compatible way.

---

## Phase 1: Enhance `TVMGraphBuilder` (Attributes Only)

**Goal:**  
Enrich node attributes with performance-related information while **keeping the existing graph topology** and public interfaces unchanged.

All enhancements happen inside `tvm_graph_builder.py` (and potentially minor, backward-compatible additions to `LoopInfo`).

### 1.1. Loop-Level Attributes

**Objective:**  
For each loop node (`loop:var`), augment attributes beyond `extent` to capture:

- Loop nesting depth
- Reduction vs. spatial role
- Parallelization / vectorization / unrolling / thread binding

#### 1.1.1. Implementation Strategy

1. **Replace or wrap the existing visitor to track nesting depth.**
   - Instead of relying solely on `stmt_functor.post_order_visit`, implement a small recursive visitor that:
     - Maintains a `current_depth: int`
     - On entering a `tir.For`, increments depth; on leaving, decrements depth.
   - For each `tir.For`:
     - Capture:
       - `loop_var: str(stmt.loop_var)`
       - `extent: int(_to_int(stmt.extent))`
       - `depth: int(current_depth)`
       - `for_kind: ForKind` (e.g., `kSerial`, `kParallel`, `kVectorized`, `kUnrolled`, `kThreadBinding`)

2. **Record loop metadata in `_LoopCapture`.**
   - Extend `_LoopCapture` to include:
     - `var: str`
     - `extent: int`
     - `depth: int`
     - `for_kind: int` (encode `tir.ForKind` as an integer)
   - Keep this change internal to `tvm_graph_builder.py`. No changes are required to the external `LoopInfo` type yet.

3. **Classify loops via boolean flags.**
   - When constructing `GraphNode` for loops, attach additional numeric attributes:
     - `"depth": depth`
     - `"is_parallel": 0 or 1`
     - `"is_vectorized": 0 or 1`
     - `"is_unrolled": 0 or 1`
     - `"is_thread_bound": 0 or 1`
   - These attributes should be derived from `for_kind`:
     - `kParallel` → `is_parallel = 1`
     - `kVectorized` → `is_vectorized = 1`
     - `kUnrolled` → `is_unrolled = 1`
     - `kThreadBinding` → `is_thread_bound = 1`
   - All new attributes are numeric (int/float) and thus compatible with `GraphEncoder`.

4. **(Optional but recommended) Reduction vs. spatial loops.**
   - Use TIR `Block` information to detect reduction axes:
     - Inspect each `tir.Block`:
       - Look at `block.iter_vars` or `block.reduce_iter_vars` / iteration types (implementation may differ based on TVM version).
       - Map loop variables to iter types (data-parallel vs. reduction).
   - For each loop, set:
     - `"is_reduction": 1` if its loop var is associated with a reduction axis in any block.
     - Otherwise `"is_reduction": 0`.

#### 1.1.2. Acceptance Criteria

- `TVMGraphBuilder.build(...)` still returns a `ProgramGraph` with the same node/edge topology (same number of nodes and edges as before).
- Each `loop:` node has at least the following attributes:
  - `extent` (existing)
  - `depth`
  - `is_parallel`
  - `is_vectorized`
  - `is_unrolled`
  - `is_thread_bound`
  - `is_reduction` (if implemented)
- `GraphEncoder.encode(...)` continues to work without modification, and the new attributes appear as additional feature dimensions.

---

### 1.2. Buffer-Level Attributes

**Objective:**  
For each buffer node (`buffer:name`), capture memory-related properties:

- Memory scope (e.g., global/shared/local/etc.)
- Element and total size
- Approximate read/write counts

#### 1.2.1. Implementation Strategy

1. **Track buffer definitions and scopes.**
   - While visiting the TIR:
     - For each `tir.Buffer` encountered (in `BufferLoad`, `BufferStore`, `Block.reads/writes`):
       - Retrieve `buffer.scope()` (string) and map it to an integer code, e.g.:
         - `"global"` → `0`
         - `"shared"` → `1`
         - `"local"` → `2`
         - `"warp"` → `3`
         - unknown / other → `-1`
     - Store a mapping:
       - `buffer_name -> scope_code`

2. **Estimate element size and total buffer size.**
   - For each `tir.Buffer`:
     - Element bytes:
       - `elem_bytes = buffer.dtype.bits // 8`
     - Number of elements:
       - Compute product of `_to_int(dim)` for each `dim` in `buffer.shape`.
       - If any dimension cannot be converted, treat size as `-1`.
     - Total bytes:
       - `total_bytes = elem_bytes * num_elements` (or `-1` if unknown).
   - Store:
     - `buffer_name -> elem_bytes`
     - `buffer_name -> total_bytes`

3. **Count read/write accesses per buffer.**
   - During traversal:
     - On `tir.BufferLoad` for buffer `B`:
       - Increment `read_count[B]` by 1.
     - On `tir.BufferStore` for buffer `B`:
       - Increment `write_count[B]` by 1.
     - Optionally, also account for `Block.reads` and `Block.writes` by incrementing coarse counts (but avoid double counting if not needed).
   - Initialize counts to 0 if a buffer is first seen.

4. **Attach attributes to buffer nodes.**
   - When creating `GraphNode` for buffers (currently `attrs={"name_len": len(name)}`), extend with:
     - `"name_len": len(name)` (existing)
     - `"scope_code": scope_code`
     - `"elem_bytes": elem_bytes`
     - `"total_bytes": total_bytes`
     - `"read_count": read_count`
     - `"write_count": write_count`

#### 1.2.2. Acceptance Criteria

- Buffer nodes continue to be named `buffer:<name>`.
- For each buffer node, the attributes include:
  - `name_len` (existing)
  - `scope_code`
  - `elem_bytes`
  - `total_bytes`
  - `read_count`
  - `write_count`
- All attributes are numeric (int or float), and encoding succeeds with the existing `GraphEncoder`.

---

### 1.3. Compute Node Aggregated Attributes

**Objective:**  
Use the compute node as a global feature aggregator for FLOPs and bytes per memory level.

#### 1.3.1. Implementation Strategy

1. **Approximate FLOPs.**
   - For each `tir.Block` representing compute:
     - Traverse its body and count arithmetic operations:
       - `tir.Add`, `tir.Sub`, `tir.Mul`, `tir.Div`, `tir.FMA`, etc.
       - Optionally distinguish between integer and floating computations, but a single FLOP count is enough for now.
   - Accumulate totals across all blocks in the function.
   - Store the final sum as `total_flops` (int).

2. **Approximate bytes per memory scope.**
   - For each buffer access:
     - Use `scope_code` (from buffer-level attributes) and `elem_bytes`.
     - For each read or write:
       - Increment `global_bytes`, `shared_bytes`, `local_bytes`, etc. according to the buffer’s scope.
   - At the end, compute:
     - `global_bytes`
     - `shared_bytes`
     - `local_bytes`
     - `other_bytes` (if needed)
   - All counts are in bytes (int or float).

3. **Arithmetic intensity.**
   - Compute:
     - `arith_intensity = total_flops / max(global_bytes, 1)` to avoid division by zero.
   - Attach this as a float attribute.

4. **Extend compute node attributes.**
   - Current attributes:
     - `"loop_depth": len(loop_nodes)`
     - `"buffer_count": len(buffer_nodes)`
     - `"total_extent": sum(loop.attrs["extent"] for loop in loop_nodes)`
   - Extend with:
     - `"total_flops"`
     - `"global_bytes"`
     - `"shared_bytes"`
     - `"local_bytes"`
     - `"arith_intensity"`

#### 1.3.2. Acceptance Criteria

- The compute node continues to be named `"compute"`.
- In addition to existing attributes, it now has:
  - `total_flops`
  - `global_bytes`
  - `shared_bytes`
  - `local_bytes`
  - `arith_intensity`
- All attributes numeric; `GraphEncoder` remains unchanged and successfully encodes graphs.

---

## Phase 2: Upgrade Graph Structure (From Star to Loop/Dataflow Graph)

**Goal:**  
Preserve Phase 1 attributes and APIs, but enrich graph topology to better reflect loop nesting and data access patterns.

The focus is on **adding more edges and potentially more edge labels**. We will **not** remove existing edges in the first iteration to keep backward compatibility.

### 2.1. Loop Nest Edges

**Objective:**  
Model the hierarchical structure of loops via parent–child edges.

#### 2.1.1. Implementation Strategy

1. **Track parent–child relationships in the visitor.**
   - While traversing `tir.For` nodes:
     - Maintain a stack of active loops:
       - On entering a loop:
         - Let `parent_loop` be the top of the stack (if any).
         - Record `(child_var, parent_var)` or directly `(child_loop_id, parent_loop_id)` in a structure.
         - Push current loop onto the stack.
       - On leaving the loop:
         - Pop from the stack.
   - Associate a stable index for each loop (e.g., index in `self.loops`).

2. **Create loop nodes as before, but now also add loop–loop edges.**
   - After constructing `loop_nodes` (list of `GraphNode`), create edges of the form:
     - `(parent_idx, child_idx, "loop_child")`
       - `parent_idx`: index of parent loop node in `nodes`
       - `child_idx`: index of child loop node in `nodes`
       - `"loop_child"`: new edge label

3. **Keep existing compute → loop edges**
   - Do **not** remove `compute -> loop` `"iterates"` edges in the initial iteration to avoid breaking downstream assumptions.
   - These edges now co-exist with more detailed loop–loop edges.

#### 2.1.2. Acceptance Criteria

- Graphs now include additional edges labeled `"loop_child"` connecting parent and child loops.
- Existing edges (compute → loop `"iterates"`) remain intact.
- `GraphEncoder` continues to encode edges; the new edge label `"loop_child"` gets a new `edge_type_id` automatically.

---

### 2.2. Loop–Buffer Access Edges

**Objective:**  
Capture which loops actually touch which buffers, rather than only connecting buffers to the compute node.

#### 2.2.1. Implementation Strategy

1. **Record buffer accesses with active loops.**
   - During traversal:
     - Maintain a stack of active loop variables or loop IDs, as in 2.1.
     - On each `BufferLoad` or `BufferStore` for buffer `B`:
       - For each active loop in the stack:
         - Record a relation `(loop_id, buffer_name)` in a set or mapping to avoid duplicate edges later.

2. **Construct loop–buffer edges.**
   - After building the list of loop and buffer nodes:
     - For each recorded `(loop_id, buffer_name)`:
       - Map `loop_id` to its node index `loop_idx`.
       - Map `buffer_name` to its `buffer_idx` in `nodes`.
       - Add an edge: `(loop_idx, buffer_idx, "accesses")`.
   - This edge label `"accesses"` may conflict with the existing compute → buffer edges; to avoid confusion, there are two options:
     - Option A (recommended):
       - Use a new label, e.g. `"loop_accesses"`, for loop–buffer edges.
       - Keep `"accesses"` for compute → buffer edges for backward compatibility.
     - Option B:
       - Reuse `"accesses"` for both compute → buffer and loop → buffer edges and rely on source/target node type to distinguish.  
       - **If implementing for an external agent, use Option A to keep semantics explicit.**

3. **Keep existing compute → buffer edges (initially).**
   - As with loops, do **not** remove compute → buffer `"accesses"` edges in the first iteration.
   - This ensures older models or scripts that expect a star-shaped graph still function.

#### 2.2.2. Acceptance Criteria

- Graphs now contain edges from loop nodes to buffer nodes, labeled (preferably) `"loop_accesses"`.
- Existing compute → buffer `"accesses"` edges are still present.
- `GraphEncoder.encode` continues to work; `"loop_accesses"` is assigned a new edge type ID.

---

### 2.3. Compute Node as Global Aggregator

**Objective:**  
Clarify the role of the compute node in the enriched graph.

#### 2.3.1. Implementation Strategy

- Preserve the compute node but primarily use it as:
  - A container for global attributes (from Phase 1: total FLOPs, memory bytes per scope, arithmetic intensity, loop/buffer counts, etc.).
  - An optional root node that can still connect to loops and buffers:
    - Existing edges:
      - `compute -> loop` (`"iterates"`)
      - `compute -> buffer` (`"accesses"`)
- Do **not** introduce additional semantics or dependencies on compute-node edges in Phase 2; graph consumers can decide whether to use or ignore them.

#### 2.3.2. Acceptance Criteria

- The compute node remains present and named `"compute"`.
- Its edges exist as before, and new loop/loop and loop/buffer edges complement them.
- No changes are required in `GraphEncoder` or its callers.

---

## Non-Goals (For Now)

- Implementing or depending directly on TVM’s built-in `FeatureExtractor` for MetaSchedule.
  - We may later add a separate pipeline to export official TVM features for baselines or teacher models, but it is **not part of this plan**.
- Removing existing edges or drastically changing the semantics of the current graph.
  - Changes should be additive and backward-compatible.
- Handling every exotic TIR construct or all TVM versions.
  - The focus is on TIR produced by our MetaSchedule-based pipelines and typical dense compute kernels (e.g., GEMM/BMM/conv).

---

## Summary of Agent Tasks

An AI agent (or contributor) implementing this plan should:

1. **Phase 1: TVMGraphBuilder Attribute Enhancements**
   - [x] Extend `_TIRVisitor` (or equivalent) to track:
     - Loop depth, loop kind (`ForKind`), and reduction vs. spatial.
     - Buffer scopes, sizes, and read/write counts.
     - Total FLOPs and bytes per memory level for the compute node.
   - [x] Update `TVMGraphBuilder.build(...)` to:
     - Attach new numeric attributes to loop, buffer, and compute nodes.
   - [x] Ensure backward compatibility with `ProgramGraph` and `GraphEncoder`.

2. **Phase 2: Graph Topology Enrichment**
   - [x] Add loop–loop edges representing nesting (`"loop_child"`).
   - [x] Add loop–buffer access edges (e.g., `"loop_accesses"`).
   - [x] Preserve existing compute → loop and compute → buffer edges for compatibility.

3. **Validation**
   - [x] Run existing unit tests (`PYTHONPATH=src pytest tests/test_graph_builder.py tests/test_graph_encoder.py`).
   - [x] Add new tests to:
     - Assert the presence of new attributes and edge types (`tests/test_tvm_graph_builder.py`).
     - Verify that structurally similar TIR programs produce consistent graphs. **TODO** (not yet covered explicitly).
     - Confirm that `GraphEncoder` continues to encode graphs without errors.

## Progress Notes

- Phase 1 and 2 completed in `src/tvm_cost_model/features/tvm_graph_builder.py` with a recursive visitor, enriched loop/buffer/compute attributes, and new `loop_child`/`loop_accesses` edges while keeping existing edges and the `GraphEncoder` API intact.
- Added coverage in `tests/test_tvm_graph_builder.py` for new attributes and topology; tests run via `PYTHONPATH=src pytest tests/test_tvm_graph_builder.py tests/test_graph_builder.py tests/test_graph_encoder.py`.
- Remaining follow-up: add an explicit invariance test to compare structurally similar TIR programs under the enriched builder.

This roadmap is designed to be feasible and incremental: Phase 1 enriches features while keeping graph structure fixed, and Phase 2 enriches topology while keeping APIs stable. Both phases can be implemented gradually with small, testable changes.
