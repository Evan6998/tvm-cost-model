Plan: Feature Engineering Enhancements in GraphEncoder

This plan defines the concrete steps required to upgrade the GraphEncoder so the model receives richer and better-scaled node features. The goal is to stabilize training, improve ranking accuracy, and inject minimal structural information without introducing a full GNN.

⸻

1. Objectives

1.1 Numeric stabilization

Apply log1p() transformation to large-magnitude features (bytes, extents, flops, etc.) so that node feature vectors fall into a predictable numeric range and avoid exploding gradients.

1.2 Derived “combo” features

Introduce lightweight human-designed features that capture common TVM/TIR patterns (e.g., “deep loop with large extent”, “buffer traffic intensity”).

1.3 Lightweight structure features

Incorporate graph structure information into the node features without modifying the model architecture:
	•	per-node in_degree, out_degree, total_degree
	•	per edge-type degree (deg_in:loop_child, deg_out:iterates, etc.)

1.4 Preserve backward compatibility

No changes to ProgramGraph, GraphEncoding schemas, or model API.
Only additions inside GraphEncoder.

⸻

2. Overview of Code Areas to Modify

All changes occur inside:

src/tvm_cost_model/features/graph_encoder.py

The following methods must be updated:
	•	_collect_feature_names
	•	prime_feature_names
	•	encode
	•	(optional) add helper functions inside GraphEncoder

⸻

3. Feature Inventory

3.1 Raw attributes currently emitted by TVMGraphBuilder

Loop nodes
	•	extent, depth
	•	is_parallel, is_unrolled, is_vectorized, is_thread_bound, is_reduction

Buffer nodes
	•	elem_bytes, total_bytes
	•	read_count, write_count

Compute nodes
	•	loop_depth, buffer_count
	•	total_extent, total_flops
	•	global_bytes, shared_bytes, local_bytes, other_bytes
	•	arith_intensity

These must be treated consistently across all graphs.

⸻

4. Log-Scaled Features

Create a fixed list of attributes that will be transformed via:

log1p(max(value, 0))

Recommended log-scaled attributes:
	•	extent
	•	total_extent
	•	total_bytes, global_bytes, shared_bytes, local_bytes, other_bytes
	•	total_flops
	•	read_count, write_count (optional if distribution is wide)

Feature names

For each raw feature X, insert a derived feature named:

log1p:X

Example:

log1p:extent
log1p:total_bytes
log1p:total_flops

These derived names must be included in feature_names.

⸻

5. Combo Features (Lightweight Handcrafted)

Recommended set:

Loop-related
	•	loop:depth_x_log_extent
→ depth * log1p(extent)
	•	loop:parallel_log_extent
→ is_parallel * log1p(extent)

Buffer-related
	•	buffer:traffic_bytes
→ (read_count + write_count) * elem_bytes
	•	buffer:traffic_ratio
→ (traffic_bytes / max(total_bytes, 1)) clipped to [0,1]

Compute-related

Start with none (we already have arith_intensity).
Optional later:
	•	compute:log_flops_minus_log_bytes

Feature name conventions

Use literal string identifiers, e.g.:

loop:depth_x_log_extent
buffer:traffic_bytes
buffer:traffic_ratio

These names must also appear in feature_names.

⸻

6. Structural Features (Degrees)

6.1 Basic degrees

For each node:
	•	deg_in
	•	deg_out
	•	deg_total = deg_in + deg_out

Values may be raw counts or log1p counts. Initially: raw counts.

6.2 Per-edge-type degrees

For every edge type E seen by GraphEncoder.edge_type_to_id:

Add the following derived feature names:

deg_in:E
deg_out:E

Example:

deg_in:iterates
deg_out:iterates
deg_in:loop_child
deg_out:loop_child

The encoder must compute these counts from the graph’s edge list.

⸻

7. Required Changes to GraphEncoder

7.1 _collect_feature_names

Modify to:
	1.	Collect raw attribute names (current behavior).
	2.	Add log1p:* names for all configured log-scaled attributes.
	3.	Add combo feature names (Section 5).
	4.	Add structural feature names:
	•	deg_in, deg_out, deg_total
	•	deg_in:<edge_type>, deg_out:<edge_type>
	5.	Sort and store into self._feature_names.

7.2 prime_feature_names

Must use the same logic as _collect_feature_names:
	•	Union raw attributes across all graphs
	•	Then add all derived names as above

7.3 encode

Modify the feature-row construction:
	1.	Precompute degree statistics:
	•	Initialize arrays: deg_in, deg_out, deg_total
	•	Initialize dicts: deg_in_by_type[(node, etype)], deg_out_by_type[...]
	•	Fill these while scanning edges
	2.	For each node, generate a complete feature_vector:
	•	For raw features: attrs.get(name, 0)
	•	For log1p features: log1p(attrs[field])
	•	For combo features: compute per formulas in Sections 5
	•	For structural features:
	•	deg_in[node]
	•	deg_out[node]
	•	deg_total[node]
	•	deg_in:<etype> / deg_out:<etype>
	3.	Append to node_features.

⸻

8. Validation Checklist (Must Do)

After implementing:
	•	Verify feature_names contains:
	•	all raw names
	•	all log1p:*
	•	all combo feature names
	•	deg_in, deg_out, deg_total
	•	all deg_in:etype and deg_out:etype
	•	Build a tiny hand-constructed ProgramGraph and manually verify features:
	•	Check degree counts match expectations
	•	Check log1p values numerically
	•	Check combo features computed correctly
	•	Ensure GraphCostModel still trains without API changes.

⸻

9. Execution Order

Recommended implementation sequence:
	1.	Implement & test log1p features (most important for training stability)
	2.	Implement combo features
	3.	Implement degree features
	4.	Re-run feature-name union tests via prime_feature_names
	5.	Train on a small dataset and confirm the model can overfit a tiny set (>90% pairwise accuracy)
	6.	Scale training to full dataset
