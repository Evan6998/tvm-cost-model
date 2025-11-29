Goal

Replace the current MLP-based NodeMLPRanker with a GraphSAGE-style multi-relational GNN that is aligned with the current SOTA results from TVM MetaSchedule research (e.g., CALO-GNN).
The new model must:
	•	Support multi-edge-type message passing (loop_child, iterates, accesses, loop_accesses).
	•	Accept the existing GraphEncoding (dense numeric node features + node types + edge indices + edge types).
	•	Produce a graph-level score compatible with both pointwise regression and pairwise ranking.
	•	Become the foundation for extending toward evidential uncertainty modeling later.

⸻

1. Architecture Overview

This GNN replaces the current per-node MLP with a 2-layer GraphSAGE encoder plus attention-based graph pooling.
It maintains full compatibility with the rest of the cost-model pipeline.

Components
	1.	Node Encoder
	•	Inputs: encoding.node_features (dense float matrix), encoding.node_types (categorical).
	•	Steps:
	•	Linear( feature_dim → hidden_dim )
	•	Add learned node_type_embedding[node_type] (size = hidden_dim)
	•	Nonlinearity: ReLU
	•	Optional: small dropout (0.05–0.1)
	•	Output: initial node representation h0.
	2.	Multi-relational GraphSAGE Message Passing
	•	Two layers recommended: GraphSAGE-Mean or Mean+Max concat.
	•	Each layer aggregates per edge-type messages.
	•	Implementation options:
	•	Use PyG HeteroConv with separate SAGEConv for each edge type, OR
	•	Implement a “Relational GraphSAGE” manually:

h_i^{(l+1)} = σ( W_self h_i^{(l)}  +  Σ_{r∈edge_types}  W_r * Agg_r( neighbors_of_type_r(i) ) )


	•	Hidden dimension: 64 (balanced accuracy vs inference-time).
	•	Number of layers: 2 (3 causes over-smoothing and higher cost).

	3.	Graph-level Readout (Pooling)
	•	Use attention pooling for interpretability and stability.
	•	Each node computes a scalar attention logit:

α_i = MLP_attn(h_i)
attn_weights = softmax(α)
graph_repr = Σ_i attn_weights[i] * h_i


	•	This replaces the current simple attention module in the MLP ranker.

	4.	Prediction Head
	•	Graph embedding → 2–3 layer MLP → scalar score.
	•	Output: single scalar latency score.
	•	Must support:
	•	Pointwise regression (MSE)
	•	Pairwise margin ranking
	5.	Optional SOTA Extension (future)
	•	Replace the scalar MLP with a Normal-Inverse-Gamma evidential regression head (4 outputs).
	•	Enables mean prediction + epistemic uncertainty.

⸻

2. Multi-relational Message Passing Requirements

To capture full ProgramGraph semantics, the GNN must treat edge types differently.

Edge types to support
	•	"loop_child"
	•	"iterates"
	•	"accesses"
	•	"loop_accesses"
	•	Any additional edge labels discovered in future graph-builder versions.

Requirements
	•	Each edge type must have its own message transform (e.g., a unique linear layer or SAGEConv).
	•	Aggregation per layer is the sum of relation-specific aggregated messages plus the self-node transform.
	•	All relations share the same hidden dimension (64).

⸻

3. Training Objectives

The model supports two training modes:

1. Pairwise (Primary Training Path)

Loss: MarginRankingLoss(margin=0.1)
Inputs: (better_graph, worse_graph)
Objective: score(better) > score(worse)

2. Pointwise (Compatibility)

Loss: MSE(pred, true_latency)
Use alongside pairwise loss as multi-task training if desired.

⸻

4. Inference-Time Constraints
	•	Must support batching of multiple graphs via padding or PyG’s mini-batching APIs.
	•	Target inference latency per batch (20–200 graphs): < 20 ms budget (same as CALO-GNN).
	•	Use TorchScript or TorchDynamo compilation for final deployment.

⸻

5. Feature Scaling & Normalization (Required)

Even with engineered features (log1p extent, degree counts, traffic metrics), modern GNN cost models require explicit normalization to reach SOTA.

Required changes
	1.	Per-workload feature standardization
	•	For each workload (operator shape or TIR module), compute mean/std of float features across all nodes and training samples in that workload.
	•	Apply:

x_norm = (x - mean) / (std + 1e-6)


	•	Store statistics inside GraphEncoder or preprocess step.

	2.	Graph-level normalization
	•	Apply LayerNorm on node embeddings after:
	•	Node encoder
	•	Each GraphSAGE layer
	•	This matches CALO-GNN and prevents exploding ranges across workloads.
	3.	Edge-type frequency normalization (optional but helpful)
	•	For each edge type r, scale aggregated messages by 1 / sqrt(deg_r).
	4.	Dropout
	•	Small (0.05–0.1) dropout after non-linearities to prevent overfitting on small datasets.
	5.	Preserve engineered features
	•	Keep existing log1p, structural degrees, combination features—they remain useful inputs to the GNN.

⸻

6. Implementation Steps (Actionable)
	1.	[x] Add a new module: GraphGNNRanker
	•	Replace NodeMLPRanker inside GraphCostModel.
	•	Input: GraphEncoding
	•	Output: score, per-node attention weights.
	2.	[x] Integrate relational GraphSAGE layers
	•	Use PyG or custom implementation.
	•	Hidden_dim = 64; num_layers = 2.
	3.	[x] Add feature normalization
	•	Stage 1: in GraphEncoder.encode() produce raw feature vectors.
	•	Stage 2: normalize features before GNN forward pass.
	4.	[x] Replace existing attention pooling with GNN-attention pooling
	•	Needs learnable MLP_attn.
	•	Ensures backward compatibility with existing attribution feature.
	5.	[x] Wire into GraphCostModel
	•	_ensure_model() must initialize the new GNN ranker.
	•	score_pair() logic remains unchanged.
	•	Training loops (pointwise + pairwise) do not need structural changes.
	6.	[ ] (Future) Add evidential regression head
	•	Implement 4-parameter NIG head.
	•	Replace scalar score output only when needed.

⸻

7. Expected Improvements
	•	Higher pairwise accuracy / top-K recall (SOTA level).
	•	Better generalization across workloads.
	•	Smooth integration with MetaSchedule’s batching execution model.
	•	Maintain interpretability via per-node attention weights.
