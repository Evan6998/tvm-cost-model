### Low-Level Kernel Cost Model

Harivallabha Rangarajan (hrangara), Yi Shi (yishi3), Evan Huang (yongfenh)

---

### 1. Motivation

Optimizing GPU kernels such as matrix multiplication and convolution is essential for accelerating deep learning workloads. State-of-the-art autotuners (TVM AutoTVM/Ansor/MetaSchedule) still rely on thousands of on-device measurements to rank candidate schedules, limiting practicality for fast-moving models or restricted hardware access. Learned cost models are the only scalable alternative, yet current approaches are brittle: they overfit to kernels and GPU models seen during training, are sensitive to syntactic schedule variations, and operate as opaque black boxes. We aim to develop a transferable, invariant, and explainable cost model that can be dropped into TVM MetaSchedule to drastically reduce measurements while maintaining accuracy.

### 2. Problem Definition & Goals

We target low-level GPU kernel programs (e.g., GEMM, convolution, depthwise convolution, batched GEMM) compiled via TVM. The model must map tensor IR (TIR) schedules to runtime on NVIDIA GPUs while generalizing across workloads, input shapes, and nearby GPU generations.

**Goals & success criteria**
- Reduce TVM MetaSchedule measurement counts by ≥5× compared to the default XGBoost cost model while keeping final kernel latency within 5% of the baseline best-found kernel.
- Achieve strong ranking quality (Kendall Tau ≥0.7 on held-out schedules) for kernels unseen during training and ≥0.6 when transferring across GPU generations; absolute runtime prediction is out-of-scope.
- Provide per-program factor attributions (e.g., bottleneck loops, memory tiles) with fidelity validated via counterfactual edits.

### 3. Related Work & Gaps

**Tensor compilers and learned cost models**. AutoTVM introduced gradient-boosted tree cost models with hand-crafted features [1]. Ansor replaced templates with evolutionary search guided by a learned model [2]. MetaSchedule exposes pluggable PyCostModel interfaces and TIR feature extraction [3].

**Transferable models and datasets**. TenSet [4], Transfer-Tuning [5], and ATFormer [6] study cross-task transfer but still rely on sequence or tree features that miss fine-grained schedule semantics.

**Structured representations**. ProGraML [7], Inst2Vec [8], Ithemal [9], and DNNPerf [10] demonstrate that graph neural networks (GNNs) over compiler IRs yield better inductive bias for performance prediction, yet they focus on CPU blocks or high-level NN graphs.

**Analytical/heuristic baselines**. Roofline [11] and Hong-Kim [12] GPU CPI models offer upper bounds but ignore schedule-specific effects. Halide autoschedulers [13, 14], FlexTensor [15], and Tensor Comprehensions [16] confirm the need for learned ranking.

**Gaps addressed here**
1. **Cross-kernel generalization**: boost tree/token models fail to extrapolate [1,4,6]; our graph representation enforces invariance to loop reordering and tiling choices.
2. **Explainability**: SHAP-like post-hoc analyses [17] lag behind the need for actionable feedback; we embed explanation heads directly into the model.
3. **Integration-readiness**: few graph-based models target MetaSchedule’s PyCostModel API despite its maturity [3].

### 4. Proposed Approach

**4.1 Data generation & curation**
- Use MetaSchedule to sample schedules for GEMM, Conv2D, Depthwise Conv, BMM, LayerNorm, and Softmax across diverse shapes.
- Collect runtimes on two NVIDIA GPUs (one Ampere, one Ada) to enable cross-hardware experiments. Measurement metadata (schedule config, hardware counters, occupancy estimates) is stored in an Arrow/Parquet dataset for reproducibility.
- Maintain a reproducible data-ingestion toolkit (PyArrow writers + CLI scripts) so collaborators can regenerate measurement artifacts locally before uploading to the shared dataset registry.
- Augment with TenSet [4] traces when license-compatible to widen coverage.

**4.2 Program representation**
- Convert TIR to a heterogeneous graph capturing loop nests, memory buffers, thread bindings, and arithmetic ops.
- Apply canonicalization passes (loop normalization, storage flattening, access-index hashing) so semantically equivalent schedules map to similar graphs, improving invariance.
- Encode hardware context (SM count, shared memory size, memory bandwidth) as node/edge attributes so the model can learn cross-GPU transfer.

**4.3 Model architecture**
- Base model: relational graph attention network (R-GAT) that aggregates loop, memory, and compute nodes.
- Two-head design: (a) ranking head producing a scalar score for ordering schedules; (b) attribution head producing normalized importance scores over loop/memory nodes. Multi-task training enforces consistency between heads via gradient alignment.
- Implementation stack: PyTorch + PyTorch Geometric for the GNN backbone, enabling rapid experimentation with relational attention layers.
- Incorporate lightweight analytical priors by concatenating features such as arithmetic intensity and estimated occupancy.
- Interim baseline: start with a light Node-MLP ranker over encoded graphs to sanity check the ranking pipeline before swapping in the R-GAT.

**4.4 Training strategy**
- Curriculum: begin with within-operator splits, then leave-one-operator-out, finally cross-GPU transfer.
 - Loss: ranking-only (pairwise hinge or listwise) to preserve ordering for MetaSchedule use; no regression objective.
 - Ranking pairs: start with easy pairs (large runtime gaps) for stable early training, then introduce hard/close pairs; optionally use curriculum sampling that increases pair difficulty as validation Kendall Tau plateaus.
 - Pair construction utilities: implement offline pair miners that label easy/medium/hard pairs from measurement deltas so training can schedule difficulty over epochs.
- Regularize via invariance constraints (contrastive loss between semantically equivalent schedules) and knowledge distillation from MetaSchedule’s XGBoost model for cold-start stability.

**4.5 Integration into MetaSchedule**
- Implement `GraphCostModel(PyCostModel)` that consumes TIR modules, runs feature extraction/graph building, and serves `predict` + `update` APIs.
- Provide failure-safe fallback to XGBoost when models are uncalibrated and expose attribution output to MetaSchedule logs/visualizations for interpretability.

### 5. Evaluation Plan
- **Predictive accuracy**: ranking metrics only (Kendall Tau, Normalized Discounted Cumulative Gain) on held-out schedules; ablations with/without invariance loss and attribution head.
- **Search efficiency**: number of on-device measurements required to reach within 5% of oracle latency for each operator on both GPUs.
- **Transfer tests**: train on Ampere, evaluate on Ada without fine-tuning; train on subset of operators, test on unseen ones.
- **Explainability validation**: perturb top-attributed loops/tiles and verify predicted ranking shifts correlate with actual measurements (fidelity ≥0.6).
- **Overheads**: measure cost-model inference latency to ensure it adds <5% overhead to MetaSchedule search time.

### 6. Implementation Plan & Milestones
1. **Dataset & tooling (Weeks 1–2)**: finalize list of operators/shapes, implement reproducible measurement scripts, collect ≥50k labeled schedules per GPU.
2. **Representation & invariance (Weeks 2–3)**: build TIR-to-graph pipeline, canonicalization, and contrastive data augmentation; deliver inspectable graph dumps.
3. **Model prototyping (Weeks 3–5)**: implement R-GAT backbone, run initial training on GEMM/Conv, compare against TVM XGBoost baseline.
4. **Attribution & explainability (Weeks 5–6)**: add attribution head, visualization notebooks, and fidelity tests.
5. **Integration (Weeks 6–7)**: wrap model in MetaSchedule PyCostModel, plug into search loop, add fallback logic.
6. **Evaluation & write-up (Weeks 7–8)**: full cross-operator/GPU experiments, ablations, documentation, and release of dataset + scripts.

### 7. Risks & Mitigations
- **Insufficient cross-GPU coverage**: mitigate by logging hardware counters and using analytic-derived features so the model can extrapolate; if second GPU access is delayed, simulate via occupancy models and validate once hardware becomes available.
- **Graph extraction overhead**: cache intermediate representations and parallelize feature extraction with Rust TVM passes; fall back to lightweight feature sets for rapid iterations.
- **Explainability fidelity**: if attribution head underperforms, employ SHAP/Integrated Gradients as cross-checks and precompute attributions for critical kernels.
- **Data imbalance across operators**: enforce stratified sampling and cost-sensitive losses so high-variance kernels do not dominate training.

### 8. Dataset Sources
- Survey public low-level GPU kernel datasets (e.g., TenSet) and supplement with internally generated MetaSchedule metadata stored as Arrow tables. Release cleaned dataset (subject to licensing) for reproducibility.

### 9. Repository Scaffold
- Python package layout under `src/tvm_cost_model/` with modules for data, features, models, training, integration, and utilities, plus CLI scripts for dataset bootstrapping and training.
- `pyproject.toml` defines dependencies (PyYAML, pytest, ruff) and ensures editable installs for active development.
- `.venv` virtual environment checked into the repository root for reproducible local runs; instructions in `README.md` describe activation and installation workflows.
- Placeholder tests (`tests/test_pipeline.py`) validate that the training pipeline skeleton wires up successfully, providing a base for future regression tests.
