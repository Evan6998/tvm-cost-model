# Benchmark Plan: GraphPyCostModel vs TVM XGBoost Cost Model

## 0. TL;DR

We want a **clean, reproducible benchmark** comparing:

1. **Offline predictive quality** of cost models (GraphPy vs TVM XGBoost).
2. **Online tuning efficiency** when plugged into MetaSchedule.
3. **Generalization** to unseen operators / shapes / hardware.

This plan assumes the existing CLI harness:

```bash
python scripts/run_metaschedule_tuning.py \
    --operator gemm --shape '{"m":1024,"n":1024,"k":1024}' \
    --target "cuda" --cost-model graph|xgb ...

An “agent” should be able to follow this document and implement everything.

⸻

1. Goals & Non-Goals

1.1 Goals
	•	G1: Predictive quality
Quantify how well GraphPy ranks candidate schedules vs XGBoost, using:
	•	Pairwise accuracy
	•	Top-K recall@{1, 5, 20}
	•	Correlation with true latency (Spearman / Kendall)
	•	G2: Tuning efficiency
When used inside MetaSchedule:
	•	Best achieved latency vs # of measured candidates
	•	Wall-clock tuning time
	•	Speedup over:
	•	TVM’s XGBoost cost model
	•	Vendor libraries (e.g., cuBLAS / cuDNN / oneDNN / Eigen where applicable)
	•	G3: Generalization
	•	Train on a subset of workloads/shapes and test on:
	•	Unseen shapes of same operator
	•	Unseen operators
	•	(Optional) A different device of same family (e.g., V100 → A10)
	•	G4: Ablation & robustness
	•	Study key design choices (GNN features, loss, sampling strategy).
	•	Effect of noisy measurements.

1.2 Non-Goals
	•	Not trying to beat every recent paper (TLP, TenSet-MLP, etc.) on their own datasets. ￼
	•	Not building a completely new search strategy; we keep MetaSchedule search as-is and swap only the cost model.

⸻

2. Baselines & Related Work

2.1 Baselines
	•	B1: TVM MetaSchedule + default XGBModel
	•	B2: GraphPyCostModel (your GNN model integrated via GraphPyCostModel)
	•	(Optional, later) B3: TenSet-style MLP if we ever import their code or dataset.

2.2 Benchmark practices from recent work (for reference, not to implement 1:1)
	•	TenSet: large-scale tensor program performance dataset (52M measurements, 6 platforms). Proposed pairwise accuracy and top-k recall as core metrics, and showed that pre-trained cost models can speed up search in TVM by up to 10×. ￼
	•	TLP, Felix, etc.: evaluate cost models via top-k recall across hardware, and then show tuning speedups vs Ansor/MetaSchedule in terms of best latency vs time / #measurements. ￼
	•	AutoTVM / Ansor / MetaSchedule: typically report
	•	Pairwise accuracy & top-k recall for ranking metrics, ￼
	•	Speedup over vendor libraries and prior auto-tuning frameworks, with tuning time breakdowns. ￼

We will mirror these metrics, but keep the setup light-weight.

⸻

3. Workload & Hardware Matrix

3.1 Operators (aligned with your script)

Use the built-in operators from run_metaschedule_tuning.py:
	•	vecadd
	•	gemm
	•	bmm
	•	conv2d_nchw
	•	depthwise_conv2d
	•	layernorm
	•	softmax

3.2 Shape sets

Canonical matrix (also captured in configs/benchmark_workloads.yaml):
	•	vecadd
		– A/train (vecadd_a): n=262144
		– B/id_test (vecadd_b): n=1048576
		– C/ood_test (vecadd_c): n=16777216
	•	gemm
		– A/train (gemm_a): m=n=k=256
		– B/train (gemm_b): m=n=k=1024
		– B/id_test (gemm_b_id): m=768, n=1024, k=512
		– C/ood_test (gemm_c): m=n=k=4096
	•	bmm
		– A/train (bmm_a): batch=4, m=n=k=64
		– B/train (bmm_b): batch=16, m=n=k=128
		– B/id_test (bmm_b_id): batch=8, m=n=k=96
		– C/ood_test (bmm_c): batch=32, m=n=k=256
	•	conv2d_nchw
		– A/train (conv2d_a): n=1, ci=32, co=32, h=w=56, kh=kw=3
		– B/train (conv2d_b): n=1, ci=64, co=128, h=w=28, kh=kw=3
		– B/id_test (conv2d_b_id): n=1, ci=96, co=128, h=w=28, kh=kw=1
		– C/ood_test (conv2d_c): n=1, ci=128, co=256, h=w=14, kh=kw=3
	•	depthwise_conv2d
		– A/train (depthwise_a): n=1, ci=32, h=w=112, kh=kw=3
		– B/train (depthwise_b): n=1, ci=64, h=w=56, kh=kw=3
		– B/id_test (depthwise_b_id): n=1, ci=96, h=w=28, kh=kw=3
		– C/ood_test (depthwise_c): n=1, ci=160, h=w=14, kh=kw=3
	•	layernorm
		– A/train (layernorm_a): n=64, hidden=256
		– B/train (layernorm_b): n=256, hidden=1024
		– B/id_test (layernorm_b_id): n=192, hidden=768
		– C/ood_test (layernorm_c): n=512, hidden=4096
	•	softmax
		– A/train (softmax_a): n=256, k=512
		– B/train (softmax_b): n=512, k=1024
		– B/id_test (softmax_b_id): n=384, k=896
		– C/ood_test (softmax_c): n=1024, k=2048

Train/val/test splits are random within the workloads marked “train”; id_test and ood_test workloads are held out wholesale for generalization checks.

3.3 Hardware targets

Minimum:
	•	CPU: e.g. llvm -num-cores=8 on your dev machine
	•	GPU: e.g. cuda -arch=sm_80 (A100 / 3090 / etc.)

Optional extensions:
	•	A second GPU type (e.g. T4 vs A100) for cross-device generalization.
	•	Mobile / ARM CPU if you already have infra.

⸻

4. Data Collection Pipeline

We want a single unified collection pipeline that produces:

{
  "workload_id": "gemm_m1024_n1024_k1024",
  "candidate_id": "...",
  "schedule_repr": "...",        // some stable string or hash
  "features_graphpy": {...},     // optional cached features
  "measured_latency_ms": 0.123,
  "device": "cuda:sm80",
  "cost_model_type": "xgb|graph|none",
  "meta": {
     "operator": "gemm",
     "shape": {"m":1024,"n":1024,"k":1024},
     "seed": 42,
     "run_id": "..."
  }
}

4.1 Generate candidate schedules + measurements

Reuse your existing tuning harness:
	•	For each (operator, shape, target):
	1.	Run run_metaschedule_tuning.py with MetaSchedule search and no/weak cost model (or XGB with random exploration) to generate a large pool of measured schedules.
	•	One option: run XGB with a high exploration ratio; log all schedules and latencies.
	•	Another option: implement a “collect mode” where the cost model is not used (pure random / evolutionary search) and everything is measured.
	2.	Ensure each schedule is uniquely identified (e.g., hash its TIR / schedule trace).
	•	Store raw logs (JSONL) per (operator, shape, device).

Target scale (reasonable for you):
	•	~ 3–5 workloads per operator × 7 operators × ~400–1000 measured schedules each → order of 10^4–10^5 measured points.

4.2 Train / validation split

Per (operator, device):
	•	Shuffle schedules.
	•	Split into:
	•	70% train
	•	15% validation
	•	15% test

For generalization experiments:
	•	Additionally mark:
	•	Some shapes entirely train-only.
	•	Some shapes test-only (never appear in training).

⸻

5. Evaluation Metrics

Exactly what the agent must implement.

5.1 Latency preprocessing
	•	Use median latency over repeats from runner_result_to_latency_ms.
	•	Discard runs with:
	•	Invalid result
	•	Extreme outliers (e.g., > 5× median of rest for that workload)
	•	Convert to throughput or inverse-latency if needed (but we can just use latency as “true cost”).

5.2 Offline ranking metrics

For each (operator, shape, device) and cost model:
	1.	Pairwise accuracy
	•	Sample random pairs (i, j) where |lat_i - lat_j| / min(lat_i, lat_j) ≥ δ (e.g., δ = 0.05).
	•	Accuracy = fraction where model and ground truth agree on which is faster.
	•	Report mean ± std across workloads.
	2.	Top-K recall@K
	•	For each workload:
	•	Sort schedules by true latency (ascending).
	•	Let R = set of “top-T” true schedules (e.g., T = 20).
	•	Sort schedules by predicted latency.
	•	For K ∈ {1, 5, 10, 20}, compute recall@K = |R ∩ TopK_pred| / |R|. ￼
	•	Aggregate across workloads (average or weighted by #candidates).
	3.	Correlation metrics
	•	Compute Spearman or Kendall τ between predicted cost and true latency per workload.
	•	Useful to see global ranking quality, not just local.
	4.	Calibration
	•	Scatter plot / summary of predicted vs true latency (optional for report, but useful for debugging).

5.3 Online tuning metrics (on-policy)

For each cost model & workload:
	•	During tuning runs, log:
	•	n_measured candidates so far
	•	current best latency
	•	wall-clock time since start
	•	From this, derive:
	1.	Best latency vs #measured curve
	•	For n ∈ {16, 32, 64, 128, 256, ...} record best latency.
	•	Plot and compare GraphPy vs XGB.
	2.	Best latency vs wall-clock time curve
	•	Same but against real time.
	3.	Final speedup metrics
	•	Speedup over:
	•	XGB baseline (MetaSchedule + XGB with same max_trials).
	•	Vendor library (e.g., cuBLAS / cuDNN / oneDNN where available). ￼
	•	Report speedup = BaselineLatency / OurLatency.
	4.	Measurement savings
	•	Given a latency target (e.g., 95% of best XGB’s latency), how many measurements are needed?

5.4 Generalization metrics
	•	Train cost model on:
	•	Subset of operators / shapes / devices.
	•	Evaluate offline and online metrics on:
	•	Held-out shapes for same operator.
	•	Held-out operators.
	•	(If possible) Held-out device (e.g. train on V100, test on A100).

Compare performance drop vs in-distribution test.

⸻

6. Experiment Design

6.1 Offline benchmarking (static dataset)
	1.	Dataset preparation
	•	run_online_benchmark.py drives run_metaschedule_tuning.py across config grids; each run emits JSONL with per-candidate latency + trace.
	•	scripts/collect_dataset.py ingests those logs, honors the config splits (train/id_test/ood_test), and writes train/val/test JSONL plus per-workload raw logs.
	2.	Feature extraction
	•	scripts/extract_features.py builds GraphPy encodings once per unique candidate_id and caches torch tensors under artifacts/benchmarks/features.
	3.	Training & evaluation loop
	•	scripts/run_offline_benchmark.py trains GraphPy (pairwise) or an XGB regressor on aggregated GraphEncoder features; evaluates on test/id_test/ood_test and writes metrics JSON.
	•	Metrics: pairwise accuracy, recall@{1,5,10,20}, Spearman/Kendall (scores vs true latency).
	4.	Reporting
	•	scripts/summarize_offline_metrics.py aggregates the JSON into CSV + Markdown tables for quick comparison.

6.2 Online tuning benchmark (MetaSchedule integration)
	1.	Harness
	•	run_metaschedule_tuning.py now seeds RNGs and logs every measurement with trace hash + scheduled TIR when --log-json-path is set.
	•	run_online_benchmark.py iterates (workload, target, cost_model, seed) from configs/benchmark_workloads.yaml and writes logs under artifacts/benchmarks/online/.
	2.	Protocol
	•	Run N seeds per workload (config default: seeds=[0,1,2]) for both cost models with identical max_trials/trials_per_iter/number/repeat.
	•	Logs already contain measure_idx, latency_ms, elapsed_sec, candidate_id, trace JSON, and best-so-far.
	3.	Metrics extraction
	•	scripts/analyze_online_tuning.py builds best_latency vs measurement/time curves, reports best_latency_ms, and computes measurements_to_target for target multipliers (95% / 90% of global optimum by workload).
	4.	Baselines vs vendor libraries
	•	TODO hook in cuBLAS/oneDNN references for GEMM/Conv2D; use summary JSON to report speedup once vendor measurements are available.

6.3 Generalization experiments
	1.	Hold-out design

Define:
	•	Shape generalization:
	•	Train on small/medium shapes; test on large shapes.
	•	Operator generalization:
	•	Train on {gemm, conv2d, bmm}, test on {layernorm, softmax} or vice-versa.
	•	Device generalization (optional):
	•	Train on GPU1; test on GPU2.

	2.	Offline generalization

	•	Train cost model on train subset.
	•	Evaluate offline metrics on:
	•	In-distribution test
	•	OOD test (hold-out shapes/operators)
	•	Compare drop in pairwise accuracy / top-k recall.

	3.	Online generalization

	•	Use trained model to initialize cost model in MetaSchedule for OOD workloads.
	•	Compare:
	•	Convergence speed vs random/XGB model.
	•	How quickly they reach acceptable latency.

6.4 Ablation studies

Potential ablations:
	•	Feature ablation
	•	Remove some GNN features (e.g., buffer size info, loop extents) and re-run offline metrics.
	•	Loss function / sampling
	•	Compare:
	•	Pairwise ranking loss
	•	Pointwise regression loss
	•	Different pair sampling strategies (hard/medium/easy).
	•	Noise robustness
	•	Simulate noise by adding Gaussian noise to latencies and re-evaluating.

Agent should pick 2–3 ablations that are easiest to implement and show clear effects.

⸻

7. Implementation Checklist (for the Agent)

7.1 Logging & dataset
	•	run_metaschedule_tuning.py: --seed seeding + --log-json-path JSONL with candidate_id, trace, scheduled_tir, latency_ms, best_latency_ms, elapsed_sec.
	•	collect_dataset.py: ingests multiple JSONL logs, normalizes workload_id/operator/shape/target, respects config splits (train/id_test/ood_test), writes train/val/test + metadata.json.

7.2 Offline training & evaluation
	•	extract_features.py: runs TVMGraphBuilder + GraphEncoder once per candidate_id, caches torch tensors under artifacts/benchmarks/features/{split}.pt.
	•	run_offline_benchmark.py: trains GraphPy (pairwise) or XGB regressor on aggregated GraphEncoder features; reports pairwise accuracy, recall@K, Spearman/Kendall for test/id_test/ood_test splits.
	•	summarize_offline_metrics.py: converts metrics JSON into CSV/Markdown tables.

7.3 Online tuning benchmark
	•	run_online_benchmark.py: Python grid driver honoring configs/benchmark_workloads.yaml (workloads/targets/seeds/cost models); writes logs to artifacts/benchmarks/online/.
	•	analyze_online_tuning.py: parses logs, builds best-latency curves, and reports measurements_to_target for 95%/90% of overall best per workload.

7.4 Generalization & ablations
	•	Workload splits for ID/OOD are encoded in configs/benchmark_workloads.yaml; reuse the same config for online runs and offline splits.
	•	Open follow-ups: add ablation flags (feature drops, loss variants) and optional multi-config runner for operator/device generalization sweeps.

⸻

8. Deliverables & Timeline

8.1 Deliverables
	1.	Code
	•	Logging, dataset builder, training, evaluation scripts.
	•	Configs for different benchmark suites.
	2.	Data
	•	Collected measurement datasets (JSONL / parquet).
	•	Train/val/test splits.
	3.	Results
	•	Offline metrics tables:
	•	Pairwise accuracy, top-k recall, correlation (GraphPy vs XGB).
	•	Online metrics:
	•	Speedup curves and summary tables per workload.
	•	Generalization & ablation results.
	4.	Short written report (optional)
	•	2–3 pages summarizing key findings:
	•	Where GraphPy wins
	•	Where it’s similar
	•	Where it regresses and hypotheses why

8.2 Suggested order of work
	1.	Phase 1 (Infra): Logging + dataset builder + basic offline evaluation on 1–2 workloads.
	2.	Phase 2 (Scale): Full workload/hardware matrix; offline benchmark for all.
	3.	Phase 3 (Online): Tuning benchmark with both cost models.
	4.	Phase 4 (Gen + Ablations): Generalization experiments and 1–2 key ablations.
	5.	Phase 5 (Report): Summaries, plots, and write-up.
