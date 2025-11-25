# Low-Level Kernel Cost Model Plan

This file tracks the state of the project so that any collaborator or future session can immediately understand progress.

## Completed
- **Proposal overhaul & scope clarification**: introduced success criteria, risks, and detailed methodology sections explaining graph-based modeling, invariance, and explainability requirements.
- **Evaluation rubric definition**: locked in quantitative metrics (nRMSE, Kendall Tau, measurement reduction targets, attribution fidelity) that downstream work must satisfy.
- **Repository scaffold & tooling bootstrap**: initialized the Python package structure (`src/tvm_cost_model`), CLI scripts, editable installation metadata, placeholder tests, and a `.venv` for reproducible development.
- **Synthetic measurement pipeline**: PyArrow-backed dataset builder with synthetic sampler/evaluator and CLI (`scripts/bootstrap_dataset.py`) validated export flow to Parquet + regression tests.

## In Progress
- **Hardware & dataset provisioning**: finalizing access to two target NVIDIA GPUs (Ampere + Ada) and consolidating public traces (e.g., TenSet) into a unified schema; blocked only on confirming measurement quotas.
- **Operator coverage selection**: iterating on the exact list of kernels and shape distributions with the TVM MetaSchedule benchmarking scripts to ensure ≥50k labeled schedules per GPU.
- **Graph extraction & canonicalization**: TVM-free canonical graph builder is in place; TVMGraphBuilder now parses real TIR via Python visitors. Next: enrich node/edge semantics with thread bindings/memory scopes.
- **Graph encoding for models**: added ProgramGraph encoder that stabilizes node/edge vocab IDs and aligns dense feature vectors to prep data for the upcoming R-GAT prototype.
- **Ranking-only objective**: pivoted the model plan to focus on schedule ordering (Kendall Tau / NDCG) instead of absolute runtime regression; code skeleton now emits scores.
- **Pair mining plan**: documented curriculum-style pair construction (easy-to-hard pairs) for ranking training to stabilize early epochs and improve discrimination on close schedules.

## New
- **Pair sampling utilities**: added helpers to generate easy/medium/hard ranking pairs from measurement records to feed the upcoming ranking losses.
- **Torch baseline ranker**: introduced a Node-MLP ranker with per-node attribution and encoded-pair dataset builder to exercise the ranking pipeline before plugging in the full R-GAT.
- **TVM integration scaffold**: TVMGraphBuilder now uses Python `stmt_functor.post_order_visit` to parse real TIR; MetaSchedule sampler emits design-space schedules and measures locally via `tvm.build` + `time_evaluator`.
- **Encoded ranking pairs**: pipeline from MeasurementRecords through GraphBuilder/GraphEncoder to produce model-ready pairs for the ranking head.

## Immediate Next Steps
- Wire MetaSchedule sampler + measurement into the dataset builder scripts to replace synthetic sampling.
- Extend measurement to handle function inputs (generate NDArrays from workload shapes) and add optional remote runner support.
- Begin PyTorch/PyG R-GAT prototyping using encoded graphs and mined pairs; keep Node-MLP as a baseline.

## Pending / Upcoming
- **Model prototyping**: implement the R-GAT backbone with ranking + attribution heads, then benchmark against TVM’s XGBoost cost model (Step 3).
- **Explainability validation**: visualization tooling and fidelity experiments for attribution signals (Step 4).
- **MetaSchedule integration**: PyCostModel wrapper, fallback logic, and telemetry plumbing (Step 5).
- **Full evaluation + release**: cross-operator/GPU experiments, ablations, and packaging of datasets/scripts (Step 6).

## Coordination Notes
- Risks around cross-GPU access and feature-extraction overhead are logged in `Project Proposal.md` under Section 7; revisit mitigations before entering each milestone.
- Record any changes to success metrics or hardware targets here so other sessions can reconcile differences quickly.
