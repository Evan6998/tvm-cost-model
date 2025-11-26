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
- **TVM integration scaffold**: TVMGraphBuilder now uses Python `stmt_functor.post_order_visit` to parse real TIR; MetaSchedule sampler emits design-space schedules and measures locally via `tvm.tir.build` + `time_evaluator`.
- **Encoded ranking pairs**: pipeline from MeasurementRecords through GraphBuilder/GraphEncoder to produce model-ready pairs for the ranking head.
- **MetaSchedule measurement plumbing**: measurement now supports input synthesis from `workload_shape`, optional runner hooks, and a `MetaScheduleRuntimeEvaluator` that slots into `DatasetBuilder`.
- **Dataset bootstrap CLI (MetaSchedule mode)**: `scripts/bootstrap_dataset.py` now defaults to MetaSchedule and ships built-in TVMScript workloads (vecadd, gemm, bmm, conv2d, depthwise, layernorm, softmax) with shape overrides, dtype-aware input gen, and optional RPC runner wiring for remote measurement.
- **Bootstrap review decisions**: confirmed current pipeline is functional but lacks alignment with TVM `TuningRecord` fields and needs broader workload/shape/hardware coverage plus ranking-friendly sampling.
- **Schema enrichment**: dataset records now capture target strings, workload keys (structural hash), and original vs post-schedule TIR to better mirror TVM `TuningRecord` needs.

## Immediate Next Steps
- **Workload diversity + shape sweeps**: add NHWC/1x1/grouped conv, broadcast elementwise, reductions, transformer micro-kernels; introduce shape sampling modes (grid/random ranges) and multi-operator runs.
- **Multi-workload sweep driver**: add a `sweep_workloads.py` that iterates operators/shapes/targets, invokes the bootstrapper, and merges Parquet shards.
- **Ranking-friendly sampling**: extend MetaScheduleSampler to inject random/mutated schedules as hard negatives and guarantee sufficient variety per workload.
- **Hardware metadata**: start collecting numeric hardware features (cores, memory, cache, clocks) alongside `hardware_id` for cross-hardware training.
- **Measurement on real hardware**: validate RPC/remote runner path on Ampere + Ada; target ≥50k labeled schedules per GPU with enriched metadata.
- **Model bring-up**: start PyTorch/PyG R-GAT prototype on encoded graphs and mined pairs; keep Node-MLP as a regression baseline and track Kendall Tau/NDCG on held-out pairs.

## Pending / Upcoming
- **Explainability validation**: visualization tooling and fidelity experiments for attribution signals (Step 4).
- **MetaSchedule integration**: PyCostModel wrapper, fallback logic, and telemetry plumbing (Step 5).
- **Full evaluation + release**: cross-operator/GPU experiments, ablations, and packaging of datasets/scripts (Step 6).

## Coordination Notes
- Risks around cross-GPU access and feature-extraction overhead are logged in `Project Proposal.md` under Section 7; revisit mitigations before entering each milestone.
- Record any changes to success metrics or hardware targets here so other sessions can reconcile differences quickly.
