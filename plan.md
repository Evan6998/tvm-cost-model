# Low-Level Kernel Cost Model Plan

This file tracks the state of the project so that any collaborator or future session can immediately understand progress.

## Completed
- Proposal + evaluation guardrails are set: success criteria, risks, and quantitative metrics (nRMSE, Kendall Tau, measurement reduction, attribution fidelity) are locked.
- Repository/tooling scaffold is stable: Python package layout, CLI scripts, editable install metadata, placeholder tests, and PyArrow-backed DatasetBuilder exporting Parquet with target/workload_key/hardware_features.
- MetaSchedule data path works end-to-end: sampler dedupes measure candidates and errors on empty traces; runtime evaluator builds TIR + `time_evaluator`; `scripts/bootstrap_dataset.py` defaults to MetaSchedule workloads (vecadd/gemm/bmm/conv2d/depthwise/layernorm/softmax) and `scripts/sweep_workloads.py` merges shards across runs.
- TIR graph extraction/encoding upgraded: TVMGraphBuilder visits real TIR via `stmt_functor.post_order_visit`, capturing loop depth/kind/reduction flags, buffer scope/byte stats, flop counts, and loop-child/access edges; GraphEncoder now emits log1p/combo/degree structural features with stable vocab priming.
- Ranking pipeline ready: MeasurementRecord → ProgramGraph → GraphEncoding → difficulty-labeled pairs; curriculum-aware TrainingPipeline stages (easy/medium/hard) with per-stage sampling; GraphCostModel now defaults to GraphGNNRanker (relational GraphSAGE + attention pooling + margin ranking) while keeping Node-MLP as a baseline. Pair sampling utilities support difficulty classification and seeded random sampling without enumerating all combinations.

## In Progress
- Hardware + dataset provisioning and operator coverage: still need real runs on target Ampere/Ada GPUs and shape distributions beyond the built-in workloads to hit ≥50k labeled schedules per GPU.
- MetaSchedule cost-model alignment: the current adapter is still a placeholder; need a PyCostModel-compatible wrapper that ingests TuneContext/MeasureCandidate/RunnerResult and trains/predicts in batches.
- Pair mining calibration: easy/hard gap heuristics and curriculum fractions are unvalidated on real runtime distributions; expect tuning once new measurements land.
- Feature/edge richness: thread-binding/memory-scope semantics exist in attrs, but we may still need additional edge types and closer alignment with `TuningRecord` fields.

## Immediate Next Steps
- Collect a small measurement shard via `scripts/sweep_workloads.py` or `scripts/bootstrap_dataset.py` on available GPUs to exercise the curriculum training path and report Kendall Tau/NDCG on held-out pairs.
- Implement a PyCostModel-compatible wrapper (see `training_plan.md`) that batches MeasureCandidates, builds graphs via TVMGraphBuilder/GraphEncoder, and supports online ranking updates from RunnerResults.
- Validate models on new data: compare GraphGNNRanker vs NodeMLPRanker, tune `easy_frac`/`hard_frac` and curriculum epochs, and add quick metric dumps (train/val loss, pair accuracy, Kendall Tau if feasible).

## Pending / Upcoming
- Workload diversity + shape sweeps: add NHWC/1x1/grouped conv, broadcast elementwise, reductions, and transformer micro-kernels; introduce grid/random shape sampling and multi-operator runs.
- Hardware metadata + scale: collect numeric hardware features (cores/mem/cache/clocks) and validate RPC/remote runner paths on Ampere + Ada to reach ≥50k labeled schedules per GPU.
- Explainability validation: visualization + fidelity experiments for attribution signals once the ranker stabilizes.
- Full evaluation + release: cross-operator/GPU ablations, MetaSchedule-in-loop experiments, and packaging of datasets/scripts.

## Coordination Notes
- Risks around cross-GPU access and feature-extraction overhead are logged in `Project Proposal.md` Section 7; revisit mitigations before entering each milestone.
- MetaSchedule API mismatches and integration tasks are outlined in `training_plan.md`; prioritize that wrapper before attempting in-loop tuning.
- Record any changes to success metrics or hardware targets here so other sessions can reconcile differences quickly.
