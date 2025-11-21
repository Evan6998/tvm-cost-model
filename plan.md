# Low-Level Kernel Cost Model Plan

This file tracks the state of the project so that any collaborator or future session can immediately understand progress.

## Completed
- **Proposal overhaul & scope clarification**: introduced success criteria, risks, and detailed methodology sections explaining graph-based modeling, invariance, and explainability requirements.
- **Evaluation rubric definition**: locked in quantitative metrics (nRMSE, Kendall Tau, measurement reduction targets, attribution fidelity) that downstream work must satisfy.
- **Repository scaffold & tooling bootstrap**: initialized the Python package structure (`src/tvm_cost_model`), CLI scripts, editable installation metadata, placeholder tests, and a `.venv` for reproducible development.

## In Progress
- **Hardware & dataset provisioning**: finalizing access to two target NVIDIA GPUs (Ampere + Ada) and consolidating public traces (e.g., TenSet) into a unified schema; blocked only on confirming measurement quotas.
- **Operator coverage selection**: iterating on the exact list of kernels and shape distributions with the TVM MetaSchedule benchmarking scripts to ensure ≥50k labeled schedules per GPU.
- **Measurement tooling build-out**: synthetic sampler + PyArrow writer (`DatasetBuilder`, `scripts/bootstrap_dataset.py`) validate the collection/export flow; needs replacement with real MetaSchedule runners once GPU access is confirmed.

## Pending / Upcoming
- **Graph extraction & canonicalization**: TIR-to-graph pipeline plus invariance augmentations (Step 2).
- **Model prototyping**: implement the R-GAT backbone with regression and attribution heads, then benchmark against TVM’s XGBoost cost model (Step 3).
- **Explainability validation**: visualization tooling and fidelity experiments for attribution signals (Step 4).
- **MetaSchedule integration**: PyCostModel wrapper, fallback logic, and telemetry plumbing (Step 5).
- **Full evaluation + release**: cross-operator/GPU experiments, ablations, and packaging of datasets/scripts (Step 6).

## Coordination Notes
- Risks around cross-GPU access and feature-extraction overhead are logged in `Project Proposal.md` under Section 7; revisit mitigations before entering each milestone.
- Record any changes to success metrics or hardware targets here so other sessions can reconcile differences quickly.
