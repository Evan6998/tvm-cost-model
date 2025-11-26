# TVM Low-Level Kernel Cost Model

This repository contains scaffolding for a transferable, invariant, and explainable GPU kernel cost model that integrates with TVM MetaSchedule.

## Layout
- `Project Proposal.md`: detailed research proposal and goals.
- `plan.md`: rolling status tracker.
- `src/tvm_cost_model/`: Python package with modules for data collection, feature extraction, modeling, training, integration, and utilities.
- `configs/`: experiment and dataset configuration files.
- `scripts/`: executable entry points for dataset generation, training, and deployment.
- `tests/`: placeholder tests.
- `notebooks/`: exploration workspace (empty placeholder).

## Getting Started
1. Create and activate the virtual environment (already bootstrapped as `.venv`).
2. Install the package in editable mode with dev dependencies:
   ```bash
   source .venv/bin/activate
   python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

   pip install -e .[dev]
   ```
3. Run stub workflows:
   ```bash
   python scripts/bootstrap_dataset.py
   python scripts/train_cost_model.py
   ```

## Next Steps
Implementation tasks are tracked in `plan.md` and mirrored in the proposal milestones. Flesh out each module with the actual TVM MetaSchedule integrations, data export logic, and learning pipeline.
