# TVM Low-Level Kernel Cost Model

Transferable, invariant, and explainable GPU kernel cost model research scaffold that plugs into TVM MetaSchedule. The code builds graph representations from TIR, mines ranking pairs, and trains lightweight ranking heads to order schedules.

## Current Status
- Dataset builder produces Parquet measurement shards from MetaSchedule workloads (vecadd, gemm, bmm, conv2d, depthwise, layernorm, softmax) with hardware metadata and shape overrides.
- Graph extraction and encoding paths are online (`TVMGraphBuilder`, `GraphEncoder`, ranking dataset builders); coverage is expanding to more loop/thread attributes.
- Ranking-first modeling flow exists with a baseline Node-MLP ranker and encoded pair sampling; R-GAT bring-up is next.
- Near-term work: broaden workload/shape sweeps, add hard-negative sampling for schedules, and run measurements on real Ampere/Ada GPUs (see `plan.md`).

## Repository Layout
- `src/tvm_cost_model/features/graph_builder.py`: base graph and node/edge data structures.
- `src/tvm_cost_model/features/tvm_graph_builder.py`: TVM-backed TIR visitor extracting loop/buffer graphs and flop/byte stats.
- `src/tvm_cost_model/features/graph_encoder.py`: stable vocab + dense feature encoding for program graphs.
- `scripts/bootstrap_dataset.py`: bootstrap measurement datasets from built-in workloads; exports Parquet.
- `scripts/sweep_workloads.py`: multi-workload sweep driver with shape overrides and shard merging.
- `scripts/train_cost_model.py`: entry point for the ranking baseline.
- `configs/`: dataset/model configuration examples.
- `tests/`: sanity tests for feature extraction (expand as functionality grows).
- `Project Proposal.md`, `plan.md`: research plan and rolling status tracker.

## Setup
1. **Prereqs:** Python 3.10+, PyTorch 2.3+, PyArrow 20.0. Install a TVM Python wheel separately (e.g., `pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly-cpu`) or build TVM from source with TIR/MetaSchedule enabled.
2. **Create env & install:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .[dev]
   ```
3. **(Optional) Verify TVM import:** `python - <<'PY'\nimport tvm\nprint(tvm.__version__)\nPY`

## Quickstart
- Bootstrap a small dataset (writes Parquet under `artifacts/` by default):
  ```bash
  python scripts/bootstrap_dataset.py --workloads vecadd gemm --target llvm -o artifacts/vecadd_gemm
  ```
- Sweep multiple workloads/shapes:
  ```bash
  python scripts/sweep_workloads.py --workloads conv2d depthwise --target cuda -o artifacts/conv_sweep
  ```
- Train the baseline ranker on encoded pairs:
  ```bash
  python scripts/train_cost_model.py --config configs/sample_ranker.yaml
  ```

## Testing
- Run the focused feature tests (requires TVM available in the environment):
  ```bash
  pytest tests/test_tvm_graph_builder.py
  ```

## Contributing Notes
- Keep feature extraction deterministic across sessions; update `GraphEncoder` vocab only via controlled additions.
- Prefer extending config files under `configs/` for new workloads/targets rather than hard-coding paths in scripts.
- Track roadmap and open questions in `plan.md`; major scope shifts should also be reflected in `Project Proposal.md`.
