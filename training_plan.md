# TVM Cost Model – Training & MetaSchedule Integration Plan

This document summarizes:

1. The intended training / prediction behavior for a TVM-style cost model (including both pointwise and pairwise training).
2. Issues / misalignments in the current implementation (`GraphCostModel`, `TrainingPipeline`, `MetaScheduleAdapter`).
3. The missing or incorrect parts in the external interface for integrating with TVM MetaSchedule (`PyCostModel`).

The goal is to drive concrete refactors that make the model usable as a proper MetaSchedule cost model while keeping the current offline training utilities.

---

## 1. Design Goals and Reference Behavior

### 1.1 TVM MetaSchedule cost model API (ground truth)

According to TVM’s Python API, a MetaSchedule cost model is represented by `tvm.meta_schedule.CostModel` with the following key methods:  [oai_citation:0‡Apache Git Repositories](https://apache.googlesource.com/tvm/%2B/refs/heads/main/python/tvm/meta_schedule/cost_model/cost_model.py)

- `CostModel.update(context: TuneContext, candidates: List[MeasureCandidate], results: List[RunnerResult]) -> None`  
  Update the cost model given profiling results.

- `CostModel.predict(context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray`  
  Predict **normalized scores** (larger = better) for a batch of measure candidates.

For Python-side customization, TVM exposes `PyCostModel`, which is the user-facing abstract base:  [oai_citation:1‡Apache Git Repositories](https://apache.googlesource.com/tvm/%2B/refs/heads/main/python/tvm/meta_schedule/cost_model/cost_model.py)

```python
class PyCostModel:
    def load(self, path: str) -> None: ...
    def save(self, path: str) -> None: ...
    def update(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None: ...
    def predict(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
    ) -> np.ndarray: ...
```

Important behavior:
	•	predict must return an np.ndarray of float64, one score per candidate.
	•	Scores are normalized and larger is better. (The TVM C++ docs for PyCostModelNode::Predict say “Predict the normalized score (the larger the better)” for given measure candidates.)  ￼

1.2 What MetaSchedule passes into update / predict
	•	TuneContext carries global tuning info (module, target, space generator, etc.).  ￼
	•	MeasureCandidate wraps a tvm.tir.Schedule (sch) and args_info metadata.  ￼
	•	RunnerResult contains:
	•	run_secs: Optional[List[FloatImm]] – runtimes from repeats;
	•	error_msg: Optional[str] – if any error occurred.  ￼

TVM’s internal utilities compute a representative cost (e.g., median runtime in milliseconds) from RunnerResult.run_secs.  ￼

1.3 Existing ML cost models in TVM & recent papers
	•	AutoTVM XGBoost cost model:
	•	Supports both "reg" and "rank" loss types.
	•	"reg": regression loss; predicts normalized FLOPs (or runtime proxy).
	•	"rank": pairwise rank loss; predicts relative ranking scores.
	•	MetaSchedule / TLP-style models (MLP, etc.) are primarily regression-based latency predictors; they still serve scores per candidate and are used for ranking / search.  ￼
	•	CALO-GNN (2025) is a graph cost model for TVM that performs graph regression with uncertainty (evidential regression), predicting latency distributions instead of just point estimates.

Conclusion for design:
	•	The external interface is always:
per-candidate score (float), larger is better, via predict(context, candidates).
	•	The training can be:
	•	Pointwise regression (predict latency / FLOPs, etc.), or
	•	Pairwise ranking (margin ranking loss, etc.), or
	•	A combination (pretrain with regression, finetune with ranking).

⸻

2. Current Implementation Overview

2.1 GraphCostModel

Key pieces:
	•	Prediction dataclass:

@dataclass
class Prediction:
    score: float
    attribution: dict[str, float]


	•	Model constructor:

class GraphCostModel:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        margin: float = 0.1,
        weight_decay: float = 0.0,
        hidden_dim: int = 64,
    ) -> None:
        self.encoder = GraphEncoder()
        self._model: NodeMLPRanker | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._margin_loss = nn.MarginRankingLoss(margin=margin)
        ...


	•	Prediction path:

def predict(self, graph: ProgramGraph) -> Prediction:
    encoding = self.encoder.encode(graph)
    self._ensure_model(len(encoding.feature_names))
    output: RankerOutput = self._model(encoding)
    return Prediction(
        score=float(output.score.detach().item()),
        attribution={...},
    )

	•	One scalar score per graph (larger is better).
	•	Also returns node-level attribution from NodeMLPRanker.

	•	Pointwise update (regression):

def update(self, graphs: Sequence[ProgramGraph], scores: Sequence[float]) -> None:
    self.encoder.prime_feature_names(graphs)
    encodings = [self.encoder.encode(graph) for graph in graphs]
    self._ensure_model(feature_dim)
    preds = torch.stack([self._model(enc).score for enc in encodings])
    target = torch.tensor(list(scores), dtype=torch.float32)
    loss = mean((preds - target) ** 2)
    optimizer.step()

	•	Standard MSE regression on score vs. target float.

	•	Pairwise training (ranking):

def train_on_pairs(self, pairs: Sequence[EncodedPair], epochs: int, batch_size: int = 32) -> float:
    # EncodedPair has .better, .worse, .difficulty
    for epoch in range(epochs):
        for batch in batches:
            better_score, worse_score = self._model.score_pair(...)
            loss = MarginRankingLoss(margin)(better_score, worse_score, target=+1)

	•	Margin ranking loss: encourages score(better) > score(worse) + margin.
	•	Uses encoded features (via GraphEncoder) rather than raw ProgramGraph.

2.2 TrainingPipeline

class TrainingPipeline:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.builder = TVMGraphBuilder()
        self.model = GraphCostModel(...)

	•	Pointwise path (fit):

def fit(self, tir_modules: Iterable[str], scores: Iterable[float]) -> None:
    graphs = [self._build_graph(tir) for tir in tir_modules]
    self.model.update(graphs, list(scores))


	•	Prediction utility (single TIR module):

def predict(self, tir_module: str) -> Prediction:
    graph = self._build_graph(tir_module)
    return self.model.predict(graph)


	•	Pairwise ranking path (fit_measurements):

def fit_measurements(self, measurements: Sequence[MeasurementRecord]) -> int:
    graphs = [self._build_graph(m.scheduled_tir or m.original_tir) for m in measurements]
    self.model.encoder.prime_feature_names(graphs)
    encodings = {id(m): self.model.encoder.encode(g) for m, g in zip(measurements, graphs)}

    pairs = sample_ranking_pairs(measurements, num_pairs=..., easy_frac=..., hard_frac=..., seed=...)
    encoded_pairs: list[EncodedPair] = []
    for pair in pairs:
        better_enc = encodings.get(id(pair.better))
        worse_enc = encodings.get(id(pair.worse))
        ...
        encoded_pairs.append(EncodedPair(better=better_enc, worse=worse_enc, difficulty=pair.difficulty))

    self.model.train_on_pairs(encoded_pairs, epochs=self.config.epochs, batch_size=self.config.batch_size)

	•	Uses MeasurementRecord (from the dataset builder) and sample_ranking_pairs to generate ranking pairs.
	•	This is the main path for pairwise training.

2.3 MetaScheduleAdapter (current)

class MetaScheduleAdapter:
    """Minimal interface mirroring PyCostModel expectations."""

    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()

    def predict(self, context: Any) -> float:
        """Return a dummy score (higher is better) for a MetaSchedule trace."""
        tir_module = getattr(context, "tir", "")
        prediction = self.pipeline.predict(tir_module)
        return prediction.score

    def update(self, context: Any, measured_cost: float) -> None:
        """Placeholder update hook converting runtime to a ranking score."""
        score = -measured_cost  # lower runtime => higher score
        self.pipeline.fit([getattr(context, "tir", "")], [score])

	•	This adapter is not yet wired into tvm.meta_schedule.cost_model.PyCostModel; it is just a standalone class.
	•	It takes a single context and (for update) a single float measured_cost.

⸻

3. Issues / Misalignments in the Current Implementation

3.1 MetaScheduleAdapter API does not match PyCostModel

Problem:
	•	Current adapter methods:

def predict(self, context: Any) -> float
def update(self, context: Any, measured_cost: float) -> None


	•	Required PyCostModel methods:  ￼

def update(
    self,
    context: TuneContext,
    candidates: List[MeasureCandidate],
    results: List[RunnerResult],
) -> None

def predict(
    self,
    context: TuneContext,
    candidates: List[MeasureCandidate],
) -> np.ndarray



Impact:
	•	This adapter cannot be plugged directly into MetaSchedule as a cost model.
	•	It cannot handle batched prediction / update, which is how MetaSchedule actually calls the cost model.

Required fix:
	•	Introduce a new adapter (or refactor this one) to strictly implement PyCostModel’s update/predict signatures.
	•	Handle lists of candidates and results; return an np.ndarray of scores.

3.2 Adapter ignores MeasureCandidate and RunnerResult

Problem:
	•	Current predict and update work only with context and assume context.tir is a string.
	•	It does not accept or process MeasureCandidate objects (which actually contain the schedules to be measured).  ￼
	•	It does not use RunnerResult (which contains the actual measured runtimes).  ￼

Impact:
	•	There is no way to access:
	•	The actual Schedule objects (candidate.sch) from MetaSchedule.
	•	The detailed runtime information (result.run_secs) and error codes (result.error_msg).
	•	As a result, the model cannot be trained online using MetaSchedule’s real measurement stream.

Required fix:
	•	In update(context, candidates, results):
	•	For each (candidate, result):
	•	Extract the scheduled TIR (e.g., from candidate.sch.mod or a PrimFunc inside that module).
	•	Extract a scalar cost from result.run_secs (e.g., median runtime in seconds or ms; if run_secs is missing or error_msg is non-empty, consider ignoring / marking as bad sample).  ￼
	•	Convert them into internal MeasurementRecord objects or directly call a new TrainingPipeline.fit_from_meta_schedule(...) helper that:
	•	Builds ProgramGraph from the scheduled TIR;
	•	Uses either fit_measurements (pairwise) or a dedicated pointwise path.

3.3 Pairwise vs pointwise training not aligned with MetaSchedule path

Problem:
	•	TrainingPipeline.fit_measurements(...) is designed as the primary pairwise ranking training path, using MeasurementRecord + sample_ranking_pairs.
	•	MetaScheduleAdapter.update(...) currently calls pipeline.fit (pointwise regression) with:

score = -measured_cost
self.pipeline.fit([tir], [score])



Impact:
	•	The main MetaSchedule integration path would only train the model pointwise, and only with 1-sample mini-batches, which:
	•	Ignores the more sophisticated pairwise ranking you’ve already implemented.
	•	May be unstable and inefficient (updating after every single sample).
	•	The gap between ~“research design” (pairwise ranking based on runtime gaps) and the actual integration path is large.

Required fix:
	•	Decide the policy for MetaSchedule-driven training. Suggested:
	1.	Primary path: use fit_measurements with ranking pairs.
	•	Accumulate a buffer of (candidate, result) pairs.
	•	Periodically (e.g., every N new results), create MeasurementRecord objects from this buffer and call pipeline.fit_measurements(...).
	2.	Optional: add a pointwise regression path for pretraining or ablations:
	•	Implement a helper that maps RunnerResult → scalar score and calls GraphCostModel.update(...) for multiple graphs at once.
	•	This should be a batch update, not one sample at a time.
	•	Update the new PyCostModel.update(...) implementation to follow (1) and/or (2).

3.4 Input type mismatch in _build_graph

Current signature:

def _build_graph(self, tir_module: str) -> ProgramGraph:
    return self.builder.build(tir_module)

Problem:
	•	For offline dataset builder, tir_module: str (TIR script text) may be appropriate.
	•	For MetaSchedule integration, MetaSchedule will give you:
	•	MeasureCandidate.sch: tvm.tir.Schedule
	•	TuneContext.mod: IRModule (original TIR).  ￼
The code currently assumes it always receives a string.

Impact:
	•	Integration code will either:
	•	Need to convert Schedule / IRModule into a TIR script string to satisfy this API, or
	•	Change _build_graph / GraphBuilder to accept these more natural TVM objects.

Required fix:
	•	Generalize _build_graph and GraphBuilder (and TVMGraphBuilder) to accept broader input types, e.g.:

def _build_graph(self, tir_module: Any) -> ProgramGraph:
    return self.builder.build(tir_module)


	•	Then let TVMGraphBuilder.build(...) handle:
	•	str (TIR script text) for offline dataset.
	•	IRModule / PrimFunc / Schedule for online MetaSchedule use.

(Agent should check existing GraphBuilder interface and update type hints + implementations accordingly.)

3.5 Scoring direction and normalization not clearly defined

Current practice:
	•	In MetaScheduleAdapter.update, score = -measured_cost implies:
	•	Lower runtime → higher score.
	•	In ranking training:
	•	Pairs are constructed via sample_ranking_pairs(measurements, easy_frac, hard_frac, ...) (relative latency differences).
	•	In regression training:
	•	scores are arbitrary floats (user-provided).

Potential issues:
	•	No single, canonical transformation from latency (run_secs or MeasurementRecord.cost) to model target:
	•	Should the regression target be latency, -latency, log(latency), normalized latency, etc.?
	•	Should ranking pairs be filtered or weighted by difficulty (difficulty is present but currently not used in the loss)?

Required fix:
	•	Decide and document a measurement_to_score policy, e.g.:

def measurement_to_score(latency_ms: float) -> float:
    # Example: negative log-latency so larger is better
    return -math.log(latency_ms + 1e-6)


	•	Ensure consistency between:
	•	Pairwise sampling (what is “better” vs “worse”).
	•	Pointwise regression targets.
	•	MetaSchedule’s expectation that higher predicted value = better candidate.

⸻

4. Target Interface and Refactors

4.1 Define a PyCostModel-compatible wrapper

Create a new class, e.g. GraphPyCostModel, that subclasses tvm.meta_schedule.cost_model.PyCostModel and internally uses TrainingPipeline / GraphCostModel.

Rough shape (agent should fill in real code):

from tvm.meta_schedule.cost_model import PyCostModel
from tvm.meta_schedule import TuneContext, MeasureCandidate, RunnerResult

class GraphPyCostModel(PyCostModel):
    def __init__(self, config: TrainingConfig | None = None) -> None:
        super().__init__()
        self.pipeline = TrainingPipeline(config)
        # Optionally: internal buffers for online training
        self._pending_measurements: list[MeasurementRecord] = []

    def update(
        self,
        context: TuneContext,
        candidates: list[MeasureCandidate],
        results: list[RunnerResult],
    ) -> None:
        # 1) Convert (candidate, result) -> MeasurementRecord list
        # 2) Append to self._pending_measurements
        # 3) Periodically call self.pipeline.fit_measurements(...)
        ...

    def predict(
        self,
        context: TuneContext,
        candidates: list[MeasureCandidate],
    ) -> np.ndarray:
        # 1) For each candidate, extract scheduled TIR / IRModule
        # 2) Build ProgramGraph using self.pipeline._build_graph(...)
        # 3) Call self.pipeline.model.predict(...) or a batched helper
        # 4) Return np.array of scores, dtype float64
        ...

Notes:
	•	Use TVM’s @derived_object decorator if needed to register this as a derived object (same pattern as FeatureExtractor / other MetaSchedule customization).  ￼
	•	Make sure predict returns np.ndarray with dtype=="float64" (TVM asserts this).  ￼

4.2 Implement (candidate, result) -> MeasurementRecord conversion

Agent should implement a helper (in some shared module, e.g. integration/utils.py):

def runner_result_to_latency_ms(result: RunnerResult) -> float | None:
    # Use median of run_secs if available; otherwise return None
    ...

def candidate_to_tir(candidate: MeasureCandidate) -> tvm.IRModule | str:
    # Extract scheduled TIR from candidate.sch (tvm.tir.Schedule)
    # For now, using candidate.sch.mod is reasonable
    ...

def pack_measurements(
    candidates: list[MeasureCandidate],
    results: list[RunnerResult],
) -> list[MeasurementRecord]:
    # Build MeasurementRecord objects with fields:
    # - original_tir / scheduled_tir
    # - cost / latency
    # - metadata (target, shape, etc.) if available
    ...

	•	The runner_result_to_latency_ms should handle error_msg and missing run_secs (skip such samples or mark them as invalid).  ￼
	•	The cost should be oriented such that lower latency is better; then use the measurement_to_score function to convert to a model score where higher is better.

4.3 Wire fit_measurements as the main online training path

Within GraphPyCostModel.update(...):
	•	Append newly constructed MeasurementRecord objects to an internal buffer.
	•	When buffer size is ≥ some threshold (configurable, e.g. 128 or 256 samples):

used = self.pipeline.fit_measurements(self._pending_measurements)
# Optionally drop used samples or keep a sliding window
self._pending_measurements.clear()


	•	This ensures:
	•	Training is pairwise ranking-based, consistent with your research goal.
	•	Each training call has enough data to build diverse pairs.

4.4 Keep GraphCostModel.update for pointwise regression

We keep GraphCostModel.update(graphs, scores) as a pointwise regression entry point:
	•	Use cases:
	•	Offline pretraining on TenSet / historical MetaSchedule records (if you load them into MeasurementRecord and then derive a scalar target).
	•	Ablation experiments (pure regression vs pure ranking vs mixed).

Add a dedicated pipeline helper for pointwise MetaSchedule training (optional):

def fit_pointwise_from_meta_schedule(
    self,
    candidates: list[MeasureCandidate],
    results: list[RunnerResult],
) -> None:
    # Transform to graphs + scores and call self.model.update(...)
    ...

This helper is not required for correctness, but useful for experiments.

4.5 Normalize types in TrainingPipeline and GraphBuilder

Refactor TrainingPipeline._build_graph and GraphBuilder.build to support both:
	•	Offline: str (TIR script).
	•	Online: IRModule, PrimFunc, or Schedule.

Concretely:
	•	Change type hints from str to Any or a Union[...].
	•	In TVMGraphBuilder, implement type dispatch:
	•	If input is str: parse via tvm.script.from_source (or similar).
	•	If input is Schedule: get schedule.mod or extract the scheduled PrimFunc.
	•	If input is IRModule: use as-is.

This makes the same pipeline usable for both dataset builder and MetaSchedule integration.

⸻

5. Actionable Checklist for Implementation

For the AI agent, the main tasks are:
	1.	Create a MetaSchedule-compatible cost model wrapper
	•	Implement GraphPyCostModel(PyCostModel) with update / predict signatures exact to TVM.
	•	Ensure predict returns np.ndarray of float64.
	•	Ensure scores follow “larger is better”.
	2.	Implement conversion from MetaSchedule objects to internal data
	•	Implement runner_result_to_latency_ms(result: RunnerResult) -> Optional[float].
	•	Implement candidate_to_tir(candidate: MeasureCandidate) to extract scheduled TIR.
	•	Implement pack_measurements(candidates, results) -> list[MeasurementRecord].
	3.	Plug pairwise training into update
	•	Maintain an internal buffer of MeasurementRecord in GraphPyCostModel.
	•	When buffer length ≥ threshold, call TrainingPipeline.fit_measurements and clear / shrink the buffer.
	•	Make sure MeasurementRecord fields (original_tir, scheduled_tir, cost) are populated consistently with offline dataset builder.
	4.	Clarify / implement score transformation
	•	Add a single measurement_to_score(latency_ms) helper.
	•	Use this both in regression targets and to define “better” vs “worse” in ranking pairs (if not already handled by sample_ranking_pairs).
	•	Confirm scoring direction is consistent: higher = better everywhere.
	5.	Generalize TrainingPipeline._build_graph & TVMGraphBuilder
	•	Update type hints to accept str and TVM TIR objects.
	•	Implement internal dispatch in TVMGraphBuilder.build(...).
	•	Ensure encoder priming (encoder.prime_feature_names(graphs)) works for all input types.
	6.	Remove / deprecate the old MetaScheduleAdapter
	•	Either:
	•	Replace it with the new GraphPyCostModel, or
	•	Make MetaScheduleAdapter a thin wrapper used only inside GraphPyCostModel, but not exposed as an external API.
	•	Ensure there is a single, clearly documented entry point for MetaSchedule integration.

⸻

Once these changes are made, the system will:
	•	Use pairwise ranking on MeasurementRecords as the main training signal.
	•	Support pointwise regression via GraphCostModel.update for pretraining / ablations.
	•	Expose a proper PyCostModel to TVM MetaSchedule, fully aligned with the official update/predict interface.
