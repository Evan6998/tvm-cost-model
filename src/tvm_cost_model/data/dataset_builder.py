"""Data generation and ingestion utilities for GPU kernel schedules."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ScheduleSample:
    """Represents a single sampled schedule candidate."""

    operator: str
    schedule_json: str
    tir: str
    workload_shape: Dict[str, tuple[int, ...]]


@dataclass
class MeasurementRecord:
    """Represents a measurement ready to be written to disk."""

    operator: str
    schedule_json: str
    tir: str
    workload_shape: Dict[str, tuple[int, ...]]
    runtime_ms: float
    hardware_id: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "operator": self.operator,
            "schedule_json": self.schedule_json,
            "tir": self.tir,
            "workload_shape": json.dumps(self.workload_shape),
            "runtime_ms": self.runtime_ms,
            "hardware_id": self.hardware_id,
        }


class ScheduleSampler(Protocol):
    """Interface for anything that can produce schedule candidates."""

    def sample(self, operator: str, batch: int) -> Iterable[ScheduleSample]:
        """Yield serialized schedules for the given operator."""
        ...


class RuntimeEvaluator(Protocol):
    """Interface for evaluating a sampled schedule."""

    def evaluate(self, sample: ScheduleSample, hardware_id: str) -> float:
        """Return a runtime prediction in milliseconds."""
        ...


class DatasetBuilder:
    """Builds Arrow/Parquet datasets from MetaSchedule samples."""

    def __init__(
        self,
        sampler: ScheduleSampler,
        evaluator: RuntimeEvaluator,
        output_dir: Path,
    ) -> None:
        self._sampler = sampler
        self._evaluator = evaluator
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        operator: str,
        batches: int,
        batch_size: int,
        hardware_id: str,
    ) -> List[MeasurementRecord]:
        """Collect measurement records for the requested operator."""

        measurements: List[MeasurementRecord] = []
        for _ in range(batches):
            for sample in self._sampler.sample(operator, batch=batch_size):
                runtime_ms = self._evaluator.evaluate(sample, hardware_id)
                measurements.append(
                    MeasurementRecord(
                        operator=sample.operator,
                        schedule_json=sample.schedule_json,
                        tir=sample.tir,
                        workload_shape=sample.workload_shape,
                        runtime_ms=runtime_ms,
                        hardware_id=hardware_id,
                    )
                )
        return measurements

    def export(self, measurements: List[MeasurementRecord], artifact_name: str) -> Path:
        """Write measurements to a Parquet file and return its path."""

        if not measurements:
            raise ValueError("No measurements provided for export")
        output = self._output_dir / f"{artifact_name}.parquet"
        table = pa.Table.from_pylist([m.as_dict() for m in measurements])
        pq.write_table(table, output) # type: ignore
        return output


class SyntheticScheduleSampler:
    """Produces pseudo-random schedules to unblock tooling work."""
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def sample(self, operator: str, batch: int) -> Iterable[ScheduleSample]:
        for idx in range(batch):
            shape = {
                "m": (self._rng.randint(64, 4096),),
                "n": (self._rng.randint(64, 4096),),
                "k": (self._rng.randint(64, 4096),),
            }
            schedule = {
                "tile_m": self._rng.choice([16, 32, 64]),
                "tile_n": self._rng.choice([16, 32, 64]),
                "unroll_k": self._rng.choice([2, 4, 8]),
            }
            tir = f"// synthetic {operator} schedule {idx}\n// shape={shape}\n"
            yield ScheduleSample(
                operator=operator,
                schedule_json=json.dumps(schedule),
                tir=tir,
                workload_shape=shape,
            )


class SyntheticRuntimeEvaluator:
    """Generates deterministic runtimes from schedule metadata."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def evaluate(self, sample: ScheduleSample, hardware_id: str) -> float:
        scale = sum(sum(dim) for dim in sample.workload_shape.values()) / 1024.0
        knob_penalty = json.loads(sample.schedule_json)["unroll_k"] * 0.1
        noise = self._rng.random() * 0.05
        hardware_factor = 0.8 if "ada" in hardware_id.lower() else 1.0
        return (scale / hardware_factor) + knob_penalty + noise
