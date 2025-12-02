"""Data generation and ingestion utilities for GPU kernel schedules."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ScheduleSample:
    """Represents a single sampled schedule candidate."""

    operator: str
    schedule_json: str
    original_tir: str
    scheduled_tir: str  # post-schedule TIR
    workload_shape: Dict[str, tuple[int, ...]]
    workload_key: str | None = None
    target: str | None = None


@dataclass
class MeasurementRecord:
    """Represents a measurement ready to be written to disk."""

    operator: str
    schedule_json: str
    original_tir: str
    scheduled_tir: str  # post-schedule TIR
    workload_shape: Dict[str, tuple[int, ...]]
    runtime_ms: float
    hardware_id: str
    target: str | None = None
    workload_key: str | None = None
    hardware_features: Dict[str, float] | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "operator": self.operator,
            "schedule_json": self.schedule_json,
            "scheduled_tir": self.scheduled_tir,
            "workload_shape": json.dumps(self.workload_shape),
            "runtime_ms": self.runtime_ms,
            "hardware_id": self.hardware_id,
            "target": self.target,
            "workload_key": self.workload_key,
            "original_tir": self.original_tir,
            "hardware_features": json.dumps(self.hardware_features or {}),
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
        artifact_name: str,
        hardware_features: Dict[str, float] | None = None,
        flush_ratio: float = 0.1,
    ) -> Path:
        """Collect measurement records and stream them to a Parquet artifact."""

        if not (0 < flush_ratio <= 1):
            raise ValueError("flush_ratio must be within (0, 1]")

        chunk_size = max(1, int(batch_size * flush_ratio))
        output = self._output_dir / f"{artifact_name}.parquet"
        if output.exists():
            output.unlink()

        writer: pq.ParquetWriter | None = None
        chunk: List[MeasurementRecord] = []

        def flush_chunk() -> None:
            nonlocal writer
            if not chunk:
                return
            table = pa.Table.from_pylist([m.as_dict() for m in chunk])
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema) # type: ignore
            writer.write_table(table) # type: ignore
            chunk.clear()

        try:
            for _ in range(batches):
                samples = self._sampler.sample(operator, batch_size)
                with tqdm(total=batch_size, desc=f"Evaluating {operator}", unit="sched") as pbar:
                    for sample in samples:
                        try:
                            runtime_ms = self._evaluator.evaluate(sample, hardware_id)
                            chunk.append(
                                MeasurementRecord(
                                    operator=sample.operator,
                                    schedule_json=sample.schedule_json,
                                    scheduled_tir=sample.scheduled_tir,
                                    workload_shape=sample.workload_shape,
                                    runtime_ms=runtime_ms,
                                    hardware_id=hardware_id,
                                    target=sample.target,
                                    workload_key=sample.workload_key,
                                    original_tir=sample.original_tir,
                                    hardware_features=hardware_features,
                                )
                            )
                            if len(chunk) >= chunk_size:
                                flush_chunk()
                        except Exception as e:
                            print(f"Error evaluating sample: {e}")
                        pbar.update(1)
            flush_chunk()
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            raise ValueError("No measurements collected for export")

        return output

    def export(self, measurements: List[MeasurementRecord], artifact_name: str) -> Path:
        """Write measurements to a Parquet file and return its path."""

        if not measurements:
            raise ValueError("No measurements provided for export")
        output = self._output_dir / f"{artifact_name}.parquet"
        table = pa.Table.from_pylist([m.as_dict() for m in measurements])
        pq.write_table(table, output) # type: ignore
        return output


def load_measurement_records(path: Path) -> List[MeasurementRecord]:
    """Load MeasurementRecords from a Parquet file."""

    table = pq.read_table(path) # type: ignore
    records: List[MeasurementRecord] = []
    for row in table.to_pylist():
        hardware_features_raw = row.get("hardware_features")
        hardware_features = None
        if hardware_features_raw:
            parsed = json.loads(hardware_features_raw)
            hardware_features = {str(k): float(v) for k, v in parsed.items()}
        records.append(
            MeasurementRecord(
                operator=row.get("operator", ""),
                schedule_json=row.get("schedule_json", "") or "",
                original_tir=row.get("original_tir", "") or "",
                scheduled_tir=row.get("scheduled_tir", "") or "",
                workload_shape=json.loads(row.get("workload_shape", "{}")),
                runtime_ms=float(row.get("runtime_ms", 0.0) or 0.0),
                hardware_id=row.get("hardware_id", "") or "",
                target=row.get("target"),
                workload_key=row.get("workload_key"),
                hardware_features=hardware_features,
            )
        )
    return records


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
                original_tir=tir,
                scheduled_tir=tir,
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
