"""Data generation and ingestion utilities for GPU kernel schedules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol


@dataclass
class ScheduleMeasurement:
    """Represents a single schedule measurement entry."""

    schedule_json: str
    runtime_ms: float
    hardware_id: str
    metadata_path: Path | None = None


class ScheduleSampler(Protocol):
    """Interface for anything that can produce schedule candidates."""

    def sample(self, operator: str, batch: int) -> Iterable[str]:
        """Yield serialized schedules for the given operator."""
        ...


class DatasetBuilder:
    """Builds Arrow/Parquet datasets from MetaSchedule samples."""

    def __init__(self, sampler: ScheduleSampler, output_dir: Path) -> None:
        self._sampler = sampler
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, operator: str, batches: int) -> List[ScheduleMeasurement]:
        """Placeholder collection routine; integrate TVM runtime later."""

        measurements: List[ScheduleMeasurement] = []
        for _ in range(batches):
            for schedule in self._sampler.sample(operator, batch=32):
                measurements.append(
                    ScheduleMeasurement(
                        schedule_json=schedule,
                        runtime_ms=0.0,
                        hardware_id="unknown",
                    )
                )
        return measurements

    def export(self, measurements: List[ScheduleMeasurement]) -> Path:
        """Stub for exporting to disk."""

        output = self._output_dir / "placeholder.txt"
        output.write_text(f"dumped {len(measurements)} measurements\n", encoding="utf-8")
        return output
