"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    dataset_root: Path
    output_root: Path


def load_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        dataset_root=Path(data["dataset_root"]),
        output_root=Path(data["output_root"]),
    )
