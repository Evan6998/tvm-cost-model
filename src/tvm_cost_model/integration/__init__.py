"""Integration helpers for MetaSchedule with lazy imports to avoid cycles."""

from importlib import import_module
from typing import Any

__all__ = ["GraphPyCostModel", "MetaScheduleAdapter"] # type: ignore[assignment]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("tvm_cost_model.integration.metaschedule_adapter")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
