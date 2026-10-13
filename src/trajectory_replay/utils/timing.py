"""Wall-clock timing helpers for run metrics dumps."""

from __future__ import annotations

import time
from typing import Any


def training_timing_record(
    *,
    wall_seconds: float,
    epochs: int,
    steps_per_epoch: int,
    finetune_strategy: str | None = None,
    replay_backend: str | None = None,
    flow_map_steps: int | None = None,
    mean_flow_steps: int | None = None,
    val_monitor_runs: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable timing block for ``metrics.json``."""
    train_steps = int(epochs) * int(steps_per_epoch)
    wall = float(wall_seconds)
    record: dict[str, Any] = {
        "train_wall_seconds": wall,
        "train_epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "train_steps": train_steps,
        "seconds_per_epoch": wall / max(int(epochs), 1),
        "seconds_per_step": wall / max(train_steps, 1),
        "finetune_strategy": finetune_strategy,
        "replay_backend": replay_backend,
        "flow_map_steps": flow_map_steps,
        "mean_flow_steps": mean_flow_steps,
        "val_monitor_runs": int(val_monitor_runs),
    }
    if extra:
        record.update(extra)
    return record


class WallTimer:
    """Context manager wrapping :func:`time.perf_counter`."""

    def __enter__(self) -> WallTimer:
        self._start = time.perf_counter()
        self._end: float | None = None
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return float(end - self._start)
