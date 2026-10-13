"""Beta cooling schedule shared by the gradual fine-tuning trainers."""

from __future__ import annotations

import math

GFT_BETA_SCHEDULES = ("sigmoid", "linear", "constant")


def gft_beta_at_step(
    step: int,
    total_steps: int,
    *,
    beta_start: float,
    beta_end: float,
    schedule: str = "sigmoid",
    sigmoid_midpoint: float = 0.5,
    sigmoid_steepness: float = 10.0,
) -> float:
    """Cooling schedule from ``beta_start`` (step 0) to ``beta_end`` (last step).

    ``schedule`` is ``sigmoid`` (inverse sigmoid, default), ``linear``, or ``constant``.
    For ``constant``, ``beta_start`` is used for every step (``beta_end`` ignored).
    """
    if schedule == "constant":
        return float(beta_start)
    if total_steps <= 1:
        return float(beta_end)
    s = step / (total_steps - 1)
    if schedule == "linear":
        return float(beta_start + (beta_end - beta_start) * s)
    if schedule == "sigmoid":
        x = sigmoid_steepness * (s - sigmoid_midpoint)
        weight = 1.0 / (1.0 + math.exp(x))
        return float(beta_end + (beta_start - beta_end) * weight)
    raise ValueError(
        f"Unknown GFT beta schedule {schedule!r}; expected 'sigmoid', 'linear', or 'constant'."
    )
