"""Shared CLI helpers for experiment entrypoints."""

from __future__ import annotations

import sys

import torch


def parse_plot_proj(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--plot-proj must be two comma-separated integers, e.g. 0,1")
    return int(parts[0]), int(parts[1])


def resolve_device(s: str) -> torch.device:
    d = torch.device(s)
    if d.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return d
