#!/usr/bin/env python3
"""Dimension vs FT wall-time lineplot (trajectory vs generative).

Pulls completed Slurm jobs via ``sacct`` for ``ft_*_lam0.5_e{epochs}_seed*``.

Example:
  uv run python experiments/sim_mixture/plot_dim_vs_time.py --epochs 400
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_mixture.common import REPO_ROOT

METHOD_STYLES = {
    "trajectory": ("Trajectory replay", "#457b9d"),
    "generative": ("Generative replay", "#c1121f"),
}

# Fallback if sacct is unavailable (e400 λ=0.5, seeds 0–2 means from 2026-07 runs).
FALLBACK_E400_SEC: dict[tuple[int, str], list[float]] = {
    (2, "trajectory"): [96, 89, 98],
    (2, "generative"): [110, 115, 112],
    (8, "trajectory"): [99, 88, 104],
    (8, "generative"): [116, 113, 107],
    (16, "trajectory"): [88, 92, 88],
    (16, "generative"): [108, 120, 108],
    (32, "trajectory"): [99, 108, 102],
    (32, "generative"): [122, 123, 120],
    (64, "trajectory"): [100, 100, 101],
    (64, "generative"): [120, 123, 124],
    (128, "trajectory"): [104, 103, 99],
    (128, "generative"): [125, 123, 122],
    (256, "trajectory"): [105, 117, 111],
    (256, "generative"): [125, 127, 136],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 8, 16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--starttime", default="2026-07-20", help="sacct --starttime")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: results_dir/sim_mixture/figures/dim_vs_wall_time_e{epochs}.png)",
    )
    p.add_argument("--use-fallback", action="store_true", help="Skip sacct; use baked-in e400 times.")
    return p.parse_args()


def _parse_elapsed(s: str) -> float | None:
    s = s.strip()
    if not s or s == "Unknown":
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    bits = [int(x) for x in s.split(":")]
    if len(bits) == 3:
        h, m, sec = bits
    elif len(bits) == 2:
        h = 0
        m, sec = bits
    else:
        return None
    return float(days * 86400 + h * 3600 + m * 60 + sec)


def collect_from_sacct(
    *,
    epochs: int,
    dims: set[int],
    seeds: set[int],
    starttime: str,
) -> dict[tuple[int, str], list[float]]:
    """Return ``{(dim, strategy): [wall_seconds, ...]}`` from completed jobs."""
    user = os.environ.get("USER", "")
    out = subprocess.check_output(
        [
            "sacct",
            "-u",
            user,
            f"--starttime={starttime}",
            "--format=JobID,JobName%50,Elapsed,State",
            "--parsable2",
            "-X",
        ],
        text=True,
    )
    # simmix-trajectory-d256-s0  (e400 dim-scaling)
    # simmix-trajectory-d256-e2564-s0
    # simmix-trajectory-d256-lr1e4-s0  (skip)
    pat_ep = re.compile(
        rf"^simmix-(trajectory|generative)-d(\d+)-e{epochs}-s(\d+)$"
    )
    pat_plain = re.compile(r"^simmix-(trajectory|generative)-d(\d+)-s(\d+)$")
    rows: dict[tuple[int, str], list[float]] = defaultdict(list)
    for line in out.strip().splitlines()[1:]:
        _jid, name, elapsed, state = line.split("|")
        if state != "COMPLETED":
            continue
        if "lr1e4" in name or "lr1e-4" in name:
            continue
        sec = _parse_elapsed(elapsed)
        if sec is None:
            continue
        m = pat_ep.match(name)
        if m is None and epochs == 400:
            m = pat_plain.match(name)
        if m is None:
            continue
        strat, dim, seed = m.group(1), int(m.group(2)), int(m.group(3))
        if dim not in dims or seed not in seeds:
            continue
        rows[(dim, strat)].append(sec)
    return dict(rows)


def plot_dim_vs_time(
    path: Path,
    times: dict[tuple[int, str], list[float]],
    *,
    dims: list[int],
    epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for strat in ("trajectory", "generative"):
        label, color = METHOD_STYLES[strat]
        xs, ys, yerr = [], [], []
        for d in dims:
            vals = times.get((d, strat), [])
            if not vals:
                continue
            xs.append(d)
            ys.append(float(np.mean(vals)))
            yerr.append(float(np.std(vals)))
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color,
            marker="o",
            markersize=6.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.6,
            capsize=2.5,
            capthick=0.9,
            elinewidth=0.9,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Ambient dimension $d$")
    ax.set_ylabel("FT wall time (s)")
    ax.set_title(rf"Fine-tune wall time vs dimension ($\lambda=0.5$, {epochs} epochs)")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    dims = list(args.dims)
    out = args.out
    if out is None:
        out = (
            REPO_ROOT
            / "results_dir"
            / "sim_mixture"
            / "figures"
            / f"dim_vs_wall_time_e{args.epochs}.png"
        )

    if args.use_fallback:
        if args.epochs != 400:
            raise SystemExit("--use-fallback only has baked-in times for epochs=400")
        times = {k: v for k, v in FALLBACK_E400_SEC.items() if k[0] in dims}
    else:
        try:
            times = collect_from_sacct(
                epochs=args.epochs,
                dims=set(dims),
                seeds=set(args.seeds),
                starttime=args.starttime,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"sacct failed ({e}); falling back to baked-in e400 times")
            if args.epochs != 400:
                raise
            times = {k: v for k, v in FALLBACK_E400_SEC.items() if k[0] in dims}

    if not times:
        raise SystemExit(f"No wall times found for epochs={args.epochs}")

    print("Means (s):")
    for d in dims:
        for strat in ("trajectory", "generative"):
            vals = times.get((d, strat), [])
            if vals:
                print(f"  d={d:>3} {strat:>11}: {np.mean(vals):6.1f} ± {np.std(vals):4.1f}  (n={len(vals)})")

    plot_dim_vs_time(out, times, dims=dims, epochs=args.epochs)


if __name__ == "__main__":
    main()
