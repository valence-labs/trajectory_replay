#!/usr/bin/env python3
"""Supp panels: trajectory leaveout samples in sim_mixture thumbnail style.

One seed, trajectory replay only, one PNG per λ (no mean markers).

Example:
  uv run python experiments/sim_conditional/stage_a_uncond_leaveout/plot_traj_panels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.sim_conditional.stage_a_uncond_leaveout.common import (
    REPO_ROOT,
    SPLIT_VAL,
    load_checkpoint,
    load_stage_a_data,
    mask_rows,
    sample_unconditional,
    stage_a_uncond_data_root,
    stage_a_uncond_runs_root,
)
from experiments.sim_mixture.plot_top_panel import plot_mixture_panel
from trajectory_replay.utils.cli import resolve_device

DEFAULT_LAMBDAS = (0.1, 0.25, 0.5, 0.75, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--panel-seed", type=int, default=0)
    p.add_argument("--n-gen", type=int, default=3000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _lam_tag(lam: float) -> str:
    s = f"{lam:g}".replace(".", "p")
    return f"lam{s}"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runs = stage_a_uncond_runs_root(REPO_ROOT, dim=args.dim)
    data_dir = stage_a_uncond_data_root(REPO_ROOT, dim=args.dim)
    arrays, meta = load_stage_a_data(data_dir)
    heldout = int(meta["heldout_class"])
    pretrain_classes = [int(k) for k in meta["pretrain_classes"]]
    classes = pretrain_classes + [heldout]

    m_true = mask_rows(arrays, classes=classes, split=SPLIT_VAL)
    x_true = np.asarray(arrays["x"][m_true], dtype=np.float64)

    pad = 0.55
    x_lo, x_hi = np.quantile(x_true[:, 0], [0.001, 0.999])
    y_lo, y_hi = np.quantile(x_true[:, 1], [0.001, 0.999])
    xlim = (float(x_lo - pad), float(x_hi + pad))
    ylim = (float(y_lo - pad), float(y_hi + pad))
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    max_aspect = 1.28
    if dx / dy > max_aspect:
        extra = 0.5 * (dx / max_aspect - dy)
        ylim = (ylim[0] - extra, ylim[1] + extra)

    out_dir = args.out_dir or (runs / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for lam in args.lambdas:
        run = (
            runs
            / "finetune"
            / f"ft_trajectory_lam{lam}_e{args.epochs}_seed{args.panel_seed}"
        )
        ckpt = run / "checkpoint.pt"
        if not ckpt.exists():
            print(f"skip λ={lam}: missing {ckpt}")
            continue
        model, _ = load_checkpoint(ckpt, device)
        x_gen = sample_unconditional(
            model,
            args.n_gen,
            device=device,
            seed=10_000 + args.panel_seed + int(100 * lam),
            ode_step_size=args.ode_step_size,
        )
        out = out_dir / f"leaveout_traj_{_lam_tag(lam)}_seed{args.panel_seed}.png"
        plot_mixture_panel(
            out,
            x_gen=x_gen,
            x_true=x_true,
            xlim=xlim,
            ylim=ylim,
            with_kde=False,
            scatter_h=1.15,
            legend_fontsize=8.5,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
