#!/usr/bin/env python3
"""Render MeanFlow sample scatters at one or more sampling budgets (N steps).

Scans finetune run dirs holding a ``mean_flow.pt`` and writes
``samples_mean_flow_N{n}.png`` for each requested ``--n-steps``. Read-only w.r.t.
training artifacts; meant to run as a batch job, not on the login node.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from experiments.sim_mixture.common import (
    REPO_ROOT,
    evaluate_mean_flow,
    load_mixture,
    load_tasks,
    mixture_data_root,
    mixture_runs_root,
    save_scatter,
)
from trajectory_replay.training.mean_flow_distill import load_mean_flow_checkpoint
from trajectory_replay.utils.cli import resolve_device


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument(
        "--glob",
        default="ft_*_mf_e*_seed*",
        help="Run-dir glob under <runs>/finetune (default: all _mf runs).",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        nargs="+",
        default=[4],
        help="MeanFlow sampling budgets to render (space-separated).",
    )
    p.add_argument("--n-gen", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1999)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else mixture_data_root(REPO_ROOT, dim=args.dim)
    ds, meta = load_tasks(data_dir)
    sigma = float(meta["sigma"])
    finetune_root = mixture_runs_root(REPO_ROOT, dim=args.dim) / "finetune"

    run_dirs = sorted(glob.glob(str(finetune_root / args.glob)))
    if not run_dirs:
        raise SystemExit(f"No runs matched {finetune_root / args.glob}")

    for d in run_dirs:
        run = Path(d)
        ckpt = run / "mean_flow.pt"
        if not ckpt.exists():
            print(f"skip {run.name}: no mean_flow.pt")
            continue
        cfg = json.load(open(run / "metrics.json"))["config"]
        lam = float(cfg["lambda_mix"])
        strat = cfg["strategy"]
        x_mix = load_mixture(data_dir, lam).x
        model, _ = load_mean_flow_checkpoint(ckpt, device)
        model.eval()
        for n in args.n_steps:
            metrics, x_gen = evaluate_mean_flow(
                model,
                ds,
                device=device,
                n_gen=args.n_gen,
                seed=args.seed,
                n_steps=n,
                lambda_target=lam,
                sigma=sigma,
                x_true_mixture=x_mix,
            )
            out = run / f"samples_mean_flow_N{n}.png"
            save_scatter(
                out,
                x_gen=x_gen,
                means=ds.means,
                title=f"MeanFlow {strat} λ={lam} N={n}",
                x_true=ds.x,
            )
            print(
                f"{run.name:40s} N={n} pi0={metrics['pi0_hat']:.3f} "
                f"E={metrics['energy_mixture']:.4f} -> {out.name}"
            )


if __name__ == "__main__":
    main()
