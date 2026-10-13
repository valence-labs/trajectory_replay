#!/usr/bin/env python3
"""Build 2-mode MoG task data and reference mixtures for a λ grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sim_mixture.common import (
    DEFAULT_LAMBDAS,
    REPO_ROOT,
    build_two_task_data,
    mixture_data_root,
    save_dataset_bundle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--separation", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--samples-per-task", type=int, default=10_000)
    p.add_argument("--mixture-n-samples", type=int, default=10_000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="Mixture weights λ for reference datasets (mass on mode 0).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_mixture/data/dim<d>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or mixture_data_root(REPO_ROOT, dim=args.dim)
    ds = build_two_task_data(
        dim=args.dim,
        separation=args.separation,
        sigma=args.sigma,
        samples_per_task=args.samples_per_task,
        data_seed=args.data_seed,
    )
    meta = save_dataset_bundle(
        out_dir,
        ds,
        sigma=args.sigma,
        separation=args.separation,
        data_seed=args.data_seed,
        lambdas=tuple(float(x) for x in args.lambdas),
        mixture_n_samples=args.mixture_n_samples,
    )
    print(f"Wrote tasks + {len(args.lambdas)} mixtures under {out_dir}")
    print(f"means=\n{ds.means}")
    print(f"meta={meta}")


if __name__ == "__main__":
    main()
