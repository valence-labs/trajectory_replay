#!/usr/bin/env python3
"""Build the five-task Gaussian ring used for sequential CL experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sim_sequential.common import (
    REPO_ROOT,
    build_dataset_bundle,
    save_dataset_bundle,
    sequential_data_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--train-per-task", type=int, default=10_000)
    parser.add_argument("--val-per-task", type=int, default=2_000)
    parser.add_argument("--test-per-task", type=int, default=5_000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/data/dim<d>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or sequential_data_root(REPO_ROOT, dim=args.dim)
    splits, meta = build_dataset_bundle(
        n_tasks=args.n_tasks,
        dim=args.dim,
        radius=args.radius,
        sigma=args.sigma,
        train_per_task=args.train_per_task,
        val_per_task=args.val_per_task,
        test_per_task=args.test_per_task,
        data_seed=args.data_seed,
    )
    save_dataset_bundle(out_dir, splits, meta)
    print(f"Wrote sequential Gaussian-ring data under {out_dir}")
    print(f"task order: {meta['ordering']}")
    print(f"means:\n{splits['train'].means}")


if __name__ == "__main__":
    main()
