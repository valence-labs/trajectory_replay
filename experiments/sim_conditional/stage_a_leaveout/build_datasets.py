#!/usr/bin/env python3
"""Build K-class conditional MoG data with train/val/test splits (Stage A).

Reuses ``sample_gaussian_mixture`` / ``make_polygon_means`` from mixture_mog.
One class is marked held-out for finetune; the rest are pretrain classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.sim_conditional.common import REPO_ROOT, stage_a_data_root
from trajectory_replay.data.mixture_mog import (
    make_polygon_means,
    sample_gaussian_mixture,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument(
        "--heldout-class",
        type=int,
        default=None,
        help="Class id left out of pretrain (default: n_classes - 1).",
    )
    p.add_argument("--separation", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--n-train", type=int, default=8_000, help="Train samples per class.")
    p.add_argument("--n-val", type=int, default=1_000, help="Val samples per class.")
    p.add_argument("--n-test", type=int, default=1_000, help="Test samples per class.")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_conditional/stage_a_leaveout/data/dim<d>/",
    )
    return p.parse_args()


def _split_within_class(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return concatenated x, y, split_id (0=train,1=val,2=test) shuffled per class."""
    n_per = n_train + n_val + n_test
    xs, ys, splits = [], [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        if idx.size != n_per:
            raise ValueError(f"class {k}: expected {n_per} rows, got {idx.size}")
        order = rng.permutation(idx)
        train_i, val_i, test_i = (
            order[:n_train],
            order[n_train : n_train + n_val],
            order[n_train + n_val :],
        )
        for part, sid in ((train_i, 0), (val_i, 1), (test_i, 2)):
            xs.append(x[part])
            ys.append(y[part])
            splits.append(np.full(part.size, sid, dtype=np.int64))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(splits, axis=0),
        np.array(["train", "val", "test"], dtype=object),
    )


def main() -> None:
    args = parse_args()
    n_classes = int(args.n_classes)
    heldout = int(args.heldout_class) if args.heldout_class is not None else n_classes - 1
    if heldout < 0 or heldout >= n_classes:
        raise SystemExit(f"--heldout-class must be in [0, {n_classes - 1}], got {heldout}")
    n_train, n_val, n_test = int(args.n_train), int(args.n_val), int(args.n_test)
    n_per = n_train + n_val + n_test
    if min(n_train, n_val, n_test) < 1:
        raise SystemExit("n_train / n_val / n_test must all be >= 1")

    out_dir = args.out_dir or stage_a_data_root(REPO_ROOT, dim=args.dim)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.data_seed)
    means = make_polygon_means(
        n_classes, dim=args.dim, separation=args.separation
    )
    ds = sample_gaussian_mixture(
        means,
        samples_per_task=n_per,
        rng=rng,
        sigma=args.sigma,
    )
    # Second RNG stream for split shuffle (reproducible, independent of sampling).
    split_rng = np.random.default_rng(args.data_seed + 17)
    x, y, split, split_names = _split_within_class(
        ds.x,
        ds.task,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        rng=split_rng,
    )

    pretrain_classes = [k for k in range(n_classes) if k != heldout]
    meta = {
        "stage": "a_leaveout",
        "dim": int(args.dim),
        "n_classes": n_classes,
        "heldout_class": heldout,
        "pretrain_classes": pretrain_classes,
        "sigma": float(args.sigma),
        "separation": float(args.separation),
        "data_seed": int(args.data_seed),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "means": means.tolist(),
        "split_names": ["train", "val", "test"],
        "split_id": {"train": 0, "val": 1, "test": 2},
    }

    np.savez_compressed(
        out_dir / "classes.npz",
        x=x,
        y=y,
        split=split,
        means=means,
        cov=ds.cov,
        heldout_class=np.asarray([heldout], dtype=np.int64),
        pretrain_classes=np.asarray(pretrain_classes, dtype=np.int64),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Convenience counts for a quick sanity print / downstream loaders.
    counts = {}
    for name, sid in (("train", 0), ("val", 1), ("test", 2)):
        counts[name] = {
            str(k): int(((y == k) & (split == sid)).sum()) for k in range(n_classes)
        }
    print(f"Wrote Stage A data under {out_dir}")
    print(f"means=\n{means}")
    print(f"heldout={heldout}  pretrain={pretrain_classes}")
    print(f"per-class counts={counts}")
    print(f"meta={meta}")


if __name__ == "__main__":
    main()
