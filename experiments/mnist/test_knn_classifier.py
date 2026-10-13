#!/usr/bin/env python3
"""KNN digit-classifier sanity check on real MNIST images (no model sampling)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from experiments.mnist.mnist_common import (
    add_mnist_data_args,
    finetune_tensors,
    load_manifest,
    pretrain_support_tensors,
    resolve_processed_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_mnist_data_args(p)
    p.add_argument("--train-max", type=int, default=20_000, help="Max real train images for KNN fit.")
    p.add_argument("--test-max", type=int, default=5_000, help="Max real test images for KNN eval.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def flatten_x(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    if x.dim() == 4:
        x = x.view(x.shape[0], -1)
    return x.numpy()


def subsample(x: torch.Tensor, y: torch.Tensor, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(x) <= n:
        return x, y
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n]
    return x[idx], y[idx]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = resolve_processed_dir(args, repo_root)
    manifest = load_manifest(processed_dir)
    held_out = manifest.held_out_digit

    x_seen, _, p_seen = pretrain_support_tensors(processed_dir, held_out=held_out)
    x_ft, _, p_ft = finetune_tensors(processed_dir, held_out=held_out, use_conditioning=False)
    x_all = torch.cat([x_seen, x_ft], dim=0)
    p_all = torch.cat([p_seen, p_ft], dim=0)

    x_pool, p_pool = subsample(x_all, p_all, args.train_max + args.test_max, args.seed)
    y = p_pool.numpy().astype(np.int64)
    test_size = min(args.test_max, max(1, len(y) // 5))
    x_tr, x_te, y_tr, y_te = train_test_split(
        flatten_x(x_pool),
        y,
        test_size=test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = KNeighborsClassifier(n_neighbors=args.k, n_jobs=-1, metric="euclidean")
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = accuracy_score(y_te, pred)

    print("=" * 72)
    print(f"KNN k={args.k} on REAL MNIST (all digits incl. held-out {held_out})")
    print(f"  pool n={len(y)}  train n={len(y_tr)}  test n={len(y_te)}")
    print(f"  accuracy: {acc:.4f}")
    print("  per-class report:")
    print(classification_report(y_te, pred, digits=4, zero_division=0))
    labels = sorted(set(y_te.tolist()) | set(pred.tolist()))
    cm = confusion_matrix(y_te, pred, labels=labels)
    print("  confusion matrix (rows=true, cols=pred):")
    print(f"  labels: {labels}")
    print(cm)

    # Seen-digit-only holdout (matches future forgetting eval pool)
    seen_mask = y_te != held_out
    if seen_mask.any():
        acc_seen = accuracy_score(y_te[seen_mask], pred[seen_mask])
        print(f"\n  seen-digit test accuracy (p != {held_out}): {acc_seen:.4f}")

    held_mask = y_te == held_out
    if held_mask.any():
        acc_held = accuracy_score(y_te[held_mask], pred[held_mask])
        print(f"  held-out digit-{held_out} test accuracy: {acc_held:.4f}")


if __name__ == "__main__":
    main()
