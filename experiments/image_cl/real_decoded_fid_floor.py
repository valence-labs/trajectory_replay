#!/usr/bin/env python3
"""Estimate the VAE-decoded FID floor for ImageNet task data (per-class).

Decodes *all* task train/test latents once with the frozen SD-VAE, then reports:

1. ``train_vs_test`` — for each class, FID(all train_c, all test_c). Matches the
   reference side of model ``mean_per_class_fid`` (official val ≈ 50/class).
2. ``train_vs_train`` — for each class, split all train_c into two equal disjoint
   halves and compute FID(half_a, half_b). This is the closest balanced
   real-vs-real comparison (~500–650/side); true 1k-vs-1k is impossible because
   ImageNet-100 only has ~1.3k train images/class before the val holdout.

Neither is a model score: both estimate the practical FID floor under the same
decoded-real pipeline used by image_cl evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.image_cl.image_cl_common import (
    build_image_cl_tasks,
    resolve_latent_to_pixels,
)
from trajectory_replay.eval.image_fid import compute_fid


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = float(statistics.mean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42, help="ImageNet class-order seed.")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-experiences", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--latent-dir", type=Path, default=None)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-mse")
    p.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="RNG seed for the balanced train-vs-train half split.",
    )
    p.add_argument(
        "--pooled-repeats",
        type=int,
        default=5,
        help="Independent whole-task train half splits (stability of ~10k/10k FID).",
    )
    p.add_argument(
        "--skip-per-class",
        action="store_true",
        help="Only compute pooled task halves (faster after decode).",
    )
    p.add_argument("--decode-batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results_dir/image_cl/imagenet100/pilot/real_decoded_fid_floor_task00_per_class.json"
        ),
    )
    args = p.parse_args()
    if args.pooled_repeats < 1:
        raise ValueError("--pooled-repeats must be >= 1")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[2]
    data_args = argparse.Namespace(
        dataset="split_imagenet100",
        seed=int(args.seed),
        n_experiences=int(args.n_experiences),
        val_fraction=float(args.val_fraction),
        latent_dir=args.latent_dir,
        vae_id=str(args.vae_id),
        data_root=None,
        image_size=32,
        fid_batch_size=int(args.decode_batch_size) * 2,
    )

    tasks = build_image_cl_tasks(data_args, repo_root)
    task = tasks[int(args.task_id)]
    y_train = task.y_train.detach().long().view(-1)
    y_test = task.y_test.detach().long().view(-1)

    print(
        f"Task {args.task_id}: classes={task.classes} "
        f"train={len(task.x_train)} test={len(task.x_test)}"
    )
    print(f"Decoding all train+test with {args.vae_id} (no subsample)")
    to_pixels = resolve_latent_to_pixels(data_args, device)
    assert to_pixels is not None

    t0 = time.perf_counter()
    decoded_train = to_pixels(task.x_train)
    decode_train_seconds = time.perf_counter() - t0
    print(f"Decoded train: {tuple(decoded_train.shape)} in {decode_train_seconds:.1f}s")

    t0 = time.perf_counter()
    decoded_test = to_pixels(task.x_test)
    decode_test_seconds = time.perf_counter() - t0
    print(f"Decoded test:  {tuple(decoded_test.shape)} in {decode_test_seconds:.1f}s")

    # Whole-task balanced halves (~11.5k vs ~11.5k for task 0).
    n_train = int(decoded_train.shape[0])
    pooled_half = n_train // 2
    pooled_rows: list[dict[str, float | int]] = []
    print(
        f"Pooled task train halves: {pooled_half} vs {pooled_half} "
        f"× {args.pooled_repeats} repeats"
    )
    for repeat in range(int(args.pooled_repeats)):
        gen_p = torch.Generator().manual_seed(int(args.split_seed) + 1000 + repeat)
        perm = torch.randperm(n_train, generator=gen_p)
        a = decoded_train[perm[:pooled_half]]
        b = decoded_train[perm[pooled_half : 2 * pooled_half]]
        t0 = time.perf_counter()
        fid_p = compute_fid(a, b, device=device)
        row = {
            "repeat": repeat,
            "draw_seed": int(args.split_seed) + 1000 + repeat,
            "fid": float(fid_p),
            "n_per_half": int(pooled_half),
            "seconds": float(time.perf_counter() - t0),
        }
        pooled_rows.append(row)
        print(
            f"  pooled repeat {repeat + 1}/{args.pooled_repeats}: "
            f"FID={fid_p:.3f} ({row['seconds']:.1f}s)"
        )

    pooled_vals = [float(r["fid"]) for r in pooled_rows]
    pooled_mean, pooled_std = _mean_std(pooled_vals)

    # Also one pooled train vs full test (1000) for reference.
    t0 = time.perf_counter()
    fid_pooled_train_test = compute_fid(decoded_test, decoded_train, device=device)
    pooled_train_test_seconds = time.perf_counter() - t0
    print(
        f"Pooled train vs test: FID={fid_pooled_train_test:.3f} "
        f"(n_train={n_train} n_test={len(decoded_test)})"
    )

    result: dict = {
        "metric": "decoded_real_fid_floor",
        "interpretation": (
            "Real-vs-real FID after decoding all task latents with the same SD-VAE "
            "used by image_cl eval. pooled_train_vs_train splits the whole task "
            "train set into equal halves (~10k/10k). Optional per-class blocks "
            "match earlier floor estimates."
        ),
        "dataset": "split_imagenet100",
        "task_id": int(args.task_id),
        "classes": [int(c) for c in task.classes],
        "class_order_seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "vae_id": str(args.vae_id),
        "latent_shape": list(task.x_test.shape[1:]),
        "decoded_shape": list(decoded_test.shape[1:]),
        "n_train_total": n_train,
        "n_test_total": int(len(task.x_test)),
        "decode_train_seconds": float(decode_train_seconds),
        "decode_test_seconds": float(decode_test_seconds),
        "pooled_train_vs_train": {
            "fid_mean": pooled_mean,
            "fid_std": pooled_std,
            "fid_min": float(min(pooled_vals)),
            "fid_max": float(max(pooled_vals)),
            "n_per_half": int(pooled_half),
            "repeats": int(args.pooled_repeats),
            "rows": pooled_rows,
        },
        "pooled_train_vs_test": {
            "fid": float(fid_pooled_train_test),
            "n_train": n_train,
            "n_test": int(len(decoded_test)),
            "seconds": float(pooled_train_test_seconds),
        },
    }

    if not args.skip_per_class:
        gen = torch.Generator().manual_seed(int(args.split_seed))
        train_vs_test: dict[str, dict[str, float | int]] = {}
        train_vs_train: dict[str, dict[str, float | int]] = {}
        skipped: list[int] = []

        for cid in task.classes:
            cid = int(cid)
            train_mask = y_train == cid
            test_mask = y_test == cid
            n_tr = int(train_mask.sum().item())
            n_te = int(test_mask.sum().item())
            if n_tr < 4 or n_te < 2:
                skipped.append(cid)
                print(f"  class {cid}: skip (train={n_tr} test={n_te})")
                continue

            real_tr = decoded_train[train_mask]
            real_te = decoded_test[test_mask]

            t0 = time.perf_counter()
            fid_tt = compute_fid(real_te, real_tr, device=device)
            train_vs_test[str(cid)] = {
                "fid": float(fid_tt),
                "n_train": n_tr,
                "n_test": n_te,
                "seconds": float(time.perf_counter() - t0),
            }

            half = n_tr // 2
            perm = torch.randperm(n_tr, generator=gen)
            a = real_tr[perm[:half]]
            b = real_tr[perm[half : 2 * half]]
            t0 = time.perf_counter()
            fid_bal = compute_fid(a, b, device=device)
            train_vs_train[str(cid)] = {
                "fid": float(fid_bal),
                "n_per_half": int(half),
                "n_train": n_tr,
                "seconds": float(time.perf_counter() - t0),
            }
            print(
                f"  class {cid}: train_vs_test={fid_tt:.3f} "
                f"(n_tr={n_tr} n_te={n_te}) | "
                f"train_vs_train={fid_bal:.3f} (half={half})"
            )

        tt_vals = [float(v["fid"]) for v in train_vs_test.values()]
        bal_vals = [float(v["fid"]) for v in train_vs_train.values()]
        tt_mean, tt_std = _mean_std(tt_vals)
        bal_mean, bal_std = _mean_std(bal_vals)
        result["classes_skipped"] = skipped
        result["train_vs_test"] = {
            "fid_per_class_mean": tt_mean,
            "fid_per_class_std": tt_std,
            "fid_per_class_min": float(min(tt_vals)) if tt_vals else float("nan"),
            "fid_per_class_max": float(max(tt_vals)) if tt_vals else float("nan"),
            "n_classes": len(tt_vals),
            "per_class": train_vs_test,
        }
        result["train_vs_train"] = {
            "fid_per_class_mean": bal_mean,
            "fid_per_class_std": bal_std,
            "fid_per_class_min": float(min(bal_vals)) if bal_vals else float("nan"),
            "fid_per_class_max": float(max(bal_vals)) if bal_vals else float("nan"),
            "n_classes": len(bal_vals),
            "per_class": train_vs_train,
        }
        print(
            f"train_vs_test mean FID:  {tt_mean:.3f} ± {tt_std:.3f} "
            f"(range {result['train_vs_test']['fid_per_class_min']:.3f}–"
            f"{result['train_vs_test']['fid_per_class_max']:.3f})"
        )
        print(
            f"per-class train_vs_train mean FID: {bal_mean:.3f} ± {bal_std:.3f} "
            f"(range {result['train_vs_train']['fid_per_class_min']:.3f}–"
            f"{result['train_vs_train']['fid_per_class_max']:.3f})"
        )

    print(
        f"pooled train_vs_train: {pooled_mean:.3f} ± {pooled_std:.3f} "
        f"(range {min(pooled_vals):.3f}–{max(pooled_vals):.3f}, "
        f"half={pooled_half})"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
