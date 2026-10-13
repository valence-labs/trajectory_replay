#!/usr/bin/env python3
"""Regenerate seen-digit real/gen grids from a MNIST finetune checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from trajectory_replay.model.flow_matching import resolve_use_conditioning_from_config
from trajectory_replay.utils.cli import resolve_device

from experiments.mnist.mnist_common import (
    add_mnist_data_args,
    add_mnist_ode_args,
    finetune_tensors,
    load_manifest,
    load_mnist_checkpoint,
    pretrain_support_tensors,
    resolve_held_out_digit,
    resolve_mnist_run_checkpoint,
    resolve_processed_dir,
    save_seen_digit_sample_grids,
)
from experiments.mnist.mnist_eval import add_mnist_eval_args, save_uncond_knn_digit_sample_grids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_mnist_data_args(p)
    add_mnist_eval_args(p)
    p.add_argument("--run-name", default=None, help="Run under results_dir/ with checkpoint.pt.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--plot-n-per-digit", type=int, default=8)
    p.add_argument("--max-digits", type=int, default=3)
    p.add_argument(
        "--plot-uncond-n-samples",
        type=int,
        default=3000,
        help="Unconditional samples to draw before KNN bucketing (uncond runs only).",
    )
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    add_mnist_ode_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    device = resolve_device(args.device)
    processed_dir = resolve_processed_dir(args, repo_root)
    manifest = load_manifest(processed_dir)
    held_out = resolve_held_out_digit(args, manifest)

    ckpt_path = resolve_mnist_run_checkpoint(
        repo_root, run_name=args.run_name, checkpoint=args.checkpoint
    )
    model, cfg = load_mnist_checkpoint(ckpt_path, device)
    use_conditioning = resolve_use_conditioning_from_config(cfg)
    model.eval()

    x_sup, c_sup, p_sup = pretrain_support_tensors(processed_dir, held_out=held_out)
    ode_options = {"step_size": float(args.ode_step_size)} if args.ode_step_size is not None else None

    if use_conditioning:
        plotted = save_seen_digit_sample_grids(
            model,
            x_sup,
            c_sup,
            p_sup,
            out_dir=ckpt_path.parent,
            device=device,
            use_conditioning=True,
            plot_n_per_digit=args.plot_n_per_digit,
            max_digits=args.max_digits,
            ode_method=args.ode_method,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_options=ode_options,
        )
    else:
        x_ft, _, p_ft = finetune_tensors(
            processed_dir,
            held_out=held_out,
            use_conditioning=False,
        )
        plotted = save_uncond_knn_digit_sample_grids(
            model,
            x_sup,
            p_sup,
            x_ft,
            p_ft,
            out_dir=ckpt_path.parent,
            device=device,
            plot_n_per_digit=args.plot_n_per_digit,
            max_digits=args.max_digits,
            knn_k=args.knn_k,
            knn_train_max=args.knn_train_max,
            eval_seed=args.eval_seed,
            uncond_n_samples=args.plot_uncond_n_samples,
            uncond_batch_size=args.eval_uncond_batch_size,
            ode_method=args.ode_method,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_options=ode_options,
        )

    print(f"Saved seen-digit grids for labels {plotted} under {ckpt_path.parent}")


if __name__ == "__main__":
    main()
