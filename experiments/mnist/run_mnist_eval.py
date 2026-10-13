#!/usr/bin/env python3
"""Re-run MNIST forget/fit ODE eval from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trajectory_replay.utils.cli import resolve_device

from experiments.mnist.mnist_eval import add_mnist_eval_args, print_eval_summary, run_mnist_ode_eval
from experiments.mnist.mnist_common import (
    add_mnist_conditioning_args,
    add_mnist_data_args,
    add_mnist_ode_args,
    finetune_tensors,
    load_manifest,
    load_mnist_checkpoint,
    pretrain_support_tensors,
    resolve_held_out_digit,
    resolve_mnist_run_checkpoint,
    resolve_processed_dir,
)
from trajectory_replay.model.flow_matching import resolve_use_conditioning_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_mnist_data_args(p)
    add_mnist_ode_args(p)
    add_mnist_eval_args(p)
    add_mnist_conditioning_args(p)
    p.add_argument("--run-name", default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-seed", type=int, default=0)
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
    if args.unconditional and use_conditioning:
        raise ValueError("Checkpoint is conditional but --unconditional was passed.")
    model.eval()

    x_forget, _, p_forget = pretrain_support_tensors(processed_dir, held_out=held_out)
    x_fit, _, p_fit = finetune_tensors(
        processed_dir,
        held_out=held_out,
        use_conditioning=use_conditioning,
    )
    x_knn = torch.cat([x_forget, x_fit], dim=0)
    p_knn = torch.cat([p_forget, p_fit], dim=0)

    ode_options = {"step_size": float(args.ode_step_size)} if args.ode_step_size is not None else None
    result = run_mnist_ode_eval(
        model,
        use_conditioning=use_conditioning,
        held_out=held_out,
        x_forget=x_forget,
        p_forget=p_forget,
        x_fit=x_fit,
        p_fit=p_fit,
        x_knn=x_knn,
        p_knn=p_knn,
        device=device,
        eval_seed=args.eval_seed,
        max_true_per_digit=args.eval_max_samples_per_stratum,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options,
        knn_k=args.knn_k,
        knn_train_max=args.knn_train_max,
        uncond_n_samples=args.eval_uncond_n_samples,
        uncond_batch_size=args.eval_uncond_batch_size,
        min_gen_per_class=args.eval_min_gen_per_class,
    )
    print_eval_summary(result, held_out=held_out)

    out = ckpt_path.parent / "metrics_eval.json"
    payload = {"checkpoint": str(ckpt_path), **result.to_metrics_dict()}
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
