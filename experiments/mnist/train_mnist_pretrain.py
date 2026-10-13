#!/usr/bin/env python3
"""Pretrain conditional flow matching on MNIST digits (UNet or MLP backbone)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trajectory_replay.data.mnist_dataset import load_manifest, load_split
from trajectory_replay.io.results import run_results_dir
from trajectory_replay.training.flow_matching import train_flow_matching
from trajectory_replay.utils.cli import resolve_device

from experiments.mnist.mnist_eval import add_mnist_eval_args, print_eval_summary, run_mnist_ode_eval
from experiments.mnist.mnist_common import (
    add_mnist_conditioning_args,
    add_mnist_data_args,
    add_mnist_model_args,
    add_mnist_ode_args,
    add_mnist_train_args,
    checkpoint_config,
    make_mnist_model,
    make_pretrain_loader,
    model_uses_conditioning,
    pretrain_tensors,
    resolve_held_out_digit,
    resolve_processed_dir,
    save_digit_grid,
    save_seen_digit_sample_grids,
    save_seen_digit_sample_grids,
    finetune_tensors,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_mnist_data_args(p)
    add_mnist_model_args(p)
    add_mnist_train_args(p)
    add_mnist_ode_args(p)
    add_mnist_eval_args(p)
    add_mnist_conditioning_args(p)
    p.add_argument("--run-name", default=None)
    p.add_argument("--plot-n-per-digit", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    device = resolve_device(args.device)
    processed_dir = resolve_processed_dir(args, repo_root)
    manifest = load_manifest(processed_dir)
    held_out = resolve_held_out_digit(args, manifest)
    use_conditioning = model_uses_conditioning(args)
    split = load_split(processed_dir, "pretrain")
    x, c, p = pretrain_tensors(
        split,
        held_out=held_out,
        max_samples=args.max_samples,
        seed=args.train_seed,
    )
    loader = make_pretrain_loader(x, c, p, batch_size=args.batch_size)

    model = make_mnist_model(manifest, args).to(device)
    cond_label = "conditional" if use_conditioning else "unconditional"
    print(
        f"Pretrain MNIST: backbone={args.backbone_type}, {cond_label}, n={len(x)}, "
        f"n_pert={model.n_perturbations} (held out digit {held_out}), "
        f"image_shape={model.image_shape}"
    )

    history = train_flow_matching(
        model,
        loader,
        data_dim=manifest.data_dim,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        train_seed=args.train_seed,
    )

    suffix = cond_label
    run_name = args.run_name or f"mnist_pretrain_{args.backbone_type}_{suffix}_d{held_out}"
    rdir = run_results_dir(repo_root, run_name)
    rdir.mkdir(parents=True, exist_ok=True)

    cfg = checkpoint_config(model, manifest, args)
    cfg["stage"] = "pretrain"
    cfg["epoch_history"] = history
    cfg["eval_max_samples_per_stratum"] = args.eval_max_samples_per_stratum
    torch.save({"state_dict": model.state_dict(), "config": cfg}, rdir / "checkpoint.pt")
    print(f"checkpoint saved (pre-eval): {rdir / 'checkpoint.pt'}")

    ode_options = {"step_size": float(args.ode_step_size)} if args.ode_step_size is not None else None
    x_fit, _, p_fit = finetune_tensors(
        processed_dir,
        held_out=held_out,
        use_conditioning=use_conditioning,
    )
    x_knn = torch.cat([x, x_fit], dim=0)
    p_knn = torch.cat([p, p_fit], dim=0)

    eval_result = run_mnist_ode_eval(
        model,
        use_conditioning=use_conditioning,
        held_out=held_out,
        x_forget=x,
        p_forget=p,
        x_fit=x_fit,
        p_fit=p_fit,
        x_knn=x_knn,
        p_knn=p_knn,
        device=device,
        eval_seed=args.train_seed + 999,
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
    print_eval_summary(eval_result, held_out=held_out)
    cfg.update(eval_result.to_metrics_dict())
    # Legacy aliases
    cfg["eval_mse"] = cfg.get("eval_mse_forget")
    cfg["eval_strata"] = cfg.get("eval_strata_forget")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, rdir / "checkpoint.pt")
    with (rdir / "metrics.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    plotted = save_seen_digit_sample_grids(
        model,
        x,
        c,
        p,
        out_dir=rdir,
        device=device,
        use_conditioning=use_conditioning,
        plot_n_per_digit=args.plot_n_per_digit,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options,
    )
    if plotted:
        print(f"Saved seen-digit sample grids for labels: {plotted}")

    print(f"Saved checkpoint and plots to {rdir}")


if __name__ == "__main__":
    main()
