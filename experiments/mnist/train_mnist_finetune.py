#!/usr/bin/env python3
"""Finetune MNIST flow matching on held-out digit with optional generative replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from trajectory_replay.io.results import run_results_dir
from trajectory_replay.training.flow_matching import (
    make_frozen_model_copy,
    train_flow_matching,
    train_flow_matching_generative_replay,
)
from trajectory_replay.utils.cli import resolve_device

from experiments.mnist.mnist_eval import (
    add_mnist_eval_args,
    print_eval_summary,
    run_mnist_ode_eval,
    save_uncond_knn_digit_sample_grids,
)
from experiments.mnist.mnist_common import (
    add_mnist_conditioning_args,
    add_mnist_data_args,
    add_mnist_ode_args,
    add_mnist_train_args,
    assert_pretrain_conditioning_compatible,
    assert_pretrain_embedding_table_compatible,
    build_replay_stratum_dist,
    checkpoint_config,
    configure_mnist_finetune,
    finetune_tensors,
    load_manifest,
    load_mnist_checkpoint,
    model_uses_conditioning,
    pretrain_support_tensors,
    resolve_held_out_digit,
    resolve_pretrain_checkpoint,
    resolve_processed_dir,
    save_digit_grid,
    save_seen_digit_sample_grids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_mnist_data_args(p)
    add_mnist_train_args(p)
    add_mnist_ode_args(p)
    add_mnist_eval_args(p)
    add_mnist_conditioning_args(p)
    p.add_argument("--pretrain-run", default=None, help="Pretrain run name under results_dir/.")
    p.add_argument("--pretrain-checkpoint", type=Path, default=None)
    p.add_argument(
        "--finetune-strategy",
        choices=("standard", "generative_replay", "generative_field_replay"),
        default="generative_field_replay",
    )
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--finetune-lr", type=float, default=None, help="Default: pretrain lr from checkpoint.")
    p.add_argument("--finetune-train-vf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--finetune-train-old-embeddings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--generative-replay-fraction",
        type=float,
        default=0.25,
        help=(
            "Mixture weight λ. generative_replay: old-sample batch fraction; "
            "generative_field_replay: loss weight λ L_traj+(1-λ)L_new with 50/50 batch split."
        ),
    )
    p.add_argument("--generative-replay-ode-method", default="euler")
    p.add_argument("--generative-replay-ode-rtol", type=float, default=1e-5)
    p.add_argument("--generative-replay-ode-atol", type=float, default=1e-5)
    p.add_argument("--generative-replay-ode-step-size", type=float, default=0.05)
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

    ckpt_path = resolve_pretrain_checkpoint(repo_root, args.pretrain_run, args.pretrain_checkpoint)
    model, pre_cfg = load_mnist_checkpoint(ckpt_path, device)
    assert_pretrain_conditioning_compatible(pre_cfg, use_conditioning)
    assert_pretrain_embedding_table_compatible(pre_cfg, manifest, use_conditioning=use_conditioning)

    finetune_lr = args.finetune_lr if args.finetune_lr is not None else float(pre_cfg.get("lr", args.lr))

    x_ft, c_ft, p_ft = finetune_tensors(
        processed_dir,
        held_out=held_out,
        use_conditioning=use_conditioning,
    )
    ft_loader = DataLoader(
        TensorDataset(x_ft, c_ft, p_ft),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(x_ft) >= args.batch_size,
    )

    finetune_params, finetune_pert_mask, finetune_context_mask = configure_mnist_finetune(
        model,
        use_conditioning=use_conditioning,
        held_out=held_out,
        train_vf=args.finetune_train_vf,
        train_old_embeddings=args.finetune_train_old_embeddings,
    )
    model_ref = make_frozen_model_copy(model)

    cond_label = "conditional" if use_conditioning else "unconditional"
    print(
        f"Finetune MNIST: strategy={args.finetune_strategy}, {cond_label}, "
        f"n_ft={len(x_ft)}, held_out={held_out}, n_pert={model.n_perturbations}"
    )

    phase_kw = dict(
        data_dim=manifest.data_dim,
        device=device,
        epochs=args.finetune_epochs,
        lr=finetune_lr,
        train_seed=args.train_seed + 1,
        params=finetune_params,
        finetune_pert_index=finetune_pert_mask,
        finetune_context_index=finetune_context_mask,
    )

    if args.finetune_strategy in ("generative_replay", "generative_field_replay"):
        x_sup, c_sup, p_sup = pretrain_support_tensors(processed_dir, held_out=held_out)
        support_contexts = list(range(manifest.n_contexts))
        replay_stratum_dist = build_replay_stratum_dist(
            c_sup, p_sup, support_contexts=support_contexts
        )
        replay_mode = (
            "field" if args.finetune_strategy == "generative_field_replay" else "endpoint"
        )
        history = train_flow_matching_generative_replay(
            model,
            model_ref,
            ft_loader,
            support_contexts=support_contexts,
            n_perturbations=model_ref.n_perturbations,
            replay_fraction=args.generative_replay_fraction,
            replay_mode=replay_mode,
            replay_ode_method=args.generative_replay_ode_method,
            replay_ode_rtol=args.generative_replay_ode_rtol,
            replay_ode_atol=args.generative_replay_ode_atol,
            replay_ode_step_size=args.generative_replay_ode_step_size,
            replay_stratum_dist=replay_stratum_dist,
            **phase_kw,
        )
    else:
        history = train_flow_matching(model, ft_loader, **phase_kw)

    x_sup, c_sup, p_sup = pretrain_support_tensors(processed_dir, held_out=held_out)

    run_name = args.run_name or f"mnist_ft_{args.finetune_strategy}_{cond_label}_d{held_out}"
    rdir = run_results_dir(repo_root, run_name)
    rdir.mkdir(parents=True, exist_ok=True)

    args.epochs = args.finetune_epochs
    args.lr = finetune_lr
    cfg = checkpoint_config(model, manifest, args)
    cfg.update(
        {
            "stage": "finetune",
            "pretrain_checkpoint": str(ckpt_path),
            "finetune_strategy": args.finetune_strategy,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": finetune_lr,
            "finetune_train_vf": args.finetune_train_vf,
            "finetune_train_old_embeddings": args.finetune_train_old_embeddings,
            "generative_replay_fraction": args.generative_replay_fraction,
            "epoch_history": history,
            "eval_max_samples_per_stratum": args.eval_max_samples_per_stratum,
        }
    )
    torch.save({"state_dict": model.state_dict(), "config": cfg}, rdir / "checkpoint.pt")
    print(f"checkpoint saved (pre-eval): {rdir / 'checkpoint.pt'}")

    ode_options = {"step_size": float(args.ode_step_size)} if args.ode_step_size is not None else None
    x_knn = torch.cat([x_sup, x_ft], dim=0)
    p_knn = torch.cat([p_sup, p_ft], dim=0)

    eval_result = run_mnist_ode_eval(
        model,
        use_conditioning=use_conditioning,
        held_out=held_out,
        x_forget=x_sup,
        p_forget=p_sup,
        x_fit=x_ft,
        p_fit=p_ft,
        x_knn=x_knn,
        p_knn=p_knn,
        device=device,
        eval_seed=args.train_seed + 1999,
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
    cfg["eval_mse_seen"] = cfg.get("eval_mse_forget")
    cfg["eval_strata_seen"] = cfg.get("eval_strata_forget")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, rdir / "checkpoint.pt")
    with (rdir / "metrics.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    n_plot = min(args.plot_n_per_digit, len(x_ft))
    x_plot = x_ft[:n_plot]
    if use_conditioning:
        gen = model.sample(
            c_ft[:n_plot].to(device),
            p_ft[:n_plot].to(device),
            method=args.ode_method,
            rtol=args.ode_rtol,
            atol=args.ode_atol,
            options=ode_options,
        ).cpu()
    else:
        gen = model.sample(
            batch_size=n_plot,
            method=args.ode_method,
            rtol=args.ode_rtol,
            atol=args.ode_atol,
            options=ode_options,
        ).cpu()
    save_digit_grid(rdir / "finetune_real.png", x_plot, title=f"Finetune real digit {held_out}")
    save_digit_grid(rdir / "finetune_gen.png", gen, title=f"Finetune generated digit {held_out}")

    if use_conditioning:
        plotted = save_seen_digit_sample_grids(
            model,
            x_sup,
            c_sup,
            p_sup,
            out_dir=rdir,
            device=device,
            use_conditioning=True,
            plot_n_per_digit=args.plot_n_per_digit,
            ode_method=args.ode_method,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_options=ode_options,
        )
        if plotted:
            print(f"Saved seen-digit sample grids for labels: {plotted}")
    else:
        plotted = save_uncond_knn_digit_sample_grids(
            model,
            x_sup,
            p_sup,
            x_ft,
            p_ft,
            out_dir=rdir,
            device=device,
            plot_n_per_digit=args.plot_n_per_digit,
            knn_k=args.knn_k,
            knn_train_max=args.knn_train_max,
            eval_seed=args.train_seed + 2999,
            uncond_n_samples=min(args.eval_uncond_n_samples, 3000),
            uncond_batch_size=args.eval_uncond_batch_size,
            ode_method=args.ode_method,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_options=ode_options,
        )
        if plotted:
            print(f"Saved KNN-stratified uncond sample grids for labels: {plotted}")

    print(f"Saved checkpoint and plots to {rdir}")


if __name__ == "__main__":
    main()
