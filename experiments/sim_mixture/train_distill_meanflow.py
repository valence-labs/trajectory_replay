#!/usr/bin/env python3
"""Distill MeanFlow from a frozen sim_mixture pretrain velocity field.

Saves ``mean_flow.pt`` and ``mean_flow_metrics.json`` next to the pretrain
``checkpoint.pt`` (same pattern as TX flow-map distillation).

Example:
  uv run python experiments/sim_mixture/train_distill_meanflow.py \\
    --pretrain-checkpoint results_dir/sim_mixture/dim2/pretrain_task0_seed0/checkpoint.pt \\
    --distill-epochs 100 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.mean_flow_distill import (
    MEAN_FLOW_CKPT_NAME,
    MEAN_FLOW_METRICS_NAME,
    eval_mean_flow_vs_teacher_ode,
    make_mean_flow_from_teacher,
    save_mean_flow_checkpoint,
    train_mean_flow_distill,
)
from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    REPO_ROOT,
    load_checkpoint,
    load_tasks,
    make_x_loader,
    mixture_data_root,
    mixture_runs_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Default: results_dir/sim_mixture/dim<d>/pretrain_task0_seed0/checkpoint.pt",
    )
    p.add_argument("--distill-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument(
        "--diag-fraction",
        type=float,
        default=0.75,
        help="Fraction of batch with r=t (instantaneous / diagonal MeanFlow term).",
    )
    p.add_argument(
        "--adaptive-p",
        type=float,
        default=0.0,
        help="Adaptive loss power (0 disables; MeanFlow repo often uses 1.0).",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--mean-flow-steps", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.diag_fraction <= 1.0):
        raise SystemExit("--diag-fraction must be in [0, 1]")
    # Seed before model construction: make_mean_flow_from_teacher below draws from the
    # *global* torch RNG for weight init, and train_mean_flow_distill's internal
    # torch.manual_seed(train_seed) runs too late to cover that.
    torch.manual_seed(args.train_seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir or mixture_data_root(REPO_ROOT, dim=args.dim)
    ds, meta = load_tasks(data_dir)

    pre_ckpt = args.pretrain_checkpoint or (
        mixture_runs_root(REPO_ROOT, dim=args.dim) / "pretrain_task0_seed0" / "checkpoint.pt"
    )
    if not pre_ckpt.is_file():
        raise SystemExit(f"missing pretrain checkpoint: {pre_ckpt}")

    teacher, teacher_cfg = load_checkpoint(pre_ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    mean_flow = make_mean_flow_from_teacher(teacher)
    mean_flow.to(device)

    x_support = ds.subset(0).x
    loader = make_x_loader(
        x_support, batch_size=args.batch_size, shuffle=True, seed=args.train_seed
    )
    eval_x1 = torch.from_numpy(x_support.astype("float32"))

    out_dir = pre_ckpt.parent
    print(
        f"MeanFlow distill from {pre_ckpt}\n"
        f"  data={data_dir} n_support={len(x_support)} device={device}\n"
        f"  epochs={args.distill_epochs} lr={args.lr} diag_fraction={args.diag_fraction}"
    )
    adaptive_p = float(args.adaptive_p) if args.adaptive_p > 0 else None
    history = train_mean_flow_distill(
        mean_flow,
        teacher,
        loader,
        data_dim=int(teacher.data_dim),
        device=device,
        epochs=args.distill_epochs,
        lr=args.lr,
        train_seed=args.train_seed,
        diag_fraction=args.diag_fraction,
        adaptive_p=adaptive_p,
        unconditional=True,
        eval_x1=eval_x1,
        eval_c=None,
        eval_p=None,
        eval_every=args.eval_every,
        eval_seed=args.train_seed + 123,
        eval_n=args.eval_n,
        eval_ode_method="euler",
        eval_ode_step_size=args.ode_step_size,
        eval_mean_flow_n_steps=args.mean_flow_steps,
    )

    map_path = out_dir / MEAN_FLOW_CKPT_NAME
    save_mean_flow_checkpoint(
        map_path,
        mean_flow,
        teacher_cfg,
        parent_checkpoint=str(pre_ckpt),
        args=args,
        extra={
            "stage": "mean_flow_distill",
            "experiment": "sim_mixture",
            "data_meta": meta,
            "history": history,
        },
    )
    final_eval = eval_mean_flow_vs_teacher_ode(
        mean_flow,
        teacher,
        eval_x1,
        None,
        None,
        data_dim=int(teacher.data_dim),
        device=device,
        eval_seed=args.train_seed + 999,
        n_eval=args.eval_n,
        ode_method="euler",
        ode_step_size=args.ode_step_size,
        mean_flow_n_steps=args.mean_flow_steps,
    )
    dump_metrics(
        out_dir / MEAN_FLOW_METRICS_NAME,
        {
            "stage": "mean_flow_distill",
            "parent_checkpoint": str(pre_ckpt),
            "config": {
                "distill_epochs": args.distill_epochs,
                "lr": args.lr,
                "diag_fraction": args.diag_fraction,
                "adaptive_p": adaptive_p,
                "train_seed": args.train_seed,
                "mean_flow_steps": args.mean_flow_steps,
            },
            "final_eval": final_eval,
            "history": history,
        },
    )
    print(f"Saved {map_path}")
    print(f"mean_flow_ode_mse={final_eval['mean_flow_ode_mse']:.6f}")


if __name__ == "__main__":
    main()
