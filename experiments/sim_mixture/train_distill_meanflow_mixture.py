#!/usr/bin/env python3
"""Distill MeanFlow from a frozen oracle CFM trained *directly on the mixture*.

This isolates whether poor N=1 quality is a CL/replay artifact (two-branch
identity loss, old-branch Euler rollout from a frozen task-0 field) or an
inherent MeanFlow one-step limitation on this bimodal target: here there is
no replay, no old task, no λ-mixed branches — just single-teacher MeanFlow
distillation on the actual λ-mixture data, using the same zhuyu CIFAR
one-step time schedule as the ``_mf_cifar_noadapt`` joint-CL runs.

Example:
  uv run python experiments/sim_mixture/train_distill_meanflow_mixture.py \\
    --lambda-mix 0.1 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.mean_flow_distill import (
    eval_mean_flow_vs_teacher_ode,
    make_mean_flow_from_teacher,
    save_mean_flow_checkpoint,
    train_mean_flow_distill,
)
from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    REPO_ROOT,
    _lam_tag,
    evaluate_mean_flow,
    load_checkpoint,
    load_mixture,
    load_tasks,
    make_x_loader,
    mixture_data_root,
    mixture_runs_root,
    save_scatter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--lambda-mix", type=float, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--oracle-checkpoint",
        type=Path,
        default=None,
        help=(
            "Default: results_dir/sim_mixture/dim<d>/oracle_mixture/"
            "oracle_mixture_lam<λ>_e400_seed0/checkpoint.pt"
        ),
    )
    p.add_argument("--distill-epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument(
        "--diag-fraction",
        type=float,
        default=0.25,
        help="zhuyu CIFAR one-step default (ratio_r_not_equal_t=0.75).",
    )
    p.add_argument("--time-mu", type=float, default=-2.0)
    p.add_argument("--time-sigma", type=float, default=2.0)
    p.add_argument(
        "--corner-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of batch anchored at r=0 (half also forced to t=1); "
            "not in the MeanFlow paper. Use to fix rare one-step corner "
            "coverage under logit_normal (0 = paper-default behavior)."
        ),
    )
    p.add_argument(
        "--adaptive-p",
        type=float,
        default=0.0,
        help="0 disables (matches the _mf_cifar_noadapt CL comparison point).",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--n-gen-eval", type=int, default=4000)
    p.add_argument(
        "--nfe-grid",
        type=str,
        default="1,2,4,8",
        help="Comma-separated MeanFlow sampling steps to evaluate/plot (e.g. '1,2,4,8,16,32,64').",
    )
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--use-adamw",
        action="store_true",
        default=True,
        help="AdamW instead of Adam (zhuyu-cs/MeanFlow reference; default on).",
    )
    p.add_argument("--no-adamw", dest="use_adamw", action="store_false")
    p.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
        help="Adam/AdamW beta2 (zhuyu-cs/MeanFlow reference uses 0.95, default 0.999 in PyTorch).",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (zhuyu-cs/MeanFlow reference default: 1.0; 0 disables).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.lambda_mix < 1.0):
        raise SystemExit("--lambda-mix must be in (0, 1)")
    if not (0.0 <= args.diag_fraction <= 1.0):
        raise SystemExit("--diag-fraction must be in [0, 1]")
    if not (0.0 <= args.corner_fraction <= 1.0):
        raise SystemExit("--corner-fraction must be in [0, 1]")
    nfe_grid = [int(s) for s in args.nfe_grid.split(",") if s.strip()]
    if not nfe_grid or any(n < 1 for n in nfe_grid):
        raise SystemExit(f"--nfe-grid must be a comma-separated list of positive ints, got {args.nfe_grid!r}")
    # Seed before any model construction: make_mean_flow_from_teacher below draws from
    # the *global* torch RNG for weight init, and train_mean_flow_distill's internal
    # torch.manual_seed(train_seed) runs too late to cover that (was previously
    # uncontrolled, so --train-seed didn't make the student init reproducible).
    torch.manual_seed(args.train_seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir or mixture_data_root(REPO_ROOT, dim=args.dim)
    ds, meta = load_tasks(data_dir)
    ds_mix = load_mixture(data_dir, args.lambda_mix)

    tag = _lam_tag(args.lambda_mix)
    oracle_ckpt = args.oracle_checkpoint or (
        mixture_runs_root(REPO_ROOT, dim=args.dim)
        / "oracle_mixture"
        / f"oracle_mixture_{tag}_e400_seed0"
        / "checkpoint.pt"
    )
    if not oracle_ckpt.is_file():
        raise SystemExit(
            f"missing oracle mixture checkpoint: {oracle_ckpt}\n"
            f"(train it first with train_on_mixture.py --lambda-mix {args.lambda_mix})"
        )

    teacher, teacher_cfg = load_checkpoint(oracle_ckpt, device)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    mean_flow = make_mean_flow_from_teacher(teacher)
    mean_flow.to(device)

    loader = make_x_loader(
        ds_mix.x, batch_size=args.batch_size, shuffle=True, seed=args.train_seed
    )
    eval_x1 = torch.from_numpy(ds_mix.x.astype("float32"))
    adaptive_p = float(args.adaptive_p) if args.adaptive_p > 0 else None

    run_name = args.run_name or f"distill_mixture_{tag}_e{args.distill_epochs}_seed{args.train_seed}"
    out_dir = mixture_runs_root(REPO_ROOT, dim=args.dim) / "distill_meanflow_mixture" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"MeanFlow distill directly on mixture λ={args.lambda_mix} (no CL / no replay)\n"
        f"  teacher={oracle_ckpt}\n"
        f"  n={len(ds_mix.x)} device={device} epochs={args.distill_epochs}\n"
        f"  diag_fraction={args.diag_fraction} corner_fraction={args.corner_fraction} "
        f"time_mu={args.time_mu} time_sigma={args.time_sigma} adaptive_p={adaptive_p}\n"
        f"  use_adamw={args.use_adamw} betas=(0.9,{args.adam_beta2}) "
        f"weight_decay={args.weight_decay} max_grad_norm={args.max_grad_norm}"
    )
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
        time_mu=args.time_mu,
        time_sigma=args.time_sigma,
        corner_fraction=args.corner_fraction,
        unconditional=True,
        use_adamw=args.use_adamw,
        betas=(0.9, args.adam_beta2),
        weight_decay=args.weight_decay,
        max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
        eval_x1=eval_x1,
        eval_c=None,
        eval_p=None,
        eval_every=args.eval_every,
        eval_seed=args.train_seed + 123,
        eval_n=args.eval_n,
        eval_ode_method="euler",
        eval_ode_step_size=args.ode_step_size,
        eval_mean_flow_n_steps=1,
    )

    save_mean_flow_checkpoint(
        out_dir / "mean_flow.pt",
        mean_flow,
        teacher_cfg,
        parent_checkpoint=str(oracle_ckpt),
        args=args,
        extra={"stage": "distill_meanflow_mixture", "lambda_mix": args.lambda_mix, "data_meta": meta},
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
        mean_flow_n_steps=1,
    )

    nfe_curve: list[dict[str, float]] = []
    for n_steps in nfe_grid:
        metrics, x_gen = evaluate_mean_flow(
            mean_flow,
            ds,
            device=device,
            n_gen=args.n_gen_eval,
            seed=args.train_seed + 1999,
            n_steps=n_steps,
            lambda_target=args.lambda_mix,
            sigma=float(meta["sigma"]),
            x_true_mixture=ds_mix.x,
        )
        nfe_curve.append({"n_steps": float(n_steps), **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
        save_scatter(
            out_dir / f"samples_mean_flow_N{n_steps}.png",
            x_gen=x_gen,
            means=ds.means,
            title=f"Distilled-on-mixture MeanFlow λ={args.lambda_mix} N={n_steps} (no CL)",
            x_true=ds_mix.x,
        )
        print(f"  N={n_steps}  pi0_hat={metrics['pi0_hat']:.3f}  energy={metrics['energy_mixture']:.4f}")

    dump_metrics(
        out_dir / "metrics.json",
        {
            "stage": "distill_meanflow_mixture",
            "oracle_checkpoint": str(oracle_ckpt),
            "config": {
                "lambda_mix": args.lambda_mix,
                "distill_epochs": args.distill_epochs,
                "lr": args.lr,
                "diag_fraction": args.diag_fraction,
                "corner_fraction": args.corner_fraction,
                "nfe_grid": nfe_grid,
                "time_mu": args.time_mu,
                "time_sigma": args.time_sigma,
                "adaptive_p": adaptive_p,
                "train_seed": args.train_seed,
                "use_adamw": args.use_adamw,
                "adam_betas": [0.9, args.adam_beta2],
                "weight_decay": args.weight_decay,
                "max_grad_norm": (args.max_grad_norm if args.max_grad_norm > 0 else None),
            },
            "final_eval_vs_teacher_ode": final_eval,
            "eval_mean_flow_nfe": nfe_curve,
            "history": history,
        },
    )
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
