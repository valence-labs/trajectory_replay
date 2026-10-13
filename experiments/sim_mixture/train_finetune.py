#!/usr/bin/env python3
"""Finetune unconditional FM on task 1 with naive / generative / trajectory / real replay."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.flow_matching import make_frozen_model_copy
from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    REPO_ROOT,
    _lam_tag,
    clone_trainable,
    evaluate_mean_flow,
    evaluate_model,
    load_checkpoint,
    load_mixture,
    load_tasks,
    make_x_loader,
    mixture_data_root,
    mixture_runs_root,
    run_finetune_strategy,
    save_checkpoint,
    save_scatter,
)


def _scalars(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, bool))}


def estimated_ref_nfe_per_step(
    strategy: str,
    *,
    ode_step_size: float,
    mean_flow_n_steps: int = 1,
    replay_backend: str = "ode",
) -> float:
    """Analytical frozen-ref NFE per FT step (see plot_nfe_vs_quality.py).

    ``replay_backend='mean_flow'`` is the joint instantaneous+MeanFlow trainer:
    instantaneous still uses ODE replay, and MeanFlow's old branch adds another
    Euler rollout to ``E[r]≈1/2``.
    """
    if strategy not in ("generative", "trajectory"):
        return 0.0
    h = float(ode_step_size)
    if strategy == "generative":
        inst = 1.0 / h
    else:
        # Trajectory: E[t]/h + 1 for v_ref, E[t]≈1/2.
        inst = 0.5 / h + 1.0
    if replay_backend == "mean_flow":
        # Extra old-branch Euler to r (E[r]≈0.5) for the MeanFlow identity.
        return inst + 0.5 / h
    return inst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Default: results_dir/sim_mixture/dim<d>/pretrain_task0/checkpoint.pt",
    )
    p.add_argument(
        "--strategy",
        choices=("naive", "generative", "trajectory", "real"),
        required=True,
    )
    p.add_argument(
        "--lambda-mix",
        type=float,
        default=0.5,
        help="λ: batch old-fraction for generative/real; loss weight for trajectory. Ignored for naive.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument(
        "--replay-backend",
        choices=("ode", "mean_flow"),
        default="ode",
        help=(
            "ode: train instantaneous CFM only (frozen-ref ODE replay). "
            "mean_flow: jointly train instantaneous CFM + dual-time MeanFlow "
            "(two-branch identity; old branch rolls out frozen CFM)."
        ),
    )
    p.add_argument(
        "--mean-flow-checkpoint",
        type=Path,
        default=None,
        help="Optional init mean_flow.pt (default: random init from teacher arch).",
    )
    p.add_argument(
        "--mean-flow-steps",
        type=int,
        default=1,
        help="MeanFlow jumps used at eval when --replay-backend=mean_flow.",
    )
    p.add_argument(
        "--mf-lr",
        type=float,
        default=None,
        help="MeanFlow Adam lr (default: --lr).",
    )
    p.add_argument(
        "--diag-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of MeanFlow batch with r=t. Default 0.25 follows zhuyu "
            "CIFAR one-step (ratio_r_not_equal_t=0.75)."
        ),
    )
    p.add_argument(
        "--time-mu",
        type=float,
        default=2.0,
        help=(
            "Logit-normal mean for MeanFlow (r,t) sampling. Default +2.0 is the "
            "zhuyu CIFAR schedule mirrored into this repo's noise→data convention."
        ),
    )
    p.add_argument(
        "--time-sigma",
        type=float,
        default=2.0,
        help="Logit-normal std for MeanFlow (r,t) sampling (CIFAR one-step default).",
    )
    p.add_argument(
        "--adaptive-p",
        type=float,
        default=0.75,
        help="MeanFlow adaptive loss power (0 disables). Default 0.75 = zhuyu CIFAR.",
    )
    p.add_argument(
        "--mf-max-grad-norm",
        type=float,
        default=0.0,
        help="Grad-norm clip on the MeanFlow net (0 disables). JVP targets can spike.",
    )
    p.add_argument(
        "--mf-use-adamw",
        action="store_true",
        default=True,
        help="AdamW instead of Adam for the MeanFlow optimizer (zhuyu-cs/MeanFlow reference; default on).",
    )
    p.add_argument("--mf-no-adamw", dest="mf_use_adamw", action="store_false")
    p.add_argument(
        "--mf-adam-beta2",
        type=float,
        default=0.95,
        help="MeanFlow Adam/AdamW beta2 (zhuyu-cs/MeanFlow reference default: 0.95).",
    )
    p.add_argument("--mf-weight-decay", type=float, default=0.0)
    p.add_argument(
        "--traj-kappa",
        choices=("ot", "none"),
        default="none",
        help=(
            "Thm. 2.4 time weight on the full trajectory objective: "
            "κ(t)·(λ L_traj + (1-λ) L_CFM-new) with a shared t. "
            "none = κ≡1 (training default); ot = t/(1-t) for the linear OT path."
        ),
    )
    p.add_argument("--n-gen-eval", type=int, default=4000)
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help=(
            "If >0, run mixture eval every N epochs during FT and store an "
            "eval_curve (with estimated cumulative ref NFEs) in metrics.json."
        ),
    )
    p.add_argument(
        "--n-gen-eval-curve",
        type=int,
        default=None,
        help="Samples for mid-training evals (default: --n-gen-eval).",
    )
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.strategy != "naive" and not (0.0 < args.lambda_mix < 1.0):
        raise SystemExit("--lambda-mix must be in (0, 1) for replay strategies")
    if args.replay_backend == "mean_flow" and args.strategy not in ("generative", "trajectory"):
        raise SystemExit("--replay-backend mean_flow requires generative or trajectory")
    if args.mean_flow_steps < 1:
        raise SystemExit("--mean-flow-steps must be >= 1")
    if args.eval_every < 0:
        raise SystemExit("--eval-every must be >= 0")
    if not (0.0 <= args.diag_fraction <= 1.0):
        raise SystemExit("--diag-fraction must be in [0, 1]")
    if args.time_sigma <= 0.0:
        raise SystemExit("--time-sigma must be > 0")
    if args.adaptive_p < 0.0:
        raise SystemExit("--adaptive-p must be >= 0")
    if args.mean_flow_checkpoint is not None and args.replay_backend != "mean_flow":
        raise SystemExit("--mean-flow-checkpoint requires --replay-backend mean_flow")
    # Seed before any model construction: make_mean_flow_from_teacher below draws from
    # the *global* torch RNG for weight init, and train_unconditional_joint_mean_flow's
    # internal torch.manual_seed(train_seed) runs too late to cover that (was previously
    # uncontrolled, so --train-seed didn't make the fresh MeanFlow init reproducible).
    torch.manual_seed(args.train_seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir or mixture_data_root(REPO_ROOT, dim=args.dim)
    ds, meta = load_tasks(data_dir)

    pre_ckpt = args.pretrain_checkpoint or (
        mixture_runs_root(REPO_ROOT, dim=args.dim) / "pretrain_task0" / "checkpoint.pt"
    )
    model_ref, pre_cfg = load_checkpoint(pre_ckpt, device)
    model_ref = make_frozen_model_copy(model_ref)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad = False

    model = clone_trainable(model_ref, device)

    mean_flow = None
    mean_flow_init_path = None
    use_joint_mf = args.replay_backend == "mean_flow"
    if use_joint_mf:
        from trajectory_replay.training.mean_flow_distill import (
            load_mean_flow_checkpoint,
            make_mean_flow_from_teacher,
            save_mean_flow_checkpoint,
        )
        from trajectory_replay.training.mean_flow_cl import (
            train_unconditional_joint_mean_flow,
        )

        if args.mean_flow_checkpoint is not None:
            mean_flow_init_path = args.mean_flow_checkpoint
            mean_flow, _ = load_mean_flow_checkpoint(mean_flow_init_path, device)
            print(f"Joint MeanFlow init from {mean_flow_init_path}")
        else:
            mean_flow = make_mean_flow_from_teacher(model_ref)
            mean_flow.to(device)
            print("Joint MeanFlow: fresh init from teacher architecture")
        for p in mean_flow.parameters():
            p.requires_grad_(True)
        mean_flow.train()
        # If init comes from a distilled checkpoint, freeze a copy for old-branch
        # replay while the trainable MeanFlow continues from the same weights.
        replay_mean_flow = None
        if mean_flow_init_path is not None:
            replay_mean_flow = copy.deepcopy(mean_flow)
            replay_mean_flow.eval()
            for p in replay_mean_flow.parameters():
                p.requires_grad_(False)
            print("  frozen MeanFlow copy enabled for old-branch replay")
    else:
        replay_mean_flow = None

    x_ft = ds.subset(1).x
    loader = make_x_loader(x_ft, batch_size=args.batch_size, shuffle=True, seed=args.train_seed)
    x_support = torch.from_numpy(ds.subset(0).x.astype("float32"))
    steps_per_epoch = len(loader)
    nfe_per_step = estimated_ref_nfe_per_step(
        args.strategy,
        ode_step_size=args.ode_step_size,
        mean_flow_n_steps=args.mean_flow_steps,
        replay_backend=args.replay_backend,
    )

    lam_tag = "naive" if args.strategy == "naive" else _lam_tag(args.lambda_mix)
    backend_tag = "_mf" if use_joint_mf else ""
    run_name = args.run_name or f"ft_{args.strategy}{backend_tag}_{lam_tag}"
    out_dir = mixture_runs_root(REPO_ROOT, dim=args.dim) / "finetune" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Finetune strategy={args.strategy} λ={args.lambda_mix} "
        f"backend={args.replay_backend} n_ft={len(x_ft)} device={device} -> {out_dir}"
    )
    if args.eval_every:
        print(
            f"Eval curve every {args.eval_every} epochs "
            f"(steps/epoch={steps_per_epoch}, est. ref NFE/step={nfe_per_step:g})"
        )

    lambda_target = None if args.strategy == "naive" else float(args.lambda_mix)
    x_mix = None if lambda_target is None else load_mixture(data_dir, float(args.lambda_mix)).x
    n_gen_curve = int(args.n_gen_eval_curve or args.n_gen_eval)
    eval_curve: list[dict[str, float]] = []

    def on_epoch_end(epoch: int, _row: dict) -> None:
        if args.eval_every <= 0:
            return
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            return
        was_training = model.training
        model.eval()
        metrics, _ = evaluate_model(
            model,
            ds,
            device=device,
            n_gen=n_gen_curve,
            seed=args.train_seed + 10_000 + epoch,
            lambda_target=lambda_target,
            sigma=float(meta["sigma"]),
            x_true_mixture=x_mix,
            ode_method=args.ode_method,
            ode_step_size=args.ode_step_size,
        )
        if was_training:
            model.train()
        point = {
            "epoch": float(epoch),
            "steps": float(epoch * steps_per_epoch),
            "nfe_per_step": float(nfe_per_step),
            "cum_ref_nfe": float(epoch * steps_per_epoch * nfe_per_step),
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        eval_curve.append(point)
        print(
            f"  eval @ epoch {epoch}: pi0_hat={metrics['pi0_hat']:.3f}  "
            f"E_act={metrics.get('energy_mixture_active', float('nan')):.4f}  "
            f"cum_ref_nfe={point['cum_ref_nfe']:.3e}"
        )

    adaptive_p = float(args.adaptive_p) if args.adaptive_p > 0 else None
    if use_joint_mf:
        assert mean_flow is not None
        history = train_unconditional_joint_mean_flow(
            model,
            mean_flow,
            model_ref,
            loader,
            strategy=args.strategy,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
            lambda_mix=args.lambda_mix,
            ode_method=args.ode_method,
            ode_step_size=args.ode_step_size,
            traj_kappa=args.traj_kappa,
            replay_mean_flow=replay_mean_flow,
            mean_flow_n_steps=int(args.mean_flow_steps),
            mf_lr=args.mf_lr,
            diag_fraction=args.diag_fraction,
            adaptive_p=adaptive_p,
            time_mu=args.time_mu,
            time_sigma=args.time_sigma,
            mf_max_grad_norm=(
                float(args.mf_max_grad_norm) if args.mf_max_grad_norm > 0 else None
            ),
            mf_use_adamw=args.mf_use_adamw,
            mf_betas=(0.9, float(args.mf_adam_beta2)),
            mf_weight_decay=float(args.mf_weight_decay),
            on_epoch_end=on_epoch_end if args.eval_every > 0 else None,
        )
    else:
        history = run_finetune_strategy(
            args.strategy,
            model,
            model_ref,
            loader,
            x_support=x_support,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
            lambda_mix=args.lambda_mix,
            ode_method=args.ode_method,
            ode_step_size=args.ode_step_size,
            mean_flow=None,
            mean_flow_n_steps=args.mean_flow_steps,
            traj_kappa=args.traj_kappa,
            on_epoch_end=on_epoch_end if args.eval_every > 0 else None,
        )

    cfg = {
        **{k: pre_cfg.get(k) for k in ("data_dim", "time_emb_dim", "hidden_dim", "depth")},
        "use_conditioning": False,
        "strategy": args.strategy,
        "lambda_mix": None if args.strategy == "naive" else float(args.lambda_mix),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mf_lr": (float(args.mf_lr) if args.mf_lr is not None else float(args.lr))
        if use_joint_mf
        else None,
        "train_seed": args.train_seed,
        "pretrain_checkpoint": str(pre_ckpt),
        "replay_backend": (
            args.replay_backend if args.strategy in ("generative", "trajectory") else None
        ),
        "mean_flow_init_checkpoint": (
            str(mean_flow_init_path) if mean_flow_init_path is not None else None
        ),
        "mean_flow_steps": args.mean_flow_steps if use_joint_mf else None,
        "diag_fraction": float(args.diag_fraction) if use_joint_mf else None,
        "time_mu": float(args.time_mu) if use_joint_mf else None,
        "time_sigma": float(args.time_sigma) if use_joint_mf else None,
        "adaptive_p": adaptive_p if use_joint_mf else None,
        "mf_max_grad_norm": (
            float(args.mf_max_grad_norm) if use_joint_mf and args.mf_max_grad_norm > 0 else None
        ),
        "mf_use_adamw": bool(args.mf_use_adamw) if use_joint_mf else None,
        "mf_adam_betas": [0.9, float(args.mf_adam_beta2)] if use_joint_mf else None,
        "mf_weight_decay": float(args.mf_weight_decay) if use_joint_mf else None,
        "traj_kappa": args.traj_kappa if args.strategy == "trajectory" else None,
        "ode_method": args.ode_method,
        "ode_step_size": args.ode_step_size,
        "eval_every": args.eval_every or None,
        "steps_per_epoch": steps_per_epoch,
        "nfe_per_step_est": nfe_per_step,
        "data_dir": str(data_dir),
        "data_meta": meta,
        "phase": "finetune",
        "joint_mean_flow": bool(use_joint_mf),
    }
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)

    metrics, x_gen = evaluate_model(
        model,
        ds,
        device=device,
        n_gen=args.n_gen_eval,
        seed=args.train_seed + 999,
        lambda_target=lambda_target,
        sigma=float(meta["sigma"]),
        x_true_mixture=x_mix,
        ode_method=args.ode_method,
        ode_step_size=args.ode_step_size,
    )

    mf_metrics = None
    mf_nfe_curve: list[dict[str, float]] = []
    if use_joint_mf:
        assert mean_flow is not None
        save_mean_flow_checkpoint(
            out_dir / "mean_flow.pt",
            mean_flow,
            cfg,
            parent_checkpoint="checkpoint.pt",
            extra={"stage": "finetune_joint", "mean_flow_steps": args.mean_flow_steps},
        )
        mf_metrics, x_gen_mf = evaluate_mean_flow(
            mean_flow,
            ds,
            device=device,
            n_gen=args.n_gen_eval,
            seed=args.train_seed + 1999,
            n_steps=args.mean_flow_steps,
            lambda_target=lambda_target,
            sigma=float(meta["sigma"]),
            x_true_mixture=x_mix,
        )
        save_scatter(
            out_dir / "samples_mean_flow.png",
            x_gen=x_gen_mf,
            means=ds.means,
            title=f"MeanFlow {args.strategy} λ={args.lambda_mix} N={args.mean_flow_steps}",
            x_true=ds.x,
        )
        print(
            f"MeanFlow pi0_hat={mf_metrics['pi0_hat']:.3f}  "
            f"(steps={args.mean_flow_steps})"
        )
        # Sweeping N separates a mis-learned average-velocity field (bad at every N)
        # from an undertrained one-step corner (bad at N=1, fine as N grows).
        for n_steps in (1, 2, 4, 8):
            if n_steps == args.mean_flow_steps:
                mf_nfe_curve.append({"n_steps": float(n_steps), **_scalars(mf_metrics)})
                continue
            m_n, _ = evaluate_mean_flow(
                mean_flow,
                ds,
                device=device,
                n_gen=args.n_gen_eval,
                seed=args.train_seed + 1999,
                n_steps=n_steps,
                lambda_target=lambda_target,
                sigma=float(meta["sigma"]),
                x_true_mixture=x_mix,
            )
            mf_nfe_curve.append({"n_steps": float(n_steps), **_scalars(m_n)})
        print(
            "MeanFlow pi0_hat vs N: "
            + "  ".join(f"N={int(r['n_steps'])}:{r['pi0_hat']:.3f}" for r in mf_nfe_curve)
        )

    # Ensure final epoch is on the curve even if eval_every does not divide epochs.
    if args.eval_every > 0 and (not eval_curve or int(eval_curve[-1]["epoch"]) != args.epochs):
        eval_curve.append(
            {
                "epoch": float(args.epochs),
                "steps": float(args.epochs * steps_per_epoch),
                "nfe_per_step": float(nfe_per_step),
                "cum_ref_nfe": float(args.epochs * steps_per_epoch * nfe_per_step),
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            }
        )
    dump_metrics(
        out_dir / "metrics.json",
        {
            "eval": metrics,
            "eval_mean_flow": mf_metrics,
            "eval_mean_flow_nfe": mf_nfe_curve,
            "history": history,
            "eval_curve": eval_curve,
            "config": cfg,
        },
    )
    save_scatter(
        out_dir / "samples.png",
        x_gen=x_gen,
        means=ds.means,
        title=f"{args.strategy} λ={args.lambda_mix}",
        x_true=ds.x,
    )
    print(f"Saved {out_dir}")
    print(
        f"pi0_hat={metrics['pi0_hat']:.3f}  metrics="
        f"{ {k: metrics[k] for k in metrics if k.startswith('pi') or 'energy' in k or 'error' in k} }"
    )


if __name__ == "__main__":
    main()
