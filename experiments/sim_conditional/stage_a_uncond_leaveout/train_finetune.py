#!/usr/bin/env python3
"""Stage A uncond: finetune on held-out mode with naive / gen / traj replay."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from experiments.sim_conditional.stage_a_uncond_leaveout.common import (
    REPO_ROOT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    evaluate_fit_forget,
    evaluate_modes,
    load_checkpoint,
    load_stage_a_data,
    make_x_loader,
    mask_rows,
    sample_unconditional,
    save_checkpoint,
    save_scatter,
    stage_a_uncond_data_root,
    stage_a_uncond_runs_root,
)
from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.mixture_fm import (
    train_unconditional_fm,
    train_unconditional_generative_replay,
    train_unconditional_trajectory_replay,
)
from trajectory_replay.utils.cli import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--pretrain-checkpoint", type=Path, default=None)
    p.add_argument(
        "--strategy",
        choices=("naive", "generative", "trajectory"),
        required=True,
    )
    p.add_argument("--lambda-mix", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-gen-eval", type=int, default=2000)
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--traj-kappa", default="none")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def _default_run_name(strategy: str, lam: float, epochs: int, seed: int) -> str:
    if strategy == "naive":
        return f"ft_naive_e{epochs}_seed{seed}"
    return f"ft_{strategy}_lam{lam}_e{epochs}_seed{seed}"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = args.data_dir or stage_a_uncond_data_root(REPO_ROOT, dim=args.dim)
    arrays, meta = load_stage_a_data(data_dir)
    heldout = int(meta["heldout_class"])
    pretrain_classes = [int(k) for k in meta["pretrain_classes"]]

    ckpt = args.pretrain_checkpoint or (
        stage_a_uncond_runs_root(REPO_ROOT, dim=args.dim)
        / "pretrain_seed0"
        / "checkpoint.pt"
    )
    model, _pre_cfg = load_checkpoint(ckpt, device)
    model_ref = copy.deepcopy(model).to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    m_ft = mask_rows(arrays, classes=[heldout], split=SPLIT_TRAIN)
    loader = make_x_loader(
        arrays["x"][m_ft],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.train_seed,
    )

    print(
        f"Stage A uncond FT strategy={args.strategy} λ={args.lambda_mix} "
        f"heldout={heldout} n_ft={int(m_ft.sum())} device={device}"
    )

    if args.strategy == "naive":
        history = train_unconditional_fm(
            model,
            loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
        )
    elif args.strategy == "generative":
        if not (0.0 < float(args.lambda_mix) < 1.0):
            raise SystemExit("--lambda-mix must be in (0, 1)")
        history = train_unconditional_generative_replay(
            model,
            model_ref,
            loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
            lambda_mix=float(args.lambda_mix),
            ode_method=args.ode_method,
            ode_step_size=args.ode_step_size,
        )
    else:
        if not (0.0 < float(args.lambda_mix) < 1.0):
            raise SystemExit("--lambda-mix must be in (0, 1)")
        history = train_unconditional_trajectory_replay(
            model,
            model_ref,
            loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
            lambda_mix=float(args.lambda_mix),
            ode_method=args.ode_method,
            ode_step_size=args.ode_step_size,
            traj_kappa=args.traj_kappa,
        )

    run_name = args.run_name or _default_run_name(
        args.strategy, args.lambda_mix, args.epochs, args.train_seed
    )
    out_dir = stage_a_uncond_runs_root(REPO_ROOT, dim=args.dim) / "finetune" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "stage": "a_uncond_leaveout",
        "phase": "finetune",
        "strategy": args.strategy,
        "lambda_mix": None if args.strategy == "naive" else float(args.lambda_mix),
        "data_dim": args.dim,
        "n_classes": int(meta["n_classes"]),
        "heldout_class": heldout,
        "pretrain_classes": pretrain_classes,
        "use_conditioning": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "pretrain_checkpoint": str(ckpt),
        "traj_kappa": args.traj_kappa if args.strategy == "trajectory" else None,
        "ode_method": args.ode_method,
        "ode_step_size": args.ode_step_size,
    }
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)

    pretrained_forget_val = evaluate_modes(
        model_ref,
        arrays,
        pretrain_classes,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 3,
        n_gen=args.n_gen_eval,
    )
    val_scores = evaluate_fit_forget(
        model,
        arrays,
        heldout_class=heldout,
        pretrain_classes=pretrain_classes,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 11,
        n_gen=args.n_gen_eval,
        pretrained_forget=pretrained_forget_val,
    )
    test_scores = evaluate_fit_forget(
        model,
        arrays,
        heldout_class=heldout,
        pretrain_classes=pretrain_classes,
        split=SPLIT_TEST,
        device=device,
        seed=args.train_seed + 13,
        n_gen=args.n_gen_eval,
        pretrained_forget=evaluate_modes(
            model_ref,
            arrays,
            pretrain_classes,
            split=SPLIT_TEST,
            device=device,
            seed=args.train_seed + 5,
            n_gen=args.n_gen_eval,
        ),
    )

    dump_metrics(
        out_dir / "metrics.json",
        {
            "val": val_scores,
            "test": test_scores,
            "pretrained_forget_val": pretrained_forget_val,
            "history": history,
            "config": cfg,
        },
    )

    m_all = mask_rows(
        arrays, classes=pretrain_classes + [heldout], split=SPLIT_VAL
    )
    x_gen = sample_unconditional(
        model, 1200, device=device, seed=args.train_seed + 50
    )
    save_scatter(
        out_dir / "samples.png",
        x_true=arrays["x"][m_all],
        x_gen=x_gen,
        means=arrays["means"],
        title=f"Uncond FT {args.strategy} λ={args.lambda_mix} (heldout={heldout})",
    )

    print(f"Saved {out_dir}")
    print(
        f"VAL  fit_E={val_scores['fit_energy']:.4f}  "
        f"forget_E={val_scores['forget_energy_mean']:.4f}  "
        f"Δforget_E={val_scores.get('delta_forget_energy_mean', float('nan')):.4f}  "
        f"heldout_mass={val_scores.get('heldout_mass_joint', float('nan')):.3f}"
    )
    print(
        f"TEST fit_E={test_scores['fit_energy']:.4f}  "
        f"forget_E={test_scores['forget_energy_mean']:.4f}  "
        f"Δforget_E={test_scores.get('delta_forget_energy_mean', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
