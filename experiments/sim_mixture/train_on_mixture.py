#!/usr/bin/env python3
"""Train unconditional FM from scratch on a reference mixture dataset at fixed λ."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.mixture_fm import train_unconditional_fm
from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    REPO_ROOT,
    _lam_tag,
    build_unconditional_model,
    evaluate_model,
    load_mixture,
    load_tasks,
    make_x_loader,
    mixture_data_root,
    mixture_runs_root,
    save_checkpoint,
    save_scatter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--lambda-mix", type=float, required=True, help="Target mixture weight on mode 0.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-gen-eval", type=int, default=4000)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.lambda_mix < 1.0):
        raise SystemExit("--lambda-mix must be in (0, 1)")
    # Seed before model construction: build_unconditional_model below draws from the
    # *global* torch RNG for weight init, and train_unconditional_fm's internal
    # torch.manual_seed(train_seed) runs too late to cover that.
    torch.manual_seed(args.train_seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir or mixture_data_root(REPO_ROOT, dim=args.dim)
    ds_tasks, meta = load_tasks(data_dir)
    ds_mix = load_mixture(data_dir, args.lambda_mix)

    loader = make_x_loader(
        ds_mix.x, batch_size=args.batch_size, shuffle=True, seed=args.train_seed
    )
    model = build_unconditional_model(
        args.dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    tag = _lam_tag(args.lambda_mix)
    run_name = args.run_name or f"oracle_mixture_{tag}"
    print(
        f"Train FM on mixture λ={args.lambda_mix}: n={len(ds_mix.x)}, "
        f"device={device}, run={run_name}"
    )
    history = train_unconditional_fm(
        model,
        loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        train_seed=args.train_seed,
    )

    out_dir = mixture_runs_root(REPO_ROOT, dim=args.dim) / "oracle_mixture" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data_dim": args.dim,
        "use_conditioning": False,
        "time_emb_dim": args.time_emb_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "lambda_mix": args.lambda_mix,
        "data_dir": str(data_dir),
        "data_meta": meta,
        "phase": "oracle_mixture",
    }
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)
    metrics, x_gen = evaluate_model(
        model,
        ds_tasks,
        device=device,
        n_gen=args.n_gen_eval,
        seed=args.train_seed + 321,
        lambda_target=args.lambda_mix,
        sigma=float(meta["sigma"]),
        x_true_mixture=ds_mix.x,
    )
    dump_metrics(out_dir / "metrics.json", {"eval": metrics, "history": history, "config": cfg})
    save_scatter(
        out_dir / "samples.png",
        x_gen=x_gen,
        means=ds_tasks.means,
        title=f"Oracle FM on mixture λ={args.lambda_mix}",
        x_true=ds_mix.x,
    )
    print(f"Saved {out_dir}")
    print(
        f"pi0_hat={metrics['pi0_hat']:.3f} (target {args.lambda_mix})  "
        f"|err|={metrics['pi0_abs_error']:.3f}"
    )


if __name__ == "__main__":
    main()
