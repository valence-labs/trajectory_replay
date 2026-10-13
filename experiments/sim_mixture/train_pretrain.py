#!/usr/bin/env python3
"""Pretrain unconditional FM on MoG task 0 (old mode)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.mixture_fm import train_unconditional_fm
from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    REPO_ROOT,
    build_unconditional_model,
    evaluate_model,
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
    p.add_argument("--run-name", default="pretrain_task0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Seed before model construction: build_unconditional_model below draws from the
    # *global* torch RNG for weight init, and train_unconditional_fm's internal
    # torch.manual_seed(train_seed) runs too late to cover that.
    torch.manual_seed(args.train_seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir or mixture_data_root(REPO_ROOT, dim=args.dim)
    ds, meta = load_tasks(data_dir)
    if ds.dim != args.dim:
        raise ValueError(f"data dim {ds.dim} != --dim {args.dim}")

    x0 = ds.subset(0).x
    loader = make_x_loader(x0, batch_size=args.batch_size, shuffle=True, seed=args.train_seed)
    model = build_unconditional_model(
        args.dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    print(f"Pretrain on task 0: n={len(x0)}, dim={args.dim}, device={device}")
    history = train_unconditional_fm(
        model,
        loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        train_seed=args.train_seed,
    )

    out_dir = mixture_runs_root(REPO_ROOT, dim=args.dim) / args.run_name
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
        "data_dir": str(data_dir),
        "data_meta": meta,
        "phase": "pretrain_task0",
    }
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)

    metrics, x_gen = evaluate_model(
        model,
        ds,
        device=device,
        n_gen=args.n_gen_eval,
        seed=args.train_seed + 123,
        lambda_target=1.0,
        sigma=float(meta["sigma"]),
    )
    dump_metrics(out_dir / "metrics.json", {"pretrain": metrics, "history": history})
    save_scatter(
        out_dir / "samples.png",
        x_gen=x_gen,
        means=ds.means,
        title="Pretrain gens (expect mode 0)",
        x_true=x0,
    )
    print(f"Saved {out_dir}")
    print(f"pi0_hat={metrics['pi0_hat']:.3f}  energy_mode0={metrics.get('energy_mode0')}")


if __name__ == "__main__":
    main()
