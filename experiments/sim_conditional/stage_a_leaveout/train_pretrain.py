#!/usr/bin/env python3
"""Stage A: pretrain conditional FM on all non-held-out classes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.sim_conditional.common import (
    REPO_ROOT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_conditional_model,
    evaluate_classes,
    load_stage_a_data,
    make_cp_loader,
    mask_rows,
    sample_class,
    save_checkpoint,
    save_class_scatter,
    stage_a_data_root,
    stage_a_runs_root,
)
from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.flow_matching import train_flow_matching
from trajectory_replay.utils.cli import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--cond-emb-dim", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-gen-eval", type=int, default=2000)
    p.add_argument("--run-name", default="pretrain_seed0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = args.data_dir or stage_a_data_root(REPO_ROOT, dim=args.dim)
    arrays, meta = load_stage_a_data(data_dir)
    if int(meta["dim"]) != args.dim:
        raise SystemExit(f"data dim {meta['dim']} != --dim {args.dim}")

    n_classes = int(meta["n_classes"])
    pretrain_classes = [int(k) for k in meta["pretrain_classes"]]
    heldout = int(meta["heldout_class"])

    m = mask_rows(arrays, classes=pretrain_classes, split=SPLIT_TRAIN)
    x, y = arrays["x"][m], arrays["y"][m]
    loader = make_cp_loader(
        x, y, batch_size=args.batch_size, shuffle=True, seed=args.train_seed
    )
    model = build_conditional_model(
        args.dim,
        n_classes,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    print(
        f"Stage A pretrain: n={len(x)} classes={pretrain_classes} "
        f"(heldout={heldout}) dim={args.dim} device={device}"
    )
    history = train_flow_matching(
        model,
        loader,
        data_dim=args.dim,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        train_seed=args.train_seed,
    )

    out_dir = stage_a_runs_root(REPO_ROOT, dim=args.dim) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "stage": "a_leaveout",
        "phase": "pretrain",
        "data_dim": args.dim,
        "n_classes": n_classes,
        "pretrain_classes": pretrain_classes,
        "heldout_class": heldout,
        "cond_emb_dim": args.cond_emb_dim,
        "time_emb_dim": args.time_emb_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "data_dir": str(data_dir),
    }
    save_checkpoint(
        out_dir / "checkpoint.pt", model, config=cfg, history=history
    )

    # Val metrics on pretrain classes (baseline forget reference for FT).
    forget_val = evaluate_classes(
        model,
        arrays,
        pretrain_classes,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 7,
        n_gen=args.n_gen_eval,
    )
    metrics = {
        "pretrain": {
            "n_train": int(len(x)),
            "forget_val": forget_val,
        },
        "history": history,
        "config": cfg,
    }
    dump_metrics(out_dir / "metrics.json", metrics)

    gens = {
        k: sample_class(
            model, k, min(800, args.n_gen_eval), device=device, seed=args.train_seed + k
        )
        for k in pretrain_classes
    }
    save_class_scatter(
        out_dir / "samples.png",
        arrays,
        gens,
        means=arrays["means"],
        title=f"Stage A pretrain (expect classes {pretrain_classes})",
    )
    print(f"Saved {out_dir}")
    print(
        f"val forget energy_mean={forget_val['energy_mean']:.4f} "
        f"mean_l2_mean={forget_val['mean_l2_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
