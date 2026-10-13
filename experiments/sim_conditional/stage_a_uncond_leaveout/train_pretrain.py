#!/usr/bin/env python3
"""Stage A uncond: pretrain unconditional FM on non-held-out modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sim_conditional.stage_a_uncond_leaveout.common import (
    REPO_ROOT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_unconditional_model,
    evaluate_modes,
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
from trajectory_replay.training.mixture_fm import train_unconditional_fm
from trajectory_replay.utils.cli import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
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
    data_dir = args.data_dir or stage_a_uncond_data_root(REPO_ROOT, dim=args.dim)
    arrays, meta = load_stage_a_data(data_dir)
    if int(meta["dim"]) != args.dim:
        raise SystemExit(f"data dim {meta['dim']} != --dim {args.dim}")

    pretrain_classes = [int(k) for k in meta["pretrain_classes"]]
    heldout = int(meta["heldout_class"])

    m = mask_rows(arrays, classes=pretrain_classes, split=SPLIT_TRAIN)
    x = arrays["x"][m]
    loader = make_x_loader(
        x, batch_size=args.batch_size, shuffle=True, seed=args.train_seed
    )
    model = build_unconditional_model(
        args.dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    print(
        f"Stage A uncond pretrain: n={len(x)} classes={pretrain_classes} "
        f"(heldout={heldout}) dim={args.dim} device={device}"
    )
    history = train_unconditional_fm(
        model,
        loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        train_seed=args.train_seed,
    )

    out_dir = stage_a_uncond_runs_root(REPO_ROOT, dim=args.dim) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "stage": "a_uncond_leaveout",
        "phase": "pretrain",
        "data_dim": args.dim,
        "n_classes": int(meta["n_classes"]),
        "pretrain_classes": pretrain_classes,
        "heldout_class": heldout,
        "use_conditioning": False,
        "time_emb_dim": args.time_emb_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "data_dir": str(data_dir),
    }
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)

    forget_val = evaluate_modes(
        model,
        arrays,
        pretrain_classes,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 7,
        n_gen=args.n_gen_eval,
    )
    dump_metrics(
        out_dir / "metrics.json",
        {
            "pretrain": {"n_train": int(len(x)), "forget_val": forget_val},
            "history": history,
            "config": cfg,
        },
    )

    m_val = mask_rows(arrays, classes=pretrain_classes, split=SPLIT_VAL)
    x_gen = sample_unconditional(
        model, min(1200, args.n_gen_eval), device=device, seed=args.train_seed
    )
    save_scatter(
        out_dir / "samples.png",
        x_true=arrays["x"][m_val],
        x_gen=x_gen,
        means=arrays["means"],
        title=f"Uncond pretrain (modes {pretrain_classes})",
    )
    print(f"Saved {out_dir}")
    print(
        f"val forget energy_mean={forget_val['energy_mean']:.4f} "
        f"mean_l2_mean={forget_val['mean_l2_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
