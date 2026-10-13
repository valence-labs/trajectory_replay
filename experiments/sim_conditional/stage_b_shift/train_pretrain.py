#!/usr/bin/env python3
"""Stage B: pretrain conditional FM on the pretrain-pool training split."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sim_conditional.stage_b_shift.common import (
    POOL_PRETRAIN,
    REPO_ROOT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_stage_b_model,
    evaluate_cells,
    load_stage_b_data,
    make_cp_loader,
    mask_rows,
    resolve_stage_b_roots,
    sample_cp,
    save_checkpoint,
    save_pretrain_scatter,
)
from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.flow_matching import train_flow_matching
from trajectory_replay.utils.cli import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-contexts", type=int, default=None)
    p.add_argument("--n-perturbations", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--cond-emb-dim", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-gen-eval", type=int, default=500)
    p.add_argument(
        "--run-name",
        default=None,
        help="Default: pretrain_seed{--train-seed}.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir, runs_root, meta = resolve_stage_b_roots(
        REPO_ROOT,
        dim=args.dim,
        data_dir=args.data_dir,
        n_contexts=args.n_contexts,
        n_perturbations=args.n_perturbations,
    )
    arrays, meta = load_stage_b_data(data_dir)
    if int(meta["dim"]) != args.dim:
        raise SystemExit(f"data dim {meta['dim']} != --dim {args.dim}")

    n_c = int(meta["n_contexts"])
    n_p = int(meta["n_perturbations"])
    pretrain_cells = [tuple(map(int, cp)) for cp in meta["pretrain_cells"]]
    ft_context = int(meta["ft_context"])
    heldout_pert = int(meta["heldout_pert"])

    m = mask_rows(arrays, pool=POOL_PRETRAIN, split=SPLIT_TRAIN)
    x, c, p = arrays["x"][m], arrays["c"][m], arrays["p"][m]
    loader = make_cp_loader(
        x, c, p, batch_size=args.batch_size, shuffle=True, seed=args.train_seed
    )
    model = build_stage_b_model(
        args.dim,
        n_c,
        n_p,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    print(
        f"Stage B pretrain: n={len(x)} cells={len(pretrain_cells)} "
        f"(ft_context={ft_context}, heldout_pert={heldout_pert}) "
        f"dim={args.dim} device={device}"
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

    run_name = args.run_name or f"pretrain_seed{args.train_seed}"
    out_dir = runs_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "stage": "b_shift",
        "phase": "pretrain",
        "data_dim": args.dim,
        "n_contexts": n_c,
        "n_perturbations": n_p,
        "pretrain_cells": [list(cp) for cp in pretrain_cells],
        "ft_context": ft_context,
        "heldout_pert": heldout_pert,
        "heldout_combo": meta["heldout_combo"],
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
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)

    val_scores = evaluate_cells(
        model,
        arrays,
        pretrain_cells,
        pool=POOL_PRETRAIN,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 7,
        n_gen=args.n_gen_eval,
    )
    dump_metrics(
        out_dir / "metrics.json",
        {
            "pretrain": {
                "n_train": int(len(x)),
                "val": val_scores,
            },
            "history": history,
            "config": cfg,
        },
    )

    # Scatter a few representative pretrain cells (true faint + gen).
    demo_cells = pretrain_cells[: min(6, len(pretrain_cells))]
    m_vis = mask_rows(arrays, pool=POOL_PRETRAIN, split=SPLIT_VAL)
    gens = {
        (ci, pj): sample_cp(
            model, ci, pj, 300, device=device, seed=args.train_seed + 100 + ci * 10 + pj
        )
        for ci, pj in demo_cells
    }
    save_pretrain_scatter(
        out_dir / "samples.png",
        x_true=arrays["x"][m_vis],
        c_true=arrays["c"][m_vis],
        p_true=arrays["p"][m_vis],
        gens=gens,
        title="Stage B pretrain (val true + gen)",
    )
    print(f"Saved {out_dir}")
    print(
        f"val energy_mean={val_scores['energy_mean']:.4f} "
        f"mean_l2_mean={val_scores['mean_l2_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
