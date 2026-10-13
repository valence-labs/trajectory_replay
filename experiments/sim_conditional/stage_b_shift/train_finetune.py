#!/usr/bin/env python3
"""Stage B: finetune on FT context cells with naive / gen / traj replay."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from experiments.sim_conditional.stage_b_shift.common import (
    POOL_FINETUNE,
    POOL_HELDOUT,
    POOL_PRETRAIN,
    REPO_ROOT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    evaluate_cells,
    evaluate_fit_forget,
    load_checkpoint,
    load_stage_b_data,
    make_cp_loader,
    mask_rows,
    sample_cp,
    save_checkpoint,
    save_overlay_scatter,
    save_stylized_heldout_overlay,
    save_heldout_shift_scatter,
    save_pretrain_scatter,
    resolve_stage_b_roots,
)
from trajectory_replay.io.results import dump_metrics
from trajectory_replay.training.flow_matching import (
    train_flow_matching,
    train_flow_matching_generative_replay,
)
from trajectory_replay.utils.cli import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-contexts", type=int, default=None)
    p.add_argument("--n-perturbations", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--pretrain-checkpoint", type=Path, default=None)
    p.add_argument(
        "--strategy",
        choices=("naive", "generative", "trajectory"),
        required=True,
    )
    p.add_argument("--lambda-mix", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-gen-eval", type=int, default=500)
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
    ft_context = int(meta["ft_context"])
    heldout_pert = int(meta["heldout_pert"])
    heldout_cell = tuple(map(int, meta["heldout_combo"]))
    pretrain_contexts = [int(c) for c in meta["pretrain_contexts"]]
    pretrain_cells = [tuple(map(int, cp)) for cp in meta["pretrain_cells"]]
    finetune_cells = [tuple(map(int, cp)) for cp in meta["finetune_cells"]]

    # Pair with the pretrain run of the same seed unless told otherwise.
    ckpt = args.pretrain_checkpoint or (
        runs_root / f"pretrain_seed{args.train_seed}" / "checkpoint.pt"
    )
    if not ckpt.exists():
        raise SystemExit(f"missing pretrain checkpoint: {ckpt}")
    model, pre_cfg = load_checkpoint(ckpt, device)
    if int(pre_cfg.get("n_contexts", n_c)) != n_c:
        raise SystemExit("pretrain n_contexts mismatch vs data")
    if int(pre_cfg.get("n_perturbations", n_p)) != n_p:
        raise SystemExit("pretrain n_perturbations mismatch vs data")

    model_ref = copy.deepcopy(model).to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    m_ft = mask_rows(arrays, pool=POOL_FINETUNE, split=SPLIT_TRAIN)
    loader = make_cp_loader(
        arrays["x"][m_ft],
        arrays["c"][m_ft],
        arrays["p"][m_ft],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.train_seed,
    )

    print(
        f"Stage B FT strategy={args.strategy} λ={args.lambda_mix} "
        f"ft_c={ft_context} heldout={heldout_cell} n_ft={int(m_ft.sum())} "
        f"device={device}"
    )

    if args.strategy == "naive":
        history = train_flow_matching(
            model,
            loader,
            data_dim=args.dim,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
            finetune_context_index=ft_context,
        )
    else:
        if not (0.0 < float(args.lambda_mix) < 1.0):
            raise SystemExit("--lambda-mix must be in (0, 1) for replay strategies")
        replay_mode = "endpoint" if args.strategy == "generative" else "field"
        history = train_flow_matching_generative_replay(
            model,
            model_ref,
            loader,
            data_dim=args.dim,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            train_seed=args.train_seed,
            support_contexts=pretrain_contexts,
            n_perturbations=n_p,
            replay_fraction=float(args.lambda_mix),
            replay_mode=replay_mode,
            replay_ode_method=args.ode_method,
            replay_ode_step_size=args.ode_step_size,
            replay_condition_sampling="uniform",
            finetune_context_index=ft_context,
            traj_kappa=args.traj_kappa,
        )

    run_name = args.run_name or _default_run_name(
        args.strategy, args.lambda_mix, args.epochs, args.train_seed
    )
    out_dir = runs_root / "finetune" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "stage": "b_shift",
        "phase": "finetune",
        "strategy": args.strategy,
        "lambda_mix": None if args.strategy == "naive" else float(args.lambda_mix),
        "data_dim": args.dim,
        "n_contexts": n_c,
        "n_perturbations": n_p,
        "ft_context": ft_context,
        "heldout_pert": heldout_pert,
        "heldout_combo": list(heldout_cell),
        "pretrain_contexts": pretrain_contexts,
        "pretrain_cells": [list(cp) for cp in pretrain_cells],
        "finetune_cells": [list(cp) for cp in finetune_cells],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "pretrain_checkpoint": str(ckpt),
        "traj_kappa": args.traj_kappa if args.strategy == "trajectory" else None,
        "ode_method": args.ode_method,
        "ode_step_size": args.ode_step_size,
        "max_grad_norm": None,
        "cond_emb_dim": int(pre_cfg.get("cond_emb_dim", 32)),
        "time_emb_dim": int(pre_cfg.get("time_emb_dim", 64)),
        "hidden_dim": int(pre_cfg.get("hidden_dim", 256)),
        "depth": int(pre_cfg.get("depth", 4)),
    }
    save_checkpoint(out_dir / "checkpoint.pt", model, config=cfg, history=history)

    pretrained_forget_val = evaluate_cells(
        model_ref,
        arrays,
        pretrain_cells,
        pool=POOL_PRETRAIN,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 3,
        n_gen=args.n_gen_eval,
    )
    val_scores = evaluate_fit_forget(
        model,
        arrays,
        finetune_cells=finetune_cells,
        pretrain_cells=pretrain_cells,
        heldout_cell=heldout_cell,
        split=SPLIT_VAL,
        device=device,
        seed=args.train_seed + 11,
        n_gen=args.n_gen_eval,
        pretrained_forget=pretrained_forget_val,
    )
    # The heldout cell (c*, p*) is the only cell with test rows, so it carries the
    # single transfer readout; fit/forget stay on val. Seed matches the historical
    # evaluate_fit_forget(split=test) path so published numbers stay reproducible.
    heldout_test = evaluate_cells(
        model,
        arrays,
        [heldout_cell],
        pool=POOL_HELDOUT,
        split=SPLIT_TEST,
        device=device,
        seed=args.train_seed + 13 + 211,
        n_gen=args.n_gen_eval,
    )
    ho_key = f"{heldout_cell[0]},{heldout_cell[1]}"
    ho_per = heldout_test["per_cell"].get(ho_key, {})
    heldout_energy = float(ho_per.get("energy", float("nan")))
    heldout_mean_l2 = float(ho_per.get("mean_l2", float("nan")))
    val_scores["heldout"] = heldout_test
    val_scores["heldout_energy"] = heldout_energy
    val_scores["heldout_mean_l2"] = heldout_mean_l2
    val_scores["heldout_split"] = "test"

    test_scores = {
        "heldout_energy": heldout_energy,
        "heldout_mean_l2": heldout_mean_l2,
        "heldout": heldout_test,
        "note": (
            "fit/forget cells have no test rows; test covers the heldout cell only"
        ),
    }

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

    demo = list(finetune_cells) + [heldout_cell] + pretrain_cells[:2]
    m_vis = mask_rows(arrays, pool=POOL_FINETUNE, split=SPLIT_VAL)
    m_pt = mask_rows(arrays, pool=POOL_PRETRAIN, split=SPLIT_VAL)
    m_ho = mask_rows(arrays, pool=POOL_HELDOUT, split=SPLIT_TEST)
    gens = {
        (ci, pj): sample_cp(
            model, ci, pj, 250, device=device, seed=args.train_seed + 50 + ci * 10 + pj
        )
        for ci, pj in demo
    }
    # Sample a larger heldout cloud for the shift plot.
    x_gen_heldout = sample_cp(
        model,
        heldout_cell[0],
        heldout_cell[1],
        600,
        device=device,
        seed=args.train_seed + 777,
    )
    save_pretrain_scatter(
        out_dir / "samples.png",
        x_true=np.concatenate(
            [arrays["x"][m_vis], arrays["x"][m_pt], arrays["x"][m_ho]], axis=0
        ),
        c_true=np.concatenate(
            [arrays["c"][m_vis], arrays["c"][m_pt], arrays["c"][m_ho]], axis=0
        ),
        p_true=np.concatenate(
            [arrays["p"][m_vis], arrays["p"][m_pt], arrays["p"][m_ho]], axis=0
        ),
        gens=gens,
        title=f"Stage B FT {args.strategy} λ={args.lambda_mix}",
    )
    # Dedicated heldout test overlay (compact Stage A–style panel).
    strat_labels = {
        "naive": "Naive FT",
        "generative": "Generative replay",
        "trajectory": "Trajectory replay",
    }
    strat_colors = {
        "naive": "#6c757d",
        "generative": "#c1121f",
        "trajectory": "#1f6aa5",
    }
    save_stylized_heldout_overlay(
        out_dir / "heldout_test_overlay.png",
        x_true=arrays["x"][m_ho],
        x_gen=x_gen_heldout,
        gen_label=strat_labels.get(args.strategy, "Generated"),
        gen_color=strat_colors.get(args.strategy, "#1f6aa5"),
    )
    # Gray all-cells + red generated heldout (visual shift readout).
    save_heldout_shift_scatter(
        out_dir / "heldout_shift.png",
        x_true=np.concatenate(
            [arrays["x"][m_vis], arrays["x"][m_pt], arrays["x"][m_ho]], axis=0
        ),
        x_gen_heldout=x_gen_heldout,
        title=rf"Heldout gen vs all real  |  {args.strategy} $\lambda$={args.lambda_mix}",
    )

    print(f"Saved {out_dir}")
    print(
        f"VAL  fit_E={val_scores['fit_energy_mean']:.4f}  "
        f"forget_E={val_scores['forget_energy_mean']:.4f}  "
        f"Δforget_E={val_scores.get('delta_forget_energy_mean', float('nan')):.4f}  "
        f"heldout_E={val_scores['heldout_energy']:.4f}"
    )
    print(f"TEST heldout_E={test_scores['heldout_energy']:.4f}")


if __name__ == "__main__":
    main()
