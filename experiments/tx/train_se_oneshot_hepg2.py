#!/usr/bin/env python3
"""One-shot CFM on Replogle STATE SE, then HepG2-only Pearson-Δ / CellEval.

Trains the same MLP CFM as sequential CL on **all non-test cells** (every cell
line, ``train`` + ``finetune``) in one go. Fits a **single** linear SE→gene
decoder on that pool. Scores only HepG2:

* fit — HepG2 train/finetune (capped)
* heldout — HepG2 STATE-test, with train/finetune controls injected

This isolates sequential-CL wiring from “CFM on SE + linear decoder”.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import numpy as np
import torch

from experiments.tx.eval_cl_celleval import evaluate_indices
from experiments.tx.train_cl import (
    DEFAULTS,
    dump_json,
    epochs_for_target_steps,
    eval_track,
    fit_decoder_for_task,
    make_loader,
    make_model,
    save_checkpoint,
    steps_per_epoch,
)
from trajectory_replay.data.tx_cl_data import (
    LATENT_STANDARDIZER_NAME,
    apply_latent_radial_scale,
    apply_latent_standardizer,
    fit_latent_standardizer,
    load_latent_standardizer,
    load_tx_cl_arrays,
    save_latent_standardizer,
)
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import (
    DEFAULT_BATCH_SIZE as DECODER_BATCH_SIZE,
    DEFAULT_DEPTH as DECODER_DEPTH,
    DEFAULT_EPOCHS as DECODER_EPOCHS,
    DEFAULT_HIDDEN as DECODER_HIDDEN,
    DEFAULT_LR as DECODER_LR,
    TaskDecoderBank,
)
from trajectory_replay.training.flow_matching import train_flow_matching


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("replogle",), default="replogle")
    p.add_argument("--h5ad-path", type=Path)
    p.add_argument("--se-path", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--run-name", default="oneshot_se_hepg2")
    p.add_argument(
        "--representation",
        choices=("se", "scimilarity"),
        default="se",
        help="Which frozen embedding sidecar --se-path points at (config tag only).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--target-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-cells", type=int, default=0)
    p.add_argument("--cond-emb-dim", type=int, default=128)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    p.add_argument("--decoder-lr", type=float, default=DECODER_LR)
    p.add_argument("--decoder-batch-size", type=int, default=DECODER_BATCH_SIZE)
    p.add_argument("--decoder-seed", type=int, default=0)
    p.add_argument(
        "--decoder-hidden",
        type=int,
        default=DECODER_HIDDEN,
        help="Hidden width of the GELU MLP decoder; 0 keeps it linear.",
    )
    p.add_argument(
        "--decoder-depth",
        type=int,
        default=DECODER_DEPTH,
        help="Number of hidden layers when --decoder-hidden > 0.",
    )
    p.add_argument("--decoder-max-fit-cells", type=int, default=0)
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument(
        "--pred-control",
        choices=("decoded", "observed"),
        default="decoded",
        help="Gene-space basal for predicted deltas; decoded uses D(z_ctrl).",
    )
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-max-cells-per-stratum", type=int, default=500)
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-celleval", action="store_true")
    p.add_argument(
        "--standardize-latent",
        action="store_true",
        help="Whiten each latent coordinate on non-test cells so x1 matches N(0,I).",
    )
    p.add_argument(
        "--radial-scale-latent",
        action="store_true",
        help="Multiply rows by sqrt(d) (keep angles). Mutually exclusive with whitening.",
    )
    p.add_argument("--save-h5ad", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def _strip_rows(block: dict[str, Any]) -> dict[str, Any]:
    out = dict(block)
    out.pop("rows", None)
    return out


def _celleval_mean_pearson_delta(result: dict[str, Any]) -> float:
    vals: list[float] = []
    for sl in result.get("slices", {}).values():
        for row in sl.get("perturbation_results", []):
            v = row.get("pearson_delta")
            if v is not None and np.isfinite(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = make_parser().parse_args()
    defaults = DEFAULTS[args.dataset]
    args.h5ad_path = args.h5ad_path or defaults["h5ad"]
    if args.se_path is None:
        args.se_path = (
            defaults["scimilarity"]
            if str(args.representation) == "scimilarity"
            else defaults["se"]
        )
    args.control_perturbation = defaults["control"]
    args.covariate_cols = list(defaults["covariates"])
    out_dir = args.out_dir or (defaults["bucket"] / args.run_name)
    args.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.smoke:
        args.target_steps = 0
        args.epochs = 1
        args.decoder_epochs = 2
        args.max_train_cells = min(args.max_train_cells or 4096, 4096)
        args.eval_max_cells_per_stratum = 32

    stream = stream_for(args.dataset)
    arrays = load_tx_cl_arrays(
        args.h5ad_path,
        args.se_path,
        stream,
        covariate_cols=tuple(args.covariate_cols),
        control_perturbation=args.control_perturbation,
    )
    if str(args.representation) == "scimilarity":
        if Path(args.se_path).name == "se.npy":
            raise ValueError(
                f"representation=scimilarity but se_path is STATE {args.se_path}; "
                "pass data_dir/replogle/scimilarity.npy"
            )
        if int(arrays.se_dim) != 128:
            raise ValueError(
                f"SCimilarity sidecar dim is {arrays.se_dim}, expected 128 "
                f"({args.se_path})"
            )
    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    train_idx = np.flatnonzero(split != TEST_SPLIT)
    hepg2_fit = np.flatnonzero((context == hepg2) & (split != TEST_SPLIT))
    hepg2_heldout = np.flatnonzero((context == hepg2) & (split == TEST_SPLIT))
    print(
        f"train cells (non-test, all lines)={train_idx.size:,}  "
        f"hepg2 fit={hepg2_fit.size:,}  hepg2 test={hepg2_heldout.size:,}  "
        f"representation={args.representation} se_dim={arrays.se_dim} "
        f"se_path={args.se_path}",
        flush=True,
    )
    if train_idx.size == 0:
        raise ValueError("No non-test cells to train on")

    if args.standardize_latent and args.radial_scale_latent:
        raise ValueError("Use only one of --standardize-latent and --radial-scale-latent")

    scale_stats: dict[str, float] | None = None
    if args.radial_scale_latent:
        scale_stats = apply_latent_radial_scale(arrays.se)
        print(
            f"radial-scaled latent by sqrt({arrays.se_dim})={scale_stats['radius']:.3f}  "
            f"row-norm {scale_stats['pre_row_norm_mean']:.3f} -> "
            f"{scale_stats['post_row_norm_mean']:.3f} "
            f"(prior ~{scale_stats['prior_norm_expected']:.3f})",
            flush=True,
        )
    elif args.standardize_latent:
        scale_path = out_dir / LATENT_STANDARDIZER_NAME
        if args.skip_train:
            if not scale_path.is_file():
                raise FileNotFoundError(
                    f"--standardize-latent --skip-train requires {scale_path}"
                )
            mean, std = load_latent_standardizer(scale_path)
        else:
            mean, std = fit_latent_standardizer(arrays.se, train_idx)
            save_latent_standardizer(scale_path, mean, std)
        scale_stats = apply_latent_standardizer(arrays.se, mean, std)
        print(
            f"standardized latent on {train_idx.size:,} non-test cells  "
            f"std median={scale_stats['std_median']:.5f}  "
            f"post row-norm={scale_stats['post_row_norm_mean']:.3f} "
            f"(prior ~{scale_stats['prior_norm_expected']:.3f})",
            flush=True,
        )

    n_loader = train_idx.size if args.max_train_cells <= 0 else min(
        train_idx.size, int(args.max_train_cells)
    )
    spe = steps_per_epoch(n_loader, args.batch_size)
    epochs = (
        args.epochs
        if args.smoke or int(args.target_steps) <= 0
        else epochs_for_target_steps(n_loader, args.batch_size, args.target_steps)
    )
    print(
        f"steps/epoch={spe}  epochs={epochs}  ~{spe * epochs:,} Adam steps",
        flush=True,
    )

    dump_json(
        out_dir / "config.json",
        {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "n_train_cells": int(train_idx.size),
            "steps_per_epoch": spe,
            "epochs_used": epochs,
            "approx_adam_steps": int(spe * epochs),
            "eval_context": hepg2,
            "latent_standardizer": (
                None
                if not args.standardize_latent
                else {"path": LATENT_STANDARDIZER_NAME, **scale_stats}
            ),
            "latent_radial_scale": None if not args.radial_scale_latent else scale_stats,
        },
    )

    ckpt_path = out_dir / "checkpoint.pt"
    decoder_dir = out_dir / "decoders"
    adata = ad.read_h5ad(args.h5ad_path, backed="r")

    if args.skip_train and ckpt_path.is_file() and (decoder_dir / "decoder_task00.npz").is_file():
        from experiments.tx.train_cl import model_from_checkpoint

        model, _ = model_from_checkpoint(ckpt_path, device)
        bank = TaskDecoderBank.load_dir(decoder_dir)
        print(f"Loaded {ckpt_path} and {decoder_dir}", flush=True)
    else:
        print("Fitting one decoder on all non-test cells", flush=True)
        decoder = fit_decoder_for_task(arrays, adata, 0, train_idx, args, device)
        bank = TaskDecoderBank()
        bank.add(decoder)
        bank.save_dir(decoder_dir)
        dump_json(
            decoder_dir / "manifest.json",
            {
                "format_version": 1,
                "dataset": args.dataset,
                "kind": "oneshot_all_nontest",
                "decoder_hidden": int(args.decoder_hidden),
                "decoder_depth": int(args.decoder_depth),
                "n_fit": int(decoder.n_fit),
                "n_genes": int(decoder.n_genes),
                "se_dim": int(decoder.se_dim),
            },
        )

        print(f"Training one-shot CFM on {args.representation}", flush=True)
        model = make_model(arrays, args).to(device)
        loader = make_loader(arrays, train_idx, args, seed=args.seed)
        history = train_flow_matching(
            model,
            loader,
            data_dim=arrays.se_dim,
            device=device,
            epochs=epochs,
            lr=args.lr,
            train_seed=args.seed,
        )
        dump_json(out_dir / "history.json", history)
        save_checkpoint(
            ckpt_path,
            model,
            {
                "experiment": "tx_se_oneshot_hepg2",
                "dataset": args.dataset,
                "data_dim": model.data_dim,
                "n_contexts": model.n_contexts,
                "n_perturbations": model.n_perturbations,
                "cond_emb_dim": model.cond_emb_dim,
                "time_emb_dim": model.time_emb_dim,
                "hidden_dim": model.hidden_dim,
                "depth": model.depth,
                "covariate_sizes": arrays.encoding.covariate_sizes(),
                "encoding": arrays.encoding.to_jsonable(),
            },
        )
        print(f"Wrote {ckpt_path}", flush=True)

    task_ids = np.zeros(arrays.n_cells, dtype=np.int16)
    pearson = {
        "fit": _strip_rows(
            eval_track(
                model,
                arrays,
                adata,
                bank,
                hepg2_fit,
                task_ids,
                args,
                device,
                seed=args.seed + 30_000,
            )
        ),
        "heldout": _strip_rows(
            eval_track(
                model,
                arrays,
                adata,
                bank,
                hepg2_heldout,
                task_ids,
                args,
                device,
                seed=args.seed + 30_001,
                control_reference=hepg2_fit,
            )
        ),
    }
    dump_json(out_dir / "hepg2_pearson.json", pearson)
    print(
        "HepG2 Pearson-Δ  "
        f"fit={pearson['fit'].get('pearson_delta_macro')}  "
        f"heldout={pearson['heldout'].get('pearson_delta_macro')}",
        flush=True,
    )

    if not args.skip_celleval:
        eval_ns = Namespace(**vars(args))
        celleval = {
            "profile": "minimal",
            "eval_context": hepg2,
            "fit": evaluate_indices(
                model,
                arrays,
                adata,
                bank,
                hepg2_fit,
                task_ids,
                eval_ns,
                device,
                seed=args.seed + 31_000,
                output_dir=out_dir / "cell_eval_minimal" / "hepg2_fit",
            ),
            "heldout": evaluate_indices(
                model,
                arrays,
                adata,
                bank,
                hepg2_heldout,
                task_ids,
                eval_ns,
                device,
                seed=args.seed + 31_001,
                output_dir=out_dir / "cell_eval_minimal" / "hepg2_heldout",
                control_reference=hepg2_fit,
            ),
        }
        dump_json(out_dir / "hepg2_celleval.json", celleval)
        print(
            "HepG2 CellEval mean Pearson-Δ  "
            f"fit={_celleval_mean_pearson_delta(celleval['fit']):.4f}  "
            f"heldout={_celleval_mean_pearson_delta(celleval['heldout']):.4f}",
            flush=True,
        )

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Done. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
