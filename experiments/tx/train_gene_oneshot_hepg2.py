#!/usr/bin/env python3
"""One-shot CFM on Replogle gene expression, then HepG2-only Pearson-Δ / CellEval.

Same protocol as ``train_se_oneshot_hepg2.py`` except the vector field lives in
h5ad ``X`` (2000 HVGs). There is no SE sidecar and no learned decoder: ODE
samples are genes, and predicted control basal is the observed control genes
(identity reconstruction).
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
    make_loader,
    make_model,
    save_checkpoint,
    steps_per_epoch,
)
from trajectory_replay.data.tx_cl_data import load_tx_cl_gene_arrays
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank, identity_task_decoder
from trajectory_replay.training.flow_matching import train_flow_matching


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("replogle",), default="replogle")
    p.add_argument("--h5ad-path", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--run-name", default="oneshot_gene_hepg2")
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
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-max-cells-per-stratum", type=int, default=500)
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-celleval", action="store_true")
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
    args.control_perturbation = defaults["control"]
    args.covariate_cols = list(defaults["covariates"])
    out_dir = args.out_dir or (defaults["bucket"] / args.run_name)
    args.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.smoke:
        args.target_steps = 0
        args.epochs = 1
        args.max_train_cells = min(args.max_train_cells or 4096, 4096)
        args.eval_max_cells_per_stratum = 32

    stream = stream_for(args.dataset)
    arrays = load_tx_cl_gene_arrays(
        args.h5ad_path,
        stream,
        covariate_cols=tuple(args.covariate_cols),
        control_perturbation=args.control_perturbation,
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
        f"n_genes={arrays.se_dim}",
        flush=True,
    )
    if train_idx.size == 0:
        raise ValueError("No non-test cells to train on")

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
            "representation": "genes",
            "n_genes": int(arrays.se_dim),
            "n_train_cells": int(train_idx.size),
            "steps_per_epoch": spe,
            "epochs_used": epochs,
            "approx_adam_steps": int(spe * epochs),
            "eval_context": hepg2,
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
        print("Using identity decoder (CFM already in gene space)", flush=True)
        decoder = identity_task_decoder(
            task_id=0, n_genes=arrays.se_dim, n_fit=int(train_idx.size)
        )
        bank = TaskDecoderBank()
        bank.add(decoder)
        bank.save_dir(decoder_dir)
        dump_json(
            decoder_dir / "manifest.json",
            {
                "format_version": 1,
                "dataset": args.dataset,
                "kind": "oneshot_gene_identity",
                "n_fit": int(decoder.n_fit),
                "n_genes": int(decoder.n_genes),
                "se_dim": int(decoder.se_dim),
            },
        )

        print("Training one-shot CFM on genes", flush=True)
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
                "experiment": "tx_gene_oneshot_hepg2",
                "dataset": args.dataset,
                "representation": "genes",
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
            "representation": "genes",
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
