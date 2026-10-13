#!/usr/bin/env python3
"""Raw vs standardized latent → gene reconstruction (no CFM).

Fits the same Adam linear decoder used by one-shot training, on a train
subsample, for both the original sidecar and a whitened copy. Scores
per-cell Pearson and R² on HepG2 fit / heldout. Unregularized linear maps
are affine-equivalent, so a large drop here would mean the standardizer
or decoder optimiser is numerically broken — not that whitening destroys
signal.
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

from experiments.tx.train_cl import fit_decoder_for_task
from trajectory_replay.data.tx_cl_data import (
    apply_latent_standardizer,
    fit_latent_standardizer,
    load_gene_rows,
    load_tx_cl_arrays,
)
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
)
from trajectory_replay.eval.tx_cl_pearson import pearson_vector


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--se-path", type=Path, required=True)
    p.add_argument("--representation", default="se")
    p.add_argument("--n-fit", type=int, default=80_000)
    p.add_argument("--n-eval", type=int, default=15_000)
    p.add_argument("--decoder-epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--decoder-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--decoder-lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", type=Path, default=None)
    return p


def score_recon(true: np.ndarray, recon: np.ndarray, n_cell: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = min(n_cell, true.shape[0])
    take = rng.choice(true.shape[0], n, replace=False) if n else np.empty(0, int)
    r = np.array([pearson_vector(true[i], recon[i]) for i in take])
    ss_res = float(((true - recon) ** 2).sum())
    ss_tot = float(((true - true.mean(0)) ** 2).sum())
    return {
        "n_cells": int(true.shape[0]),
        "per_cell_recon_pearson": float(np.nanmean(r)) if r.size else float("nan"),
        "decoder_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def eval_variant(
    arrays,
    adata,
    train_idx: np.ndarray,
    pools: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    *,
    label: str,
) -> dict[str, Any]:
    ns = Namespace(
        decoder_max_fit_cells=int(args.n_fit),
        decoder_seed=int(args.seed),
        representation=str(args.representation),
        decoder_hidden=0,
        decoder_depth=1,
        decoder_lr=float(args.decoder_lr),
        decoder_batch_size=int(args.decoder_batch_size),
        decoder_epochs=int(args.decoder_epochs),
    )
    print(f"\n== {label}  se row-norm={float(np.linalg.norm(arrays.se[train_idx[:2048]], axis=1).mean()):.3f}", flush=True)
    dec = fit_decoder_for_task(arrays, adata, 0, train_idx, ns, device)
    out: dict[str, Any] = {}
    for name, idx in pools.items():
        genes = load_gene_rows(adata, idx)
        recon = dec.decode(np.asarray(arrays.se[idx]))
        block = score_recon(genes, recon, min(2000, idx.size), args.seed)
        out[name] = block
        print(
            f"  {name:12s} reconP={block['per_cell_recon_pearson']:.4f} "
            f"R2={block['decoder_r2']:+.4f} n={block['n_cells']:,}",
            flush=True,
        )
    return out


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device(args.device)
    stream = stream_for("replogle")
    arrays = load_tx_cl_arrays(
        args.h5ad_path,
        args.se_path,
        stream,
        covariate_cols=(),
        control_perturbation="non-targeting",
    )
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    rng = np.random.default_rng(args.seed)
    train = np.flatnonzero(split != TEST_SPLIT)
    hepg2_fit = np.flatnonzero((context == REPLOGLE_FEWSHOT_CELL_LINE) & (split != TEST_SPLIT))
    hepg2_test = np.flatnonzero((context == REPLOGLE_FEWSHOT_CELL_LINE) & (split == TEST_SPLIT))
    fit_idx = np.sort(rng.choice(train, size=min(args.n_fit, train.size), replace=False))
    pools = {
        "hepg2_fit": np.sort(rng.choice(hepg2_fit, size=min(args.n_eval, hepg2_fit.size), replace=False)),
        "hepg2_test": np.sort(rng.choice(hepg2_test, size=min(args.n_eval, hepg2_test.size), replace=False)),
    }
    raw = eval_variant(arrays, adata, train, pools, args, device, label="raw")
    mean, std = fit_latent_standardizer(arrays.se, train)
    stats = apply_latent_standardizer(arrays.se, mean, std)
    print(
        f"whitened std_median={stats['std_median']:.5f} "
        f"row-norm={stats['post_row_norm_mean']:.3f} prior~{stats['prior_norm_expected']:.3f}",
        flush=True,
    )
    std_metrics = eval_variant(arrays, adata, train, pools, args, device, label="standardized")
    payload = {
        "se_path": str(args.se_path),
        "n_fit": int(fit_idx.size),
        "n_eval": int(args.n_eval),
        "decoder_epochs": int(args.decoder_epochs),
        "standardizer": stats,
        "raw": raw,
        "standardized": std_metrics,
    }
    out = args.out_json or (
        REPO_ROOT / "results_dir/tx_results/replogle" / f"diag_std_decoder_{args.representation}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
