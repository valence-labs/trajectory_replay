#!/usr/bin/env python3
"""Sanity baselines for the SCimilarity/STATE reconstruction benchmark.

Answers "are the decode metrics wired correctly, or just bad?" by scoring, on
the same pools and the same metric code as ``bench_scimilarity_recon.py``:

* ``mean_train``   — predict the task-0 train gene mean for every cell (no
  embedding at all). Any decoder must beat this.
* ``mean_pool``    — predict the pool's own gene mean (oracle baseline; upper
  bound for anything that only gets the cell line right).
* ``pca256``       — the task-0 PCA sidecar inverse, a known-good decoder.
* ``sci_ridge`` / ``se_ridge`` — same fits as the benchmark, for reference.

Also reports ``r2_centered``: R^2 after removing the per-gene mean offset of
each pool, which separates "wrong baseline for this cell line" from "wrong
per-cell structure".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge

from trajectory_replay.data.pca_decode import pca_gene_decoder_from_basis
from trajectory_replay.data.tx_cl_data import load_gene_rows, load_se_rows
from trajectory_replay.data.tx_preprocess import load_pca_basis
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE


def pearson_rows(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=1, keepdims=True)
    r = recon - recon.mean(axis=1, keepdims=True)
    num = (t * r).sum(axis=1)
    den = np.sqrt((t * t).sum(axis=1) * (r * r).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def pearson_cols(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=0, keepdims=True)
    r = recon - recon.mean(axis=0, keepdims=True)
    num = (t * r).sum(axis=0)
    den = np.sqrt((t * t).sum(axis=0) * (r * r).sum(axis=0))
    return num / np.maximum(den, 1e-12)


def subsample(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if max_n <= 0 or idx.size <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(max_n), replace=False))


def score(true: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    err = true - recon
    ss_res = float((err * err).sum())
    ss_tot = float(((true - true.mean(axis=0, keepdims=True)) ** 2).sum())
    # centered: strip each pool's per-gene mean offset from both sides
    tc = true - true.mean(axis=0, keepdims=True)
    rc = recon - recon.mean(axis=0, keepdims=True)
    ss_res_c = float(((tc - rc) ** 2).sum())
    return {
        "n_cells": int(true.shape[0]),
        "mean_mse": float((err * err).mean()),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "r2_centered": float(1.0 - ss_res_c / max(ss_tot, 1e-12)),
        "mean_pearson_cell": float(pearson_rows(true, recon).mean()),
        "mean_pearson_gene": float(np.nanmean(pearson_cols(true, recon))),
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--sci-path", type=Path, default=REPO_ROOT / "data_dir/replogle/scimilarity.npy")
    p.add_argument("--se-path", type=Path, default=REPO_ROOT / "data_dir/replogle/se.npy")
    p.add_argument(
        "--pca-basis", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_basis.npz"
    )
    p.add_argument(
        "--pca-embedding", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_embedding.npy"
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/sci_recon_bench/baselines.json",
    )
    p.add_argument("--max-fit-cells", type=int, default=100_000)
    p.add_argument("--eval-cells", type=int, default=20_000)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = make_parser().parse_args()
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(adata.obs["split"]).astype(str)
    context = np.asarray(adata.obs["cell_line"]).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE

    train = np.flatnonzero(split == "train")
    fit_idx = subsample(train, args.max_fit_cells, args.seed)
    pools = {
        "train_heldout": subsample(
            np.setdiff1d(train, fit_idx), args.eval_cells, args.seed + 1
        ),
        "hepg2_fit": subsample(
            np.flatnonzero((context == hepg2) & (split != "test")), args.eval_cells, args.seed + 2
        ),
        "hepg2_test": subsample(
            np.flatnonzero((context == hepg2) & (split == "test")), args.eval_cells, args.seed + 3
        ),
    }
    print(
        f"fit={fit_idx.size:,}  " + "  ".join(f"{k}={v.size:,}" for k, v in pools.items()),
        flush=True,
    )

    x_fit = load_gene_rows(adata, fit_idx).astype(np.float64)
    train_mean = x_fit.mean(axis=0, keepdims=True)

    z_sci = np.load(args.sci_path, mmap_mode="r")
    z_se = np.load(args.se_path, mmap_mode="r")
    print("fitting ridge decoders", flush=True)
    sci_ridge = Ridge(alpha=args.ridge_alpha).fit(load_se_rows(z_sci, fit_idx), x_fit)
    se_ridge = Ridge(alpha=args.ridge_alpha).fit(load_se_rows(z_se, fit_idx), x_fit)

    pca_dec = pca_gene_decoder_from_basis(load_pca_basis(args.pca_basis))
    x_pca = np.load(args.pca_embedding, mmap_mode="r")
    print(f"pca basis {pca_dec.pcs.shape} embedding {x_pca.shape}", flush=True)

    models: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "mean_train": lambda idx: np.repeat(train_mean, idx.size, axis=0),
        "mean_pool": lambda idx: None,  # filled per pool below
        "pca256": lambda idx: pca_dec.decode(np.asarray(x_pca[idx], dtype=np.float64)),
        "sci_ridge": lambda idx: sci_ridge.predict(load_se_rows(z_sci, idx)),
        "se_ridge": lambda idx: se_ridge.predict(load_se_rows(z_se, idx)),
    }

    results: dict[str, Any] = {}
    for pool_name, idx in pools.items():
        true = load_gene_rows(adata, idx).astype(np.float64)
        pool_mean = true.mean(axis=0, keepdims=True)
        results[pool_name] = {}
        for model_name, fn in models.items():
            recon = (
                np.repeat(pool_mean, idx.size, axis=0)
                if model_name == "mean_pool"
                else np.asarray(fn(idx), dtype=np.float64)
            )
            block = score(true, recon)
            results[pool_name][model_name] = block
            print(
                f"  {pool_name:14s} {model_name:12s} "
                f"cellP={block['mean_pearson_cell']:.4f} "
                f"geneP={block['mean_pearson_gene']:.4f} "
                f"R2={block['r2']:+.4f} R2c={block['r2_centered']:+.4f} "
                f"mse={block['mean_mse']:.5f}",
                flush=True,
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "n_fit": int(fit_idx.size),
                "eval_cells": int(args.eval_cells),
                "ridge_alpha": float(args.ridge_alpha),
                "metrics": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
