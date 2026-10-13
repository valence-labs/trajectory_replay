#!/usr/bin/env python3
"""Is the tied inverse-PCA readout mis-scaled on a new cell line?

Task-0 PCA encoder throughout; only the gene-space readout changes. All maps are
calibrated on HepG2 non-test and scored on HepG2 STATE-test:

* ``tied``        — inverse PCA ``z P_0 + mu_0`` (0 new parameters)
* ``tied_scalar`` — ``a * tied + b``, one scalar pair (2 parameters)
* ``tied_affine`` — per-gene ``a_g * tied + b_g`` (2 x n_genes parameters)
* ``d1_linear``   — full linear ``256 -> genes`` (CL ``D_1``)

Per-gene squared error splits exactly into mean offset, standard-deviation
mismatch, and residual shape:

    MSE_g = (mu_t - mu_r)^2 + (sd_t - sd_r)^2 + 2 sd_t sd_r (1 - rho_g)

If the first two terms dominate, the tied decoder is mis-calibrated rather than
short of capacity, and the cheap calibrations recover most of ``d1_linear``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad

from experiments.tx.bench_scimilarity_recon import fit_torch
from experiments.tx.diag_recon_baselines import subsample
from experiments.tx.plot_recon_umap import (
    pearson_cell,
    r2_gene,
    scatter_overlay,
    umap_true_recon,
)
from trajectory_replay.data.pca_decode import pca_gene_decoder_from_basis
from trajectory_replay.data.tx_cl_data import load_gene_rows
from trajectory_replay.data.tx_preprocess import load_pca_basis
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE

DecodeFn = Callable[[np.ndarray], np.ndarray]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--pca-basis", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_basis.npz")
    p.add_argument(
        "--pca-embedding", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_embedding.npy"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/sci_recon_bench/umap_cl",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/diag_pca_decoder_scale.json",
    )
    p.add_argument("--umap-cells", type=int, default=4_000)
    p.add_argument(
        "--metric-cells",
        type=int,
        default=20_000,
        help="Cells for the numeric table (0 = all HepG2 test).",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--umap-seed",
        type=int,
        default=101,
        help="101 reproduces the pca256/hepg2_test panel of plot_recon_umap.py.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-umap", action="store_true")
    return p


def error_decomposition(true: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    """Split total MSE into mean offset, sd mismatch, and residual shape."""
    mu_t, mu_r = true.mean(axis=0), recon.mean(axis=0)
    sd_t, sd_r = true.std(axis=0), recon.std(axis=0)
    cov = ((true - mu_t) * (recon - mu_r)).mean(axis=0)
    bias = float(((mu_t - mu_r) ** 2).sum())
    scale = float(((sd_t - sd_r) ** 2).sum())
    shape = float((2.0 * (sd_t * sd_r - cov)).sum())
    total = float(((true - recon) ** 2).mean(axis=0).sum())
    denom = max(total, 1e-12)
    return {
        "mse_total": total,
        "mse_mean_offset": bias,
        "mse_sd_mismatch": scale,
        "mse_shape": shape,
        "frac_mean_offset": bias / denom,
        "frac_sd_mismatch": scale / denom,
        "frac_shape": shape / denom,
    }


def scale_stats(true: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    norm_t = float(np.linalg.norm(true, axis=1).mean())
    norm_r = float(np.linalg.norm(recon, axis=1).mean())
    sd_t = float(true.std(axis=0).mean())
    sd_r = float(recon.std(axis=0).mean())
    alpha = float((true * recon).sum() / max((recon * recon).sum(), 1e-12))
    return {
        "row_norm_true": norm_t,
        "row_norm_recon": norm_r,
        "row_norm_ratio": norm_r / max(norm_t, 1e-12),
        "gene_sd_true": sd_t,
        "gene_sd_recon": sd_r,
        "gene_sd_ratio": sd_r / max(sd_t, 1e-12),
        "mean_shift_l2": float(np.linalg.norm(true.mean(axis=0) - recon.mean(axis=0))),
        "best_global_scalar": alpha,
    }


def fit_scalar(base_fit: np.ndarray, x_fit: np.ndarray) -> tuple[float, float]:
    """Least-squares ``a, b`` for ``a * base + b`` pooled over all genes/cells."""
    b_mean, x_mean = float(base_fit.mean()), float(x_fit.mean())
    bc = base_fit - b_mean
    a = float((bc * (x_fit - x_mean)).sum() / max((bc * bc).sum(), 1e-12))
    return a, x_mean - a * b_mean


def fit_affine(base_fit: np.ndarray, x_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene least-squares slope/intercept for ``a_g * base + b_g``."""
    b_mean = base_fit.mean(axis=0)
    x_mean = x_fit.mean(axis=0)
    bc = base_fit - b_mean
    var = (bc * bc).mean(axis=0)
    cov = (bc * (x_fit - x_mean)).mean(axis=0)
    a = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), 0.0)
    return a, x_mean - a * b_mean


def main() -> None:
    args = make_parser().parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(adata.obs["split"]).astype(str)
    context = np.asarray(adata.obs["cell_line"]).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    fit_idx = np.flatnonzero((context == hepg2) & (split != "test"))
    test_all = np.flatnonzero((context == hepg2) & (split == "test"))
    umap_idx = subsample(test_all, args.umap_cells, args.seed + 3)
    metric_idx = subsample(test_all, args.metric_cells, args.seed + 7)
    print(
        f"device={device}  hepg2 calib={fit_idx.size:,}  hepg2 test={test_all.size:,}  "
        f"metric={metric_idx.size:,}  umap={umap_idx.size:,}",
        flush=True,
    )

    pca_dec = pca_gene_decoder_from_basis(load_pca_basis(args.pca_basis))
    z_pca = np.load(args.pca_embedding, mmap_mode="r")

    def tied(idx: np.ndarray) -> np.ndarray:
        return pca_dec.decode(np.asarray(z_pca[idx], dtype=np.float64))

    x_fit = load_gene_rows(adata, fit_idx).astype(np.float64)
    base_fit = tied(fit_idx)

    a_s, b_s = fit_scalar(base_fit, x_fit)
    a_g, b_g = fit_affine(base_fit, x_fit)
    print(
        f"global calibration a={a_s:.4f} b={b_s:+.4f}   per-gene slope "
        f"mean={a_g.mean():.4f} p05={np.quantile(a_g, 0.05):.4f} "
        f"p95={np.quantile(a_g, 0.95):.4f}",
        flush=True,
    )

    print(f"fitting d1_linear (256 -> {x_fit.shape[1]}) on {fit_idx.size:,} cells", flush=True)
    d1 = fit_torch(
        np.asarray(z_pca[fit_idx], dtype=np.float32),
        x_fit.astype(np.float32),
        hidden=0,
        epochs=args.epochs,
        batch_size=8192,
        lr=1e-3,
        seed=args.seed,
        device=device,
    )

    decoders: dict[str, DecodeFn] = {
        "tied": tied,
        "tied_scalar": lambda idx: a_s * tied(idx) + b_s,
        "tied_affine": lambda idx: a_g * tied(idx) + b_g,
        "d1_linear": lambda idx: np.asarray(d1(np.asarray(z_pca[idx], dtype=np.float32)), np.float64),
    }
    n_params = {
        "tied": 0,
        "tied_scalar": 2,
        "tied_affine": 2 * int(x_fit.shape[1]),
        "d1_linear": int(z_pca.shape[1] + 1) * int(x_fit.shape[1]),
    }

    x_metric = load_gene_rows(adata, metric_idx).astype(np.float64)
    rows: list[dict[str, Any]] = []
    print(
        f"\n{'readout':<12} {'params':>9} {'cellP':>8} {'geneR2':>8} "
        f"{'sdRatio':>8} {'%offset':>8} {'%sd':>7} {'%shape':>7}",
        flush=True,
    )
    for name, fn in decoders.items():
        recon = np.asarray(fn(metric_idx), dtype=np.float64)
        row = {
            "readout": name,
            "n_new_params": n_params[name],
            "cell_pearson": pearson_cell(x_metric, recon),
            "gene_r2": r2_gene(x_metric, recon),
            **scale_stats(x_metric, recon),
            **error_decomposition(x_metric, recon),
        }
        rows.append(row)
        print(
            f"{name:<12} {row['n_new_params']:>9,} {row['cell_pearson']:>8.4f} "
            f"{row['gene_r2']:>8.4f} {row['gene_sd_ratio']:>8.4f} "
            f"{row['frac_mean_offset']:>8.3f} {row['frac_sd_mismatch']:>7.3f} "
            f"{row['frac_shape']:>7.3f}",
            flush=True,
        )

    if not args.skip_umap:
        x_umap = load_gene_rows(adata, umap_idx).astype(np.float64)
        for name, fn in decoders.items():
            recon = np.asarray(fn(umap_idx), dtype=np.float64)
            coords = umap_true_recon(x_umap, recon, seed=args.umap_seed)
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            scatter_overlay(ax, coords, x_umap.shape[0], s=8)
            ax.set_title(f"pca256_{name} · hepg2_test")
            ax.legend(frameon=False, markerscale=2)
            fig.tight_layout()
            png = args.out_dir / f"pca256_{name}_hepg2_test.png"
            fig.savefig(png, dpi=140)
            plt.close(fig)
            print(f"wrote {png}", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "h5ad_path": str(args.h5ad_path),
                "pca_basis": str(args.pca_basis),
                "pca_embedding": str(args.pca_embedding),
                "n_calibration_cells": int(fit_idx.size),
                "n_metric_cells": int(metric_idx.size),
                "global_scalar": {"a": a_s, "b": b_s},
                "readouts": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {args.out_json}", flush=True)
    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
