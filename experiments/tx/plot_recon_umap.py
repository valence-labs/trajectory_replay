#!/usr/bin/env python3
"""UMAP overlays of true vs reconstructed HVG profiles.

Task-0 train leftover uses decoders fit on task-0 ``train`` (jurkat/k562/rpe1).
HepG2 test uses SCimilarity/STATE decoders fit on HepG2 ``finetune`` — the CL
``D_1`` protocol — not the task-0 map. PCA stays the task-0 inverse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad

from experiments.tx.bench_scimilarity_recon import fit_torch, parse_mlp_specs
from experiments.tx.diag_recon_baselines import subsample
from trajectory_replay.data.pca_decode import pca_gene_decoder_from_basis
from trajectory_replay.data.tx_cl_data import load_gene_rows, load_se_rows
from trajectory_replay.data.tx_preprocess import load_pca_basis
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE


def umap_true_recon(
    x_true: np.ndarray,
    x_recon: np.ndarray,
    *,
    seed: int,
    n_neighbors: int = 15,
) -> np.ndarray:
    X = np.concatenate([x_true, x_recon], axis=0).astype(np.float32, copy=False)
    adata = ad.AnnData(X=X)
    n_pcs = min(50, adata.n_vars - 1, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata, random_state=seed)
    return np.asarray(adata.obsm["X_umap"])


def scatter_overlay(ax, coords: np.ndarray, n_true: int, *, s: float = 6.0) -> None:
    ax.scatter(
        coords[:n_true, 0],
        coords[:n_true, 1],
        s=s,
        c="#4c78a8",
        alpha=0.45,
        linewidths=0,
        label="true",
        rasterized=True,
    )
    ax.scatter(
        coords[n_true:, 0],
        coords[n_true:, 1],
        s=s,
        c="#f58518",
        alpha=0.45,
        linewidths=0,
        label="recon",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def pearson_cell(true: np.ndarray, recon: np.ndarray) -> float:
    t = true - true.mean(axis=1, keepdims=True)
    r = recon - recon.mean(axis=1, keepdims=True)
    num = (t * r).sum(axis=1)
    den = np.sqrt((t * t).sum(axis=1) * (r * r).sum(axis=1))
    return float(np.mean(num / np.maximum(den, 1e-12)))


def r2_gene(true: np.ndarray, recon: np.ndarray) -> float:
    ss_res = float(((true - recon) ** 2).sum())
    ss_tot = float(((true - true.mean(axis=0, keepdims=True)) ** 2).sum())
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def fit_learned(
    *,
    z_sci,
    z_se,
    fit_idx: np.ndarray,
    x_fit: np.ndarray,
    args,
    device: torch.device,
    tag: str,
) -> dict:
    """Ridge / Adam linear / MLP maps from embeddings → HVGs on ``fit_idx``."""
    models: dict = {}
    z_sci_fit = load_se_rows(z_sci, fit_idx)
    print(f"[{tag}] fitting sci_ridge / se_ridge on {fit_idx.size:,} cells", flush=True)
    sci_ridge = Ridge(alpha=args.ridge_alpha).fit(z_sci_fit, x_fit)
    se_ridge = Ridge(alpha=args.ridge_alpha).fit(load_se_rows(z_se, fit_idx), x_fit)
    models["sci_ridge"] = lambda idx, _m=sci_ridge: _m.predict(load_se_rows(z_sci, idx))
    models["se_ridge"] = lambda idx, _m=se_ridge: _m.predict(load_se_rows(z_se, idx))

    print(f"[{tag}] fitting sci_linear_adam", flush=True)
    lin_dec = fit_torch(
        z_sci_fit,
        x_fit,
        hidden=0,
        epochs=args.epochs,
        batch_size=8192,
        lr=1e-3,
        seed=args.seed,
        device=device,
    )
    models["sci_linear_adam"] = lambda idx, _d=lin_dec: _d(load_se_rows(z_sci, idx))
    if not args.skip_mlp:
        for hidden, depth in parse_mlp_specs(args.mlp_specs):
            name = f"sci_mlp_h{hidden}_d{depth}"
            print(f"[{tag}] fitting {name}", flush=True)
            dec = fit_torch(
                z_sci_fit,
                x_fit,
                hidden=hidden,
                depth=depth,
                epochs=args.epochs,
                batch_size=8192,
                lr=1e-3,
                seed=args.seed,
                device=device,
            )
            models[name] = lambda idx, _d=dec: _d(load_se_rows(z_sci, idx))
    return models


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--sci-path", type=Path, default=REPO_ROOT / "data_dir/replogle/scimilarity.npy")
    p.add_argument("--se-path", type=Path, default=REPO_ROOT / "data_dir/replogle/se.npy")
    p.add_argument("--pca-basis", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_basis.npz")
    p.add_argument(
        "--pca-embedding", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_embedding.npy"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/sci_recon_bench/umap_cl",
    )
    p.add_argument("--max-fit-cells", type=int, default=100_000)
    p.add_argument("--umap-cells", type=int, default=4_000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--mlp-specs", default="256x1,512x1,1024x1,1024x2")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-mlp", action="store_true")
    return p


def main() -> None:
    args = make_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(adata.obs["split"]).astype(str)
    context = np.asarray(adata.obs["cell_line"]).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    train = np.flatnonzero(split == "train")
    hepg2_ft = np.flatnonzero((context == hepg2) & (split == "finetune"))
    fit_idx = subsample(train, args.max_fit_cells, args.seed)
    pools = {
        "train_heldout": subsample(
            np.setdiff1d(train, fit_idx), args.umap_cells, args.seed + 1
        ),
        "hepg2_test": subsample(
            np.flatnonzero((context == hepg2) & (split == "test")),
            args.umap_cells,
            args.seed + 3,
        ),
    }
    print(
        f"device={device} D0_fit={fit_idx.size:,} D_hepg2_ft={hepg2_ft.size:,}  "
        + "  ".join(f"{k}={v.size:,}" for k, v in pools.items()),
        flush=True,
    )

    x_fit = load_gene_rows(adata, fit_idx)
    train_mean = x_fit.mean(axis=0, keepdims=True).astype(np.float64)
    z_sci = np.load(args.sci_path, mmap_mode="r")
    z_se = np.load(args.se_path, mmap_mode="r")

    pca_dec = pca_gene_decoder_from_basis(load_pca_basis(args.pca_basis))
    x_pca = np.load(args.pca_embedding, mmap_mode="r")
    shared = {
        "mean_train": lambda idx: np.repeat(train_mean, idx.size, axis=0),
        "pca256": lambda idx: pca_dec.decode(np.asarray(x_pca[idx], dtype=np.float64)),
    }
    d0 = fit_learned(
        z_sci=z_sci,
        z_se=z_se,
        fit_idx=fit_idx,
        x_fit=x_fit,
        args=args,
        device=device,
        tag="D0 task0-train",
    )
    d_ft = fit_learned(
        z_sci=z_sci,
        z_se=z_se,
        fit_idx=hepg2_ft,
        x_fit=load_gene_rows(adata, hepg2_ft),
        args=args,
        device=device,
        tag="D1 HepG2-finetune",
    )
    method_names = list(shared) + list(d0)
    per_pool = {
        "train_heldout": {**shared, **d0},
        "hepg2_test": {**shared, **d_ft},
    }
    n_row, n_col = len(method_names), len(pools)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(4.2 * n_col, 3.4 * n_row),
        squeeze=False,
    )
    pool_names = list(pools)

    for j, pool_name in enumerate(pool_names):
        idx = pools[pool_name]
        x_true = load_gene_rows(adata, idx).astype(np.float64)
        models = per_pool[pool_name]
        for i, method in enumerate(method_names):
            print(f"UMAP {method} / {pool_name}", flush=True)
            x_recon = np.asarray(models[method](idx), dtype=np.float64)
            print(
                f"  cellP={pearson_cell(x_true, x_recon):.4f}  "
                f"R2={r2_gene(x_true, x_recon):+.4f}",
                flush=True,
            )
            coords = umap_true_recon(x_true, x_recon, seed=args.seed + 100 * i + j)
            ax = axes[i, j]
            scatter_overlay(ax, coords, x_true.shape[0])
            if i == 0:
                title = pool_name.replace("_", " ")
                if pool_name == "hepg2_test":
                    title += " (sci/SE: D on FT)"
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(method.replace("_", "\n"), fontsize=8)
            png = args.out_dir / f"{method}_{pool_name}.png"
            fig_one, ax_one = plt.subplots(figsize=(5.2, 4.4))
            scatter_overlay(ax_one, coords, x_true.shape[0], s=8)
            ax_one.set_title(f"{method} · {pool_name}")
            ax_one.legend(frameon=False, markerscale=2)
            fig_one.tight_layout()
            fig_one.savefig(png, dpi=140)
            plt.close(fig_one)
            print(f"  wrote {png}", flush=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, markerscale=2)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    grid_path = args.out_dir / "true_vs_recon_grid.png"
    fig.savefig(grid_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {grid_path}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
