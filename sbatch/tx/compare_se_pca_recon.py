#!/usr/bin/env python3
"""Compare PCA vs SE gene-space reconstruction (streamed; UMAP on a subset).

Works for Replogle-scale and PBMC-scale (~10M cells): metrics are computed over
**all** cells via batched reads; UMAPs only materialize ``--umap-max-cells``.

Pipeline
--------
1. Open expression AnnData (backed) and SE embeddings (``.h5ad`` with
   ``obsm[X_state]`` or a raw ``.npy`` matrix, same cell order).
2. Fit PCA on a subsample of fit-splits (``sc.pp.pca``); project/decode in
   batches over every eval cell.
3. Fit Ridge ``X_state -> genes`` on a fit subsample; decode in batches.
4. Report MSE / R^2 / per-cell Pearson by split (full split, streamed).
5. Dump UMAPs on a small subset only.

Example
-------
  python sbatch/tx/compare_se_pca_recon.py \\
    --expr-h5ad data_dir/pbmc_parse_with_splits.h5ad \\
    --se-h5ad results_dir/arc_state/se/pbmc_parse_with_splits_se.npy \\
    --split-col split3 \\
    --n-pcs 1000 \\
    --out-dir results_dir/arc_state/pbmc_se_vs_pca_recon_pcs1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge

from trajectory_replay.data.pca_decode import PcaGeneDecoder


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def pearson_per_cell(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=1, keepdims=True)
    r = recon - recon.mean(axis=1, keepdims=True)
    num = (t * r).sum(axis=1)
    den = np.sqrt((t * t).sum(axis=1) * (r * r).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def recon_metrics_from_running(
    *,
    n_cells: int,
    n_genes: int,
    sum_sq_err: float,
    gene_sum: np.ndarray,
    gene_sumsq: np.ndarray,
    cell_mse: np.ndarray,
    cell_pearson: np.ndarray,
) -> dict[str, float]:
    """Finalize metrics after a streamed pass over a split."""
    mean_mse = float(sum_sq_err / max(n_cells * n_genes, 1))
    # SS_tot = sum_i,j (x_ij - mean_j)^2 = sum_j (sumsq_j - n * mean_j^2)
    gene_mean = gene_sum / max(n_cells, 1)
    ss_tot = float(np.sum(gene_sumsq - n_cells * (gene_mean**2)))
    r2 = float(1.0 - sum_sq_err / max(ss_tot, 1e-12))
    return {
        "mean_mse": mean_mse,
        "median_mse": float(np.median(cell_mse)),
        "r2": r2,
        "mean_pearson": float(cell_pearson.mean()),
        "median_pearson": float(np.median(cell_pearson)),
    }


def subsample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def iter_index_batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def load_expr_batch(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    """Dense float64 gene matrix for absolute integer indices (sorted preferred)."""
    order = np.argsort(idx)
    sorted_idx = idx[order]
    x = _to_dense(adata.X[sorted_idx])
    # restore caller order
    out = np.empty_like(x)
    out[order] = x
    return out


def load_se_batch(z_se, idx: np.ndarray) -> np.ndarray:
    order = np.argsort(idx)
    sorted_idx = idx[order]
    z = np.asarray(z_se[sorted_idx], dtype=np.float32)
    out = np.empty_like(z)
    out[order] = z
    return out


def open_se_matrix(path: Path, *, se_key: str, n_obs: int, obs_names_head: list[str]):
    """Return (z_handle, se_dim, closer). z_handle supports integer fancy indexing."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        z = np.load(path, mmap_mode="r")
        if z.shape[0] != n_obs:
            raise ValueError(f"SE npy n_cells={z.shape[0]} != expr n_cells={n_obs}")
        print(f"  SE npy mmap shape={z.shape} dtype={z.dtype}", flush=True)
        return z, int(z.shape[1]), (lambda: None)

    se = ad.read_h5ad(path, backed="r")
    if se_key not in se.obsm:
        se.file.close()
        raise KeyError(f"obsm[{se_key!r}] missing in {path}")
    if se.n_obs != n_obs:
        se.file.close()
        raise ValueError(f"Cell count mismatch: expr={n_obs} vs se={se.n_obs}")
    if list(se.obs_names[:5]) != obs_names_head:
        se.file.close()
        raise ValueError(
            "obs_names diverge between expr and SE h5ads (first 5 differ); "
            "refusing to align by position."
        )
    z = se.obsm[se_key]
    print(f"  SE h5ad backed {se_key} shape={z.shape}", flush=True)

    def _close() -> None:
        if hasattr(se, "file") and se.file is not None:
            se.file.close()

    return z, int(z.shape[1]), _close


def fit_se_linear_decoder(
    z_fit: np.ndarray,
    x_fit: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> Ridge:
    print(
        f"Fitting Ridge(alpha={alpha}) on {z_fit.shape[0]:,} cells "
        f"({z_fit.shape[1]}-d -> {x_fit.shape[1]} genes)",
        flush=True,
    )
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
    t0 = time.time()
    model.fit(z_fit, x_fit)
    print(f"  ridge fit done in {time.time() - t0:.1f}s", flush=True)
    return model


def make_true_recon_umap(
    x_true: np.ndarray,
    x_recon: np.ndarray,
    *,
    title: str,
    out_png: Path,
    seed: int,
) -> None:
    X = np.concatenate([x_true, x_recon], axis=0)
    obs = pd.DataFrame(
        {"dataset_type": (["true"] * len(x_true)) + (["recon"] * len(x_recon))}
    )
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs)
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata, random_state=seed)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc.pl.umap(adata, color="dataset_type", ax=ax, show=False, s=8, frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


def make_latent_umap(
    z: np.ndarray,
    labels: pd.Series,
    *,
    title: str,
    color_key: str,
    out_png: Path,
    seed: int,
) -> None:
    obs = pd.DataFrame({color_key: labels.astype(str).to_numpy()})
    adata = ad.AnnData(X=z.astype(np.float32), obs=obs)
    if adata.n_vars > 50:
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    else:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
    sc.tl.umap(adata, random_state=seed)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc.pl.umap(adata, color=color_key, ax=ax, show=False, s=8, frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


def pick_color_key(obs: pd.DataFrame, split_col: str) -> str:
    for key in ("cell_line", "cell_type", "donor", split_col):
        if key in obs.columns:
            return key
    return split_col


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--expr-h5ad",
        type=Path,
        default=Path("data_dir/replogle_with_splits.h5ad"),
        help="Expression AnnData with splits",
    )
    p.add_argument(
        "--se-h5ad",
        type=Path,
        default=Path("results_dir/arc_state/se/replogle_with_splits_se.h5ad"),
        help="SE embeddings: .h5ad with obsm[X_state], or .npy (n_cells, se_dim)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results_dir/arc_state/se_vs_pca_recon"),
    )
    p.add_argument("--split-col", default="split")
    p.add_argument("--fit-splits", nargs="+", default=["train", "finetune"])
    p.add_argument("--eval-splits", nargs="+", default=["train", "finetune", "val", "test"])
    p.add_argument("--se-key", default="X_state")
    p.add_argument(
        "--n-pcs",
        type=int,
        default=1000,
        help="Number of PCs to fit with sc.pp.pca on fit subsample",
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument(
        "--max-ridge-fit",
        type=int,
        default=100_000,
        help="Max fit-split cells for Ridge probe",
    )
    p.add_argument(
        "--max-pca-fit",
        type=int,
        default=100_000,
        help="Max fit-split cells for sc.pp.pca (memory cap)",
    )
    p.add_argument("--decode-batch-size", type=int, default=8192)
    p.add_argument(
        "--umap-max-cells",
        type=int,
        default=20_000,
        help="Cells used for UMAPs only (metrics still use full splits)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def stream_recon_metrics(
    *,
    adata: ad.AnnData,
    z_se,
    indices: np.ndarray,
    pca_decoder: PcaGeneDecoder,
    pcs_varm: np.ndarray,
    mean: np.ndarray,
    se_decoder: Ridge,
    batch_size: int,
) -> dict[str, float]:
    """Compute recon metrics for PCA and SE over all ``indices`` (absolute)."""
    n_genes = adata.n_vars
    n = len(indices)
    # Accumulate separately for pca / se
    stats = {
        "pca": {
            "sse": 0.0,
            "gene_sum": np.zeros(n_genes, dtype=np.float64),
            "gene_sumsq": np.zeros(n_genes, dtype=np.float64),
            "cell_mse": np.empty(n, dtype=np.float64),
            "cell_pearson": np.empty(n, dtype=np.float64),
        },
        "se": {
            "sse": 0.0,
            "gene_sum": np.zeros(n_genes, dtype=np.float64),
            "gene_sumsq": np.zeros(n_genes, dtype=np.float64),
            "cell_mse": np.empty(n, dtype=np.float64),
            "cell_pearson": np.empty(n, dtype=np.float64),
        },
    }
    # gene_sum/sumsq are identical for both (from true X); fill once via pca path
    filled_gene = False
    cursor = 0
    for batch_idx in iter_index_batches(indices, batch_size):
        x_true = load_expr_batch(adata, batch_idx)
        z_pca = ((x_true - mean) @ pcs_varm).astype(np.float64)
        x_pca = pca_decoder.decode(z_pca)
        z_batch = load_se_batch(z_se, batch_idx)
        x_se = se_decoder.predict(z_batch)

        bs = len(batch_idx)
        sl = slice(cursor, cursor + bs)

        if not filled_gene:
            # will accumulate over all batches
            filled_gene = True
        for name, x_hat in (("pca", x_pca), ("se", x_se)):
            resid = x_true - x_hat
            sq = resid**2
            stats[name]["sse"] += float(sq.sum())
            stats[name]["cell_mse"][sl] = sq.mean(axis=1)
            stats[name]["cell_pearson"][sl] = pearson_per_cell(x_true, x_hat)
        # gene moments from true (once per batch)
        stats["pca"]["gene_sum"] += x_true.sum(axis=0)
        stats["pca"]["gene_sumsq"] += np.square(x_true).sum(axis=0)

        cursor += bs
        if cursor % (batch_size * 20) < batch_size or cursor == n:
            print(f"    streamed {cursor:,} / {n:,}", flush=True)

    # share gene moments with se
    stats["se"]["gene_sum"] = stats["pca"]["gene_sum"]
    stats["se"]["gene_sumsq"] = stats["pca"]["gene_sumsq"]

    out = {}
    for name in ("pca", "se"):
        out[name] = recon_metrics_from_running(
            n_cells=n,
            n_genes=n_genes,
            sum_sq_err=stats[name]["sse"],
            gene_sum=stats[name]["gene_sum"],
            gene_sumsq=stats[name]["gene_sumsq"],
            cell_mse=stats[name]["cell_mse"],
            cell_pearson=stats[name]["cell_pearson"],
        )
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.se_h5ad.is_file():
        raise FileNotFoundError(
            f"SE embeddings missing: {args.se_h5ad}\n"
            "Wait for encode job to finish, or pass --se-h5ad."
        )

    print(f"Loading expression (backed): {args.expr_h5ad}", flush=True)
    adata = ad.read_h5ad(args.expr_h5ad, backed="r")
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes", flush=True)

    print(f"Opening SE embeddings: {args.se_h5ad}", flush=True)
    z_se, se_dim, close_se = open_se_matrix(
        args.se_h5ad,
        se_key=args.se_key,
        n_obs=adata.n_obs,
        obs_names_head=list(adata.obs_names[:5]),
    )

    try:
        split = adata.obs[args.split_col].astype(str)
        fit_mask = split.isin(args.fit_splits).to_numpy()
        fit_idx = np.flatnonzero(fit_mask)
        print(
            "split counts:\n",
            split.value_counts().to_string(),
            f"\nfit cells ({'+'.join(args.fit_splits)}): {len(fit_idx):,}",
            flush=True,
        )

        # --- PCA fit on subsample ---
        pca_fit_idx = fit_idx[subsample_idx(len(fit_idx), args.max_pca_fit, args.seed)]
        n_pcs = min(int(args.n_pcs), adata.n_vars - 1, len(pca_fit_idx) - 1)
        if n_pcs != args.n_pcs:
            print(f"Capping n_pcs {args.n_pcs} -> {n_pcs}", flush=True)
        print(
            f"Fitting sc.pp.pca(n_comps={n_pcs}) on {len(pca_fit_idx):,} fit cells...",
            flush=True,
        )
        t_pca = time.time()
        # Materialize only the PCA-fit subset into memory
        adata_fit = adata[pca_fit_idx].to_memory()
        sc.pp.pca(adata_fit, n_comps=n_pcs)
        pcs_varm = np.asarray(adata_fit.varm["PCs"], dtype=np.float64)
        mean = _to_dense(adata_fit.X).mean(axis=0, keepdims=True)
        pca_decoder = PcaGeneDecoder(
            pcs=pcs_varm.T.copy(),
            mean=mean,
            n_components=n_pcs,
        )
        print(
            f"PCA ready: n_comps={n_pcs}, genes={pcs_varm.shape[0]} "
            f"in {time.time() - t_pca:.1f}s",
            flush=True,
        )
        del adata_fit

        # --- SE Ridge on subsample ---
        ridge_fit_idx = fit_idx[
            subsample_idx(len(fit_idx), args.max_ridge_fit, args.seed + 11)
        ]
        print(f"Loading Ridge fit matrices ({len(ridge_fit_idx):,} cells)...", flush=True)
        x_ridge = load_expr_batch(adata, ridge_fit_idx)
        z_ridge = load_se_batch(z_se, ridge_fit_idx)
        se_decoder = fit_se_linear_decoder(
            z_ridge,
            x_ridge,
            alpha=args.ridge_alpha,
            seed=args.seed,
        )
        del x_ridge, z_ridge

        rows: list[dict] = []
        umap_payload: dict[str, np.ndarray] | None = None
        umap_split = None
        preferred_umap = "test" if "test" in args.eval_splits else args.eval_splits[0]

        for sp in args.eval_splits:
            mask = (split == sp).to_numpy()
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                print(f"skip empty split {sp}", flush=True)
                continue
            print(f"Streaming metrics split={sp} (n={idx.size:,})...", flush=True)
            t_sp = time.time()
            metrics = stream_recon_metrics(
                adata=adata,
                z_se=z_se,
                indices=idx,
                pca_decoder=pca_decoder,
                pcs_varm=pcs_varm,
                mean=mean,
                se_decoder=se_decoder,
                batch_size=args.decode_batch_size,
            )
            m_pca, m_se = metrics["pca"], metrics["se"]
            rows.append({"method": "pca", "split": sp, "n_cells": int(idx.size), **m_pca})
            rows.append(
                {"method": "se_ridge", "split": sp, "n_cells": int(idx.size), **m_se}
            )
            print(
                f"  PCA:      MSE={m_pca['mean_mse']:.4f}  R2={m_pca['r2']:.4f}  "
                f"Pearson={m_pca['mean_pearson']:.4f}  ({time.time() - t_sp:.1f}s)",
                flush=True,
            )
            print(
                f"  SE-ridge: MSE={m_se['mean_mse']:.4f}  R2={m_se['r2']:.4f}  "
                f"Pearson={m_se['mean_pearson']:.4f}",
                flush=True,
            )

            if sp == preferred_umap or umap_payload is None:
                umap_split = sp
                u_idx = idx[subsample_idx(len(idx), args.umap_max_cells, args.seed + 7)]
                print(
                    f"  caching UMAP subset for split={sp} (n={len(u_idx):,})",
                    flush=True,
                )
                x_true = load_expr_batch(adata, u_idx)
                z_pca_u = ((x_true - mean) @ pcs_varm).astype(np.float64)
                x_pca = pca_decoder.decode(z_pca_u)
                z_u = load_se_batch(z_se, u_idx)
                x_se = se_decoder.predict(z_u)
                umap_payload = {
                    "true": x_true,
                    "pca": x_pca,
                    "se": np.asarray(x_se, dtype=np.float64),
                    "z_pca": z_pca_u.astype(np.float32),
                    "z_se": z_u,
                    "obs_idx": u_idx,
                }

        metrics_df = pd.DataFrame(rows)
        metrics_path = args.out_dir / "recon_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote {metrics_path}", flush=True)

        assert umap_payload is not None and umap_split is not None
        print(f"Building gene-space UMAPs on split={umap_split} subset...", flush=True)
        make_true_recon_umap(
            umap_payload["true"],
            umap_payload["pca"],
            title=f"Gene space: true vs PCA recon ({umap_split})",
            out_png=args.out_dir / f"umap_gene_true_vs_pca_{umap_split}.png",
            seed=args.seed,
        )
        make_true_recon_umap(
            umap_payload["true"],
            umap_payload["se"],
            title=f"Gene space: true vs SE-ridge recon ({umap_split})",
            out_png=args.out_dir / f"umap_gene_true_vs_se_{umap_split}.png",
            seed=args.seed + 1,
        )

        color_key = pick_color_key(adata.obs, args.split_col)
        # Latent UMAPs: global subsample (not just one split)
        lat_idx = subsample_idx(adata.n_obs, args.umap_max_cells, args.seed + 2)
        print(
            f"Building latent UMAPs on {len(lat_idx):,} cells by {color_key}...",
            flush=True,
        )
        x_lat = load_expr_batch(adata, lat_idx)
        z_pca_lat = ((x_lat - mean) @ pcs_varm).astype(np.float32)
        z_se_lat = load_se_batch(z_se, lat_idx)
        labels = adata.obs[color_key].iloc[lat_idx]
        make_latent_umap(
            z_pca_lat,
            labels,
            title=f"PCA latent ({n_pcs}-d) by {color_key}",
            color_key=color_key,
            out_png=args.out_dir / f"umap_latent_pca_by_{color_key}.png",
            seed=args.seed + 2,
        )
        make_latent_umap(
            z_se_lat,
            labels,
            title=f"SE latent ({se_dim}-d) by {color_key}",
            color_key=color_key,
            out_png=args.out_dir / f"umap_latent_se_by_{color_key}.png",
            seed=args.seed + 3,
        )

        meta = {
            "expr_h5ad": str(args.expr_h5ad),
            "se_h5ad": str(args.se_h5ad),
            "se_dim": int(se_dim),
            "pca_dim": int(n_pcs),
            "n_pcs_requested": int(args.n_pcs),
            "n_genes": int(adata.n_vars),
            "n_cells": int(adata.n_obs),
            "split_col": args.split_col,
            "fit_splits": list(args.fit_splits),
            "max_pca_fit": args.max_pca_fit,
            "max_ridge_fit": args.max_ridge_fit,
            "umap_max_cells": args.umap_max_cells,
            "umap_split": umap_split,
            "color_key": color_key,
            "ridge_alpha": args.ridge_alpha,
            "elapsed_sec": float(time.time() - t0),
        }
        (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Done in {meta['elapsed_sec']:.1f}s -> {args.out_dir}", flush=True)
    finally:
        close_se()
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()


if __name__ == "__main__":
    main()
