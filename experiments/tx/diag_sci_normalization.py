#!/usr/bin/env python3
"""Does re-normalizing after gene alignment change SCimilarity embeddings?

Runs inside the SCimilarity venv. The pipeline feeds ``align_dataset`` output
straight to ``get_embeddings``, but the panel is log1p(tp10k) over the *full*
Replogle transcriptome, and only 6481/28231 of SCimilarity's genes are present.
So the aligned rows sum to ~8.5k in expm1 space, not the 10k the encoder was
trained on.

Variant A — current pipeline: align only.
Variant B — align, then renormalize each row to tp10k over the aligned space.

Both are ridge-decoded to the 2000 HVGs and scored on held-out train + HepG2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]


def pearson_rows(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=1, keepdims=True)
    r = recon - recon.mean(axis=1, keepdims=True)
    num = (t * r).sum(axis=1)
    den = np.sqrt((t * t).sum(axis=1) * (r * r).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def score(true: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    err = true - recon
    ss_res = float((err * err).sum())
    centered = true - true.mean(axis=0, keepdims=True)
    ss_tot = float((centered**2).sum())
    ss_res_c = float(((centered - (recon - recon.mean(axis=0, keepdims=True))) ** 2).sum())
    return {
        "mean_pearson_cell": float(pearson_rows(true, recon).mean()),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "r2_centered": float(1.0 - ss_res_c / max(ss_tot, 1e-12)),
        "mean_mse": float((err * err).mean()),
    }


def renorm_tp10k(x: sp.spmatrix | np.ndarray) -> np.ndarray:
    """log1p(tp10k) recomputed over whatever genes survived alignment."""
    dense = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)
    counts = np.expm1(dense)
    totals = counts.sum(axis=1, keepdims=True)
    counts *= 10_000.0 / np.maximum(totals, 1e-12)
    return np.log1p(counts).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-path", type=Path, default=REPO_ROOT / "data_dir/replogle/replogle.h5ad")
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--model-dir", type=Path, default=REPO_ROOT / "data/scimilarity/model")
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/sci_recon_bench/normalization.json",
    )
    p.add_argument("--n-fit", type=int, default=40_000)
    p.add_argument("--n-eval", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from scimilarity.cell_embedding import CellEmbedding
    from scimilarity.utils import align_dataset

    rng = np.random.default_rng(args.seed)
    hvg = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(hvg.obs["split"]).astype(str)
    context = np.asarray(hvg.obs["cell_line"]).astype(str)
    train = np.flatnonzero(split == "train")
    hepg2 = np.flatnonzero((context == "hepg2") & (split == "test"))
    fit_idx = np.sort(rng.choice(train, size=min(args.n_fit, train.size), replace=False))
    rest = np.setdiff1d(train, fit_idx)
    eval_train = np.sort(rng.choice(rest, size=min(args.n_eval, rest.size), replace=False))
    eval_hepg2 = np.sort(rng.choice(hepg2, size=min(args.n_eval, hepg2.size), replace=False))
    sel = np.concatenate([fit_idx, eval_train, eval_hepg2])
    order = np.argsort(sel)
    sel_sorted = sel[order]
    # map original blocks onto rows of the sorted matrix
    inv = np.empty(sel.size, dtype=np.int64)
    inv[order] = np.arange(sel.size)
    n_fit, n_tr = fit_idx.size, eval_train.size
    pos = {
        "fit": inv[:n_fit],
        "train_heldout": inv[n_fit : n_fit + n_tr],
        "hepg2_test": inv[n_fit + n_tr :],
    }
    print(f"fit={fit_idx.size:,} train_heldout={eval_train.size:,} hepg2={eval_hepg2.size:,}", flush=True)

    x_genes = hvg.X[sel_sorted]
    x_genes = np.asarray(x_genes.toarray() if sp.issparse(x_genes) else x_genes, dtype=np.float64)
    if hasattr(hvg, "file") and hvg.file is not None:
        hvg.file.close()

    panel = ad.read_h5ad(args.panel_path, backed="r")
    sub = panel[sel_sorted].to_memory()
    if hasattr(panel, "file") and panel.file is not None:
        panel.file.close()

    ce = CellEmbedding(model_path=str(args.model_dir), use_gpu=True)
    aligned = align_dataset(sub, ce.gene_order, keep_obsm=False)
    raw = aligned.X
    totals = np.expm1(np.asarray(raw.toarray() if sp.issparse(raw) else raw, dtype=np.float64)).sum(axis=1)
    print(
        f"aligned expm1 rowsum: median {np.median(totals):.1f} "
        f"p5 {np.percentile(totals, 5):.1f} p95 {np.percentile(totals, 95):.1f}",
        flush=True,
    )

    variants = {
        "align_only": raw,
        "align_then_tp10k": renorm_tp10k(raw),
    }
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, matrix in variants.items():
        emb = np.asarray(ce.get_embeddings(matrix, buffer_size=4096), dtype=np.float64)
        ridge = Ridge(alpha=1.0).fit(emb[pos["fit"]], x_genes[pos["fit"]])
        results[name] = {}
        for pool, rows in pos.items():
            block = score(x_genes[rows], ridge.predict(emb[rows]))
            results[name][pool] = block
            print(
                f"  {name:18s} {pool:14s} cellP={block['mean_pearson_cell']:.4f} "
                f"R2={block['r2']:+.4f} R2c={block['r2_centered']:+.4f}",
                flush=True,
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "n_fit": int(fit_idx.size),
                "n_eval": int(args.n_eval),
                "aligned_rowsum_median": float(np.median(totals)),
                "metrics": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
