"""Compare old-only vs old+new-train PCA fitting populations on Replogle.

Measures gene-space reconstruction quality (MSE, R^2, per-cell Pearson
correlation) on held-out HepG2 TEST cells, under two candidate PCA bases:

  - old-only:      fit on all non-hepg2 (Jurkat/K562/RPE1/...) cells
  - old+new-train: fit on non-hepg2 + hepg2-finetune (excludes hepg2 test genes),
                   matching current production practice in
                   notebooks/bio_data/analyze_and_preprocess_replogle.ipynb

Both are restricted to the same 2000 HVG columns already flagged in the raw
h5ad, and use n_comps=256 to match the production pipeline.
"""

import os
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

_REPO_ROOT = Path(__file__).resolve().parents[2]
H5AD_PATH = os.environ.get(
    "REPLOGLE_H5AD",
    str(_REPO_ROOT / "data_dir" / "replogle.h5ad"),
)
TEST_GENES_PATH = "configs/replogle_hepg2_state_test_genes.txt"
N_FIT = 100_000
N_EVAL = 20_000
N_COMPS = 256


def subsample_sorted(idx, n, seed):
    if len(idx) <= n:
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=n, replace=False))


def load_hvg_rows(X, row_idx, hvg_cols):
    rows = X[row_idx, :]
    return rows[:, hvg_cols].astype(np.float32)


def fit_pca(X_fit, n_comps):
    pca = PCA(n_components=n_comps, svd_solver="randomized", random_state=0)
    pca.fit(X_fit)
    return pca


def metrics(true, recon):
    resid = true - recon
    mse = float(np.mean(resid**2))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    tc = true - true.mean(axis=1, keepdims=True)
    rc = recon - recon.mean(axis=1, keepdims=True)
    num = (tc * rc).sum(axis=1)
    den = np.sqrt((tc**2).sum(axis=1) * (rc**2).sum(axis=1)) + 1e-8
    per_cell_corr = num / den
    return mse, r2, float(per_cell_corr.mean()), float(per_cell_corr.std())


def main() -> None:
    t0 = time.time()
    f = h5py.File(H5AD_PATH, "r")
    X = f["X"]
    print("shape", X.shape, flush=True)

    cl_codes = f["obs/cell_line/codes"][:]
    cl_cats = f["obs/cell_line/categories"][:].astype(str)
    cell_line = cl_cats[cl_codes]

    g_codes = f["obs/gene/codes"][:]
    g_cats = f["obs/gene/categories"][:].astype(str)
    gene = g_cats[g_codes]

    hvg_mask = f["var/highly_variable"][:].astype(bool)
    hvg_cols = np.where(hvg_mask)[0]
    print("n_hvg", len(hvg_cols), flush=True)

    hepg2_test_genes = {
        g
        for g in open(TEST_GENES_PATH).read().splitlines()
        if g and not g.startswith("#")
    }

    is_hepg2 = cell_line == "hepg2"
    is_test_gene = np.isin(gene, list(hepg2_test_genes))

    old_only_idx = np.where(~is_hepg2)[0]
    old_new_train_idx = np.where((~is_hepg2) | (is_hepg2 & ~is_test_gene))[0]
    hepg2_test_idx = np.where(is_hepg2 & is_test_gene)[0]
    print(
        "pools:",
        "old_only", len(old_only_idx),
        "old_new_train", len(old_new_train_idx),
        "hepg2_test", len(hepg2_test_idx),
        flush=True,
    )

    old_only_fit_idx = subsample_sorted(old_only_idx, N_FIT, seed=1)
    old_new_train_fit_idx = subsample_sorted(old_new_train_idx, N_FIT, seed=2)
    hepg2_test_eval_idx = subsample_sorted(hepg2_test_idx, N_EVAL, seed=3)

    print("loading old-only fit rows...", time.time() - t0, flush=True)
    X_old_only = load_hvg_rows(X, old_only_fit_idx, hvg_cols)
    print("loading old+new-train fit rows...", time.time() - t0, flush=True)
    X_old_new_train = load_hvg_rows(X, old_new_train_fit_idx, hvg_cols)
    print("loading hepg2 test eval rows...", time.time() - t0, flush=True)
    X_hepg2_test = load_hvg_rows(X, hepg2_test_eval_idx, hvg_cols)
    print("loaded, shapes:", X_old_only.shape, X_old_new_train.shape, X_hepg2_test.shape, time.time() - t0, flush=True)

    print("fitting old-only PCA...", time.time() - t0, flush=True)
    pca_old_only = fit_pca(X_old_only, N_COMPS)
    print("fitting old+new-train PCA...", time.time() - t0, flush=True)
    pca_old_new_train = fit_pca(X_old_new_train, N_COMPS)
    print("done fitting, time", time.time() - t0, flush=True)

    print(
        "\nvariance explained (fit population): old-only=",
        pca_old_only.explained_variance_ratio_.sum(),
        "old+new-train=",
        pca_old_new_train.explained_variance_ratio_.sum(),
        flush=True,
    )

    recon_old_only = pca_old_only.inverse_transform(pca_old_only.transform(X_hepg2_test))
    recon_old_new_train = pca_old_new_train.inverse_transform(
        pca_old_new_train.transform(X_hepg2_test)
    )

    mse_a, r2_a, corr_mean_a, corr_std_a = metrics(X_hepg2_test, recon_old_only)
    mse_b, r2_b, corr_mean_b, corr_std_b = metrics(X_hepg2_test, recon_old_new_train)

    print("\n=== HepG2 TEST reconstruction quality (n_eval={}) ===".format(len(X_hepg2_test)), flush=True)
    print(
        f"old-only PCA:       MSE={mse_a:.4f}  R2={r2_a:.4f}  "
        f"per-cell-corr={corr_mean_a:.4f}+-{corr_std_a:.4f}",
        flush=True,
    )
    print(
        f"old+new-train PCA:  MSE={mse_b:.4f}  R2={r2_b:.4f}  "
        f"per-cell-corr={corr_mean_b:.4f}+-{corr_std_b:.4f}",
        flush=True,
    )
    print("\ntotal time", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
