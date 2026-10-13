"""Shared preprocessing for transcriptomics dumps: split-aware PCA and SE sidecars.

The dataset dump scripts (``scripts/dump_replogle_dataset.py``,
``scripts/dump_pbmc_dataset.py``) use these helpers so both datasets get an
identical PCA contract:

* the basis is fit **only** on the phases a model is allowed to see
  (``train`` + ``finetune``), never on ``val`` / ``test``. Large ``n_obs``
  uses the gene–gene covariance (or IncrementalPCA) instead of ``sc.tl.pca``,
  which overflows 32-bit LAPACK indexes when ``n_obs * n_comps > 2^31``;
* ``varm['PCs']`` (n_genes, n_comps) and ``uns['means_train']`` (n_genes,) are
  stored so :mod:`trajectory_replay.data.pca_decode` can invert the
  embedding back to log-normalized gene space;
* ``obsm['X_pca']`` holds the projection of *every* cell, including held-out
  ones (they are embedded with the train basis, just not used to fit it).

SE-600M embeddings are far too large to live inside the h5ad (~80 GB for PBMC),
so they stay in a ``.npy`` matrix sharing the h5ad's cell order and are
referenced from ``uns``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

SE_OBSM_KEY = "X_state"
PCA_OBSM_KEY = "X_pca"
PCS_VARM_KEY = "PCs"
MEANS_UNS_KEY = "means_train"


@dataclass(frozen=True)
class PcaBasis:
    """Zero-centered PCA basis fit on a subset of cells."""

    pcs: np.ndarray  # (n_genes, n_comps), scanpy varm['PCs'] layout
    means: np.ndarray  # (n_genes,) mean of the fit cells
    variance_ratio: np.ndarray  # (n_comps,)
    n_fit_cells: int
    fit_phases: tuple[str, ...]

    @property
    def n_comps(self) -> int:
        return int(self.pcs.shape[1])

    @property
    def n_genes(self) -> int:
        return int(self.pcs.shape[0])


def phase_mask(labels: np.ndarray, phases: tuple[str, ...]) -> np.ndarray:
    """Boolean mask of cells whose split label is in ``phases``."""
    return np.isin(np.asarray(labels).astype(str), list(phases))


def fit_indices(
    mask: np.ndarray,
    *,
    fraction: float = 1.0,
    max_cells: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Indices to fit PCA on: ``mask`` cells, optionally thinned.

    Thinning is uniform over the eligible cells (not stratified) and sorted, so
    the fit stays a contiguous-friendly read on the h5ad.
    """
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        raise ValueError("No cells selected for the PCA fit.")
    target = idx.size
    if 0.0 < fraction < 1.0:
        target = max(1, int(round(idx.size * fraction)))
    if max_cells is not None and max_cells > 0:
        target = min(target, max_cells)
    if target >= idx.size:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=target, replace=False))


# SciPy's svds path (scanpy/sklearn) uses 32-bit LAPACK indexes; n_obs * n_comps
# above this overflows (PBMC T0 train × 512).
_LAPACK_INT32_MAX = 2**31 - 1
_SCANPY_OBS_CAP = 1_000_000
_COVARIANCE_VARS_CAP = 10_000
PCA_SOLVERS = ("auto", "scanpy", "covariance", "incremental")


def resolve_pca_solver(
    n_obs: int,
    n_vars: int,
    n_comps: int,
    method: str = "auto",
) -> str:
    """Pick a PCA backend. ``auto`` avoids the LAPACK int32 overflow on large n.

    * ``scanpy``: ``sc.tl.pca`` (exact-enough randomized/full SVD). Fine when
      ``n_obs`` is Replogle-scale.
    * ``covariance``: eigendecompose the gene–gene covariance in batches.
      Same as sklearn's full PCA when ``n_genes`` is small (PBMC HVGs). Exact,
      no ``n_obs * n_comps`` workspace.
    * ``incremental``: sklearn ``IncrementalPCA``. Use when both ``n_obs`` and
      ``n_genes`` are large; it is an approximation.
    """
    method = str(method).lower()
    if method not in PCA_SOLVERS:
        raise ValueError(f"method must be one of {PCA_SOLVERS}, got {method!r}")
    if method != "auto":
        return method
    n_obs = int(n_obs)
    n_vars = int(n_vars)
    n_comps = int(n_comps)
    if n_obs <= _SCANPY_OBS_CAP and n_obs * n_comps < _LAPACK_INT32_MAX:
        return "scanpy"
    if n_vars <= _COVARIANCE_VARS_CAP:
        return "covariance"
    return "incremental"


def _as_dense_float64(chunk) -> np.ndarray:
    if issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk, dtype=np.float64, order="C")


def _iter_fit_chunks(adata: ad.AnnData, fit_idx: np.ndarray, batch_size: int):
    """Yield dense float64 batches of ``adata.X[fit_idx]`` without copying all cells."""
    fit_idx = np.asarray(fit_idx)
    batch_size = max(int(batch_size), 1)
    for start in range(0, fit_idx.size, batch_size):
        rows = fit_idx[start : start + batch_size]
        if rows.size and int(rows[-1]) - int(rows[0]) + 1 == rows.size:
            chunk = adata.X[int(rows[0]) : int(rows[-1]) + 1]
        else:
            chunk = adata.X[rows]
        yield _as_dense_float64(chunk)


def _flip_pc_signs(pcs: np.ndarray) -> np.ndarray:
    """Make the largest-magnitude loading in each component positive (sklearn)."""
    out = np.array(pcs, copy=True)
    for j in range(out.shape[1]):
        i = int(np.argmax(np.abs(out[:, j])))
        if out[i, j] < 0:
            out[:, j] *= -1.0
    return out


def _fit_pca_scanpy(adata: ad.AnnData, fit_idx: np.ndarray, n_comps: int, seed: int):
    sub = adata[fit_idx].copy()
    sc.tl.pca(sub, n_comps=n_comps, random_state=seed)
    pcs = np.asarray(sub.varm[PCS_VARM_KEY], dtype=np.float32)
    means = np.asarray(sub.X.mean(axis=0), dtype=np.float32).ravel()
    variance_ratio = np.asarray(
        sub.uns.get("pca", {}).get("variance_ratio", np.zeros(n_comps)), dtype=np.float32
    )
    del sub
    return pcs, means, variance_ratio


def _fit_pca_covariance(
    adata: ad.AnnData, fit_idx: np.ndarray, n_comps: int, batch_size: int
):
    from scipy.linalg import eigh

    fit_idx = np.asarray(fit_idx)
    n = int(fit_idx.size)
    p = int(adata.n_vars)
    mean = np.zeros(p, dtype=np.float64)
    for x in _iter_fit_chunks(adata, fit_idx, batch_size):
        mean += x.sum(axis=0)
    mean /= n

    gram = np.zeros((p, p), dtype=np.float64)
    for x in _iter_fit_chunks(adata, fit_idx, batch_size):
        xc = x - mean
        gram += xc.T @ xc
    cov = gram / max(n - 1, 1)
    evals, evecs = eigh(cov, subset_by_index=[p - n_comps, p - 1])
    pcs = _flip_pc_signs(evecs[:, ::-1])
    total = float(np.trace(cov))
    evals = np.clip(evals[::-1], 0.0, None)
    ratio = evals / total if total > 0 else np.zeros(n_comps, dtype=np.float64)
    return (
        np.asarray(pcs, dtype=np.float32),
        np.asarray(mean, dtype=np.float32).ravel(),
        np.asarray(ratio, dtype=np.float32),
    )


def _fit_pca_incremental(
    adata: ad.AnnData, fit_idx: np.ndarray, n_comps: int, batch_size: int
):
    from sklearn.decomposition import IncrementalPCA

    fit_idx = np.asarray(fit_idx)
    if fit_idx.size < n_comps:
        raise ValueError(
            f"IncrementalPCA needs n_fit_cells={fit_idx.size} >= n_comps={n_comps}"
        )
    bs = max(int(batch_size), int(n_comps))
    ipca = IncrementalPCA(n_components=n_comps, batch_size=bs)
    buf: list[np.ndarray] = []
    n_buf = 0
    fitted = False

    def _flush(force: bool) -> None:
        nonlocal buf, n_buf, fitted
        if n_buf == 0:
            return
        if not fitted and n_buf < n_comps and not force:
            return
        x = np.vstack(buf)
        buf = []
        n_buf = 0
        if not fitted and x.shape[0] < n_comps:
            raise ValueError(
                f"IncrementalPCA first batch has {x.shape[0]} rows < n_comps={n_comps}"
            )
        if x.shape[0] == 0:
            return
        if fitted and x.shape[0] < n_comps:
            return
        ipca.partial_fit(x)
        fitted = True

    for x in _iter_fit_chunks(adata, fit_idx, bs):
        buf.append(x)
        n_buf += x.shape[0]
        if n_buf >= bs:
            _flush(force=False)
    _flush(force=True)
    if not fitted:
        raise ValueError("IncrementalPCA did not receive a large enough batch")
    pcs = _flip_pc_signs(np.asarray(ipca.components_.T, dtype=np.float64))
    return (
        np.asarray(pcs, dtype=np.float32),
        np.asarray(ipca.mean_, dtype=np.float32).ravel(),
        np.asarray(ipca.explained_variance_ratio_, dtype=np.float32),
    )


def fit_pca_basis(
    adata: ad.AnnData,
    fit_idx: np.ndarray,
    *,
    n_comps: int = 256,
    seed: int = 0,
    fit_phases: tuple[str, ...] = (),
    method: str = "auto",
    batch_size: int = 8192,
) -> PcaBasis:
    """Fit a PCA basis on ``adata[fit_idx]`` without requiring a full SVD of n×d.

    ``method='auto'`` uses scanpy when that is safe, otherwise the gene–gene
    covariance (PBMC-scale ``n_obs``, HVG-scale ``n_genes``). See
    :func:`resolve_pca_solver`.
    """
    if n_comps >= min(len(fit_idx), adata.n_vars):
        raise ValueError(
            f"n_comps={n_comps} must be < min(n_fit_cells={len(fit_idx)}, "
            f"n_genes={adata.n_vars})"
        )
    solver = resolve_pca_solver(len(fit_idx), adata.n_vars, n_comps, method=method)
    print(
        f"[pca] solver={solver} n_fit={len(fit_idx):,} n_genes={adata.n_vars} "
        f"n_comps={n_comps} batch_size={batch_size}",
        flush=True,
    )
    if solver == "scanpy":
        pcs, means, variance_ratio = _fit_pca_scanpy(adata, fit_idx, n_comps, seed)
    elif solver == "covariance":
        pcs, means, variance_ratio = _fit_pca_covariance(
            adata, fit_idx, n_comps, batch_size
        )
    else:
        pcs, means, variance_ratio = _fit_pca_incremental(
            adata, fit_idx, n_comps, batch_size
        )
    return PcaBasis(
        pcs=pcs,
        means=means,
        variance_ratio=variance_ratio,
        n_fit_cells=int(len(fit_idx)),
        fit_phases=tuple(fit_phases),
    )


def project_pca(
    adata: ad.AnnData,
    basis: PcaBasis,
    *,
    batch_size: int = 100_000,
) -> np.ndarray:
    """Embed every cell with ``basis``: ``(X - means) @ PCs``, batched.

    Sparse ``X`` is multiplied without densifying (the mean shift is folded into
    a constant offset), which keeps peak memory at one batch of the *output*.
    """
    if basis.n_genes != adata.n_vars:
        raise ValueError(
            f"basis has {basis.n_genes} genes but adata has {adata.n_vars}; "
            "gene order/subset must match the fit."
        )
    out = np.empty((adata.n_obs, basis.n_comps), dtype=np.float32)
    offset = basis.means @ basis.pcs
    for start in range(0, adata.n_obs, batch_size):
        stop = min(start + batch_size, adata.n_obs)
        chunk = adata.X[start:stop]
        proj = chunk @ basis.pcs if issparse(chunk) else np.asarray(chunk) @ basis.pcs
        out[start:stop] = np.asarray(proj, dtype=np.float32) - offset
    return out


def store_pca(adata: ad.AnnData, basis: PcaBasis, x_pca: np.ndarray) -> None:
    """Write the basis and embedding into the AnnData slots readers expect."""
    adata.varm[PCS_VARM_KEY] = basis.pcs
    adata.uns[MEANS_UNS_KEY] = basis.means
    adata.obsm[PCA_OBSM_KEY] = x_pca
    adata.uns["pca_obsm_key"] = PCA_OBSM_KEY
    adata.uns["pca_variance_ratio"] = basis.variance_ratio
    adata.uns["pca_fit"] = {
        "n_comps": basis.n_comps,
        "n_fit_cells": basis.n_fit_cells,
        "fit_phases": list(basis.fit_phases),
        "cumulative_variance_ratio": float(basis.variance_ratio.sum()),
    }


DEFAULT_PCA_DIM = 256


def pca_sidecar_names(n_comps: int) -> tuple[str, str]:
    """Basis and embedding filenames for a task-0 PCA width.

    ``build_replogle.py`` writes the canonical dump as unlabeled
    ``pca_basis.npz`` / ``pca_embedding.npy`` at :data:`DEFAULT_PCA_DIM`.
    Other widths from ``fit_pca_task0_widths.py`` are ``pca{d}_*``.
    """
    dim = int(n_comps)
    if dim < 1:
        raise ValueError(f"n_comps must be >= 1, got {n_comps}")
    stem = "pca" if dim == DEFAULT_PCA_DIM else f"pca{dim}"
    return f"{stem}_basis.npz", f"{stem}_embedding.npy"


def slice_pca_basis(basis: PcaBasis, n_comps: int) -> PcaBasis:
    """Keep the leading ``n_comps`` of a wider task-0 SVD (nested widths)."""
    k = int(n_comps)
    if k < 1 or k > basis.n_comps:
        raise ValueError(f"n_comps={k} must be in [1, {basis.n_comps}]")
    if k == basis.n_comps:
        return basis
    return PcaBasis(
        pcs=np.ascontiguousarray(basis.pcs[:, :k], dtype=np.float32),
        means=np.asarray(basis.means, dtype=np.float32).ravel(),
        variance_ratio=np.ascontiguousarray(basis.variance_ratio[:k], dtype=np.float32),
        n_fit_cells=int(basis.n_fit_cells),
        fit_phases=tuple(basis.fit_phases),
    )


def save_pca_basis(path: Path | str, basis: PcaBasis) -> Path:
    """Write ``basis`` to a ``.npz`` sidecar readable by :func:`load_pca_basis`.

    Sidecars exist so a basis can be fit on an arbitrary cell subset (e.g. only
    the contexts of CL task 0) without rewriting the h5ad, which would force a
    full copy of the dump for a 256-column embedding.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pcs=np.asarray(basis.pcs, dtype=np.float32),
        means=np.asarray(basis.means, dtype=np.float32).ravel(),
        variance_ratio=np.asarray(basis.variance_ratio, dtype=np.float32),
        n_fit_cells=np.int64(basis.n_fit_cells),
        fit_phases=np.asarray(list(basis.fit_phases), dtype=object),
    )
    return path


def load_pca_basis(path: Path | str) -> PcaBasis:
    """Read a basis written by :func:`save_pca_basis`."""
    with np.load(Path(path), allow_pickle=True) as z:
        return PcaBasis(
            pcs=np.asarray(z["pcs"], dtype=np.float32),
            means=np.asarray(z["means"], dtype=np.float32).ravel(),
            variance_ratio=np.asarray(z["variance_ratio"], dtype=np.float32),
            n_fit_cells=int(z["n_fit_cells"]),
            fit_phases=tuple(str(p) for p in z["fit_phases"].tolist()),
        )


def reconstruction_pearson(
    adata: ad.AnnData,
    basis: PcaBasis,
    x_pca: np.ndarray,
    *,
    max_cells: int = 20_000,
    seed: int = 0,
) -> float:
    """Mean per-cell Pearson between true and PCA-reconstructed expression."""
    n = min(max_cells, adata.n_obs)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False)) if n < adata.n_obs else np.arange(n)
    x_true = adata.X[idx]
    x_true = x_true.toarray() if issparse(x_true) else np.asarray(x_true)
    x_true = x_true.astype(np.float32)
    x_recon = x_pca[idx] @ basis.pcs.T + basis.means
    a = x_true - x_true.mean(axis=1, keepdims=True)
    b = x_recon - x_recon.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return float(np.mean(num / np.maximum(den, 1e-12)))


def verify_se_npy(path: Path | str, n_obs: int) -> tuple[int, int]:
    """Check an SE ``.npy`` sidecar matches ``n_obs`` rows; return its shape."""
    arr = np.load(Path(path), mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"SE npy {path} must be 2D, got shape {arr.shape}")
    if arr.shape[0] != n_obs:
        raise ValueError(
            f"SE npy {path} has {arr.shape[0]} rows but the h5ad has {n_obs} cells; "
            "re-encode from the current dump."
        )
    return int(arr.shape[0]), int(arr.shape[1])


def record_se_sidecar(
    adata: ad.AnnData,
    se_npy: Path | str,
    *,
    repo_root: Path | None = None,
    key: str = SE_OBSM_KEY,
) -> bool:
    """Point ``uns`` at the SE ``.npy`` sidecar, verifying it when it exists.

    Returns whether the sidecar was found on disk. The path is stored relative
    to ``repo_root`` when possible so dumps stay portable.
    """
    se_npy = Path(se_npy)
    stored = se_npy
    if repo_root is not None:
        try:
            stored = se_npy.resolve().relative_to(Path(repo_root).resolve())
        except ValueError:
            stored = se_npy
    adata.uns["se_obsm_key"] = key
    adata.uns["se_npy_path"] = str(stored)
    if not se_npy.exists():
        adata.uns["se_dim"] = 0
        return False
    _, dim = verify_se_npy(se_npy, adata.n_obs)
    adata.uns["se_dim"] = dim
    return True


def load_se_matrix(
    se_npy: Path | str,
    *,
    n_obs: int | None = None,
    mmap: bool = True,
) -> np.ndarray:
    """Load an SE sidecar, optionally checking it against ``n_obs``."""
    path = Path(se_npy)
    if n_obs is not None:
        verify_se_npy(path, n_obs)
    return np.load(path, mmap_mode="r" if mmap else None)


def recode_merge_category(
    codes: np.ndarray,
    categories: list[str],
    *,
    src: str,
    dst: str,
) -> tuple[np.ndarray, list[str]]:
    """Map categorical ``src`` onto existing ``dst`` and drop ``src`` from levels."""
    cats = [str(c) for c in categories]
    if src not in cats:
        return np.asarray(codes).copy(), cats
    if dst not in cats:
        raise ValueError(f"destination category {dst!r} not in {cats}")
    new_cats = [c for c in cats if c != src]
    old_to_new = np.empty(len(cats), dtype=np.intp)
    for i, cat in enumerate(cats):
        old_to_new[i] = new_cats.index(dst if cat == src else cat)
    codes_arr = np.asarray(codes)
    out = codes_arr.copy()
    valid = codes_arr >= 0
    out[valid] = old_to_new[codes_arr[valid]]
    return out.astype(codes_arr.dtype, copy=False), new_cats


def _obs_column_order(obs) -> list[str]:
    return [str(c) for c in obs.attrs["column-order"]]


def _set_obs_column_order(obs, order: list[str]) -> None:
    obs.attrs["column-order"] = np.array(order, dtype=object)


def drop_h5ad_obs_columns(path: Path | str, columns: tuple[str, ...]) -> list[str]:
    """Delete ``obs`` columns in place (does not touch ``X`` / ``obsm``)."""
    import h5py

    dropped: list[str] = []
    with h5py.File(path, "r+") as f:
        obs = f["obs"]
        for col in columns:
            if col in obs:
                del obs[col]
                dropped.append(col)
        order = [c for c in _obs_column_order(obs) if c not in columns]
        _set_obs_column_order(obs, order)
    return dropped


def read_h5ad_categorical(path: Path | str, column: str) -> tuple[np.ndarray, list[str]]:
    """Read an AnnData 0.2.0 categorical ``obs`` column without loading ``X``."""
    import h5py

    with h5py.File(path, "r") as f:
        group = f["obs"][column]
        codes = np.asarray(group["codes"][:])
        raw_cats = group["categories"][:]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in raw_cats]
    return codes, cats


def write_h5ad_categorical(
    path: Path | str,
    column: str,
    codes: np.ndarray,
    categories: list[str],
) -> None:
    """Overwrite one categorical ``obs`` column in place."""
    import h5py
    import pandas as pd
    from anndata.io import write_elem

    labels = np.asarray(categories, dtype=object)[np.asarray(codes, dtype=int)]
    categorical = pd.Categorical(labels, categories=list(categories), ordered=False)
    with h5py.File(path, "r+") as f:
        obs = f["obs"]
        if column in obs:
            del obs[column]
        write_elem(obs, column, categorical)
        order = _obs_column_order(obs)
        if column not in order:
            order.append(column)
            _set_obs_column_order(obs, order)


def merge_h5ad_obs_phase(
    path: Path | str,
    column: str,
    *,
    src: str,
    dst: str,
) -> dict[str, int]:
    """Relabel ``src`` cells as ``dst`` in ``obs[column]`` and drop the ``src`` level."""
    codes, cats = read_h5ad_categorical(path, column)
    n_src = int(np.sum(codes == cats.index(src))) if src in cats else 0
    new_codes, new_cats = recode_merge_category(codes, cats, src=src, dst=dst)
    if new_cats != cats:
        write_h5ad_categorical(path, column, new_codes, new_cats)
    return {"n_src": n_src, "n_dst_after": int(np.sum(new_codes == new_cats.index(dst)))}


def decode_h5ad_categorical(path: Path | str, column: str) -> np.ndarray:
    """String labels for one categorical ``obs`` column."""
    codes, cats = read_h5ad_categorical(path, column)
    return np.asarray(cats, dtype=object)[np.asarray(codes, dtype=int)]


def h5ad_obs_columns(path: Path | str) -> list[str]:
    import h5py

    with h5py.File(path, "r") as f:
        return [str(c) for c in f["obs"].attrs["column-order"]]


def rename_h5ad_obs_column(path: Path | str, src: str, dst: str) -> None:
    """Rename an ``obs`` column in place."""
    if src == dst:
        return
    cols = h5ad_obs_columns(path)
    if src not in cols:
        raise KeyError(f"{path} has no obs column {src!r} (have {cols})")
    if dst in cols:
        raise ValueError(f"{path} already has obs column {dst!r}")
    codes, cats = read_h5ad_categorical(path, src)
    write_h5ad_categorical(path, dst, codes, cats)
    drop_h5ad_obs_columns(path, (src,))


def _write_split_labels(path: Path | str, column: str, labels: np.ndarray) -> None:
    order = [c for c in ("finetune", "test", "train", "val") if c in set(map(str, labels))]
    inv = {cat: i for i, cat in enumerate(order)}
    codes = np.fromiter((inv[str(x)] for x in labels), dtype=np.int8, count=labels.size)
    write_h5ad_categorical(path, column, codes, order)


def fold_pbmc_val_cytokines_into_finetune(path: Path | str, split_col: str) -> int:
    """Move STATE val cytokines on finetune donors from ``test``/``val`` into ``finetune``."""
    from trajectory_replay.data.tx_state_splits import (
        PBMC_FINETUNE_DONORS,
        PBMC_STATE_VAL_CYTOKINES,
    )

    labels = decode_h5ad_categorical(path, split_col).copy()
    donors = decode_h5ad_categorical(path, "donor")
    cytokines = decode_h5ad_categorical(path, "cytokine")
    mask = np.isin(donors, PBMC_FINETUNE_DONORS) & np.isin(
        cytokines, PBMC_STATE_VAL_CYTOKINES
    )
    n = int(mask.sum())
    if n:
        labels[mask] = "finetune"
        _write_split_labels(path, split_col, labels)
    return n


def _split_counts(path: Path | str, column: str) -> dict[str, int]:
    codes, cats = read_h5ad_categorical(path, column)
    return {cat: int(np.sum(codes == i)) for i, cat in enumerate(cats)}


def simplify_replogle_dump(path: Path | str) -> dict[str, object]:
    """Keep one ``split`` column (STATE hepg2 few-shot)."""
    dropped = drop_h5ad_obs_columns(path, ("split1",))
    cols = h5ad_obs_columns(path)
    if "split2" in cols:
        rename_h5ad_obs_column(path, "split2", "split")
    return {"dropped": dropped, "split_counts": _split_counts(path, "split")}


def simplify_pbmc_dump(path: Path | str) -> dict[str, object]:
    """Keep one ``split`` column; fold STATE val cytokines into finetune."""
    dropped = drop_h5ad_obs_columns(path, ("split1", "split2"))
    cols = h5ad_obs_columns(path)
    split_col = "split3" if "split3" in cols else "split"
    if "val" in (read_h5ad_categorical(path, split_col)[1]):
        merge_h5ad_obs_phase(path, split_col, src="val", dst="finetune")
    n_folded = fold_pbmc_val_cytokines_into_finetune(path, split_col)
    if split_col != "split":
        rename_h5ad_obs_column(path, split_col, "split")
    return {
        "dropped": dropped,
        "n_val_cytokine_cells_into_finetune": n_folded,
        "split_counts": _split_counts(path, "split"),
    }
