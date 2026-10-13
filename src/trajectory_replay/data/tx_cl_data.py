"""SE + obs arrays for sequential transcriptomics CL.

The SE sidecar is loaded **once into RAM** (PBMC ~80 GB float32). Gene-space
oneshot jobs load h5ad ``X`` instead (~5 GB for Replogle HVGs). Slurm jobs
should request ~512 GB so the matrix, gene ``X`` batches, and the CFM fit
together. Do not call :func:`load_tx_cl_arrays` on a login node.

Categorical obs columns stay as integer arrays. Gene expression is still
opened backed and read only for decoder fitting / eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from trajectory_replay.data.tx_cl_tasks import TxClStream, load_stream_obs
from trajectory_replay.data.tx_dataset import TxEncoding, build_encoding, encode_labels
from trajectory_replay.data.tx_preprocess import decode_h5ad_categorical


@dataclass(frozen=True)
class TxClArrays:
    """Cell-aligned obs encodings and SE sidecar metadata."""

    h5ad_path: Path
    se_path: Path
    stream: TxClStream
    encoding: TxEncoding
    context: np.ndarray
    perturbation: np.ndarray
    split: np.ndarray
    c: np.ndarray
    p: np.ndarray
    z: dict[str, np.ndarray]
    se: np.ndarray
    se_dim: int
    n_cells: int

    @property
    def covariate_col_order(self) -> tuple[str, ...]:
        return self.encoding.covariate_col_order


def _obs_encoding(
    h5ad_path: Path,
    stream: TxClStream,
    *,
    covariate_cols: tuple[str, ...],
    control_perturbation: str,
) -> tuple[dict[str, np.ndarray], TxEncoding, np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    obs = load_stream_obs(h5ad_path, stream)
    cov = {col: decode_h5ad_categorical(h5ad_path, col) for col in covariate_cols}
    encoding = build_encoding(
        obs["context"],
        obs["perturbation"],
        covariate_arrays=cov,
        covariate_col_order=covariate_cols,
        control_perturbation=control_perturbation,
    )
    c, p, z = encode_labels(
        obs["context"],
        obs["perturbation"],
        encoding,
        covariate_arrays=cov,
    )
    n = int(obs["split"].shape[0])
    return obs, encoding, c, p, z, n


def _as_float32_matrix(x: np.ndarray, n_expected: int, *, label: str) -> np.ndarray:
    if x.ndim != 2 or int(x.shape[0]) != n_expected:
        raise ValueError(f"{label} shape {x.shape} does not match h5ad n_obs={n_expected}")
    if x.dtype != np.float32 or not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, dtype=np.float32)
    gb = x.nbytes / (1024**3)
    print(f"  {label} in RAM: shape={x.shape} dtype={x.dtype} ({gb:.1f} GiB)", flush=True)
    return x


def load_dense_gene_matrix(h5ad_path: Path | str, n_expected: int) -> np.ndarray:
    """Load h5ad ``X`` densely into float32 RAM (Replogle HVGs are ~5 GiB)."""
    h5ad_path = Path(h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        return _as_float32_matrix(np.asarray(x), n_expected, label="genes")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def load_tx_cl_arrays(
    h5ad_path: Path | str,
    se_path: Path | str,
    stream: TxClStream,
    *,
    covariate_cols: tuple[str, ...] = (),
    control_perturbation: str,
) -> TxClArrays:
    """Load obs arrays and the full SE sidecar into RAM (never reads h5ad ``X``)."""
    h5ad_path = Path(h5ad_path)
    se_path = Path(se_path)
    obs, encoding, c, p, z, n = _obs_encoding(
        h5ad_path,
        stream,
        covariate_cols=covariate_cols,
        control_perturbation=control_perturbation,
    )
    se = _as_float32_matrix(np.load(se_path), n, label="SE")
    return TxClArrays(
        h5ad_path=h5ad_path,
        se_path=se_path,
        stream=stream,
        encoding=encoding,
        context=obs["context"],
        perturbation=obs["perturbation"],
        split=obs["split"],
        c=c,
        p=p,
        z=z,
        se=se,
        se_dim=int(se.shape[1]),
        n_cells=n,
    )


def load_cl_representation_arrays(
    h5ad_path: Path | str,
    stream: TxClStream,
    *,
    se_path: Path | str | None = None,
    covariate_cols: tuple[str, ...] = (),
    control_perturbation: str,
    representation: str = "se",
) -> TxClArrays:
    """Load CL arrays in STATE SE, SCimilarity, or gene ``X`` (same obs encodings)."""
    kind = str(representation or "se")
    if kind == "genes":
        return load_tx_cl_gene_arrays(
            h5ad_path,
            stream,
            covariate_cols=covariate_cols,
            control_perturbation=control_perturbation,
        )
    if kind == "pca":
        if se_path is None:
            raise ValueError("se_path (PCA embedding .npy) is required when representation='pca'")
        return load_tx_cl_pca_arrays(
            h5ad_path,
            stream,
            covariate_cols=covariate_cols,
            control_perturbation=control_perturbation,
            pca_path=se_path,
        )
    if kind not in ("se", "scimilarity"):
        raise ValueError(
            f"representation must be 'se', 'scimilarity', 'pca', or 'genes', got {representation!r}"
        )
    if se_path is None:
        raise ValueError(f"se_path is required when representation={kind!r}")
    return load_tx_cl_arrays(
        h5ad_path,
        se_path,
        stream,
        covariate_cols=covariate_cols,
        control_perturbation=control_perturbation,
    )


def load_tx_cl_pca_arrays(
    h5ad_path: Path | str,
    stream: TxClStream,
    *,
    covariate_cols: tuple[str, ...] = (),
    control_perturbation: str,
    obsm_key: str = "X_pca",
    pca_path: Path | str | None = None,
) -> TxClArrays:
    """Same obs encodings as :func:`load_tx_cl_arrays`, but ``se`` is a PCA embedding.

    With ``pca_path`` the embedding is read from a ``.npy`` sidecar sharing the
    h5ad's cell order; otherwise it comes from ``obsm[obsm_key]``.

    The dump's ``obsm['X_pca']`` excludes ``test`` cells from the fit but
    *includes* the HepG2 ``finetune`` phase, so it leaks a future task and is
    oneshot-only. The sidecar produced by ``scripts/build_replogle.py`` is fit
    on task-0 contexts alone and has no such leak.
    """
    h5ad_path = Path(h5ad_path)
    obs, encoding, c, p, z, n = _obs_encoding(
        h5ad_path,
        stream,
        covariate_cols=covariate_cols,
        control_perturbation=control_perturbation,
    )
    if pca_path is not None:
        pca = _as_float32_matrix(np.load(Path(pca_path)), n, label="PCA")
    else:
        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            if obsm_key not in adata.obsm:
                raise KeyError(f"obsm[{obsm_key!r}] missing in {h5ad_path}")
            pca = _as_float32_matrix(np.asarray(adata.obsm[obsm_key]), n, label="PCA")
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
    return TxClArrays(
        h5ad_path=h5ad_path,
        se_path=Path(pca_path) if pca_path is not None else h5ad_path,
        stream=stream,
        encoding=encoding,
        context=obs["context"],
        perturbation=obs["perturbation"],
        split=obs["split"],
        c=c,
        p=p,
        z=z,
        se=pca,
        se_dim=int(pca.shape[1]),
        n_cells=n,
    )


def load_tx_cl_gene_arrays(
    h5ad_path: Path | str,
    stream: TxClStream,
    *,
    covariate_cols: tuple[str, ...] = (),
    control_perturbation: str,
) -> TxClArrays:
    """Same obs encodings as :func:`load_tx_cl_arrays`, but ``se`` is gene ``X``."""
    h5ad_path = Path(h5ad_path)
    obs, encoding, c, p, z, n = _obs_encoding(
        h5ad_path,
        stream,
        covariate_cols=covariate_cols,
        control_perturbation=control_perturbation,
    )
    genes = load_dense_gene_matrix(h5ad_path, n)
    return TxClArrays(
        h5ad_path=h5ad_path,
        se_path=h5ad_path,
        stream=stream,
        encoding=encoding,
        context=obs["context"],
        perturbation=obs["perturbation"],
        split=obs["split"],
        c=c,
        p=p,
        z=z,
        se=genes,
        se_dim=int(genes.shape[1]),
        n_cells=n,
    )


class IndexedSeDataset(Dataset):
    """Dataset over selected rows of the in-RAM SE matrix plus conditions."""

    def __init__(self, arrays: TxClArrays, indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self._se = torch.from_numpy(np.ascontiguousarray(arrays.se))

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        i = int(self.indices[item])
        x = self._se[i].clone()
        out: list[torch.Tensor] = [
            x,
            torch.tensor(int(self.arrays.c[i]), dtype=torch.long),
            torch.tensor(int(self.arrays.p[i]), dtype=torch.long),
        ]
        out.extend(
            torch.tensor(int(self.arrays.z[col][i]), dtype=torch.long)
            for col in self.arrays.covariate_col_order
        )
        return tuple(out)


def stratified_cap_indices(
    indices: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    z: dict[str, np.ndarray],
    covariate_col_order: tuple[str, ...],
    *,
    max_per_stratum: int,
    seed: int,
) -> np.ndarray:
    """Deterministically cap rows per ``(c,p[,z...])`` stratum."""
    idx = np.asarray(indices, dtype=np.int64)
    if max_per_stratum <= 0 or idx.size == 0:
        return np.sort(idx)
    cols = [c[idx], p[idx], *(z[col][idx] for col in covariate_col_order)]
    keys = np.stack(cols, axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for group in range(int(inverse.max()) + 1):
        members = idx[inverse == group]
        if members.size > max_per_stratum:
            members = rng.choice(members, size=max_per_stratum, replace=False)
        keep.append(np.asarray(members, dtype=np.int64))
    return np.sort(np.concatenate(keep)) if keep else np.empty(0, dtype=np.int64)


def load_gene_rows(adata: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    """Read backed h5ad ``X`` rows in sorted order and return dense float32."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, int(adata.n_vars)), dtype=np.float32)
    order = np.argsort(idx)
    sorted_idx = idx[order]
    x = adata.X[sorted_idx]
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    out[order] = x
    return out


def load_se_rows(se: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Gather selected SE rows (already in RAM) as float32."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, int(se.shape[1])), dtype=np.float32)
    return np.ascontiguousarray(se[idx], dtype=np.float32)


LATENT_STANDARDIZER_NAME = "latent_standardizer.npz"


def fit_latent_standardizer(
    se: np.ndarray,
    indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate mean/std on ``indices``, matching a unit Gaussian prior."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("Cannot fit a latent standardizer on an empty index set")
    block = np.asarray(se[idx], dtype=np.float64)
    mean = block.mean(axis=0)
    std = np.maximum(block.std(axis=0), float(eps))
    return mean.astype(np.float32), std.astype(np.float32)


def apply_latent_radial_scale(se: np.ndarray) -> dict[str, float]:
    """In-place ``z <- sqrt(d) * z`` so unit-sphere rows match ``N(0,I)`` norms."""
    radius = float(np.sqrt(se.shape[1]))
    pre = (
        float(np.linalg.norm(se, axis=1).mean()) if se.shape[0] else float("nan")
    )
    se *= np.float32(radius)
    return {
        "kind": "radial",
        "radius": radius,
        "pre_row_norm_mean": pre,
        "post_row_norm_mean": float(np.linalg.norm(se, axis=1).mean()) if se.shape[0] else float("nan"),
        "prior_norm_expected": radius,
        "post_coord_std_median": float(np.median(se.std(axis=0))) if se.shape[0] else float("nan"),
    }


def apply_latent_standardizer(
    se: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    """In-place ``(z - mean) / std``. Returns scale diagnostics."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (se.shape[1],) or std.shape != (se.shape[1],):
        raise ValueError(
            f"standardizer dim mean={mean.shape} std={std.shape} != se_dim={se.shape[1]}"
        )
    se -= mean
    se /= std
    sample = se[0] if se.shape[0] else np.zeros(0, np.float32)
    return {
        "mean_abs_mean": float(np.mean(np.abs(mean))),
        "std_median": float(np.median(std)),
        "std_min": float(np.min(std)),
        "std_max": float(np.max(std)),
        "post_coord_std_median": float(np.median(se.std(axis=0))) if se.shape[0] else float("nan"),
        "post_row_norm_mean": float(np.linalg.norm(se, axis=1).mean()) if se.shape[0] else float("nan"),
        "prior_norm_expected": float(np.sqrt(se.shape[1])),
        "n_zero_std_clipped": int((std <= 1e-6 + 1e-12).sum()),
        "probe_norm": float(np.linalg.norm(sample)),
    }


def save_latent_standardizer(path: Path | str, mean: np.ndarray, std: np.ndarray) -> Path:
    path = Path(path)
    np.savez(path, mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))
    return path


def load_latent_standardizer(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(Path(path))
    return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)

