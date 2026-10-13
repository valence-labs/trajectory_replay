"""Decode PCA embeddings back to gene expression (log-normalized space)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np


def _means_train_from_uns(adata: ad.AnnData, *, key: str = "means_train") -> np.ndarray:
    """Load per-gene means stored at preprocessing time (shape ``[1, n_genes]``)."""
    if key not in adata.uns:
        raise KeyError(f"uns[{key!r}] missing in AnnData; available: {list(adata.uns.keys())}")
    mean = np.asarray(adata.uns[key], dtype=np.float64)
    if mean.ndim == 1:
        mean = mean[np.newaxis, :]
    elif mean.ndim != 2 or mean.shape[0] != 1:
        raise ValueError(
            f"uns[{key!r}] must have shape (n_genes,) or (1, n_genes), got {mean.shape}"
        )
    return mean


@dataclass(frozen=True)
class PcaGeneDecoder:
    """Linear inverse PCA: ``genes = pca @ pcs.T + mean``."""

    pcs: np.ndarray  # (n_components, n_genes), scanpy varm['PCs'].T layout
    mean: np.ndarray  # (1, n_genes)
    n_components: int

    def decode(self, x_pca: np.ndarray) -> np.ndarray:
        x = np.asarray(x_pca, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x_pca must be 2D, got shape {x.shape}")
        if x.shape[1] > self.n_components:
            x = x[:, : self.n_components]
        elif x.shape[1] < self.n_components:
            pad = np.zeros((x.shape[0], self.n_components - x.shape[1]), dtype=np.float64)
            x = np.concatenate([x, pad], axis=1)
        return x @ self.pcs + self.mean


def pca_gene_decoder_from_basis(basis, *, n_components: int | None = None) -> PcaGeneDecoder:
    """Build a decoder from a :class:`~.tx_preprocess.PcaBasis` sidecar.

    Used when the basis was fit on a cell subset the h5ad slots do not describe,
    e.g. only the contexts of CL task 0.
    """
    pcs_varm = np.asarray(basis.pcs, dtype=np.float64)
    pcs = pcs_varm.T.copy()
    n_comp = pcs.shape[0] if n_components is None else min(int(n_components), pcs.shape[0])
    pcs = pcs[:n_comp, :]
    mean = np.asarray(basis.means, dtype=np.float64).reshape(1, -1)
    if mean.shape[1] != pcs.shape[1]:
        raise ValueError(f"basis means n_genes={mean.shape[1]} != PCs n_genes={pcs.shape[1]}")
    return PcaGeneDecoder(pcs=pcs, mean=mean, n_components=n_comp)


def pca_inverse_as_task_decoder(
    pca: PcaGeneDecoder,
    *,
    task_id: int,
    n_fit: int = 0,
):
    """Frozen ``TaskLinearDecoder`` with the same map as :meth:`PcaGeneDecoder.decode`.

    Safe for sequential CL only when the basis was fit on cells the stream is
    already allowed to see at that point; the dump's ``varm['PCs']`` is *not*
    (it includes the HepG2 finetune phase, i.e. a future task).
    """
    from trajectory_replay.data.tx_task_decoder import TaskLinearDecoder

    intercept = np.asarray(pca.mean, dtype=np.float64).reshape(-1)
    return TaskLinearDecoder(
        task_id=int(task_id),
        weight=np.asarray(pca.pcs.T, dtype=np.float64),
        intercept=intercept,
        n_fit=int(n_fit),
        lr=0.0,
        weight_decay=0.0,
        epochs=0,
        identity=False,
    )


def calibrate_tied_affine(tied, z_fit: np.ndarray, x_fit: np.ndarray):
    """Per-gene least squares ``a_g * D0(z) + b_g``, baked into a linear map.

    Fit only on the current task's allowed cells. Does not recover discarded PCs.
    """
    from trajectory_replay.data.tx_task_decoder import TaskLinearDecoder

    base = tied.decode(z_fit)
    x = np.asarray(x_fit, dtype=np.float64)
    b_mean = base.mean(axis=0)
    x_mean = x.mean(axis=0)
    bc = base - b_mean
    var = (bc * bc).mean(axis=0)
    cov = (bc * (x - x_mean)).mean(axis=0)
    a = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), 0.0)
    b = x_mean - a * b_mean
    print(
        f"  tied_affine slopes mean={float(a.mean()):.4f} "
        f"p05={float(np.quantile(a, 0.05)):.4f} p95={float(np.quantile(a, 0.95)):.4f}",
        flush=True,
    )
    return TaskLinearDecoder(
        task_id=tied.task_id,
        weight=tied.weight * a[:, None],
        intercept=a * tied.intercept + b,
        n_fit=int(np.asarray(z_fit).shape[0]),
        lr=0.0,
        weight_decay=0.0,
        epochs=0,
        identity=False,
    )


def load_pca_gene_decoder(
    h5ad_path: Path | str,
    *,
    pcs_key: str = "PCs",
    obsm_key: str = "X_pca",
    means_key: str = "means_train",
) -> PcaGeneDecoder:
    """Load PCA loadings and ``uns['means_train']`` from preprocessed AnnData."""
    path = Path(h5ad_path)
    adata = ad.read_h5ad(path, backed="r")
    try:
        if pcs_key not in adata.varm:
            raise KeyError(f"varm[{pcs_key!r}] missing in {path}")
        if obsm_key not in adata.obsm:
            raise KeyError(f"obsm[{obsm_key!r}] missing in {path}")

        pcs_varm = np.asarray(adata.varm[pcs_key], dtype=np.float64)
        # scanpy: varm['PCs'] is (n_genes, n_components)
        pcs = pcs_varm.T.copy()
        n_pca = int(adata.obsm[obsm_key].shape[1])
        n_comp = min(n_pca, pcs.shape[0])
        pcs = pcs[:n_comp, :]

        mean = _means_train_from_uns(adata, key=means_key)
        if mean.shape[1] != pcs.shape[1]:
            raise ValueError(
                f"means_train n_genes={mean.shape[1]} != varm[{pcs_key!r}] n_genes={pcs.shape[1]}"
            )

        return PcaGeneDecoder(pcs=pcs, mean=mean, n_components=n_comp)
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
