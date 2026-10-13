"""Tests for PCA gene-space decoding."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

_REPO = Path(__file__).resolve().parents[1]

from trajectory_replay.data.pca_decode import load_pca_gene_decoder  # noqa: E402


def test_decoder_uses_uns_means_train_not_global_x_mean(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_obs, n_genes, n_comp = 20, 8, 3
    x = rng.normal(size=(n_obs, n_genes)).astype(np.float32)
    pcs = rng.normal(size=(n_genes, n_comp))
    means_train = x[:5].mean(axis=0, keepdims=True)
    x_pca = (x - means_train) @ pcs

    adata = ad.AnnData(X=x.copy())
    adata.obsm["X_pca"] = x_pca
    adata.varm["PCs"] = pcs
    adata.uns["means_train"] = means_train.astype(np.float32)

    path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(path)

    decoder = load_pca_gene_decoder(path)
    decoded = decoder.decode(x_pca)

    expected = x_pca @ pcs.T + means_train
    wrong = x_pca @ pcs.T + x.mean(axis=0, keepdims=True)

    assert np.allclose(decoded, expected)
    assert not np.allclose(decoded, wrong)


def test_decoder_accepts_1d_means_train(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n_obs, n_genes, n_comp = 5, 4, 2
    x = rng.normal(size=(n_obs, n_genes)).astype(np.float32)
    pcs = rng.normal(size=(n_genes, n_comp))
    means_train = x[:2].mean(axis=0)
    x_pca = (x - means_train) @ pcs

    adata = ad.AnnData(X=x)
    adata.obsm["X_pca"] = x_pca
    adata.varm["PCs"] = pcs
    adata.uns["means_train"] = means_train.astype(np.float32)

    path = tmp_path / "tiny_1d_mean.h5ad"
    adata.write_h5ad(path)

    decoder = load_pca_gene_decoder(path)
    assert decoder.mean.shape == (1, n_genes)
    assert np.allclose(decoder.decode(x_pca), x_pca @ pcs.T + means_train)


def test_pca_inverse_matches_task_linear_decoder(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n_obs, n_genes, n_comp = 11, 6, 3
    x = rng.normal(size=(n_obs, n_genes)).astype(np.float32)
    pcs = rng.normal(size=(n_genes, n_comp))
    means_train = x[:4].mean(axis=0, keepdims=True)
    x_pca = (x - means_train) @ pcs

    adata = ad.AnnData(X=x)
    adata.obsm["X_pca"] = x_pca
    adata.varm["PCs"] = pcs
    adata.uns["means_train"] = means_train.astype(np.float32)
    path = tmp_path / "tiny_pca_task.h5ad"
    adata.write_h5ad(path)

    from trajectory_replay.data.pca_decode import pca_inverse_as_task_decoder

    pca = load_pca_gene_decoder(path)
    lin = pca_inverse_as_task_decoder(pca, task_id=0, n_fit=n_obs)
    np.testing.assert_allclose(lin.decode(x_pca), pca.decode(x_pca))
    assert lin.se_dim == n_comp
    assert lin.n_genes == n_genes
    assert not lin.identity


def test_sidecar_basis_round_trips_to_same_decoder(tmp_path: Path) -> None:
    from trajectory_replay.data.pca_decode import pca_gene_decoder_from_basis
    from trajectory_replay.data.tx_preprocess import (
        PcaBasis,
        load_pca_basis,
        save_pca_basis,
    )

    rng = np.random.default_rng(3)
    n_obs, n_genes, n_comp = 9, 7, 4
    pcs = rng.normal(size=(n_genes, n_comp)).astype(np.float32)
    means = rng.normal(size=n_genes).astype(np.float32)
    x_pca = rng.normal(size=(n_obs, n_comp)).astype(np.float32)

    basis = PcaBasis(
        pcs=pcs,
        means=means,
        variance_ratio=np.full(n_comp, 1.0 / n_comp, dtype=np.float32),
        n_fit_cells=n_obs,
        fit_phases=("train",),
    )
    path = save_pca_basis(tmp_path / "basis.npz", basis)
    loaded = load_pca_basis(path)

    assert loaded.n_comps == n_comp
    assert loaded.n_genes == n_genes
    assert loaded.fit_phases == ("train",)
    np.testing.assert_allclose(loaded.pcs, pcs)
    np.testing.assert_allclose(loaded.means, means)

    decoder = pca_gene_decoder_from_basis(loaded)
    np.testing.assert_allclose(decoder.decode(x_pca), x_pca @ pcs.T + means, rtol=1e-5)


def test_sidecar_decoder_truncates_to_embedding_width(tmp_path: Path) -> None:
    from trajectory_replay.data.pca_decode import pca_gene_decoder_from_basis
    from trajectory_replay.data.tx_preprocess import PcaBasis

    rng = np.random.default_rng(4)
    n_genes, n_comp, used = 6, 5, 3
    basis = PcaBasis(
        pcs=rng.normal(size=(n_genes, n_comp)).astype(np.float32),
        means=np.zeros(n_genes, dtype=np.float32),
        variance_ratio=np.full(n_comp, 0.2, dtype=np.float32),
        n_fit_cells=10,
        fit_phases=("train",),
    )
    decoder = pca_gene_decoder_from_basis(basis, n_components=used)
    assert decoder.n_components == used
    assert decoder.pcs.shape == (used, n_genes)


def test_pca_sidecar_names_label_non_default_widths() -> None:
    from trajectory_replay.data.tx_preprocess import (
        DEFAULT_PCA_DIM,
        pca_sidecar_names,
    )

    assert DEFAULT_PCA_DIM == 256
    assert pca_sidecar_names(256) == ("pca_basis.npz", "pca_embedding.npy")
    assert pca_sidecar_names(512) == ("pca512_basis.npz", "pca512_embedding.npy")


def test_slice_pca_basis_keeps_leading_components() -> None:
    from trajectory_replay.data.tx_preprocess import PcaBasis, slice_pca_basis

    rng = np.random.default_rng(0)
    full = PcaBasis(
        pcs=rng.normal(size=(10, 8)).astype(np.float32),
        means=rng.normal(size=10).astype(np.float32),
        variance_ratio=np.linspace(0.4, 0.01, 8, dtype=np.float32),
        n_fit_cells=100,
        fit_phases=("train",),
    )
    slim = slice_pca_basis(full, 3)
    assert slim.n_comps == 3
    np.testing.assert_array_equal(slim.pcs, full.pcs[:, :3])
    np.testing.assert_array_equal(slim.variance_ratio, full.variance_ratio[:3])
    np.testing.assert_array_equal(slim.means, full.means)


def test_resolve_pca_artifacts_pbmc_canonical_256() -> None:
    from experiments.tx.train_cl import DEFAULTS, resolve_pca_artifacts

    defaults = DEFAULTS["pbmc"]
    root = defaults["pca"].parent
    emb, basis, dim = resolve_pca_artifacts(defaults)
    assert dim == 256
    assert emb == root / "pca_embedding.npy"
    assert basis == root / "pca_basis.npz"
    emb50, basis50, dim50 = resolve_pca_artifacts(defaults, pca_dim=50)
    assert dim50 == 50
    assert emb50 == root / "pca50_embedding.npy"
    assert basis50 == root / "pca50_basis.npz"


def test_resolve_pca_artifacts_selects_width_without_mixing_banks() -> None:
    from experiments.tx.train_cl import DEFAULTS, resolve_pca_artifacts

    defaults = DEFAULTS["replogle"]
    root = defaults["pca"].parent
    emb, basis, dim = resolve_pca_artifacts(defaults)
    assert dim == 256
    assert emb == root / "pca_embedding.npy"
    assert basis == root / "pca_basis.npz"

    emb512, basis512, dim512 = resolve_pca_artifacts(defaults, pca_dim=512)
    assert dim512 == 512
    assert emb512 == root / "pca512_embedding.npy"
    assert basis512 == root / "pca512_basis.npz"

    custom = Path("/tmp/custom_pca.npy")
    custom_basis = Path("/tmp/custom_pca_basis.npz")
    emb_c, basis_c, dim_c = resolve_pca_artifacts(
        defaults, se_path=custom, pca_basis=custom_basis
    )
    assert dim_c is None
    assert emb_c == custom
    assert basis_c == custom_basis
    _, _, dim_match = resolve_pca_artifacts(
        defaults, pca_dim=512, se_path=custom, pca_basis=custom_basis
    )
    assert dim_match == 512
