"""Tests for split-aware PCA helpers and in-place dump simplification."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from trajectory_replay.data.tx_preprocess import (
    fit_pca_basis,
    recode_merge_category,
    resolve_pca_solver,
    simplify_pbmc_dump,
    simplify_replogle_dump,
)
from trajectory_replay.data.tx_state_splits import (
    expected_pbmc_phase,
    expected_replogle_phase,
)


def test_recode_merge_category_maps_src_onto_dst() -> None:
    cats = ["finetune", "test", "train", "val"]
    codes = np.array([0, 1, 2, 3, 3, 1], dtype=np.int8)
    new_codes, new_cats = recode_merge_category(codes, cats, src="val", dst="test")
    assert new_cats == ["finetune", "test", "train"]
    np.testing.assert_array_equal(new_codes, np.array([0, 1, 2, 1, 1, 1], dtype=np.int8))


def test_expected_pbmc_phase_folds_val_into_finetune() -> None:
    assert expected_pbmc_phase("Donor2", "G-CSF") == "train"
    assert expected_pbmc_phase("Donor1", "G-CSF") == "finetune"
    assert expected_pbmc_phase("Donor1", "C5a") == "test"
    assert expected_pbmc_phase("Donor1", "PBS") == "finetune"


def test_expected_replogle_phase() -> None:
    test_genes = frozenset({"GPS1", "non-targeting"})
    assert expected_replogle_phase("k562", "GPS1", test_genes=test_genes) == "train"
    assert expected_replogle_phase("hepg2", "GPS1", test_genes=test_genes) == "test"
    assert (
        expected_replogle_phase(
            "hepg2", "non-targeting", test_genes=test_genes
        )
        == "finetune"
    )
    assert expected_replogle_phase("hepg2", "AAA", test_genes=test_genes) == "finetune"


def _write_split_h5ad(path: Path, splits: dict[str, list[str]]) -> None:
    n = len(next(iter(splits.values())))
    adata = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    for col, labels in splits.items():
        adata.obs[col] = pd.Categorical(labels)
    adata.write_h5ad(path)


def test_simplify_replogle_dump_renames_to_split(tmp_path: Path) -> None:
    path = tmp_path / "replogle.h5ad"
    _write_split_h5ad(
        path,
        {
            "gene": ["a", "b", "c"],
            "split1": ["train", "finetune", "train"],
            "split2": ["train", "finetune", "test"],
        },
    )
    report = simplify_replogle_dump(path)
    assert report["dropped"] == ["split1"]
    loaded = ad.read_h5ad(path)
    assert list(loaded.obs.columns) == ["gene", "split"]
    assert list(loaded.obs["split"]) == ["train", "finetune", "test"]


def test_simplify_pbmc_dump_folds_val_into_finetune_and_renames(tmp_path: Path) -> None:
    path = tmp_path / "pbmc.h5ad"
    _write_split_h5ad(
        path,
        {
            "donor": ["Donor1", "Donor1", "Donor2", "Donor1"],
            "cytokine": ["G-CSF", "C5a", "PBS", "PBS"],
            "split1": ["train", "train", "finetune", "train"],
            "split2": ["train", "train", "train", "train"],
            "split3": ["val", "test", "train", "finetune"],
        },
    )
    report = simplify_pbmc_dump(path)
    assert set(report["dropped"]) == {"split1", "split2"}
    loaded = ad.read_h5ad(path)
    assert "split3" not in loaded.obs.columns
    assert list(loaded.obs["split"]) == ["finetune", "test", "train", "finetune"]
    assert set(loaded.obs["split"].cat.categories) <= {"train", "finetune", "test"}


def test_resolve_pca_solver_avoids_lapack_overflow() -> None:
    assert resolve_pca_solver(500_000, 2000, 256) == "scanpy"
    assert resolve_pca_solver(6_164_993, 2000, 512) == "covariance"
    assert resolve_pca_solver(6_164_993, 50_000, 512) == "incremental"
    assert resolve_pca_solver(100, 20, 5, method="incremental") == "incremental"


def test_pca_covariance_matches_scanpy_on_small_data() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 24)).astype(np.float32)
    adata = ad.AnnData(X=x)
    idx = np.arange(adata.n_obs)
    scanpy_basis = fit_pca_basis(adata, idx, n_comps=6, seed=0, method="scanpy")
    cov_basis = fit_pca_basis(adata, idx, n_comps=6, seed=0, method="covariance")
    inc_basis = fit_pca_basis(
        adata, idx, n_comps=6, seed=0, method="incremental", batch_size=32
    )
    np.testing.assert_allclose(scanpy_basis.means, cov_basis.means, rtol=1e-5, atol=1e-5)
    for j in range(6):
        corr = np.abs(np.corrcoef(scanpy_basis.pcs[:, j], cov_basis.pcs[:, j])[0, 1])
        assert corr > 0.999, (j, corr)
    assert inc_basis.pcs.shape == scanpy_basis.pcs.shape
    inc0 = np.abs(np.corrcoef(scanpy_basis.pcs[:, 0], inc_basis.pcs[:, 0])[0, 1])
    assert inc0 > 0.85, inc0


if __name__ == "__main__":
    import tempfile

    test_recode_merge_category_maps_src_onto_dst()
    test_expected_pbmc_phase_folds_val_into_finetune()
    test_expected_replogle_phase()
    with tempfile.TemporaryDirectory() as tmp:
        test_simplify_replogle_dump_renames_to_split(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_simplify_pbmc_dump_folds_val_into_finetune_and_renames(Path(tmp))
    test_resolve_pca_solver_avoids_lapack_overflow()
    test_pca_covariance_matches_scanpy_on_small_data()
    print("ok")
