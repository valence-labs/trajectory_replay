"""Tests for Cell-Eval AnnData export helpers."""

from __future__ import annotations

import numpy as np
import torch

from experiments.tx.cell_eval_export import (
    CELL_EVAL_EXPORT_PROTOCOL,
    CELL_EVAL_GENE_MAX,
    clip_gene_matrix_for_celleval,
    context_indices_in_adata,
    contexts_missing_in_phase_adata,
    contexts_missing_controls_for_export_slices,
    decode_obsm_to_genes,
    gene_matrix_for_adata,
    load_control_reference_adata,
    merge_control_reference_adata,
    read_raw_gene_matrix,
    select_control_cells_for_stratum,
    export_slice_has_controls,
    _pert_label_series,
    subsample_indices_per_stratum,
)
from trajectory_replay.data.pca_decode import PcaGeneDecoder
from trajectory_replay.data.tx_strata import (
    default_aggregate_covariate,
    iter_aggregate_covariate_slices,
)
from trajectory_replay.data.tx_dataset import TxEncoding
from trajectory_replay.training.flow_matching import iter_observed_strata_pairs


def test_subsample_indices_matches_stratum_cap() -> None:
    c = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    p = np.array([0, 0, 1, 0, 0], dtype=np.int64)
    sel = subsample_indices_per_stratum(c, p, max_per_stratum=2, seed=0)
    assert len(sel) == 5
    assert len(np.unique(sel)) == len(sel)


def test_clip_gene_matrix_for_celleval_clips_to_state_range() -> None:
    x = np.array([[-1.5, 0.0, 2.0, 20.0]], dtype=np.float32)
    out = clip_gene_matrix_for_celleval(x)
    assert out.tolist() == [[0.0, 0.0, 2.0, CELL_EVAL_GENE_MAX]]
    assert CELL_EVAL_GENE_MAX == 14.0


def test_clip_gene_matrix_for_celleval_clips_negatives() -> None:
    x = np.array([[-1.5, 0.0, 2.0]], dtype=np.float32)
    out = clip_gene_matrix_for_celleval(x)
    assert out.tolist() == [[0.0, 0.0, 2.0]]


def test_decode_obsm_to_genes_uses_decoder_and_clips() -> None:
    import anndata as ad

    decoder = PcaGeneDecoder(
        pcs=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        mean=np.array([[-0.5, 1.0]], dtype=np.float64),
        n_components=2,
    )
    adata = ad.AnnData(
        X=np.zeros((1, 2)),
        obsm={"X_pca": np.array([[-1.0, 0.5]], dtype=np.float32)},
    )
    out = decode_obsm_to_genes(adata, decoder, "X_pca")
    assert out.shape == (1, 2)
    assert out.tolist() == [[0.0, 1.5]]


def test_iter_observed_strata_pairs_unique_pairs() -> None:
    c = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    p = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    pairs = iter_observed_strata_pairs(c, p)
    assert pairs == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_subsample_indices_unlimited_when_cap_zero() -> None:
    c = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    p = np.array([0, 0, 1, 0, 1], dtype=np.int64)
    sel = subsample_indices_per_stratum(c, p, max_per_stratum=0, seed=0)
    assert sel.tolist() == list(range(5))


def test_export_protocol_is_current() -> None:
    assert CELL_EVAL_EXPORT_PROTOCOL == 10


def test_read_raw_gene_matrix_clips_negatives() -> None:
    import anndata as ad

    adata = ad.AnnData(X=np.array([[-0.5, 1.0, 2.0]], dtype=np.float32))
    out = read_raw_gene_matrix(adata)
    assert out.tolist() == [[0.0, 1.0, 2.0]]


def test_gene_matrix_for_adata_raw_vs_decode() -> None:
    import anndata as ad

    decoder = PcaGeneDecoder(
        pcs=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        mean=np.array([[-0.5, 1.0]], dtype=np.float64),
        n_components=2,
    )
    adata = ad.AnnData(
        X=np.array([[0.25, 1.75]], dtype=np.float32),
        obsm={"X_pca": np.array([[-1.0, 0.5]], dtype=np.float32)},
    )
    raw = gene_matrix_for_adata(adata, decoder, "X_pca", use_raw_x=True)
    decoded = gene_matrix_for_adata(adata, decoder, "X_pca", use_raw_x=False)
    assert raw.tolist() == [[0.25, 1.75]]
    assert decoded.tolist() == [[0.0, 1.5]]


def test_pert_label_series_keeps_long_control_name() -> None:
    series = _pert_label_series(["ATP6V1G1", "non-targeting"])
    assert series.iloc[1] == "non-targeting"
    assert series.dtype == object


def test_context_indices_in_adata_filters_labels() -> None:
    import anndata as ad

    enc = TxEncoding(
        context_categories=("A", "B"),
        perturbation_categories=("PBS", "IL-2"),
        context_to_idx={"A": 0, "B": 1},
        perturbation_to_idx={"PBS": 0, "IL-2": 1},
        control_perturbation="PBS",
    )
    adata = ad.AnnData(
        X=np.zeros((3, 2)),
        obs={"donor": ["A", "A", "A"], "cytokine": ["PBS", "IL-2", "IL-2"]},
    )
    idx = context_indices_in_adata(adata, enc, "donor", context_labels=("A",))
    assert idx == [0]
    idx_all = context_indices_in_adata(adata, enc, "donor")
    assert idx_all == [0]


def test_contexts_missing_in_phase_adata_with_covariates() -> None:
    import anndata as ad

    enc = TxEncoding(
        context_categories=("Donor1", "Donor2"),
        perturbation_categories=("PBS", "IL-2"),
        context_to_idx={"Donor1": 0, "Donor2": 1},
        perturbation_to_idx={"PBS": 0, "IL-2": 1},
        covariate_col_order=("cell_type",),
        covariate_categories={"cell_type": ("CD4", "CD8")},
        covariate_to_idx={"cell_type": {"CD4": 0, "CD8": 1}},
        control_perturbation="PBS",
    )
    adata = ad.AnnData(
        X=np.zeros((4, 2)),
        obs={
            "donor": ["Donor1", "Donor1", "Donor2", "Donor2"],
            "cytokine": ["IL-2", "IL-2", "IL-2", "PBS"],
            "cell_type": ["CD4", "CD8", "CD4", "CD4"],
        },
    )
    missing = contexts_missing_in_phase_adata(
        adata,
        enc,
        control_pert=0,
        context_col="donor",
        perturbation_col="cytokine",
    )
    assert missing == frozenset({0})


def test_contexts_missing_controls_for_export_slices_per_cell_type() -> None:
    import anndata as ad

    enc = TxEncoding(
        context_categories=("Donor1",),
        perturbation_categories=("PBS", "IL-2"),
        context_to_idx={"Donor1": 0},
        perturbation_to_idx={"PBS": 0, "IL-2": 1},
        covariate_col_order=("cell_type",),
        covariate_categories={"cell_type": ("CD4", "CD8")},
        covariate_to_idx={"cell_type": {"CD4": 0, "CD8": 1}},
        control_perturbation="PBS",
    )
    # Donor1 has PBS for CD4 but not CD8 — context-level check would miss this.
    adata = ad.AnnData(
        X=np.zeros((3, 2)),
        obs={
            "donor": ["Donor1", "Donor1", "Donor1"],
            "cytokine": ["PBS", "IL-2", "IL-2"],
            "cell_type": ["CD4", "CD4", "CD8"],
        },
    )
    missing = contexts_missing_controls_for_export_slices(
        adata,
        enc,
        control_pert=0,
        context_col="donor",
        perturbation_col="cytokine",
    )
    assert missing == frozenset({0})


def test_replogle_test_loads_control_reference_from_finetune() -> None:
    from pathlib import Path

    h5ad = Path(__file__).resolve().parents[1] / "data_dir/replogle_with_splits.h5ad"
    if not h5ad.is_file():
        return
    from trajectory_replay.data.tx_dataset import build_encoding_from_h5ad
    from experiments.tx.cell_eval_export import load_phase_adata

    enc = build_encoding_from_h5ad(
        h5ad,
        context_col="cell_line",
        perturbation_col="gene",
        control_perturbation="non-targeting",
    )
    ctrl = enc.control_pert_index
    assert ctrl is not None
    finetune = load_phase_adata(h5ad, split_col="split", phase="finetune")
    finetune_missing = contexts_missing_in_phase_adata(
        finetune,
        enc,
        ctrl,
        context_col="cell_line",
        perturbation_col="gene",
    )
    assert not finetune_missing

    test = load_phase_adata(h5ad, split_col="split", phase="test")
    test_missing = contexts_missing_in_phase_adata(
        test,
        enc,
        ctrl,
        context_col="cell_line",
        perturbation_col="gene",
    )
    assert test_missing

    ref = load_control_reference_adata(
        h5ad, split_col="split", reference_phases=("finetune",)
    )
    assert ref is not None
    hepg2_ctrl = (
        (ref.obs["cell_line"].astype(str) == "hepg2")
        & (ref.obs["gene"].astype(str) == "non-targeting")
    ).sum()
    assert hepg2_ctrl > 0


def test_select_control_cells_requires_same_cell_type() -> None:
    import anndata as ad
    import pandas as pd

    encoding = TxEncoding(
        context_categories=("Donor5",),
        perturbation_categories=("PBS", "IL-2"),
        context_to_idx={"Donor5": 0},
        perturbation_to_idx={"PBS": 0, "IL-2": 1},
        control_perturbation="PBS",
        covariate_col_order=("cell_type",),
        covariate_categories={"cell_type": ("Plasmablast", "NKT")},
        covariate_to_idx={"cell_type": {"Plasmablast": 0, "NKT": 1}},
    )
    obs = pd.DataFrame(
        {
            "donor": ["Donor5", "Donor5", "Donor5"],
            "cytokine": ["IL-2", "PBS", "PBS"],
            "cell_type": ["Plasmablast", "NKT", "NKT"],
        }
    )
    adata = ad.AnnData(X=np.zeros((3, 2)), obs=obs)
    picked = select_control_cells_for_stratum(
        adata,
        context_label="Donor5",
        control_perturbation="PBS",
        context_col="donor",
        perturbation_col="cytokine",
        encoding=encoding,
        covariate_indices={"cell_type": 0},
    )
    assert picked is None
    assert not export_slice_has_controls(
        adata,
        context_label="Donor5",
        control_perturbation="PBS",
        context_col="donor",
        perturbation_col="cytokine",
        encoding=encoding,
        covariate_indices={"cell_type": 0},
    )
    nkt = select_control_cells_for_stratum(
        adata,
        context_label="Donor5",
        control_perturbation="PBS",
        context_col="donor",
        perturbation_col="cytokine",
        encoding=encoding,
        covariate_indices={"cell_type": 1},
    )
    assert nkt is not None and nkt.n_obs == 2


def test_default_aggregate_covariate_pbmc_and_replogle() -> None:
    assert default_aggregate_covariate(
        context_col="donor",
        covariate_col_order=("cell_type",),
    ) == "cell_type"
    assert default_aggregate_covariate(
        context_col="cell_line",
        covariate_col_order=(),
    ) == "cell_line"


def test_iter_aggregate_covariate_slices_pools_donors_per_cell_type() -> None:
    import anndata as ad

    enc = TxEncoding(
        context_categories=("Donor1", "Donor2"),
        perturbation_categories=("PBS", "IL-2"),
        context_to_idx={"Donor1": 0, "Donor2": 1},
        perturbation_to_idx={"PBS": 0, "IL-2": 1},
        covariate_col_order=("cell_type",),
        covariate_categories={"cell_type": ("CD4", "CD8")},
        covariate_to_idx={"cell_type": {"CD4": 0, "CD8": 1}},
        control_perturbation="PBS",
    )
    adata = ad.AnnData(
        X=np.zeros((4, 2)),
        obs={
            "donor": ["Donor1", "Donor2", "Donor1", "Donor2"],
            "cytokine": ["IL-2", "IL-2", "IL-2", "IL-2"],
            "cell_type": ["CD4", "CD4", "CD8", "CD8"],
        },
    )
    slices = iter_aggregate_covariate_slices(
        adata,
        enc,
        aggregate_covariate="cell_type",
        context_col="donor",
    )
    assert len(slices) == 2
    assert all(ci is None for ci, _, _ in slices)
    assert {suffix for _, _, suffix in slices} == {"CD4", "CD8"}


def test_select_control_cells_pool_contexts() -> None:
    import anndata as ad
    import pandas as pd

    encoding = TxEncoding(
        context_categories=("Donor1", "Donor2"),
        perturbation_categories=("PBS", "IL-2"),
        context_to_idx={"Donor1": 0, "Donor2": 1},
        perturbation_to_idx={"PBS": 0, "IL-2": 1},
        control_perturbation="PBS",
        covariate_col_order=("cell_type",),
        covariate_categories={"cell_type": ("CD4", "CD8")},
        covariate_to_idx={"cell_type": {"CD4": 0, "CD8": 1}},
    )
    obs = pd.DataFrame(
        {
            "donor": ["Donor1", "Donor2", "Donor2"],
            "cytokine": ["PBS", "PBS", "IL-2"],
            "cell_type": ["CD4", "CD4", "CD8"],
        }
    )
    adata = ad.AnnData(X=np.zeros((3, 2)), obs=obs)
    picked = select_control_cells_for_stratum(
        adata,
        context_label="CD4",
        control_perturbation="PBS",
        context_col="donor",
        perturbation_col="cytokine",
        encoding=encoding,
        covariate_indices={"cell_type": 0},
        pool_contexts=True,
    )
    assert picked is not None and picked.n_obs == 2
