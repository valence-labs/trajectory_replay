"""Covariate / stratum-key behavior for transcriptomics conditioning."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from trajectory_replay.data.tx_dataset import TxEncoding, build_encoding, encode_labels
from trajectory_replay.data.tx_strata import (
    control_indices_by_prefix,
    iter_observed_strata,
    parse_covariate_cols,
    stratum_mask,
    subsample_per_stratum,
)
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import _concat_z_dicts, _z_dict_to_device
from experiments.tx.tx_common import sync_encoding_args


def test_parse_covariate_cols_empty_and_multi() -> None:
    assert parse_covariate_cols("") == ()
    assert parse_covariate_cols("cell_type") == ("cell_type",)
    assert parse_covariate_cols("cell_type,batch") == ("cell_type", "batch")


def test_replogle_encoding_unchanged_without_covariates() -> None:
    enc = build_encoding(
        np.array(["A", "B"]),
        np.array(["g1", "g2", "ctrl"]),
        control_perturbation="ctrl",
    )
    assert enc.covariate_col_order == ()
    assert enc.n_covariates == 0
    c, p, z = encode_labels(
        np.array(["A", "B"]),
        np.array(["g1", "g2"]),
        enc,
    )
    assert z == {}
    model = ConditionalFlowMatching(
        data_dim=8,
        n_contexts=enc.n_contexts,
        n_perturbations=enc.n_perturbations,
        cond_emb_dim=16,
        hidden_dim=32,
        depth=2,
    )
    x = torch.randn(4, 8)
    t = torch.rand(4)
    out = model(x, t, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    assert out.shape == (4, 8)


def test_pbmc_style_stratum_keys_length_three() -> None:
    enc = build_encoding(
        np.array(["donor1", "donor1"]),
        np.array(["IL-2", "PBS"]),
        covariate_arrays={"cell_type": np.array(["CD4", "CD8"])},
        covariate_col_order=("cell_type",),
    )
    c = torch.tensor([0, 0])
    p = torch.tensor([0, 1])
    z = {"cell_type": torch.tensor([0, 1])}
    strata = iter_observed_strata(c, p, z=z, covariate_col_order=("cell_type",))
    assert len(strata) == 2
    assert all(len(key) == 3 for key in strata)

    pools = control_indices_by_prefix(
        c, p, control_pert=1, z=z, covariate_col_order=("cell_type",)
    )
    assert (0, 1) in pools  # donor0, cell_type index 1 has PBS


def test_subsample_per_stratum_respects_covariates() -> None:
    x = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    c = torch.tensor([0, 0, 0, 1, 1, 1])
    p = torch.tensor([0, 0, 1, 0, 1, 1])
    z = {"cell_type": torch.tensor([0, 1, 0, 0, 1, 1])}
    x2, c2, p2, z2 = subsample_per_stratum(
        x, c, p, z=z, covariate_col_order=("cell_type",), max_per_stratum=1, seed=0
    )
    assert len(x2) == 5  # five unique (c,p,cell_type) strata
    assert z2 is not None
    for i in range(len(c2)):
        m = stratum_mask(c2, p2, (int(c2[i]), int(p2[i]), int(z2["cell_type"][i])), z=z2, covariate_col_order=("cell_type",))
        assert m[i]


def test_encoding_json_roundtrip_with_covariates() -> None:
    enc = build_encoding(
        np.array(["d1"]),
        np.array(["p1"]),
        covariate_arrays={"cell_type": np.array(["T"])},
        covariate_col_order=("cell_type",),
    )
    restored = TxEncoding.from_jsonable(enc.to_jsonable())
    assert restored.covariate_col_order == ("cell_type",)
    assert restored.covariate_categories["cell_type"] == ("T",)


def test_concat_z_dicts_aligns_cpu_subset_to_cuda_batch() -> None:
    cols = ("cell_type",)
    z_cpu = {"cell_type": torch.tensor([0, 1], dtype=torch.long)}
    z_cuda = {"cell_type": torch.tensor([2, 3], device="cuda" if torch.cuda.is_available() else "cpu")}
    out = _concat_z_dicts(z_cpu, z_cuda, cols)
    assert out is not None
    assert out["cell_type"].device == z_cuda["cell_type"].device
    assert out["cell_type"].tolist() == [0, 1, 2, 3]


def test_z_dict_to_device() -> None:
    cols = ("cell_type",)
    z = {"cell_type": torch.tensor([0, 1], dtype=torch.long)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = _z_dict_to_device(z, cols, dev)
    assert moved is not None
    assert moved["cell_type"].device.type == dev.type


def test_sync_encoding_args_fills_covariate_cols_from_checkpoint() -> None:
    enc = build_encoding(
        np.array(["d1"]),
        np.array(["p1"]),
        covariate_arrays={"cell_type": np.array(["T"])},
        covariate_col_order=("cell_type",),
    )
    args = argparse.Namespace(covariate_cols="")
    sync_encoding_args(args, enc)
    assert args.covariate_cols == "cell_type"
