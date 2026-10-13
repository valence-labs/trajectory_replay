#!/usr/bin/env python3
"""Precompute the shared per-task decoder bank for Tx sequential CL.

SE representation fits ``nn.Linear`` SE→genes. Gene representation writes
identity decoders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import torch

from experiments.tx.train_cl import (
    DEFAULTS,
    cl_representation,
    embedding_sidecar_path,
    load_cl_arrays,
    parse_covariates,
    prepare_decoder_bank,
    resolve_pca_artifacts,
    shared_decoder_dirname,
)
from trajectory_replay.data.tx_task_decoder import (
    DEFAULT_BATCH_SIZE as DECODER_BATCH_SIZE,
    DEFAULT_DEPTH as DECODER_DEPTH,
    DEFAULT_EPOCHS as DECODER_EPOCHS,
    DEFAULT_HIDDEN as DECODER_HIDDEN,
    DEFAULT_LR as DECODER_LR,
)
from trajectory_replay.data.tx_cl_tasks import stream_for


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("pbmc", "replogle"), required=True)
    p.add_argument("--h5ad-path", type=Path)
    p.add_argument("--se-path", type=Path)
    p.add_argument(
        "--representation",
        choices=("se", "genes", "scimilarity", "pca"),
        default="se",
        help="CFM data space; genes uses identity; pca uses tied inverse + per-task affine.",
    )
    p.add_argument("--pca-dim", type=int, default=0)
    p.add_argument("--pca-basis", type=Path)
    p.add_argument("--decoder-dir", type=Path)
    p.add_argument(
        "--require-existing-decoders",
        action="store_true",
        help="Verify a complete compatible bank; never fit or overwrite decoders.",
    )
    p.add_argument("--covariate-cols")
    p.add_argument("--control-perturbation")
    p.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    p.add_argument("--decoder-lr", type=float, default=DECODER_LR)
    p.add_argument("--decoder-batch-size", type=int, default=DECODER_BATCH_SIZE)
    p.add_argument(
        "--decoder-hidden",
        type=int,
        default=DECODER_HIDDEN,
        help="Hidden width of the GELU MLP decoder; 0 keeps it linear.",
    )
    p.add_argument(
        "--decoder-depth",
        type=int,
        default=DECODER_DEPTH,
        help="Number of hidden layers when --decoder-hidden > 0.",
    )
    p.add_argument("--decoder-max-fit-cells", type=int, default=0)
    p.add_argument("--decoder-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = make_parser().parse_args()
    defaults = DEFAULTS[args.dataset]
    stream = stream_for(args.dataset)
    representation = cl_representation(args)
    if representation == "pca":
        h5ad_path = args.h5ad_path or defaults.get("h5ad_pca") or defaults["h5ad"]
        se_path, args.pca_basis, expected_pca_dim = resolve_pca_artifacts(
            defaults,
            pca_dim=int(getattr(args, "pca_dim", 0) or 0) or None,
            se_path=args.se_path,
            pca_basis=args.pca_basis,
        )
    else:
        h5ad_path = args.h5ad_path or defaults["h5ad"]
        se_path = embedding_sidecar_path(defaults, representation, args.se_path)
        expected_pca_dim = None
    covariates = parse_covariates(args.covariate_cols, defaults["covariates"])
    control = args.control_perturbation or defaults["control"]
    arrays = load_cl_arrays(
        h5ad_path,
        se_path,
        stream,
        covariate_cols=covariates,
        control_perturbation=control,
        representation=representation,
    )
    if representation == "pca":
        if expected_pca_dim is not None and int(arrays.se_dim) != int(expected_pca_dim):
            raise ValueError(
                f"PCA sidecar {se_path} has dim {arrays.se_dim}, expected {expected_pca_dim}"
            )
        args.pca_dim = int(arrays.se_dim)
    decoder_dir = args.decoder_dir or defaults["bucket"] / shared_decoder_dirname(
        representation,
        args.decoder_hidden,
        args.decoder_depth,
        pca_dim=int(arrays.se_dim) if representation == "pca" else 0,
    )
    adata = ad.read_h5ad(h5ad_path, backed="r")
    bank = prepare_decoder_bank(
        decoder_dir,
        arrays,
        adata,
        stream,
        Path(h5ad_path),
        Path(se_path) if se_path is not None else None,
        args,
        torch.device(args.device),
    )
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Prepared {len(bank)} task decoders in {decoder_dir}", flush=True)


if __name__ == "__main__":
    main()
