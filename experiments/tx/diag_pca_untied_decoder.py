#!/usr/bin/env python3
"""Task-0 PCA encoder, tied inverse vs per-task linear readout (no CFM).

``z = PCA_0(x)`` is frozen (task-0 train). Compare on HepG2:

* ``tied``      — inverse PCA ``z P_0 + μ_0``
* ``D0_linear`` — Adam linear map fit on task-0 train (should match tied)
* ``D1_linear`` — Adam linear map fit on HepG2 finetune only (CL ``D_1``)

Same HepG2 pools / caps / seeds as the oneshot decoder oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import numpy as np
import torch

from experiments.tx.diag_oneshot_decoder_oracle import score_pool
from experiments.tx.train_cl import fit_decoder_for_task
from trajectory_replay.data.pca_decode import (
    pca_gene_decoder_from_basis,
    pca_inverse_as_task_decoder,
)
from trajectory_replay.data.tx_cl_data import load_tx_cl_pca_arrays
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_preprocess import load_pca_basis
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    TaskDecoderBank,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--pca-basis", type=Path, default=REPO_ROOT / "data_dir/replogle/pca_basis.npz")
    p.add_argument(
        "--pca-embedding",
        type=Path,
        default=REPO_ROOT / "data_dir/replogle/pca_embedding.npy",
    )
    p.add_argument("--decoder-epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--decoder-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--decoder-lr", type=float, default=DEFAULT_LR)
    p.add_argument("--eval-max-cells-per-stratum", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/diag_pca_untied_decoder.json",
    )
    return p


def _bank(dec) -> TaskDecoderBank:
    bank = TaskDecoderBank()
    bank.add(dec)
    return bank


def decoder_ns(args: argparse.Namespace) -> Namespace:
    return Namespace(
        decoder_max_fit_cells=0,
        decoder_seed=int(args.seed),
        representation="se",
        decoder_hidden=0,
        decoder_depth=1,
        decoder_lr=float(args.decoder_lr),
        decoder_batch_size=int(args.decoder_batch_size),
        decoder_epochs=int(args.decoder_epochs),
    )


def score_label(
    label: str,
    bank: TaskDecoderBank,
    arrays,
    adata,
    hepg2_fit: np.ndarray,
    hepg2_heldout: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n== {label}", flush=True)
    fit = score_pool(
        arrays, adata, bank, hepg2_fit,
        cap=args.eval_max_cells_per_stratum,
        seed=args.seed + 30_000,
        recon_sample=2000,
    )
    held = score_pool(
        arrays, adata, bank, hepg2_heldout,
        cap=args.eval_max_cells_per_stratum,
        seed=args.seed + 30_001,
        recon_sample=2000,
        control_reference=hepg2_fit,
    )
    print(
        f"{label:<12} fitΔ={fit['oracle_pearson_delta_macro']:.4f} "
        f"heldΔ={held['oracle_pearson_delta_macro']:.4f} "
        f"held R2={held['decoder_r2']:.4f} reconP={held['per_cell_recon_pearson']:.4f}",
        flush=True,
    )
    return {"spec": label, "fit": fit, "heldout": held}


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device(args.device)
    stream = stream_for("replogle")
    arrays = load_tx_cl_pca_arrays(
        args.h5ad_path,
        stream,
        pca_path=args.pca_embedding,
        covariate_cols=(),
        control_perturbation="non-targeting",
    )
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    task0 = stream.task(0)
    task0_idx = np.flatnonzero(
        np.isin(context, list(task0.contexts)) & (split == task0.train_split)
    )
    hepg2_fit = np.flatnonzero((context == hepg2) & (split != TEST_SPLIT))
    hepg2_heldout = np.flatnonzero((context == hepg2) & (split == TEST_SPLIT))
    print(
        f"task0 train={task0_idx.size:,}  hepg2 finetune={hepg2_fit.size:,}  "
        f"hepg2 test={hepg2_heldout.size:,}  pca_dim={arrays.se_dim}",
        flush=True,
    )

    basis = load_pca_basis(args.pca_basis)
    ns = decoder_ns(args)
    results = []

    tied = pca_inverse_as_task_decoder(
        pca_gene_decoder_from_basis(basis, n_components=arrays.se_dim),
        task_id=0,
        n_fit=int(task0_idx.size),
    )
    results.append(
        score_label("tied", _bank(tied), arrays, adata, hepg2_fit, hepg2_heldout, args)
    )

    print(f"\n== fitting D0_linear on {task0_idx.size:,} task-0 train cells", flush=True)
    d0 = fit_decoder_for_task(arrays, adata, 0, task0_idx, ns, device)
    results.append(
        score_label("D0_linear", _bank(d0), arrays, adata, hepg2_fit, hepg2_heldout, args)
    )

    print(f"\n== fitting D1_linear on {hepg2_fit.size:,} HepG2 finetune cells", flush=True)
    # task_id=0 so score_pool (all-zero tids) hits this decoder in a one-entry bank
    d1 = fit_decoder_for_task(arrays, adata, 0, hepg2_fit, ns, device)
    results.append(
        score_label("D1_linear", _bank(d1), arrays, adata, hepg2_fit, hepg2_heldout, args)
    )

    payload = {
        "h5ad_path": str(args.h5ad_path),
        "pca_basis": str(args.pca_basis),
        "pca_embedding": str(args.pca_embedding),
        "n_task0": int(task0_idx.size),
        "n_hepg2_fit": int(hepg2_fit.size),
        "decoders": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
