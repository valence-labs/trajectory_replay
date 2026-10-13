#!/usr/bin/env python3
"""Task-0 PCA encoder, tied inverse vs per-task linear readout (no CFM).

Latents are the frozen task-0 projection. Compare:

* ``inverse_pca`` — decode with ``P0.T`` (current oneshot)
* ``linear_hepg2`` — ``D1`` fit on HepG2 non-test only (CL-style)
* ``linear_nontest`` — one linear map on all non-test cells (oneshot-style)

Same HepG2 fit/heldout pools, cap 500, seeds as the Sci decoder oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
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
        default=REPO_ROOT / "results_dir/tx_results/replogle/diag_pca_untied_decoder_oracle.json",
    )
    return p


def _ns(args: argparse.Namespace):
    from argparse import Namespace

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


def _print_row(label: str, fit: dict[str, Any], held: dict[str, Any]) -> None:
    print(
        f"{label:<18} {fit['oracle_pearson_delta_macro']:8.4f} "
        f"{held['oracle_pearson_delta_macro']:8.4f} "
        f"{fit['decoder_r2']:8.4f} {held['decoder_r2']:8.4f} "
        f"{fit['per_cell_recon_pearson']:10.4f} {held['per_cell_recon_pearson']:11.4f}",
        flush=True,
    )


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device(args.device)
    arrays = load_tx_cl_pca_arrays(
        args.h5ad_path,
        stream_for("replogle"),
        covariate_cols=(),
        control_perturbation="non-targeting",
        pca_path=args.pca_embedding,
    )
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    train_idx = np.flatnonzero(split != TEST_SPLIT)
    hepg2_fit = np.flatnonzero(
        (context == REPLOGLE_FEWSHOT_CELL_LINE) & (split != TEST_SPLIT)
    )
    hepg2_heldout = np.flatnonzero(
        (context == REPLOGLE_FEWSHOT_CELL_LINE) & (split == TEST_SPLIT)
    )
    cap = int(args.eval_max_cells_per_stratum)
    ns = _ns(args)

    print(
        f"n_pca={arrays.se_dim}  nontest={train_idx.size:,}  "
        f"hepg2_fit={hepg2_fit.size:,}  hepg2_test={hepg2_heldout.size:,}",
        flush=True,
    )
    print(
        f"{'spec':<18} {'fitΔ':>8} {'heldΔ':>8} {'fit R2':>8} {'held R2':>8} "
        f"{'fit reconP':>10} {'held reconP':>11}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    variants: list[tuple[str, object]] = [
        (
            "inverse_pca",
            pca_inverse_as_task_decoder(
                pca_gene_decoder_from_basis(
                    load_pca_basis(args.pca_basis), n_components=arrays.se_dim
                ),
                task_id=0,
                n_fit=int(train_idx.size),
            ),
        ),
    ]
    print("\n== fitting linear D on HepG2 non-test (D1)", flush=True)
    variants.append(
        ("linear_hepg2", fit_decoder_for_task(arrays, adata, 0, hepg2_fit, ns, device))
    )
    print("\n== fitting linear D on all non-test", flush=True)
    variants.append(
        ("linear_nontest", fit_decoder_for_task(arrays, adata, 0, train_idx, ns, device))
    )

    for label, dec in variants:
        bank = TaskDecoderBank()
        bank.add(dec)
        fit = score_pool(
            arrays, adata, bank, hepg2_fit,
            cap=cap, seed=args.seed + 30_000, recon_sample=2000,
        )
        held = score_pool(
            arrays, adata, bank, hepg2_heldout,
            cap=cap, seed=args.seed + 30_001, recon_sample=2000,
            control_reference=hepg2_fit,
        )
        results.append({"spec": label, "fit": fit, "heldout": held})
        _print_row(label, fit, held)

    payload = {
        "pca_basis": str(args.pca_basis),
        "pca_embedding": str(args.pca_embedding),
        "n_pca": int(arrays.se_dim),
        "n_hepg2_fit": int(hepg2_fit.size),
        "n_nontest": int(train_idx.size),
        "architectures": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
