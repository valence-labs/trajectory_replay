#!/usr/bin/env python3
"""Score existing tied-affine PCA decoder banks on observed PCs (no CFM).

HepG2 fit/heldout use D1. T0 train uses D0 as a sanity recon check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import numpy as np

from experiments.tx.diag_oneshot_decoder_oracle import score_pool
from experiments.tx.train_cl import DEFAULTS
from trajectory_replay.data.tx_cl_data import load_tx_cl_pca_arrays
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_preprocess import pca_sidecar_names
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--widths", nargs="+", type=int, default=[50, 100])
    p.add_argument("--eval-max-cells-per-stratum", type=int, default=500)
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/diag_pca_affine_bank_oracle.json",
    )
    return p


def main() -> None:
    args = make_parser().parse_args()
    defaults = DEFAULTS["replogle"]
    h5ad_path = defaults["h5ad_pca"]
    root = Path(defaults["pca"]).parent
    bucket = defaults["bucket"]
    stream = stream_for("replogle")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    cap = int(args.eval_max_cells_per_stratum)
    rows = []
    for d in args.widths:
        basis_name, emb_name = pca_sidecar_names(d)
        decoder_dir = bucket / f"decoders_pca{d}_affine"
        print(f"\n== d={d} bank={decoder_dir}", flush=True)
        arrays = load_tx_cl_pca_arrays(
            h5ad_path,
            stream,
            covariate_cols=(),
            control_perturbation=str(defaults["control"]),
            pca_path=root / emb_name,
        )
        bank = TaskDecoderBank.load_dir(decoder_dir)
        split = np.asarray(arrays.split).astype(str)
        context = np.asarray(arrays.context).astype(str)
        t0 = stream.task(0)
        t0_train = np.flatnonzero(
            np.isin(context, list(t0.contexts)) & (split == t0.train_split)
        )
        hepg2_fit = np.flatnonzero(
            (context == REPLOGLE_FEWSHOT_CELL_LINE) & (split != TEST_SPLIT)
        )
        hepg2_held = np.flatnonzero(
            (context == REPLOGLE_FEWSHOT_CELL_LINE) & (split == TEST_SPLIT)
        )
        print("  T0 train D0", flush=True)
        t0_scored = score_pool(
            arrays, adata, bank, t0_train, cap=cap, seed=42, recon_sample=2000, task_id=0,
        )
        print("  HepG2 fit D1", flush=True)
        fit = score_pool(
            arrays, adata, bank, hepg2_fit, cap=cap, seed=42 + 30_000, recon_sample=2000, task_id=1,
        )
        print("  HepG2 heldout D1", flush=True)
        held = score_pool(
            arrays, adata, bank, hepg2_held, cap=cap, seed=42 + 30_001, recon_sample=2000,
            control_reference=hepg2_fit, task_id=1,
        )
        rows.append(
            {
                "n_comps": int(d),
                "decoder_dir": str(decoder_dir),
                "t0_train_d0": t0_scored,
                "fit_d1": fit,
                "heldout_d1": held,
            }
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"widths": rows}, indent=2) + "\n")
    print(f"\nWrote {args.out_json}", flush=True)
    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
