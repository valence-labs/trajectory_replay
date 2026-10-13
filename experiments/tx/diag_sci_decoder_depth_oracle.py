#!/usr/bin/env python3
"""Oracle Pearson-Δ for SCimilarity vs decoder depth (no CFM).

Fits one decoder per architecture on all non-test cells, then scores the
same HepG2 fit/heldout pools as the one-shot oracle (cap 500, same seeds).
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
from trajectory_replay.data.tx_cl_data import (
    apply_latent_radial_scale,
    apply_latent_standardizer,
    fit_latent_standardizer,
    load_tx_cl_arrays,
)
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    TaskDecoderBank,
)

DEFAULT_SPECS = ("linear", "256x1", "512x1", "1024x1", "1024x2", "1024x3")


def parse_spec(spec: str) -> tuple[int, int]:
    name = spec.strip().lower()
    if name in ("linear", "lin", "0"):
        return 0, 1
    if "x" not in name:
        raise ValueError(f"spec must be 'linear' or '{{hidden}}x{{depth}}', got {spec!r}")
    hidden_s, depth_s = name.split("x", 1)
    return int(hidden_s), int(depth_s)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument("--se-path", type=Path, default=REPO_ROOT / "data_dir/replogle/scimilarity.npy")
    p.add_argument("--specs", nargs="+", default=list(DEFAULT_SPECS))
    p.add_argument("--standardize-latent", action="store_true")
    p.add_argument(
        "--radial-scale-latent",
        action="store_true",
        help="Multiply rows by sqrt(d). Mutually exclusive with --standardize-latent.",
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
        default=REPO_ROOT / "results_dir/tx_results/replogle/diag_sci_decoder_depth_oracle.json",
    )
    return p


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device(args.device)
    arrays = load_tx_cl_arrays(
        args.h5ad_path,
        args.se_path,
        stream_for("replogle"),
        covariate_cols=(),
        control_perturbation="non-targeting",
    )
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    train_idx = np.flatnonzero(split != TEST_SPLIT)
    hepg2_fit = np.flatnonzero((context == REPLOGLE_FEWSHOT_CELL_LINE) & (split != TEST_SPLIT))
    hepg2_heldout = np.flatnonzero((context == REPLOGLE_FEWSHOT_CELL_LINE) & (split == TEST_SPLIT))
    if args.standardize_latent and args.radial_scale_latent:
        raise ValueError("Use only one of --standardize-latent and --radial-scale-latent")
    if args.radial_scale_latent:
        stats = apply_latent_radial_scale(arrays.se)
        print(
            f"radial radius={stats['radius']:.3f} "
            f"row-norm {stats['pre_row_norm_mean']:.3f} -> {stats['post_row_norm_mean']:.3f}",
            flush=True,
        )
    elif args.standardize_latent:
        mean, std = fit_latent_standardizer(arrays.se, train_idx)
        stats = apply_latent_standardizer(arrays.se, mean, std)
        print(
            f"whitened std_median={stats['std_median']:.5f} "
            f"row-norm={stats['post_row_norm_mean']:.3f}",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    print(
        f"{'spec':<10} {'fitΔ':>8} {'heldΔ':>8} {'fit R2':>8} {'held R2':>8} "
        f"{'fit reconP':>10} {'held reconP':>11}",
        flush=True,
    )
    for spec in args.specs:
        hidden, depth = parse_spec(spec)
        ns = Namespace(
            decoder_max_fit_cells=0,
            decoder_seed=int(args.seed),
            representation="scimilarity",
            decoder_hidden=hidden,
            decoder_depth=depth,
            decoder_lr=float(args.decoder_lr),
            decoder_batch_size=int(args.decoder_batch_size),
            decoder_epochs=int(args.decoder_epochs),
        )
        label = "linear" if hidden <= 0 else f"{hidden}x{depth}"
        print(f"\n== fitting {label} on {train_idx.size:,} cells", flush=True)
        dec = fit_decoder_for_task(arrays, adata, 0, train_idx, ns, device)
        bank = TaskDecoderBank()
        bank.add(dec)
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
        row = {
            "spec": label,
            "decoder_hidden": hidden,
            "decoder_depth": depth,
            "fit": fit,
            "heldout": held,
        }
        results.append(row)
        print(
            f"{label:<10} {fit['oracle_pearson_delta_macro']:8.4f} "
            f"{held['oracle_pearson_delta_macro']:8.4f} "
            f"{fit['decoder_r2']:8.4f} {held['decoder_r2']:8.4f} "
            f"{fit['per_cell_recon_pearson']:10.4f} {held['per_cell_recon_pearson']:11.4f}",
            flush=True,
        )

    payload = {
        "se_path": str(args.se_path),
        "standardize_latent": bool(args.standardize_latent),
        "radial_scale_latent": bool(args.radial_scale_latent),
        "decoder_epochs": int(args.decoder_epochs),
        "n_train": int(train_idx.size),
        "architectures": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
