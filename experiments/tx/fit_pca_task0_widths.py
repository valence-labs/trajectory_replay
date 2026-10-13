#!/usr/bin/env python3
"""Fit task-0 PCA at several widths and optionally score tied-inverse oracles.

Writes ``pca{d}_basis.npz`` / ``pca{d}_embedding.npy`` (256 is unlabeled
``pca_*``). ``--nested`` fits the widest SVD once and slices leading PCs, so
50/100/256/512 are the same T0 basis.

Oracle Pearson-Δ uses the few-shot contexts' non-test vs STATE-test pools
(HepG2 on Replogle; Donor1/4/9/12 on PBMC). Skip it with ``--skip-oracle``.
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
from trajectory_replay.data.pca_decode import (
    pca_gene_decoder_from_basis,
    pca_inverse_as_task_decoder,
)
from trajectory_replay.data.tx_cl_data import load_tx_cl_pca_arrays
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_preprocess import (
    DEFAULT_PCA_DIM,
    fit_pca_basis,
    load_pca_basis,
    pca_sidecar_names,
    project_pca,
    reconstruction_pearson,
    save_pca_basis,
    slice_pca_basis,
)
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("replogle", "pbmc"), default="replogle")
    p.add_argument("--h5ad-path", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--widths", nargs="+", type=int, default=[512, 1024, 2048])
    p.add_argument(
        "--nested",
        action="store_true",
        help="Fit max(widths) once and write each width as the leading PCs.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-max-cells-per-stratum", type=int, default=500)
    p.add_argument("--out-json", type=Path)
    p.add_argument("--skip-fit", action="store_true", help="Score existing sidecars only.")
    p.add_argument(
        "--slice-from",
        type=int,
        default=0,
        help="Load existing pca{d}_* (or unlabeled pca_* if d=256) and write "
        "--widths as leading columns. Does not refit and will not overwrite "
        "the unlabeled 256 dump.",
    )
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument(
        "--pca-method",
        choices=("auto", "scanpy", "covariance", "incremental"),
        default="auto",
        help="auto: scanpy if n_obs*n_comps is LAPACK-safe, else gene-gene "
        "covariance (PBMC), else IncrementalPCA.",
    )
    p.add_argument("--pca-batch-size", type=int, default=8192)
    return p


def fewshot_eval_indices(arrays) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    stream = arrays.stream
    ctxs = tuple(c for task in stream.tasks[1:] for c in task.contexts)
    spl = np.asarray(arrays.split).astype(str)
    ctx = np.asarray(arrays.context).astype(str)
    in_few = np.isin(ctx, list(ctxs))
    fit = np.flatnonzero(in_few & (spl != TEST_SPLIT))
    held = np.flatnonzero(in_few & (spl == TEST_SPLIT))
    return fit, held, ctxs


def score_tied_with_basis(basis, arrays, adata, fit_idx, held_idx, cap, seed):
    tied = pca_inverse_as_task_decoder(
        pca_gene_decoder_from_basis(basis, n_components=arrays.se_dim),
        task_id=0,
        n_fit=int(fit_idx.size),
    )
    bank = TaskDecoderBank()
    bank.add(tied)
    fit = score_pool(
        arrays, adata, bank, fit_idx,
        cap=cap, seed=seed + 30_000, recon_sample=2000,
    )
    held = score_pool(
        arrays, adata, bank, held_idx,
        cap=cap, seed=seed + 30_001, recon_sample=2000,
        control_reference=fit_idx,
    )
    return {"fit": fit, "heldout": held}


def _write_width(out_dir: Path, d: int, basis, x_pca: np.ndarray) -> tuple[Path, Path]:
    slim = slice_pca_basis(basis, d)
    basis_name, emb_name = pca_sidecar_names(d)
    basis_path = save_pca_basis(out_dir / basis_name, slim)
    emb_path = out_dir / emb_name
    np.save(emb_path, np.ascontiguousarray(x_pca[:, :d], dtype=np.float32))
    return basis_path, emb_path


def main() -> None:
    args = make_parser().parse_args()
    defaults = DEFAULTS[args.dataset]
    args.h5ad_path = args.h5ad_path or defaults.get("h5ad_pca") or defaults["h5ad"]
    args.out_dir = args.out_dir or Path(defaults["pca"]).parent
    args.out_json = args.out_json or (
        defaults["bucket"] / "diag_pca_width_oracle.json"
    )
    stream = stream_for(args.dataset)
    adata_fit = ad.read_h5ad(args.h5ad_path, backed="r")
    split = np.asarray(adata_fit.obs["split"]).astype(str)
    context = np.asarray(adata_fit.obs[stream.context_col]).astype(str)
    task0 = stream.task(0)
    fit_idx = np.flatnonzero(
        np.isin(context, list(task0.contexts)) & (split == task0.train_split)
    )
    if fit_idx.size == 0:
        raise SystemExit(
            f"No task-0 cells: contexts={task0.contexts} split={task0.train_split}"
        )
    future = tuple(c for c in np.unique(context.astype(str)) if c not in task0.contexts)
    if np.isin(context[fit_idx], list(future)).any():
        raise AssertionError("future-task contexts in the PCA fit set")
    print(
        f"dataset={args.dataset} task-0 train cells for PCA fit: {fit_idx.size:,} "
        f"excluded={future}",
        flush=True,
    )

    adata_score = None if args.skip_oracle else ad.read_h5ad(args.h5ad_path, backed="r")
    rows = []
    widths = sorted({int(d) for d in args.widths})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slice_from = int(args.slice_from or 0)
    if slice_from and args.nested:
        raise SystemExit("--slice-from is mutually exclusive with --nested")
    if slice_from and DEFAULT_PCA_DIM in widths:
        raise SystemExit(
            f"--slice-from refuses to write unlabeled {DEFAULT_PCA_DIM}-PC pca_*"
        )

    existing_basis, existing_emb = pca_sidecar_names(DEFAULT_PCA_DIM)
    existing_256 = args.out_dir / existing_emb
    if (
        not args.nested
        and not slice_from
        and existing_256.is_file()
        and DEFAULT_PCA_DIM not in widths
        and args.dataset == "replogle"
    ):
        widths = [DEFAULT_PCA_DIM, *widths]

    nested_basis = None
    nested_x = None
    if slice_from:
        src_basis_name, src_emb_name = pca_sidecar_names(slice_from)
        src_basis_path = args.out_dir / src_basis_name
        src_emb_path = args.out_dir / src_emb_name
        if not src_basis_path.is_file() or not src_emb_path.is_file():
            raise SystemExit(f"missing source sidecar {src_basis_path} or {src_emb_path}")
        print(f"\n== slice-from d={slice_from} {src_emb_name}", flush=True)
        nested_basis = load_pca_basis(src_basis_path)
        nested_x = np.load(src_emb_path, mmap_mode="r")
        if int(nested_basis.n_comps) < max(widths):
            raise SystemExit(
                f"source d={nested_basis.n_comps} is narrower than max(widths)={max(widths)}"
            )
        if int(nested_x.shape[1]) < max(widths):
            raise SystemExit(
                f"source embedding width {nested_x.shape[1]} < max(widths)={max(widths)}"
            )
    elif args.nested and not args.skip_fit:
        max_d = max(widths)
        print(f"\n== nested task-0 PCA d={max_d} on {fit_idx.size:,} cells", flush=True)
        nested_basis = fit_pca_basis(
            adata_fit, fit_idx, n_comps=max_d, seed=args.seed,
            fit_phases=(task0.train_split,),
            method=args.pca_method,
            batch_size=args.pca_batch_size,
        )
        nested_x = project_pca(adata_fit, nested_basis)
        recon = float(reconstruction_pearson(adata_fit, nested_basis, nested_x, seed=args.seed))
        print(
            f"  cumvar={float(nested_basis.variance_ratio.sum()):.4f} reconP={recon:.4f}",
            flush=True,
        )

    for d in widths:
        basis_name, emb_name = pca_sidecar_names(d)
        basis_path = args.out_dir / basis_name
        emb_path = args.out_dir / emb_name
        if nested_basis is not None and (
            slice_from or (args.nested and not args.skip_fit)
        ):
            src = f"slice-from {slice_from}" if slice_from else f"nested {nested_basis.n_comps}-PC SVD"
            print(f"\n== d={d} {src}", flush=True)
            basis_path, emb_path = _write_width(args.out_dir, d, nested_basis, nested_x)
            basis = slice_pca_basis(nested_basis, d)
            recon = float(
                reconstruction_pearson(
                    adata_fit, basis, nested_x[:, :d], seed=args.seed,
                )
            )
            print(
                f"  wrote {basis_path.name} cumvar={float(basis.variance_ratio.sum()):.4f} "
                f"reconP={recon:.4f}",
                flush=True,
            )
        elif (
            not args.nested
            and d == DEFAULT_PCA_DIM
            and existing_256.is_file()
            and (args.out_dir / existing_basis).is_file()
            and args.dataset == "replogle"
        ):
            print(f"\n== d={d} existing sidecar {emb_path.name}", flush=True)
            basis = load_pca_basis(basis_path)
        elif args.skip_fit:
            if not basis_path.is_file() or not emb_path.is_file():
                raise FileNotFoundError(f"missing {basis_path} or {emb_path}")
            print(f"\n== d={d} loading {emb_path.name}", flush=True)
            basis = load_pca_basis(basis_path)
        else:
            print(f"\n== fitting task-0 PCA d={d} on {fit_idx.size:,} cells", flush=True)
            basis = fit_pca_basis(
                adata_fit, fit_idx, n_comps=d, seed=args.seed,
                fit_phases=(task0.train_split,),
                method=args.pca_method,
                batch_size=args.pca_batch_size,
            )
            x_pca = project_pca(adata_fit, basis)
            save_pca_basis(basis_path, basis)
            np.save(emb_path, np.asarray(x_pca, dtype=np.float32))
            recon = float(reconstruction_pearson(adata_fit, basis, x_pca, seed=args.seed))
            print(
                f"  wrote {basis_path.name} cumvar={float(basis.variance_ratio.sum()):.4f} "
                f"reconP={recon:.4f}",
                flush=True,
            )

        row = {
            "n_comps": int(d),
            "basis_path": str(basis_path),
            "embedding_path": str(emb_path),
            "n_fit_cells": int(basis.n_fit_cells),
            "cumulative_variance_ratio": float(basis.variance_ratio.sum()),
            "nested": bool(args.nested),
        }
        if not args.skip_oracle:
            arrays = load_tx_cl_pca_arrays(
                args.h5ad_path,
                stream,
                covariate_cols=tuple(defaults["covariates"]),
                control_perturbation=str(defaults["control"]),
                pca_path=emb_path,
            )
            few_fit, few_held, few_ctx = fewshot_eval_indices(arrays)
            scored = score_tied_with_basis(
                basis, arrays, adata_score, few_fit, few_held,
                cap=args.eval_max_cells_per_stratum, seed=42,
            )
            gap = (
                scored["fit"]["oracle_pearson_delta_macro"]
                - scored["heldout"]["oracle_pearson_delta_macro"]
            )
            print(
                f"  d={d} eval_ctx={few_ctx} fitΔ={scored['fit']['oracle_pearson_delta_macro']:.4f} "
                f"heldΔ={scored['heldout']['oracle_pearson_delta_macro']:.4f} "
                f"gap={gap:+.4f}",
                flush=True,
            )
            row.update(scored)
        rows.append(row)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"dataset": args.dataset, "nested": bool(args.nested), "widths": rows}, indent=2)
        + "\n"
    )
    print(f"\nWrote {args.out_json}", flush=True)
    for obj in (adata_fit, adata_score):
        if obj is not None and getattr(obj, "file", None) is not None:
            obj.file.close()


if __name__ == "__main__":
    main()
