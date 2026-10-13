#!/usr/bin/env python3
"""Training-available gene-blind baselines for Replogle HepG2 heldout.

Two generic (same-for-all-strata) predictors, built only from labels a model
can see during sequential CL:

* ``hepg2_fit_mean`` — mean knockdown delta of HepG2 *finetune* genes
* ``t0_heldout_genes_mean`` — mean T0-train delta of the *same gene names*
  that appear in the HepG2 STATE-test split

Also scores the transductive leave-one-out mean of heldout deltas (uses heldout
labels; not a fair baseline) and a per-gene T0 transfer predictor (gene-specific,
still training-available).

L1 discrimination is the pairwise ranking score: for each true profile i, the
fraction of other predictions j whose L1 distance to true_i is strictly larger
than pred_i, with ties counting as 0.5. Identical predictions therefore score
exactly 0.5.
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

from experiments.tx.cell_eval_export import clip_gene_matrix_for_celleval
from experiments.tx.train_cl import DEFAULTS
from trajectory_replay.data.tx_cl_data import load_gene_rows
from trajectory_replay.data.tx_cl_tasks import (
    FINETUNE_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    REPLOGLE_PRETRAIN_CELL_LINES,
)
from trajectory_replay.data.tx_preprocess import decode_h5ad_categorical
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.eval.tx_cl_pearson import pearson_vector

CONTROL = "non-targeting"
CAP = 0
FEWSHOT = REPLOGLE_FEWSHOT_CELL_LINE


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--h5ad-path",
        type=Path,
        default=DEFAULTS["replogle"]["h5ad"],
    )
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT
        / "results_dir/tx_results/replogle/diag_geneblind_heldout.json",
    )
    return p


def _cap(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or idx.size <= cap:
        return np.sort(idx)
    return np.sort(rng.choice(idx, cap, replace=False))


def _mean_rows(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        raise ValueError("empty index")
    return load_gene_rows(adata, idx).mean(axis=0).astype(np.float64)


def _deltas_for_genes(
    adata: ad.AnnData,
    gene: np.ndarray,
    mask: np.ndarray,
    control_mean: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rows: list[np.ndarray] = []
    perts = np.unique(gene[mask])
    for g in perts:
        if str(g) == CONTROL:
            continue
        take = np.flatnonzero(mask & (gene == g))
        take = _cap(take, cap, rng)
        if take.size == 0:
            continue
        names.append(str(g))
        rows.append(_mean_rows(adata, take) - control_mean)
    if not rows:
        return [], np.empty((0, control_mean.size), dtype=np.float64)
    return names, np.stack(rows, axis=0)


def l1_discrimination(pred: np.ndarray, true: np.ndarray) -> float:
    """Pairwise L1 ranking; ties count as 0.5. Identical preds score 0.5."""
    if pred.shape != true.shape or pred.shape[0] < 2:
        return float("nan")
    n = int(pred.shape[0])
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        d_self = float(np.abs(pred[i] - true[i]).sum())
        d_all = np.abs(pred - true[i]).sum(axis=1)
        wins = float(np.sum(d_self < d_all) - 1.0)
        ties = float(np.sum(np.isclose(d_all, d_self)) - 1.0)
        scores[i] = (wins + 0.5 * ties) / (n - 1)
    return float(np.mean(scores))


def macro_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    vals = [pearson_vector(true[i], pred[i]) for i in range(true.shape[0])]
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def score_predictor(
    name: str,
    pred: np.ndarray,
    true: np.ndarray,
    true_expr: np.ndarray,
    pred_expr: np.ndarray,
    *,
    uses_heldout_labels: bool,
    gene_specific: bool,
) -> dict[str, Any]:
    clipped_true = clip_gene_matrix_for_celleval(true_expr)
    clipped_pred = clip_gene_matrix_for_celleval(pred_expr)
    return {
        "name": name,
        "uses_heldout_labels": uses_heldout_labels,
        "gene_specific": gene_specific,
        "n_strata": int(true.shape[0]),
        "pearson_delta_macro": macro_pearson(pred, true),
        "l1_discrimination_delta": l1_discrimination(pred, true),
        "l1_discrimination_clipped_expr": l1_discrimination(clipped_pred, clipped_true),
    }


def main() -> None:
    args = make_parser().parse_args()
    path = args.h5ad_path
    rng = np.random.default_rng(args.seed)
    print(f"h5ad={path}", flush=True)

    ctx = decode_h5ad_categorical(path, "cell_line")
    gene = decode_h5ad_categorical(path, "gene")
    split = decode_h5ad_categorical(path, "split")
    adata = ad.read_h5ad(path, backed="r")

    hepg2 = ctx == FEWSHOT
    t0 = np.isin(ctx, list(REPLOGLE_PRETRAIN_CELL_LINES)) & (split == TRAIN_SPLIT)
    fit_mask = hepg2 & (split == FINETUNE_SPLIT) & (gene != CONTROL)
    held_mask = hepg2 & (split == TEST_SPLIT) & (gene != CONTROL)
    hepg2_ctrl_idx = np.flatnonzero(hepg2 & (gene == CONTROL) & (split != TEST_SPLIT))
    hepg2_ctrl_idx = _cap(hepg2_ctrl_idx, args.cap, rng)
    hepg2_ctrl = _mean_rows(adata, hepg2_ctrl_idx)
    print(
        f"hepg2 controls={hepg2_ctrl_idx.size:,}  "
        f"fit_cells={int(fit_mask.sum()):,}  held_cells={int(held_mask.sum()):,}",
        flush=True,
    )

    fit_names, fit_D = _deltas_for_genes(
        adata, gene, fit_mask, hepg2_ctrl, args.cap, rng
    )
    held_names, held_D = _deltas_for_genes(
        adata, gene, held_mask, hepg2_ctrl, args.cap, rng
    )
    print(
        f"fit strata={len(fit_names)}  heldout strata={len(held_names)}",
        flush=True,
    )
    if held_D.shape[0] < 2:
        raise RuntimeError("need at least two heldout strata")

    held_set = set(held_names)
    t0_held_by_gene: dict[str, list[np.ndarray]] = {}
    for line in REPLOGLE_PRETRAIN_CELL_LINES:
        line_mask = t0 & (ctx == line)
        ctrl = np.flatnonzero(line_mask & (gene == CONTROL))
        ctrl = _cap(ctrl, args.cap, rng)
        if ctrl.size == 0:
            continue
        mean_ctrl = _mean_rows(adata, ctrl)
        n_found = 0
        for n in held_names:
            take = np.flatnonzero(line_mask & (gene == n))
            take = _cap(take, args.cap, rng)
            if take.size == 0:
                continue
            t0_held_by_gene.setdefault(n, []).append(
                _mean_rows(adata, take) - mean_ctrl
            )
            n_found += 1
        print(
            f"  T0 {line}: n_ctrl={ctrl.size:,} heldout genes present={n_found}",
            flush=True,
        )
    t0_per_gene = np.stack(
        [
            np.mean(np.stack(t0_held_by_gene[n], axis=0), axis=0)
            if n in t0_held_by_gene
            else np.zeros_like(hepg2_ctrl)
            for n in held_names
        ],
        axis=0,
    )
    n_t0_overlap = int(sum(n in t0_held_by_gene for n in held_names))
    t0_generic = (
        np.mean(np.stack([np.mean(v, 0) for v in t0_held_by_gene.values()], 0), 0)
        if t0_held_by_gene
        else np.zeros_like(hepg2_ctrl)
    )
    fit_generic = fit_D.mean(axis=0)
    loo = np.empty_like(held_D)
    n = held_D.shape[0]
    total = held_D.sum(axis=0)
    for i in range(n):
        loo[i] = (total - held_D[i]) / (n - 1)

    true_expr = held_D + hepg2_ctrl
    predictors: list[tuple[str, np.ndarray, bool, bool]] = [
        ("hepg2_fit_mean", np.broadcast_to(fit_generic, held_D.shape).copy(), False, False),
        (
            "t0_heldout_genes_mean",
            np.broadcast_to(t0_generic, held_D.shape).copy(),
            False,
            False,
        ),
        ("t0_per_gene_transfer", t0_per_gene, False, True),
        ("heldout_loo_mean", loo, True, False),
    ]

    rows = []
    for name, pred, leak, specific in predictors:
        pred_expr = pred + hepg2_ctrl
        rec = score_predictor(
            name,
            pred,
            held_D,
            true_expr,
            pred_expr,
            uses_heldout_labels=leak,
            gene_specific=specific,
        )
        rows.append(rec)
        print(
            f"{name:28s} pearsonΔ={rec['pearson_delta_macro']:.4f}  "
            f"L1Δ={rec['l1_discrimination_delta']:.4f}  "
            f"L1clip={rec['l1_discrimination_clipped_expr']:.4f}  "
            f"leak={leak} gene_specific={specific}",
            flush=True,
        )

    identical = l1_discrimination(
        np.broadcast_to(fit_generic, held_D.shape).copy(),
        held_D,
    )
    print(f"sanity identical-pred L1Δ={identical:.4f} (expect 0.5)", flush=True)

    out = {
        "h5ad": str(path),
        "cap": int(args.cap),
        "seed": int(args.seed),
        "n_fit_strata": int(len(fit_names)),
        "n_heldout_strata": int(len(held_names)),
        "n_heldout_genes_seen_at_t0": n_t0_overlap,
        "heldout_set_size": int(len(held_set)),
        "predictors": rows,
        "sanity_identical_pred_l1_discrimination_delta": identical,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
