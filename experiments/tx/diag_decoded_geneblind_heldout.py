#!/usr/bin/env python3
"""Gene-blind and T0-transfer baselines after the CL affine PCA decoder.

Native-space oracles are unfair ceilings for a PCA-256 CFM: the model can only
emit D_1(z) for the HepG2 task decoder. This job scores the same predictors on
*decoded PCA means* (task-1 affine on T0-train PCs), against true gene-space
deltas, matching ``score_gene_strata`` / ``pred_control=decoded``.
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
import pandas as pd
from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics._anndata import discrimination_score

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
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank
from trajectory_replay.eval.tx_cl_pearson import pearson_vector

CONTROL = "non-targeting"
FEWSHOT = REPLOGLE_FEWSHOT_CELL_LINE


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=DEFAULTS["replogle"]["h5ad"])
    p.add_argument("--pca-path", type=Path, default=DEFAULTS["replogle"]["pca"])
    p.add_argument(
        "--decoder-dir",
        type=Path,
        default=DEFAULTS["replogle"]["bucket"] / "decoders_pca256_affine",
    )
    p.add_argument("--cap", type=int, default=0)
    p.add_argument(
        "--control-cap",
        type=int,
        default=0,
        help="Control cells per context; <=0 keeps the complete basal pool.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT
        / "results_dir/tx_results/replogle/diag_decoded_geneblind_heldout.json",
    )
    return p


def _cap(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or idx.size <= cap:
        return np.sort(idx)
    return np.sort(rng.choice(idx, cap, replace=False))


def _mean_genes(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    return load_gene_rows(adata, idx).mean(axis=0).astype(np.float64)


def _mean_clipped_genes(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    rows = clip_gene_matrix_for_celleval(load_gene_rows(adata, idx))
    return rows.mean(axis=0).astype(np.float64)


def _mean_pca(se: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(se[np.asarray(idx, dtype=np.int64)], dtype=np.float64).mean(0)


def _finite_mean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def macro_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    return _finite_mean(
        [pearson_vector(true[i], pred[i]) for i in range(true.shape[0])]
    )


def residualize(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = D.mean(axis=0)
    return D - u, u


def decode_delta(decoder, z_delta: np.ndarray) -> np.ndarray:
    z = np.asarray(z_delta, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
        return decoder.decode(z)[0] - decoder.intercept
    return decoder.decode(z) - decoder.intercept


def decode_clipped_delta(
    decoder, z_control_rows: np.ndarray, z_delta: np.ndarray
) -> np.ndarray:
    """Cell-Eval effect for a centroid prediction and the full decoded basal.

    The baseline defines one predicted centroid per perturbation, so clipping
    that centroid is exact for the baseline. The control side follows model
    Cell-Eval: decode and clip every observed control cell, then average.
    """
    dz = np.asarray(z_delta, dtype=np.float64)
    if dz.ndim == 1:
        dz = dz[None, :]
    z0_rows = np.asarray(z_control_rows, dtype=np.float64)
    z0 = z0_rows.mean(axis=0, keepdims=True)
    control = clip_gene_matrix_for_celleval(decoder.decode(z0_rows)).mean(axis=0)
    pert = clip_gene_matrix_for_celleval(decoder.decode(z0 + dz))
    return pert - control


def celleval_l1_scores(
    pred_effects: np.ndarray,
    real_effects: np.ndarray,
    perturbations: list[str],
    feature_names: np.ndarray,
) -> np.ndarray:
    """Run Cell-Eval itself on one centroid per perturbation plus zero control."""
    pred = np.asarray(pred_effects, dtype=np.float64)
    real = np.asarray(real_effects, dtype=np.float64)
    if pred.shape != real.shape or pred.shape[0] != len(perturbations):
        raise ValueError(
            f"L1 inputs do not align: pred={pred.shape} real={real.shape} "
            f"perturbations={len(perturbations)}"
        )
    control = "__baseline_control__"
    labels = [*map(str, perturbations), control]
    obs = pd.DataFrame({"perturbation": pd.Categorical(labels)})
    var = pd.DataFrame(index=pd.Index(feature_names.astype(str)))
    real_ad = ad.AnnData(
        X=np.vstack([real, np.zeros((1, real.shape[1]))]),
        obs=obs.copy(),
        var=var.copy(),
    )
    pred_ad = ad.AnnData(
        X=np.vstack([pred, np.zeros((1, pred.shape[1]))]),
        obs=obs.copy(),
        var=var.copy(),
    )
    pair = PerturbationAnndataPair(
        real=real_ad,
        pred=pred_ad,
        pert_col="perturbation",
        control_pert=control,
    )
    result = discrimination_score(pair, metric="l1")
    return np.asarray([result[str(p)] for p in perturbations], dtype=np.float64)


def celleval_l1_discrimination(
    pred_effects: np.ndarray,
    real_effects: np.ndarray,
    perturbations: list[str],
    feature_names: np.ndarray,
) -> float:
    return float(
        celleval_l1_scores(
            pred_effects, real_effects, perturbations, feature_names
        ).mean()
    )


def reachability_projection(
    decoder, true_delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """C2: exact Pearson-reachable projection onto centered col(D1)."""
    true = np.asarray(true_delta, dtype=np.float64)
    centered_true = true - true.mean(axis=1, keepdims=True)
    A = np.asarray(decoder.weight, dtype=np.float64)
    centered_A = A - A.mean(axis=0, keepdims=True)
    z, *_ = np.linalg.lstsq(centered_A, centered_true.T, rcond=None)
    z = z.T
    return decode_delta(decoder, z), z


def collect(
    adata: ad.AnnData,
    se: np.ndarray,
    gene: np.ndarray,
    cell_mask: np.ndarray,
    ctrl_mask: np.ndarray,
    names: list[str],
    cap: int,
    control_cap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ctrl = _cap(np.flatnonzero(ctrl_mask & (gene == CONTROL)), control_cap, rng)
    if ctrl.size == 0:
        raise RuntimeError("no controls")
    z_ctrl = _mean_pca(se, ctrl)
    x_ctrl = _mean_genes(adata, ctrl)
    x_ctrl_clipped = _mean_clipped_genes(adata, ctrl)
    keep: list[str] = []
    z_rows: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []
    x_clipped_rows: list[np.ndarray] = []
    for g in names:
        take = _cap(np.flatnonzero(cell_mask & (gene == g)), cap, rng)
        if take.size == 0:
            continue
        keep.append(g)
        z_rows.append(_mean_pca(se, take) - z_ctrl)
        x_rows.append(_mean_genes(adata, take) - x_ctrl)
        x_clipped_rows.append(_mean_clipped_genes(adata, take) - x_ctrl_clipped)
    return {
        "names": keep,
        "Z": np.stack(z_rows, 0),
        "X": np.stack(x_rows, 0),
        "X_clipped": np.stack(x_clipped_rows, 0),
        "z_ctrl": z_ctrl,
        "z_ctrl_rows": np.asarray(se[ctrl], dtype=np.float64),
        "n_ctrl": int(ctrl.size),
    }


def rec(
    name: str,
    pred: np.ndarray,
    true: np.ndarray,
    *,
    l1_pred: np.ndarray | None = None,
    l1_true: np.ndarray | None = None,
    perturbations: list[str] | None = None,
    feature_names: np.ndarray | None = None,
    **extra: Any,
) -> dict[str, Any]:
    t_res, _ = residualize(true)
    p_res, _ = residualize(pred)
    residual_score = (
        0.0
        if float(np.linalg.norm(p_res)) <= 1e-12
        else macro_pearson(p_res, t_res)
    )
    out = {
        "name": name,
        "n_strata": int(true.shape[0]),
        "pearson_delta_raw": macro_pearson(pred, true),
        "pearson_delta_residual": residual_score,
        **extra,
    }
    if l1_pred is not None:
        if l1_true is None or perturbations is None or feature_names is None:
            raise ValueError("L1 scoring requires true effects, perturbations, and features")
        out["discrimination_score_l1"] = celleval_l1_discrimination(
            l1_pred, l1_true, perturbations, feature_names
        )
    print(
        f"{name:40s} rawΔ={out['pearson_delta_raw']:.4f}  "
        f"residualΔ={out['pearson_delta_residual']:.4f}",
        flush=True,
    )
    return out


def main() -> None:
    args = make_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"h5ad={args.h5ad_path}", flush=True)
    print(f"pca={args.pca_path}", flush=True)
    print(f"decoders={args.decoder_dir}", flush=True)

    ctx = decode_h5ad_categorical(args.h5ad_path, "cell_line")
    gene = decode_h5ad_categorical(args.h5ad_path, "gene")
    split = decode_h5ad_categorical(args.h5ad_path, "split")
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    se = np.load(args.pca_path, mmap_mode="r")
    bank = TaskDecoderBank.load_dir(args.decoder_dir)
    d1 = bank.get(1)
    print(f"D1 se_dim={d1.se_dim} n_genes={d1.n_genes} n_fit={d1.n_fit}", flush=True)
    if int(se.shape[1]) < d1.se_dim:
        raise ValueError(f"PCA width {se.shape[1]} < decoder se_dim {d1.se_dim}")

    held_mask = (ctx == FEWSHOT) & (split == TEST_SPLIT) & (gene != CONTROL)
    fit_mask = (ctx == FEWSHOT) & (split == FINETUNE_SPLIT) & (gene != CONTROL)
    hepg2_ctrl = (ctx == FEWSHOT) & (split != TEST_SPLIT)
    held_names = sorted({str(g) for g in gene[held_mask]})
    fit_names = sorted({str(g) for g in gene[fit_mask]})
    print(f"heldout genes={len(held_names)} fit genes={len(fit_names)}", flush=True)

    print("loading HepG2 heldout...", flush=True)
    held = collect(
        adata, se, gene, held_mask, hepg2_ctrl, held_names,
        args.cap, args.control_cap, rng
    )
    print("loading HepG2 fit...", flush=True)
    fit = collect(
        adata, se, gene, fit_mask, hepg2_ctrl, fit_names,
        args.cap, args.control_cap, rng
    )
    t0_blocks: dict[str, dict[str, Any]] = {}
    for line in REPLOGLE_PRETRAIN_CELL_LINES:
        line_mask = (ctx == line) & (split == TRAIN_SPLIT)
        print(f"loading T0 {line}...", flush=True)
        t0_blocks[line] = collect(
            adata, se, gene, line_mask, line_mask, held_names,
            args.cap, args.control_cap, rng
        )
        print(
            f"  {line}: n_ctrl={t0_blocks[line]['n_ctrl']} "
            f"n_genes={len(t0_blocks[line]['names'])}",
            flush=True,
        )

    n_t0_by_gene = {
        g: sum(g in set(block["names"]) for block in t0_blocks.values())
        for g in held["names"]
    }
    n_any = sum(n > 0 for n in n_t0_by_gene.values())
    n_all = sum(n == len(t0_blocks) for n in n_t0_by_gene.values())
    print(
        f"heldout genes with ≥1 T0 line={n_any}/{len(held['names'])} "
        f"with all {len(t0_blocks)} T0 lines={n_all}",
        flush=True,
    )

    true_all = held["X"]

    u_fit_z = fit["Z"].mean(0)
    pred_fit_blind = np.broadcast_to(decode_delta(d1, u_fit_z), true_all.shape).copy()
    pred_held_loo = decode_delta(
        d1,
        (held["Z"].sum(0, keepdims=True) - held["Z"]) / max(held["Z"].shape[0] - 1, 1),
    )

    t0_z_by_gene: dict[str, list[np.ndarray]] = {}
    for block in t0_blocks.values():
        for i, g in enumerate(block["names"]):
            t0_z_by_gene.setdefault(g, []).append(block["Z"][i])
    t0_z_all = np.stack([
        np.mean(np.stack(t0_z_by_gene[g], 0), 0)
        if g in t0_z_by_gene
        else np.zeros(d1.se_dim, dtype=np.float64)
        for g in held["names"]
    ], 0)
    u_t0_z = np.mean(
        [np.mean(np.stack(v, 0), 0) for v in t0_z_by_gene.values()],
        0,
    )
    pred_t0_blind = np.broadcast_to(decode_delta(d1, u_t0_z), true_all.shape).copy()
    pred_t0_gene = decode_delta(d1, t0_z_all)

    t0_res_z, _ = residualize(t0_z_all)
    hepg2_res_z, _ = residualize(held["Z"])
    compositional_z = u_fit_z + t0_res_z
    pred_comp = decode_delta(d1, compositional_z)
    pred_t0_res = decode_delta(d1, t0_res_z)
    pred_reach, z_reach = reachability_projection(d1, true_all)

    # Cell-Eval clips decoded expression before constructing perturbation
    # effects. True expression in this dump is already in [0, 14], but clip it
    # explicitly to pin the same contract.
    true_all_l1 = held["X_clipped"]
    l1_fit_blind = decode_clipped_delta(d1, held["z_ctrl_rows"], u_fit_z)
    l1_fit_blind = np.broadcast_to(l1_fit_blind, true_all.shape).copy()
    l1_t0_blind = decode_clipped_delta(d1, held["z_ctrl_rows"], u_t0_z)
    l1_t0_blind = np.broadcast_to(l1_t0_blind, true_all.shape).copy()
    l1_t0_gene = decode_clipped_delta(d1, held["z_ctrl_rows"], t0_z_all)
    l1_comp = decode_clipped_delta(d1, held["z_ctrl_rows"], compositional_z)
    l1_reach = decode_clipped_delta(d1, held["z_ctrl_rows"], z_reach)
    feature_names = np.asarray(adata.var_names).astype(str)

    native_fit_blind = np.broadcast_to(fit["X"].mean(0), true_all.shape).copy()
    native_t0_gene = np.stack(
        [
            np.mean(
                np.stack(
                    [
                        t0_blocks[line]["X"][t0_blocks[line]["names"].index(g)]
                        for line in t0_blocks
                        if g in t0_blocks[line]["names"]
                    ],
                    0,
                ),
                0,
            )
            if g in t0_z_by_gene
            else np.zeros(true_all.shape[1])
            for g in held["names"]
        ],
        0,
    )

    rows = [
        rec("native_hepg2_fit_mean", native_fit_blind, true_all, space="gene", leak=False),
        rec(
            "decoded_hepg2_fit_mean",
            pred_fit_blind,
            true_all,
            l1_pred=l1_fit_blind,
            l1_true=true_all_l1,
            perturbations=held["names"],
            feature_names=feature_names,
            space="pca_D1",
            leak=False,
        ),
        rec(
            "decoded_t0_heldout_genes_mean",
            pred_t0_blind,
            true_all,
            l1_pred=l1_t0_blind,
            l1_true=true_all_l1,
            perturbations=held["names"],
            feature_names=feature_names,
            space="pca_D1",
            leak=False,
        ),
        rec(
            "decoded_t0_per_gene",
            pred_t0_gene,
            true_all,
            l1_pred=l1_t0_gene,
            l1_true=true_all_l1,
            perturbations=held["names"],
            feature_names=feature_names,
            space="pca_D1",
            leak=False,
            gene_specific=True,
        ),
        rec(
            "decoded_heldout_loo_mean",
            pred_held_loo,
            true_all,
            space="pca_D1",
            leak=True,
        ),
        rec(
            "native_t0_per_gene",
            native_t0_gene,
            true_all,
            space="gene",
            leak=False,
            gene_specific=True,
        ),
        rec(
            "decoded_hepg2_fit_plus_t0_residual",
            pred_comp,
            true_all,
            l1_pred=l1_comp,
            l1_true=true_all_l1,
            perturbations=held["names"],
            feature_names=feature_names,
            space="pca_D1",
            leak=False,
            gene_specific=True,
            note=(
                "T0 residual uses the mean over T0 lines that contain each gene "
                "(1, 2, or 3); residualized over all held-out genes"
            ),
        ),
        rec(
            "decoded_pca_reachability_ceiling",
            pred_reach,
            true_all,
            l1_pred=l1_reach,
            l1_true=true_all_l1,
            perturbations=held["names"],
            feature_names=feature_names,
            space="pca_D1",
            leak=True,
            oracle=True,
            exact_projection=True,
        ),
        rec(
            "decoded_t0_residual_only",
            pred_t0_res,
            true_all,
            space="pca_D1",
            leak=False,
            gene_specific=True,
        ),
    ]
    true_res_all, _ = residualize(true_all)
    rows.append(
        rec(
            "decoded_t0_residual_vs_true_residual",
            pred_t0_res,
            true_res_all,
            space="pca_D1",
            leak=False,
            note="pred is decoded T0 PCA residual; true is native residual",
        )
    )
    hepg2_res_decoded = decode_delta(d1, hepg2_res_z)
    rows.append(
        rec(
            "decoded_t0_residual_vs_decoded_hepg2_residual",
            pred_t0_res,
            hepg2_res_decoded,
            space="pca_D1",
            leak=False,
            note="both sides decoded; fair in-subspace residual concordance",
        )
    )

    out = {
        "h5ad": str(args.h5ad_path),
        "pca": str(args.pca_path),
        "decoder_dir": str(args.decoder_dir),
        "cap": int(args.cap),
        "control_cap": int(args.control_cap),
        "l1_implementation": "cell_eval.metrics._anndata.discrimination_score",
        "seed": int(args.seed),
        "n_heldout": int(len(held["names"])),
        "n_fit": int(len(fit["names"])),
        "n_heldout_with_any_t0": int(n_any),
        "n_heldout_with_all_t0": int(n_all),
        "d1_se_dim": int(d1.se_dim),
        "predictors": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
