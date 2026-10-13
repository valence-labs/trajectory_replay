#!/usr/bin/env python3
"""Oracle Pearson-Δ from decoding observed SE (no ODE).

Scores fit / forget / heldout on an existing sequential CL run. Heldout uses
``task_id_from_context`` (test cells are never in a train task) and injects
same-context train/finetune controls, matching ``eval_track``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.tx.train_cl import control_reference_indices, task_id_from_context
from trajectory_replay.data.tx_cl_data import (
    load_gene_rows,
    load_tx_cl_arrays,
    stratified_cap_indices,
)
from trajectory_replay.data.tx_cl_tasks import step_masks, stream_for, train_task_ids
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank
from trajectory_replay.eval.tx_cl_pearson import pearson_vector, score_gene_strata

OLD_CELLEVAL = (
    REPO_ROOT
    / "results_dir/tx_results_old/replogle_split2_hepg2/metrics_celleval"
)
OLD_COMPARE_LABELS = (
    "vf1_pert1_ctx1",
    "real_replay_r25",
    "generative_replay_r25",
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results_dir/tx_results/replogle/real_grid_st100000_s42"),
    )
    p.add_argument("--max-cells-per-stratum", type=int, default=500)
    p.add_argument("--out-json", type=Path, default=None)
    return p


def cap_indices(arrays, indices: np.ndarray, cap: int, seed: int) -> np.ndarray:
    return stratified_cap_indices(
        indices,
        arrays.c,
        arrays.p,
        arrays.z,
        arrays.covariate_col_order,
        max_per_stratum=cap,
        seed=seed,
    )


def score_pool(
    arrays,
    adata,
    dec: TaskDecoderBank,
    indices: np.ndarray,
    task_ids: np.ndarray,
    *,
    cap: int,
    seed: int,
    control_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    target = cap_indices(arrays, indices, cap, seed)
    extra = np.empty(0, dtype=np.int64)
    if control_reference is not None:
        extra = control_reference_indices(
            arrays,
            target,
            np.asarray(control_reference, dtype=np.int64),
            max_per_stratum=cap,
            seed=seed + 1,
        )
    sel = np.concatenate([target, extra])
    true_genes = load_gene_rows(adata, sel)
    tids = np.asarray(task_ids)[sel]
    recon = dec.decode(np.asarray(arrays.se[sel]), tids)
    scored = score_gene_strata(
        arrays,
        sel,
        true_genes,
        recon,
        tids,
        control_pert_index=arrays.encoding.control_pert_index,
    )
    rng = np.random.default_rng(seed)
    n_samp = min(500, sel.size)
    samp = rng.choice(sel.size, n_samp, replace=False) if n_samp else np.empty(0, int)
    r_cells = (
        np.array([pearson_vector(true_genes[i], recon[i]) for i in samp])
        if samp.size
        else np.array([])
    )
    ss_res = float(((true_genes - recon) ** 2).sum())
    ss_tot = float(((true_genes - true_genes.mean(0)) ** 2).sum())
    ctrl = arrays.encoding.control_pert_index
    is_ctrl = arrays.p[sel] == ctrl if ctrl is not None else np.zeros(sel.size, bool)
    out = {
        "n_target_cells": int(target.size),
        "n_control_reference_cells": int(extra.size),
        "n_cells_scored": int(sel.size),
        "n_control_cells": int(is_ctrl.sum()),
        "task_ids": [int(x) for x in np.unique(tids)],
        "per_cell_recon_pearson": float(np.nanmean(r_cells)) if r_cells.size else float("nan"),
        "decoder_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "oracle_pearson_delta_macro": scored["pearson_delta_macro"],
        "oracle_pearson_macro": scored["pearson_macro"],
        "n_strata": scored["n_strata"],
        "n_delta_strata": scored["n_delta_strata"],
        "n_pearson_fallback_strata": scored["n_pearson_fallback_strata"],
    }
    print(
        f"  cells={out['n_cells_scored']} strata={out['n_strata']} "
        f"reconP={out['per_cell_recon_pearson']:.4f} R2={out['decoder_r2']:.4f} "
        f"oracleΔ={out['oracle_pearson_delta_macro']:.4f} "
        f"oracleP={out['oracle_pearson_macro']:.4f}",
        flush=True,
    )
    return out


def flow_macros(run: Path) -> dict[str, Any]:
    path = run / "final_metrics.json"
    if not path.is_file():
        return {}
    final = json.loads(path.read_text())["final"]
    out = {}
    for track in ("fit", "forget", "heldout"):
        block = final.get(track, {})
        out[track] = {
            "pearson_delta_macro": block.get("pearson_delta_macro"),
            "pearson_macro": block.get("pearson_macro"),
            "n_strata": block.get("n_strata"),
        }
    return out


def _read_old_csv(path: Path, label_col: str, value_col: str) -> dict[str, float]:
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            label = row.get(label_col) or row.get("label")
            if label in OLD_COMPARE_LABELS:
                out[label] = float(row[value_col])
    return out


def old_comparison() -> dict[str, Any]:
    fit = _read_old_csv(
        OLD_CELLEVAL / "replogle_split2_hepg2_methods_comparison_all_celleval_fitting.csv",
        "label",
        "fitting_pearson_delta",
    )
    ood = _read_old_csv(
        OLD_CELLEVAL / "replogle_split2_hepg2_methods_comparison_all_celleval_ood_fitting.csv",
        "label",
        "test_pearson_delta",
    )
    return {
        "note": (
            "Old CellEval Pearson-Δ. fitting = HepG2 few-shot (our fit). "
            "test/ood = HepG2 STATE-test genes (our heldout). "
            "Old forgetting.csv is a *delta vs pretrain*, not absolute Pearson-Δ."
        ),
        "fitting_pearson_delta": fit,
        "test_pearson_delta": ood,
    }


def main() -> None:
    args = make_parser().parse_args()
    run = args.run_dir
    cfg = json.loads((run / "config.json").read_text())
    stream = stream_for(cfg["dataset"])
    arrays = load_tx_cl_arrays(
        cfg["h5ad_path"],
        cfg["se_path"],
        stream,
        covariate_cols=tuple(cfg["covariate_cols"]),
        control_perturbation=cfg["control_perturbation"],
    )
    adata = ad.read_h5ad(cfg["h5ad_path"], backed="r")
    dec = TaskDecoderBank.load_dir(cfg["decoder_dir"])
    train_ids = train_task_ids(stream, arrays.context, arrays.split)
    context_ids = task_id_from_context(stream, arrays.context)
    masks = step_masks(stream, 1, arrays.context, arrays.split, arrays.perturbation)

    tracks = {}
    print("FIT (HepG2 finetune, D_1)", flush=True)
    tracks["fit"] = score_pool(
        arrays,
        adata,
        dec,
        np.flatnonzero(masks.new_train),
        train_ids,
        cap=args.max_cells_per_stratum,
        seed=0,
    )
    print("FORGET (T0 train lines, D_0)", flush=True)
    tracks["forget"] = score_pool(
        arrays,
        adata,
        dec,
        np.flatnonzero(masks.old_train),
        train_ids,
        cap=args.max_cells_per_stratum,
        seed=1,
    )
    print("HELDOUT (HepG2 STATE-test, D_context + train/finetune controls)", flush=True)
    tracks["heldout"] = score_pool(
        arrays,
        adata,
        dec,
        np.flatnonzero(masks.heldout),
        context_ids,
        cap=args.max_cells_per_stratum,
        seed=2,
        control_reference=np.flatnonzero(masks.seen_train),
    )
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    payload = {
        "run_dir": str(run),
        "decoder_dir": cfg["decoder_dir"],
        "max_cells_per_stratum": args.max_cells_per_stratum,
        "oracle": tracks,
        "flow_from_final_metrics": flow_macros(run),
        "old_celleval": old_comparison(),
    }
    out_json = args.out_json or (
        Path("results_dir/tx_results/replogle/diag_decoder_oracle_fit_heldout.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
