#!/usr/bin/env python3
"""Oracle Pearson-Δ for a one-shot run: decode observed embeddings, no ODE.

Replaces the ODE sample in ``eval_track`` with ``D(z_observed)`` while keeping
the index pools, per-stratum caps and seeds identical, so the number is directly
comparable to ``hepg2_pearson.json`` from the same run directory. It upper
bounds what any flow in this representation can score: a perfect vector field
would land exactly on the observed embedding.
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

from experiments.tx.train_cl import control_reference_indices
from trajectory_replay.data.tx_cl_data import (
    LATENT_STANDARDIZER_NAME,
    apply_latent_radial_scale,
    apply_latent_standardizer,
    load_gene_rows,
    load_latent_standardizer,
    load_tx_cl_arrays,
    load_tx_cl_gene_arrays,
    load_tx_cl_pca_arrays,
    stratified_cap_indices,
)
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank
from trajectory_replay.eval.tx_cl_pearson import pearson_vector, score_gene_strata


def arrays_from_config(cfg: dict[str, Any]):
    """Rebuild the run's latent arrays, dispatching on ``representation``."""
    representation = str(cfg.get("representation") or "se")
    stream = stream_for(cfg["dataset"])
    shared = {
        "covariate_cols": tuple(cfg["covariate_cols"]),
        "control_perturbation": cfg["control_perturbation"],
    }
    if representation == "pca":
        arrays = load_tx_cl_pca_arrays(
            cfg["h5ad_path"], stream, pca_path=cfg.get("pca_embedding"), **shared
        )
    elif representation == "genes":
        arrays = load_tx_cl_gene_arrays(cfg["h5ad_path"], stream, **shared)
    else:
        arrays = load_tx_cl_arrays(cfg["h5ad_path"], cfg["se_path"], stream, **shared)
    if cfg.get("radial_scale_latent"):
        apply_latent_radial_scale(arrays.se)
    elif cfg.get("standardize_latent"):
        scale_path = Path(cfg["out_dir"]) / LATENT_STANDARDIZER_NAME
        mean, std = load_latent_standardizer(scale_path)
        apply_latent_standardizer(arrays.se, mean, std)
    return arrays


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--max-cells-per-stratum",
        type=int,
        default=0,
        help="Override the run's eval_max_cells_per_stratum; 0 reuses it.",
    )
    p.add_argument("--recon-sample", type=int, default=2000)
    p.add_argument("--out-json", type=Path, default=None)
    return p


def score_pool(
    arrays,
    adata,
    bank: TaskDecoderBank,
    indices: np.ndarray,
    *,
    cap: int,
    seed: int,
    recon_sample: int,
    control_reference: np.ndarray | None = None,
    task_id: int = 0,
) -> dict[str, Any]:
    target = stratified_cap_indices(
        indices,
        arrays.c,
        arrays.p,
        arrays.z,
        arrays.covariate_col_order,
        max_per_stratum=cap,
        seed=seed,
    )
    extra = np.empty(0, dtype=np.int64)
    if control_reference is not None and target.size:
        extra = control_reference_indices(
            arrays,
            target,
            np.asarray(control_reference, dtype=np.int64),
            max_per_stratum=cap,
            seed=seed + 1,
        )
    sel = np.concatenate([target, extra])
    tids = np.full(sel.size, int(task_id), dtype=np.int16)

    true_genes = load_gene_rows(adata, sel)
    recon = bank.decode(np.asarray(arrays.se[sel]), tids)
    scored = score_gene_strata(
        arrays,
        sel,
        true_genes,
        recon,
        tids,
        control_pert_index=arrays.encoding.control_pert_index,
    )

    rng = np.random.default_rng(seed)
    n_samp = min(recon_sample, sel.size)
    samp = rng.choice(sel.size, n_samp, replace=False) if n_samp else np.empty(0, int)
    r_cells = np.array([pearson_vector(true_genes[i], recon[i]) for i in samp])
    ss_res = float(((true_genes - recon) ** 2).sum())
    ss_tot = float(((true_genes - true_genes.mean(0)) ** 2).sum())

    out = {
        "n_cells": int(target.size),
        "n_control_reference_cells": int(extra.size),
        "n_cells_scored": int(sel.size),
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
    path = run / "hepg2_pearson.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {
        track: {
            "pearson_delta_macro": block.get("pearson_delta_macro"),
            "pearson_macro": block.get("pearson_macro"),
            "n_strata": block.get("n_strata"),
        }
        for track, block in payload.items()
    }


def main() -> None:
    args = make_parser().parse_args()
    run = args.run_dir
    cfg = json.loads((run / "config.json").read_text())
    cap = args.max_cells_per_stratum or int(cfg["eval_max_cells_per_stratum"])
    seed = int(cfg["seed"])

    arrays = arrays_from_config(cfg)
    adata = ad.read_h5ad(cfg["h5ad_path"], backed="r")
    bank = TaskDecoderBank.load_dir(run / "decoders")

    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    hepg2_fit = np.flatnonzero((context == hepg2) & (split != TEST_SPLIT))
    hepg2_heldout = np.flatnonzero((context == hepg2) & (split == TEST_SPLIT))

    tracks = {}
    print(f"FIT (HepG2 train/finetune)  representation={cfg.get('representation')}", flush=True)
    tracks["fit"] = score_pool(
        arrays,
        adata,
        bank,
        hepg2_fit,
        cap=cap,
        seed=seed + 30_000,
        recon_sample=args.recon_sample,
    )
    print("HELDOUT (HepG2 STATE-test + injected controls)", flush=True)
    tracks["heldout"] = score_pool(
        arrays,
        adata,
        bank,
        hepg2_heldout,
        cap=cap,
        seed=seed + 30_001,
        recon_sample=args.recon_sample,
        control_reference=hepg2_fit,
    )
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    flow = flow_macros(run)
    payload = {
        "run_dir": str(run),
        "representation": cfg.get("representation"),
        "decoder_dir": str(run / "decoders"),
        "max_cells_per_stratum": cap,
        "oracle": tracks,
        "flow_from_hepg2_pearson": flow,
    }
    out_json = args.out_json or (run / "diag_decoder_oracle.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("\ntrack     oracle-Δ   flow-Δ   flow/oracle", flush=True)
    for track in ("fit", "heldout"):
        oracle = tracks[track]["oracle_pearson_delta_macro"]
        got = (flow.get(track) or {}).get("pearson_delta_macro")
        frac = f"{got / oracle:.2f}" if got and oracle else "n/a"
        got_s = f"{got:.4f}" if got is not None else "n/a"
        print(f"{track:9s} {oracle:8.4f} {got_s:>8s} {frac:>13s}", flush=True)
    print(f"\nWrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
