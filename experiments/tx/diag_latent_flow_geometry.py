#!/usr/bin/env python3
"""Split a one-shot run's Pearson-Δ loss into "flow" and "decoder" halves.

Scores the same strata three ways on identical index pools, caps and seeds:

``latent``  Pearson-Δ between true and generated latents, before any decoding.
``gene``    Pearson-Δ after decoding the generated latents (reproduces the run).
``oracle``  Pearson-Δ after decoding the *observed* latents (no ODE).

``latent`` near ``oracle`` means the decoder is the bottleneck; ``latent`` near
``gene`` means the vector field is. Also reports latent-space geometry (norms,
per-stratum centroid error vs within-stratum spread) to show whether generated
points land on the real manifold.
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

from experiments.tx.diag_oneshot_decoder_oracle import arrays_from_config
from experiments.tx.train_cl import control_reference_indices, model_from_checkpoint
from trajectory_replay.data.tx_cl_data import load_gene_rows, stratified_cap_indices
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank
from trajectory_replay.eval.tx_cl_pearson import (
    generate_se_for_indices,
    score_gene_strata,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--max-cells-per-stratum", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", type=Path, default=None)
    return p


def stratum_geometry(
    arrays, sel: np.ndarray, z_true: np.ndarray, z_pred: np.ndarray, is_ctrl: np.ndarray
) -> dict[str, float]:
    """Per-stratum centroid error of generated latents against within-stratum spread."""
    keys = np.stack([arrays.c[sel], arrays.p[sel]], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    errors: list[float] = []
    spreads: list[float] = []
    for group in range(int(inverse.max()) + 1):
        mask = (inverse == group) & ~is_ctrl
        if mask.sum() < 2:
            continue
        true_block = z_true[mask]
        centroid = true_block.mean(0)
        errors.append(float(np.linalg.norm(z_pred[mask].mean(0) - centroid)))
        spreads.append(float(np.sqrt(((true_block - centroid) ** 2).sum(1).mean())))
    if not errors:
        return {}
    err = np.asarray(errors)
    spread = np.asarray(spreads)
    return {
        "n_strata_scored": int(err.size),
        "median_centroid_error": float(np.median(err)),
        "median_within_stratum_spread": float(np.median(spread)),
        "median_error_over_spread": float(np.median(err / np.maximum(spread, 1e-12))),
    }


def score_pool(
    model,
    arrays,
    adata,
    bank: TaskDecoderBank,
    indices: np.ndarray,
    *,
    cap: int,
    seed: int,
    cfg: dict[str, Any],
    device: torch.device,
    control_reference: np.ndarray | None = None,
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
    tids = np.zeros(sel.size, dtype=np.int16)

    control = arrays.encoding.control_pert_index
    is_ctrl = (
        arrays.p[sel] == int(control)
        if control is not None
        else np.zeros(sel.size, dtype=bool)
    )
    pert_pos = np.flatnonzero(~is_ctrl)

    z_true = np.asarray(arrays.se[sel], dtype=np.float64)
    z_pred = z_true.copy()
    if pert_pos.size:
        generated = generate_se_for_indices(
            model,
            arrays,
            sel[pert_pos],
            device=device,
            batch_size=int(cfg["eval_batch_size"]),
            seed=seed,
            ode_method=str(cfg["ode_method"]),
            ode_step_size=cfg.get("ode_step_size"),
            ode_rtol=float(cfg["ode_rtol"]),
            ode_atol=float(cfg["ode_atol"]),
        )
        z_pred[pert_pos] = np.asarray(generated, dtype=np.float64)

    kwargs = {"control_pert_index": control}
    latent = score_gene_strata(arrays, sel, z_true, z_pred, tids, **kwargs)

    true_genes = load_gene_rows(adata, sel)
    gene = score_gene_strata(
        arrays, sel, true_genes, bank.decode(z_pred, tids), tids, **kwargs
    )
    oracle = score_gene_strata(
        arrays, sel, true_genes, bank.decode(z_true, tids), tids, **kwargs
    )

    out = {
        "n_cells": int(target.size),
        "n_control_reference_cells": int(extra.size),
        "n_strata": gene["n_strata"],
        "latent_pearson_delta_macro": latent["pearson_delta_macro"],
        "gene_pearson_delta_macro": gene["pearson_delta_macro"],
        "oracle_pearson_delta_macro": oracle["pearson_delta_macro"],
        "true_latent_norm_mean": float(np.linalg.norm(z_true[pert_pos], axis=1).mean()),
        "generated_latent_norm_mean": float(
            np.linalg.norm(z_pred[pert_pos], axis=1).mean()
        ),
        "prior_norm_expected": float(np.sqrt(arrays.se_dim)),
        "geometry": stratum_geometry(arrays, sel, z_true, z_pred, is_ctrl),
    }
    print(
        f"  cells={out['n_cells']} strata={out['n_strata']}  "
        f"latentΔ={out['latent_pearson_delta_macro']:.4f}  "
        f"geneΔ={out['gene_pearson_delta_macro']:.4f}  "
        f"oracleΔ={out['oracle_pearson_delta_macro']:.4f}",
        flush=True,
    )
    print(
        f"    latent norm true={out['true_latent_norm_mean']:.4f} "
        f"generated={out['generated_latent_norm_mean']:.4f} "
        f"prior~{out['prior_norm_expected']:.4f}  {out['geometry']}",
        flush=True,
    )
    return out


def main() -> None:
    args = make_parser().parse_args()
    run = args.run_dir
    cfg = json.loads((run / "config.json").read_text())
    cap = args.max_cells_per_stratum or int(cfg["eval_max_cells_per_stratum"])
    seed = int(cfg["seed"])
    device = torch.device(args.device)

    arrays = arrays_from_config(cfg)
    adata = ad.read_h5ad(cfg["h5ad_path"], backed="r")
    bank = TaskDecoderBank.load_dir(run / "decoders")
    model, _ = model_from_checkpoint(run / "checkpoint.pt", device)
    model.eval()

    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    hepg2_fit = np.flatnonzero((context == hepg2) & (split != TEST_SPLIT))
    hepg2_heldout = np.flatnonzero((context == hepg2) & (split == TEST_SPLIT))

    tracks = {}
    print(f"FIT  representation={cfg.get('representation')} dim={arrays.se_dim}", flush=True)
    tracks["fit"] = score_pool(
        model, arrays, adata, bank, hepg2_fit,
        cap=cap, seed=seed + 30_000, cfg=cfg, device=device,
    )
    print("HELDOUT", flush=True)
    tracks["heldout"] = score_pool(
        model, arrays, adata, bank, hepg2_heldout,
        cap=cap, seed=seed + 30_001, cfg=cfg, device=device,
        control_reference=hepg2_fit,
    )
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    out_json = args.out_json or (run / "diag_latent_flow_geometry.json")
    out_json.write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "representation": cfg.get("representation"),
                "latent_dim": int(arrays.se_dim),
                "max_cells_per_stratum": cap,
                "tracks": tracks,
            },
            indent=2,
        )
        + "\n"
    )

    print("\ntrack     latentΔ    geneΔ  oracleΔ", flush=True)
    for name, block in tracks.items():
        print(
            f"{name:9s} {block['latent_pearson_delta_macro']:7.4f} "
            f"{block['gene_pearson_delta_macro']:8.4f} "
            f"{block['oracle_pearson_delta_macro']:8.4f}",
            flush=True,
        )
    print(f"\nWrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
