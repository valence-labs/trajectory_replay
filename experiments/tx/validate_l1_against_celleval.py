#!/usr/bin/env python3
"""Validate centroid L1 scoring against Cell-Eval from saved per-cell data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import anndata as ad
import numpy as np
import pandas as pd
from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics._anndata import discrimination_score

from experiments.tx.diag_decoded_geneblind_heldout import celleval_l1_scores


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-h5ad", type=Path, required=True)
    p.add_argument("--pred-h5ad", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--pert-col", default="perturbation")
    p.add_argument("--control-pert", default="non-targeting")
    p.add_argument("--out-json", type=Path, required=True)
    return p


def _scores(result: dict[str, float], perturbations: list[str]) -> np.ndarray:
    return np.asarray([result[p] for p in perturbations], dtype=np.float64)


def main() -> None:
    args = parser().parse_args()
    real = ad.read_h5ad(args.real_h5ad)
    pred = ad.read_h5ad(args.pred_h5ad)
    pair = PerturbationAnndataPair(
        real=real,
        pred=pred,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
    )
    perturbations = pair.perts.astype(str).tolist()

    direct = _scores(
        discrimination_score(pair, metric="l1"),
        perturbations,
    )
    stored = pd.read_csv(args.results).set_index("perturbation")
    expected = stored.loc[
        perturbations, "discrimination_score_l1"
    ].to_numpy(np.float64)
    direct_diff = np.abs(direct - expected)

    bulk = list(pair.iter_bulk_arrays())
    real_effects = np.vstack(
        [row.perturbation_effect(which="real", abs=False) for row in bulk]
    )
    pred_effects = np.vstack(
        [row.perturbation_effect(which="pred", abs=False) for row in bulk]
    )
    centroid = celleval_l1_scores(
        pred_effects,
        real_effects,
        perturbations,
        pair.genes,
    )
    centroid_diff = np.abs(centroid - direct)

    direct_matches_stored = bool(
        np.allclose(direct, expected, rtol=0, atol=1e-12)
    )
    centroid_matches_direct = bool(
        np.allclose(centroid, direct, rtol=0, atol=1e-12)
    )
    payload = {
        "real_h5ad": str(args.real_h5ad),
        "pred_h5ad": str(args.pred_h5ad),
        "results": str(args.results),
        "n_perturbations": len(perturbations),
        "n_features": int(real.n_vars),
        "mean_direct_per_cell": float(direct.mean()),
        "mean_stored": float(expected.mean()),
        "direct_vs_stored_max_abs_error": float(direct_diff.max()),
        "direct_vs_stored_n_exact": int(np.sum(direct_diff == 0)),
        "direct_matches_stored": direct_matches_stored,
        "mean_pseudobulk_effects_zero_control": float(centroid.mean()),
        "centroid_vs_direct_max_abs_error": float(centroid_diff.max()),
        "centroid_vs_direct_n_exact": int(np.sum(centroid_diff == 0)),
        "centroid_matches_direct": centroid_matches_direct,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if not direct_matches_stored:
        raise AssertionError("Direct per-cell Cell-Eval does not reproduce results.csv")
    if not centroid_matches_direct:
        raise AssertionError(
            "Pseudobulk effects plus zero control do not reproduce per-cell Cell-Eval"
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
