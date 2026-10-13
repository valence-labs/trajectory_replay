#!/usr/bin/env python3
"""Simplify Tx dump split columns and check them against STATE tomls.

Replogle: one ``split`` column (hepg2.toml few-shot).
PBMC: one ``split`` column; STATE val cytokines on finetune donors go to
``finetune`` (few-shot train), not ``test``.

Does not rewrite ``X``, PCA, or the SE ``.npy`` sidecars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajectory_replay.data.tx_preprocess import (
    decode_h5ad_categorical,
    h5ad_obs_columns,
    read_h5ad_categorical,
    simplify_pbmc_dump,
    simplify_replogle_dump,
)
from trajectory_replay.data.tx_state_splits import (
    PBMC_CONTROL_PERTURBATION,
    PBMC_FINETUNE_DONORS,
    PBMC_STATE_TEST_CYTOKINES,
    PBMC_STATE_VAL_CYTOKINES,
    REPLOGLE_CONTROL_PERTURBATION,
    load_replogle_state_test_genes,
    mismatch_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _preview(path: Path) -> dict[str, object]:
    out: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return out
    cols = h5ad_obs_columns(path)
    out["obs_columns"] = cols
    for col in ("split1", "split2", "split3", "split"):
        if col in cols:
            codes, cats = read_h5ad_categorical(path, col)
            out[col] = {cat: int((codes == i).sum()) for i, cat in enumerate(cats)}
    return out


def validate_replogle(path: Path) -> dict[str, object]:
    test_genes = load_replogle_state_test_genes()
    cell_line = decode_h5ad_categorical(path, "cell_line")
    gene = decode_h5ad_categorical(path, "gene")
    actual = decode_h5ad_categorical(path, "split")
    expected = np.full(actual.shape, "train", dtype=object)
    hepg2 = cell_line == "hepg2"
    expected[hepg2] = "finetune"
    test_mask = (
        hepg2
        & np.isin(gene.astype(str), list(test_genes))
        & (gene.astype(str) != REPLOGLE_CONTROL_PERTURBATION)
    )
    expected[test_mask] = "test"
    hepg2 = cell_line == "hepg2"
    hepg2_genes = set(map(str, gene[hepg2]))
    return {
        **mismatch_summary(cell_line, gene, actual, expected),
        "n_toml_test_genes": len(test_genes),
        "toml_test_genes_missing_on_hepg2": sorted(test_genes - hepg2_genes),
        "hepg2_genes_labeled_test_not_in_toml": sorted(
            {str(g) for g in gene[hepg2 & (actual == "test")]} - test_genes
        ),
    }


def validate_pbmc(path: Path) -> dict[str, object]:
    donor = decode_h5ad_categorical(path, "donor")
    cytokine = decode_h5ad_categorical(path, "cytokine")
    actual = decode_h5ad_categorical(path, "split")
    expected = np.full(actual.shape, "train", dtype=object)
    ft = np.isin(donor, PBMC_FINETUNE_DONORS)
    expected[ft] = "finetune"
    expected[
        ft
        & np.isin(cytokine, PBMC_STATE_TEST_CYTOKINES)
        & (cytokine != PBMC_CONTROL_PERTURBATION)
    ] = "test"
    ft = np.isin(donor, PBMC_FINETUNE_DONORS)
    ft_cyto = set(map(str, cytokine[ft]))
    return {
        **mismatch_summary(donor, cytokine, actual, expected),
        "n_toml_test_cytokines": len(PBMC_STATE_TEST_CYTOKINES),
        "toml_test_missing_on_finetune_donors": sorted(
            set(PBMC_STATE_TEST_CYTOKINES) - ft_cyto
        ),
        "toml_val_missing_on_finetune_donors": sorted(
            set(PBMC_STATE_VAL_CYTOKINES) - ft_cyto
        ),
        "finetune_donor_test_cytokines_not_in_toml": sorted(
            {str(c) for c in cytokine[ft & (actual == "test")]}
            - set(PBMC_STATE_TEST_CYTOKINES)
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--replogle",
        type=Path,
        default=REPO_ROOT / "data_dir/replogle_with_splits.h5ad",
    )
    p.add_argument(
        "--pbmc",
        type=Path,
        default=REPO_ROOT / "data_dir/pbmc_parse_with_splits.h5ad",
    )
    p.add_argument("--skip-replogle", action="store_true")
    p.add_argument("--skip-pbmc", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()

    report: dict[str, object] = {}
    if not args.skip_replogle:
        report["replogle_before"] = _preview(args.replogle)
        if not args.dry_run and not args.validate_only:
            report["replogle"] = simplify_replogle_dump(args.replogle)
        if not args.dry_run:
            report["replogle_after"] = _preview(args.replogle)
            report["replogle_vs_state"] = validate_replogle(args.replogle)
    if not args.skip_pbmc:
        report["pbmc_before"] = _preview(args.pbmc)
        if not args.dry_run and not args.validate_only:
            report["pbmc"] = simplify_pbmc_dump(args.pbmc)
        if not args.dry_run:
            report["pbmc_after"] = _preview(args.pbmc)
            report["pbmc_vs_state"] = validate_pbmc(args.pbmc)
    print(json.dumps(report, indent=2))
    mismatches = []
    for key in ("replogle_vs_state", "pbmc_vs_state"):
        block = report.get(key)
        if isinstance(block, dict) and int(block.get("n_mismatch") or 0) != 0:
            mismatches.append(key)
    if mismatches:
        raise SystemExit(f"STATE split mismatch: {', '.join(mismatches)}")


if __name__ == "__main__":
    main()
