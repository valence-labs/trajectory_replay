"""STATE preprint few-shot specs used to label Tx dumps.

Parse donor.toml: https://github.com/ArcInstitute/state/blob/main/state_preprint_toml_files/parse_tomls/donor.toml
Replogle hepg2.toml: https://github.com/ArcInstitute/state/blob/main/state_preprint_toml_files/replogle_tomls/hepg2.toml

Our dumps map STATE cell-type few-shot onto CL phases:

* other contexts → ``train``
* few-shot context, not in the toml test list → ``finetune`` (includes STATE ``val``)
* few-shot context, toml ``test`` perturbations → ``test``
* controls are always trainable: ``train`` in pretrain contexts and
  ``finetune`` in few-shot contexts, even if STATE lists the control as test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLOGLE_TEST_GENES_PATH = REPO_ROOT / "configs/replogle_hepg2_state_test_genes.txt"

PBMC_FINETUNE_DONORS = ("Donor1", "Donor4", "Donor9", "Donor12")
PBMC_CONTROL_PERTURBATION = "PBS"
PBMC_STATE_VAL_CYTOKINES = ("G-CSF", "IFN-beta", "M-CSF", "IGF-1")
PBMC_STATE_TEST_CYTOKINES = (
    "C5a",
    "TWEAK",
    "LIF",
    "IL-17C",
    "FGF-beta",
    "BAFF",
    "FLT3L",
    "IFN-lambda2",
    "4-1BBL",
    "IL-17E",
    "IL-2",
    "IL-19",
    "PRL",
    "IL-16",
    "IL-7",
    "GM-CSF",
    "IL-17A",
    "IL-13",
    "IL-36-alpha",
    "PSPN",
    "EPO",
    "IL-9",
    "IL-31",
    "IL-11",
    "LT-alpha1-beta2",
    "TL1A",
    "IL-32-beta",
    "IFN-omega",
    "IL-8",
    "Noggin",
    "IFN-epsilon",
    "IL-17F",
    "Leptin",
    "IL-33",
    "IL-35",
    "OSM",
    "IL-34",
    "HGF",
    "IL-1Ra",
    "LIGHT",
    "CD27L",
    "SCF",
    "IL-1-alpha",
    "TGF-beta1",
    "CD30L",
    "IL-24",
    "IFN-gamma",
    "IFN-alpha1",
    "IFN-lambda1",
    "GITRL",
    "VEGF",
    "ADSF",
    "IL-15",
    "EGF",
    "IL-3",
    "IL-17B",
    "IL-6",
    "IL-4",
    "IL-10",
    "CT-1",
    "IFN-lambda3",
    "TRAIL",
)
REPLOGLE_FEWSHOT_CELL_LINE = "hepg2"
REPLOGLE_CONTROL_PERTURBATION = "non-targeting"


def load_replogle_state_test_genes(path: Path | None = None) -> frozenset[str]:
    p = Path(path) if path is not None else REPLOGLE_TEST_GENES_PATH
    genes = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return frozenset(genes)


def expected_pbmc_phase(donor: str, cytokine: str) -> str:
    """Map a PBMC cell onto our dump phases (STATE val folded into finetune)."""
    if donor not in PBMC_FINETUNE_DONORS:
        return "train"
    if cytokine == PBMC_CONTROL_PERTURBATION:
        return "finetune"
    if cytokine in PBMC_STATE_TEST_CYTOKINES:
        return "test"
    return "finetune"


def expected_replogle_phase(
    cell_line: str,
    gene: str,
    *,
    test_genes: frozenset[str] | None = None,
) -> str:
    """Map a Replogle cell onto our phases, keeping controls trainable."""
    if cell_line != REPLOGLE_FEWSHOT_CELL_LINE:
        return "train"
    if gene == REPLOGLE_CONTROL_PERTURBATION:
        return "finetune"
    genes = test_genes if test_genes is not None else load_replogle_state_test_genes()
    if gene in genes:
        return "test"
    return "finetune"


def mismatch_summary(
    context: np.ndarray,
    perturbation: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_rows: int = 12,
) -> dict[str, object]:
    """Count cells whose dump label disagrees with the STATE-derived expectation."""
    bad = actual != expected
    n_bad = int(bad.sum())
    examples: list[dict[str, str]] = []
    if n_bad:
        idx = np.flatnonzero(bad)
        seen: set[tuple[str, str, str, str]] = set()
        for i in idx:
            key = (
                str(context[i]),
                str(perturbation[i]),
                str(actual[i]),
                str(expected[i]),
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "context": key[0],
                    "perturbation": key[1],
                    "actual": key[2],
                    "expected": key[3],
                }
            )
            if len(examples) >= max_rows:
                break
    return {"n_mismatch": n_bad, "n_cells": int(actual.size), "examples": examples}
