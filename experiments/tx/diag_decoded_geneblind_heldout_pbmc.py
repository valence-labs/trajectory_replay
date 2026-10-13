#!/usr/bin/env python3
"""PBMC held-out reference predictors in decoded PCA-100 gene space.

Scores B1/B3/B4/O1/C2 on target-donor ``test`` cytokines. Donor-aware mode
uses donor/cell-type/cytokine strata and target-donor affine decoders.
Cell-type-only mode pools donors before constructing and scoring references
and uses one affine decoder calibrated across all target donors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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

from experiments.tx.train_cl import DEFAULTS
from trajectory_replay.data.tx_cl_data import load_gene_rows
from trajectory_replay.data.tx_cl_tasks import (
    FINETUNE_SPLIT,
    PBMC_FEWSHOT_ORDER,
    PBMC_PRETRAIN_DONORS,
    TEST_SPLIT,
    TRAIN_SPLIT,
)
from trajectory_replay.data.tx_preprocess import decode_h5ad_categorical
from trajectory_replay.data.tx_state_splits import PBMC_CONTROL_PERTURBATION
from trajectory_replay.data.tx_task_decoder import (
    TaskDecoderBank,
    TaskLinearDecoder,
)
from trajectory_replay.eval.tx_cl_pearson import pearson_vector

CONTROL = PBMC_CONTROL_PERTURBATION
Stratum = tuple[str, str, str]
Pool = tuple[str, str]


@dataclass(frozen=True)
class GroupMean:
    z: np.ndarray
    x: np.ndarray
    n: int


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=DEFAULTS["pbmc"]["h5ad_pca"])
    p.add_argument(
        "--pca-path",
        type=Path,
        default=REPO_ROOT / "data_dir/pbmc/pca100_embedding.npy",
    )
    p.add_argument(
        "--decoder-dir",
        type=Path,
        default=DEFAULTS["pbmc"]["bucket"] / "decoders_pca100_affine",
    )
    p.add_argument(
        "--conditioning",
        choices=("donor_cell_type", "cell_type"),
        default="donor_cell_type",
        help="Keep donor-specific strata/decoders or pool donors strictly by cell type.",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=0,
        help="Maximum perturbed cells per donor/cell-type/cytokine; <=0 keeps all.",
    )
    p.add_argument(
        "--control-cap",
        type=int,
        default=0,
        help="Maximum PBS cells per donor/cell-type; <=0 keeps all.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT
        / "results_dir/tx_results/pbmc/diag_decoded_geneblind_heldout_pca100.json",
    )
    return p


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def macro_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    return _finite_mean(
        [pearson_vector(true[i], pred[i]) for i in range(true.shape[0])]
    )


def pool_group_means_by_cell_type(
    means: dict[Stratum, GroupMean],
) -> dict[Stratum, GroupMean]:
    """Cell-count-weighted pooling over donors for each cell type/cytokine."""
    grouped: dict[tuple[str, str], list[GroupMean]] = {}
    for (_donor, cell_type, cytokine), mean in means.items():
        grouped.setdefault((cell_type, cytokine), []).append(mean)
    pooled: dict[Stratum, GroupMean] = {}
    for (cell_type, cytokine), rows in grouped.items():
        total = sum(row.n for row in rows)
        if total <= 0:
            continue
        z = sum((row.z * row.n for row in rows), start=np.zeros_like(rows[0].z))
        x = sum((row.x * row.n for row in rows), start=np.zeros_like(rows[0].x))
        pooled[("pooled", cell_type, cytokine)] = GroupMean(
            z=z / total,
            x=x / total,
            n=total,
        )
    return pooled


def residualize_within_pools(
    values: np.ndarray, pools: list[Pool]
) -> np.ndarray:
    """Center rows within each target donor/cell-type pool."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(pools):
        raise ValueError(f"values {arr.shape} do not align with {len(pools)} pools")
    out = np.empty_like(arr)
    members: dict[Pool, list[int]] = {}
    for i, pool in enumerate(pools):
        members.setdefault(pool, []).append(i)
    for idx in members.values():
        take = np.asarray(idx, dtype=np.int64)
        out[take] = arr[take] - arr[take].mean(axis=0)
    return out


def average_source_effects(
    effects: dict[Stratum, np.ndarray],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """Return B4 cell-type/cytokine donor means and B3 cell-type means."""
    by_type_cytokine: dict[tuple[str, str], list[np.ndarray]] = {}
    for (_donor, cell_type, cytokine), effect in effects.items():
        by_type_cytokine.setdefault((cell_type, cytokine), []).append(effect)
    b4 = {
        key: np.mean(np.stack(rows), axis=0)
        for key, rows in by_type_cytokine.items()
    }
    by_type: dict[str, list[np.ndarray]] = {}
    for (cell_type, _cytokine), effect in b4.items():
        by_type.setdefault(cell_type, []).append(effect)
    b3 = {
        cell_type: np.mean(np.stack(rows), axis=0)
        for cell_type, rows in by_type.items()
    }
    return b4, b3


def decode_delta(decoder: TaskLinearDecoder, z_delta: np.ndarray) -> np.ndarray:
    z = np.asarray(z_delta, dtype=np.float64)
    if z.ndim == 1:
        return decoder.decode(z[None, :])[0] - decoder.intercept
    return decoder.decode(z) - decoder.intercept


def reachability_projection(
    decoder: TaskLinearDecoder, true_delta: np.ndarray
) -> np.ndarray:
    """Exact Pearson-reachable projection onto the centered decoder columns."""
    true = np.asarray(true_delta, dtype=np.float64)
    centered_true = true - true.mean(axis=1, keepdims=True)
    weight = np.asarray(decoder.weight, dtype=np.float64)
    centered_weight = weight - weight.mean(axis=0, keepdims=True)
    z, *_ = np.linalg.lstsq(centered_weight, centered_true.T, rcond=None)
    return decode_delta(decoder, z.T)


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, labels = pd.factorize(values, sort=True)
    if np.any(codes < 0):
        raise ValueError("Missing labels are not supported")
    return np.asarray(codes, dtype=np.int64), np.asarray(labels).astype(str)


def _cap(
    indices: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    if cap <= 0 or indices.size <= cap:
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=cap, replace=False))


def collect_group_means(
    adata: ad.AnnData,
    pca: np.ndarray,
    donor_codes: np.ndarray,
    donor_labels: np.ndarray,
    cell_type_codes: np.ndarray,
    cell_type_labels: np.ndarray,
    cytokine_codes: np.ndarray,
    cytokine_labels: np.ndarray,
    mask: np.ndarray,
    *,
    cap: int,
    control_cap: int,
    rng: np.random.Generator,
    label: str,
) -> dict[Stratum, GroupMean]:
    """Group selected rows once, avoiding one full-array scan per stratum."""
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise RuntimeError(f"{label}: no selected cells")
    n_cell_types = int(cell_type_labels.size)
    n_cytokines = int(cytokine_labels.size)
    keys = (
        (donor_codes[selected] * n_cell_types + cell_type_codes[selected])
        * n_cytokines
        + cytokine_codes[selected]
    )
    order = np.argsort(keys, kind="stable")
    selected = selected[order]
    sorted_keys = keys[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1, selected.size]
    out: dict[Stratum, GroupMean] = {}
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        first = selected[start]
        donor = str(donor_labels[donor_codes[first]])
        cell_type = str(cell_type_labels[cell_type_codes[first]])
        cytokine = str(cytokine_labels[cytokine_codes[first]])
        group = selected[start:stop]
        limit = control_cap if cytokine == CONTROL else cap
        take = _cap(group, limit, rng)
        z = np.asarray(pca[take], dtype=np.float64).mean(axis=0)
        x = load_gene_rows(adata, take).mean(axis=0, dtype=np.float64)
        out[(donor, cell_type, cytokine)] = GroupMean(z=z, x=x, n=int(take.size))
    print(
        f"{label}: {selected.size:,} cells, {len(out):,} strata",
        flush=True,
    )
    return out


def effects_from_controls(
    means: dict[Stratum, GroupMean],
) -> tuple[dict[Stratum, np.ndarray], dict[Stratum, np.ndarray]]:
    controls = {
        (donor, cell_type): mean
        for (donor, cell_type, cytokine), mean in means.items()
        if cytokine == CONTROL
    }
    z_effects: dict[Stratum, np.ndarray] = {}
    x_effects: dict[Stratum, np.ndarray] = {}
    for key, mean in means.items():
        donor, cell_type, cytokine = key
        if cytokine == CONTROL:
            continue
        control = controls.get((donor, cell_type))
        if control is None:
            continue
        z_effects[key] = mean.z - control.z
        x_effects[key] = mean.x - control.x
    return z_effects, x_effects


def fit_blind_arrows(
    effects: dict[Stratum, np.ndarray],
) -> dict[Pool, np.ndarray]:
    grouped: dict[Pool, list[np.ndarray]] = {}
    for (donor, cell_type, _cytokine), effect in effects.items():
        grouped.setdefault((donor, cell_type), []).append(effect)
    return {
        pool: np.mean(np.stack(rows), axis=0) for pool, rows in grouped.items()
    }


def record(
    code: str,
    name: str,
    pred: np.ndarray,
    true: np.ndarray,
    pools: list[Pool],
    **extra: Any,
) -> dict[str, Any]:
    pred_res = residualize_within_pools(pred, pools)
    true_res = residualize_within_pools(true, pools)
    residual = (
        0.0
        if float(np.linalg.norm(pred_res)) <= 1e-12
        else macro_pearson(pred_res, true_res)
    )
    row = {
        "code": code,
        "name": name,
        "n_strata": int(true.shape[0]),
        "pearson_delta_raw": macro_pearson(pred, true),
        "pearson_delta_residual": residual,
        **extra,
    }
    print(
        f"{code:2s} {name:45s} rawΔ={row['pearson_delta_raw']:.4f} "
        f"residualΔ={row['pearson_delta_residual']:.4f}",
        flush=True,
    )
    return row


def _require_affine_decoder(bank: TaskDecoderBank, task_id: int) -> TaskLinearDecoder:
    decoder = bank.get(task_id)
    if not isinstance(decoder, TaskLinearDecoder) or decoder.identity:
        raise TypeError(f"task {task_id} decoder must be a non-identity affine decoder")
    return decoder


def main() -> None:
    args = make_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"h5ad={args.h5ad_path}", flush=True)
    print(f"pca={args.pca_path}", flush=True)
    print(f"decoders={args.decoder_dir}", flush=True)
    print(f"conditioning={args.conditioning}", flush=True)

    donor = decode_h5ad_categorical(args.h5ad_path, "donor")
    cell_type = decode_h5ad_categorical(args.h5ad_path, "cell_type")
    cytokine = decode_h5ad_categorical(args.h5ad_path, "cytokine")
    split = decode_h5ad_categorical(args.h5ad_path, "split")
    donor_codes, donor_labels = _factorize(donor)
    cell_type_codes, cell_type_labels = _factorize(cell_type)
    cytokine_codes, cytokine_labels = _factorize(cytokine)

    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    pca = np.load(args.pca_path, mmap_mode="r")
    if pca.ndim != 2 or pca.shape[0] != adata.n_obs:
        raise ValueError(f"PCA shape {pca.shape} does not match h5ad n_obs={adata.n_obs}")
    bank = TaskDecoderBank.load_dir(args.decoder_dir)
    if args.conditioning == "cell_type":
        decoders = {"pooled": _require_affine_decoder(bank, 0)}
    else:
        decoders = {
            target: _require_affine_decoder(bank, task_id)
            for task_id, target in enumerate(PBMC_FEWSHOT_ORDER, start=1)
        }
    if any(decoder.se_dim != 100 for decoder in decoders.values()):
        dims = {target: decoder.se_dim for target, decoder in decoders.items()}
        raise ValueError(f"expected PCA-100 target decoders, got {dims}")
    if any(decoder.n_genes != adata.n_vars for decoder in decoders.values()):
        sizes = {target: decoder.n_genes for target, decoder in decoders.items()}
        raise ValueError(f"decoder gene widths {sizes} do not match h5ad n_vars={adata.n_vars}")
    if pca.shape[1] != 100:
        raise ValueError(f"expected PCA width 100, got {pca.shape[1]}")

    is_target = np.isin(donor, PBMC_FEWSHOT_ORDER)
    is_t0 = np.isin(donor, PBMC_PRETRAIN_DONORS)
    target_test = is_target & (split == TEST_SPLIT) & (cytokine != CONTROL)
    target_fit = is_target & (split == FINETUNE_SPLIT)
    t0_train = is_t0 & (split == TRAIN_SPLIT)

    test_means = collect_group_means(
        adata, pca, donor_codes, donor_labels, cell_type_codes, cell_type_labels,
        cytokine_codes, cytokine_labels, target_test,
        cap=args.cap, control_cap=args.control_cap, rng=rng, label="target test",
    )
    fit_means = collect_group_means(
        adata, pca, donor_codes, donor_labels, cell_type_codes, cell_type_labels,
        cytokine_codes, cytokine_labels, target_fit,
        cap=args.cap, control_cap=args.control_cap, rng=rng, label="target finetune",
    )
    t0_means = collect_group_means(
        adata, pca, donor_codes, donor_labels, cell_type_codes, cell_type_labels,
        cytokine_codes, cytokine_labels, t0_train,
        cap=args.cap, control_cap=args.control_cap, rng=rng, label="T0 train",
    )
    if args.conditioning == "cell_type":
        test_means = pool_group_means_by_cell_type(test_means)
        fit_means = pool_group_means_by_cell_type(fit_means)
        t0_means = pool_group_means_by_cell_type(t0_means)

    # Test controls are intentionally borrowed from the same target
    # donor/cell-type finetune pool, matching the PBMC held-out protocol.
    fit_controls = {
        (d, c): mean
        for (d, c, p), mean in fit_means.items()
        if p == CONTROL
    }
    true_x: dict[Stratum, np.ndarray] = {}
    for key, mean in test_means.items():
        control = fit_controls.get((key[0], key[1]))
        if control is not None:
            true_x[key] = mean.x - control.x

    fit_z, _ = effects_from_controls(fit_means)
    t0_z_all, _ = effects_from_controls(t0_means)
    heldout_cytokines = {key[2] for key in true_x}
    t0_z = {
        key: effect
        for key, effect in t0_z_all.items()
        if key[2] in heldout_cytokines
    }
    b1 = fit_blind_arrows(fit_z)
    b4, b3 = average_source_effects(t0_z)
    source_center = {
        key: effect - b3[key[0]] for key, effect in b4.items()
    }

    all_test_keys = sorted(true_x)
    keys = [
        key
        for key in all_test_keys
        if (key[0], key[1]) in b1
        and key[1] in b3
        and (key[1], key[2]) in b4
    ]
    if not keys:
        raise RuntimeError("No target test strata have all required references")
    dropped = sorted(set(all_test_keys) - set(keys))
    print(
        f"matched test strata={len(keys):,}; dropped for missing references={len(dropped):,}",
        flush=True,
    )

    pools = [(donor_name, cell_type_name) for donor_name, cell_type_name, _ in keys]
    true = np.stack([true_x[key] for key in keys])
    z_b1 = np.stack([b1[(key[0], key[1])] for key in keys])
    z_b3 = np.stack([b3[key[1]] for key in keys])
    z_b4 = np.stack([b4[(key[1], key[2])] for key in keys])
    z_o1 = np.stack(
        [b1[(key[0], key[1])] + source_center[(key[1], key[2])] for key in keys]
    )

    def decode_by_target(z: np.ndarray) -> np.ndarray:
        out = np.empty((len(keys), decoders[keys[0][0]].n_genes), dtype=np.float64)
        for target, decoder in decoders.items():
            idx = np.asarray(
                [i for i, key in enumerate(keys) if key[0] == target],
                dtype=np.int64,
            )
            if idx.size:
                out[idx] = decode_delta(decoder, z[idx])
        return out

    pred_b1 = decode_by_target(z_b1)
    pred_b3 = decode_by_target(z_b3)
    pred_b4 = decode_by_target(z_b4)
    pred_o1 = decode_by_target(z_o1)
    pred_c2 = np.empty_like(true)
    for target, decoder in decoders.items():
        idx = np.asarray(
            [i for i, key in enumerate(keys) if key[0] == target],
            dtype=np.int64,
        )
        if idx.size:
            pred_c2[idx] = reachability_projection(decoder, true[idx])

    rows = [
        record(
            "B1", "target finetune cytokine mean", pred_b1, true, pools,
            space="pca_target_decoder", leak=False, cytokine_specific=False,
        ),
        record(
            "B3", "T0 generic heldout-cytokine mean", pred_b3, true, pools,
            space="pca_target_decoder", leak=False, cytokine_specific=False,
        ),
        record(
            "B4", "T0 per-cytokine transfer", pred_b4, true, pools,
            space="pca_target_decoder", leak=False, cytokine_specific=True,
        ),
        record(
            "O1", "target arrow plus centered T0 residual", pred_o1, true, pools,
            space="pca_target_decoder", leak=False, cytokine_specific=True,
        ),
        record(
            "C2", "exact affine-decoder reachability projection", pred_c2, true, pools,
            space="pca_target_decoder", leak=True, oracle=True,
            exact_projection=True,
        ),
    ]

    out = {
        "dataset": "pbmc",
        "h5ad": str(args.h5ad_path),
        "pca": str(args.pca_path),
        "decoder_dir": str(args.decoder_dir),
        "conditioning": str(args.conditioning),
        "cap": int(args.cap),
        "control_cap": int(args.control_cap),
        "seed": int(args.seed),
        "target_donors": list(PBMC_FEWSHOT_ORDER),
        "t0_donors": list(PBMC_PRETRAIN_DONORS),
        "heldout_split": TEST_SPLIT,
        "fit_split": FINETUNE_SPLIT,
        "context": "donor" if args.conditioning == "donor_cell_type" else "cell_type",
        "covariate": (
            "cell_type" if args.conditioning == "donor_cell_type" else None
        ),
        "perturbation": "cytokine",
        "control": CONTROL,
        "n_test_strata_observed": len(all_test_keys),
        "n_test_strata_scored": len(keys),
        "dropped_strata": [list(key) for key in dropped],
        "residualization": (
            "within each target donor/cell_type pool"
            if args.conditioning == "donor_cell_type"
            else "within each pooled cell_type"
        ),
        "l1_discrimination": {
            "included": False,
            "reason": (
                "PBMC Cell-Eval pools full real and generated cell distributions "
                "across donors within cell_type. These references define one centroid "
                "per donor/cell_type/cytokine, so centroid discrimination would not "
                "faithfully reproduce that pooling contract."
            ),
        },
        "predictors": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)
    if getattr(adata, "file", None) is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
