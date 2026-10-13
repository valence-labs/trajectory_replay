#!/usr/bin/env python3
"""Is there gene-specific T0→HepG2 signal after stripping the shared direction?

For the 945 HepG2 STATE-test genes:

* delta in HepG2 test and in each T0 train line
* subtract that line's mean delta (the gene-blind direction)
* correlate residuals across lines
* split-half reliability, Spearman-Brown to the full-mean scale
* disattenuated concordance = raw / sqrt(rel_a * rel_b)
* oracle: predict HepG2 residual from the mean T0 residual of the same gene

A gene-blind predictor scores ~0 on residual Pearson-Δ by construction.
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

from experiments.tx.train_cl import DEFAULTS
from trajectory_replay.data.tx_cl_data import load_gene_rows
from trajectory_replay.data.tx_cl_tasks import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    REPLOGLE_PRETRAIN_CELL_LINES,
)
from trajectory_replay.data.tx_preprocess import decode_h5ad_categorical
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.eval.tx_cl_pearson import pearson_vector

CONTROL = "non-targeting"
FEWSHOT = REPLOGLE_FEWSHOT_CELL_LINE


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=DEFAULTS["replogle"]["h5ad"])
    p.add_argument("--cap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT
        / "results_dir/tx_results/replogle/diag_heldout_residual_transfer.json",
    )
    return p


def _cap(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if idx.size <= cap:
        return np.sort(idx)
    return np.sort(rng.choice(idx, cap, replace=False))


def _mean_rows(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    return load_gene_rows(adata, idx).mean(axis=0).astype(np.float64)


def _finite_mean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def spearman_brown(split_half: float) -> float:
    if not np.isfinite(split_half) or split_half >= 1.0:
        return float("nan") if not np.isfinite(split_half) else 1.0
    if split_half <= -1.0:
        return float("nan")
    return float(2.0 * split_half / (1.0 + split_half))


def residualize(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = D.mean(axis=0)
    return D - u, u


def macro_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    return _finite_mean(
        [pearson_vector(true[i], pred[i]) for i in range(true.shape[0])]
    )


def paired_macro_pearson(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.shape[0], B.shape[0])
    return _finite_mean([pearson_vector(A[i], B[i]) for i in range(n)])


def collect_line(
    adata: ad.AnnData,
    gene: np.ndarray,
    line_mask: np.ndarray,
    names: list[str],
    cap: int,
    rng: np.random.Generator,
    *,
    ctrl_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    empty = {
        "names": [],
        "D": np.empty((0, adata.n_vars)),
        "n": {},
        "n_ctrl": 0,
        "half_names": [],
        "half_a": np.empty((0, adata.n_vars)),
        "half_b": np.empty((0, adata.n_vars)),
    }
    basal = line_mask if ctrl_mask is None else ctrl_mask
    ctrl = _cap(np.flatnonzero(basal & (gene == CONTROL)), cap, rng)
    if ctrl.size == 0:
        return empty
    ctrl_mean = _mean_rows(adata, ctrl)
    rows = []
    keep = []
    n_cells = {}
    half_a = []
    half_b = []
    half_names = []
    for g in names:
        take = _cap(np.flatnonzero(line_mask & (gene == g)), cap, rng)
        if take.size == 0:
            continue
        keep.append(g)
        n_cells[g] = int(take.size)
        rows.append(_mean_rows(adata, take) - ctrl_mean)
        if take.size >= 4:
            rng.shuffle(take)
            mid = take.size // 2
            half_names.append(g)
            half_a.append(_mean_rows(adata, take[:mid]) - ctrl_mean)
            half_b.append(_mean_rows(adata, take[mid:]) - ctrl_mean)
    D = np.stack(rows, axis=0) if rows else np.empty((0, ctrl_mean.size))
    Ha = np.stack(half_a, axis=0) if half_a else np.empty((0, ctrl_mean.size))
    Hb = np.stack(half_b, axis=0) if half_b else np.empty((0, ctrl_mean.size))
    return {
        "names": keep,
        "D": D,
        "n": n_cells,
        "n_ctrl": int(ctrl.size),
        "half_names": half_names,
        "half_a": Ha,
        "half_b": Hb,
    }


def align(block: dict[str, Any], names: list[str]) -> np.ndarray | None:
    idx = {g: i for i, g in enumerate(block["names"])}
    if any(g not in idx for g in names):
        return None
    return np.stack([block["D"][idx[g]] for g in names], axis=0)


def main() -> None:
    args = make_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    path = args.h5ad_path
    print(f"h5ad={path}", flush=True)

    ctx = decode_h5ad_categorical(path, "cell_line")
    gene = decode_h5ad_categorical(path, "gene")
    split = decode_h5ad_categorical(path, "split")
    adata = ad.read_h5ad(path, backed="r")

    held_mask = (
        (ctx == FEWSHOT) & (split == TEST_SPLIT) & (gene != CONTROL)
    )
    held_names = sorted({str(g) for g in gene[held_mask]})
    print(f"heldout genes={len(held_names)}", flush=True)

    masks = {
        FEWSHOT: (ctx == FEWSHOT) & (split == TEST_SPLIT),
        **{
            line: (ctx == line) & (split == TRAIN_SPLIT)
            for line in REPLOGLE_PRETRAIN_CELL_LINES
        },
    }
    blocks: dict[str, dict[str, Any]] = {}
    hepg2_ctrl_mask = (ctx == FEWSHOT) & (split != TEST_SPLIT)
    for line, mask in masks.items():
        print(f"loading {line}...", flush=True)
        blocks[line] = collect_line(
            adata,
            gene,
            mask,
            held_names,
            args.cap,
            rng,
            ctrl_mask=hepg2_ctrl_mask if line == FEWSHOT else None,
        )
        print(
            f"  {line}: n_ctrl={blocks[line]['n_ctrl']} "
            f"n_genes={len(blocks[line]['names'])} "
            f"n_split_half={len(blocks[line]['half_names'])}",
            flush=True,
        )

    common = held_names
    for block in blocks.values():
        common = [g for g in common if g in set(block["names"])]
    print(f"genes present in HepG2 + all T0 lines={len(common)}", flush=True)

    residual: dict[str, np.ndarray] = {}
    shared: dict[str, np.ndarray] = {}
    aligned: dict[str, np.ndarray] = {}
    for line, block in blocks.items():
        D = align(block, common)
        if D is None:
            continue
        R, u = residualize(D)
        aligned[line] = D
        residual[line] = R
        shared[line] = u

    rel_raw: dict[str, dict[str, float]] = {}
    rel_res: dict[str, dict[str, float]] = {}
    for line, block in blocks.items():
        Ha, Hb = block["half_a"], block["half_b"]
        if Ha.shape[0] < 2:
            continue
        raw = paired_macro_pearson(Ha, Hb)
        Ra, _ = residualize(Ha)
        Rb, _ = residualize(Hb)
        res = paired_macro_pearson(Ra, Rb)
        rel_raw[line] = {"split_half": raw, "full_mean": spearman_brown(raw)}
        rel_res[line] = {"split_half": res, "full_mean": spearman_brown(res)}
        print(
            f"reliability {line}: raw split-half={raw:.3f} full={spearman_brown(raw):.3f}  "
            f"residual split-half={res:.3f} full={spearman_brown(res):.3f}",
            flush=True,
        )

    pairs = []
    t0_lines = list(REPLOGLE_PRETRAIN_CELL_LINES)
    for a, b in [(FEWSHOT, line) for line in t0_lines] + [
        (t0_lines[i], t0_lines[j])
        for i in range(len(t0_lines))
        for j in range(i + 1, len(t0_lines))
    ]:
        if a not in residual or b not in residual:
            continue
        raw = paired_macro_pearson(aligned[a], aligned[b])
        res = paired_macro_pearson(residual[a], residual[b])
        rel_a = rel_res.get(a, {}).get("full_mean", float("nan"))
        rel_b = rel_res.get(b, {}).get("full_mean", float("nan"))
        dis = (
            float(res / np.sqrt(rel_a * rel_b))
            if np.isfinite(res) and rel_a > 0 and rel_b > 0
            else float("nan")
        )
        rec = {
            "a": a,
            "b": b,
            "n_genes": int(residual[a].shape[0]),
            "pearson_delta_raw": raw,
            "pearson_delta_residual": res,
            "disattenuated_residual": dis,
        }
        pairs.append(rec)
        print(
            f"concordance {a} vs {b}: rawΔ={raw:.3f} residualΔ={res:.3f} "
            f"disattenuated={dis:.3f}",
            flush=True,
        )

    t0_stack = np.stack([residual[line] for line in t0_lines if line in residual], 0)
    t0_mean_res = t0_stack.mean(axis=0)
    hepg2_res = residual[FEWSHOT]
    hepg2_raw = aligned[FEWSHOT]
    oracle_res = macro_pearson(t0_mean_res, hepg2_res)
    blind_res = macro_pearson(np.zeros_like(hepg2_res), hepg2_res)
    compositional = hepg2_raw.mean(axis=0) + t0_mean_res
    oracle_raw = macro_pearson(compositional, hepg2_raw)
    blind_raw = macro_pearson(
        np.broadcast_to(hepg2_raw.mean(axis=0), hepg2_raw.shape), hepg2_raw
    )
    print(
        f"oracle T0-residual → HepG2-residual Δ={oracle_res:.3f}  "
        f"blind residual Δ={blind_res:.3f}",
        flush=True,
    )
    print(
        f"oracle (HepG2 shared + T0 residual) rawΔ={oracle_raw:.3f}  "
        f"HepG2 gene-blind rawΔ={blind_raw:.3f}",
        flush=True,
    )

    out = {
        "h5ad": str(path),
        "cap": int(args.cap),
        "seed": int(args.seed),
        "n_heldout_genes": int(len(held_names)),
        "n_genes_in_all_lines": int(len(common)),
        "reliability_raw": rel_raw,
        "reliability_residual": rel_res,
        "concordance": pairs,
        "oracle": {
            "t0_mean_residual_to_hepg2_residual": oracle_res,
            "zero_to_hepg2_residual": blind_res,
            "hepg2_shared_plus_t0_residual_raw": oracle_raw,
            "hepg2_shared_only_raw": blind_raw,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
