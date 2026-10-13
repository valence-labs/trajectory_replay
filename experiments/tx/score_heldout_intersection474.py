#!/usr/bin/env python3
"""Restrict stored heldout Pearson-Δ rows to the 474-gene T0 intersection.

Does not regenerate cells. Residual macros here average the *already
945-pool-residualized* per-stratum scores on the 474 identities; that is
not the same as re-residualizing against the 474-pool mean.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from trajectory_replay.data.tx_cl_tasks import REPLOGLE_PRETRAIN_CELL_LINES
from trajectory_replay.data.tx_dataset import build_encoding
from trajectory_replay.data.tx_preprocess import decode_h5ad_categorical

CONTROL = "non-targeting"
HEPG2 = "hepg2"
TEST_SPLIT = "test"
TRAIN_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--h5ad-path",
        type=Path,
        default=Path("data_dir/replogle/adata.h5ad"),
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results_dir/tx_results/replogle"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results_dir/tx_results/replogle/diag_heldout_intersection474.json"),
    )
    return p.parse_args()


def mean_se(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    if n < 2:
        return m, float("nan")
    return m, float(arr.std(ddof=1) / math.sqrt(n))


def run_label(cfg: dict[str, Any]) -> str:
    strategy = str(cfg.get("strategy"))
    scope = str(cfg.get("finetune_scope", "full"))
    lm = str(cfg.get("lambda_mode") or "")
    be = str(cfg.get("replay_backend") or "ode")
    mf = int(cfg.get("mean_flow_steps") or 0)
    if strategy == "real":
        return f"real {lm}"
    if strategy in ("generative", "trajectory"):
        backend = f"MF-N{mf}" if be == "mean_flow" else "ODE"
        return f"{strategy} {backend} {lm}"
    if strategy == "lora":
        return f"lora r{cfg.get('lora_rank')}"
    if strategy == "gft":
        return f"gft {cfg.get('gft_beta_schedule')}"
    if strategy == "two_stage":
        return f"two-stage {cfg.get('two_stage_1_epoch_fraction')}"
    if strategy == "standard":
        p = int(bool(cfg.get("finetune_train_old_pert_embeddings")))
        c = int(bool(cfg.get("finetune_train_old_context_embeddings")))
        return f"naive {scope} p{p}c{c}"
    return f"{strategy} {scope} {lm}".strip()


def main() -> None:
    args = parse_args()
    path = args.h5ad_path
    ctx = decode_h5ad_categorical(path, "cell_line")
    gene = decode_h5ad_categorical(path, "gene")
    split = decode_h5ad_categorical(path, "split")
    encoding = build_encoding(ctx, gene, control_perturbation=CONTROL)
    pert_of = encoding.perturbation_categories

    held = {
        str(g)
        for g, c, s in zip(gene, ctx, split)
        if c == HEPG2 and s == TEST_SPLIT and str(g) != CONTROL
    }
    t0_by_line: dict[str, set[str]] = {}
    for line in REPLOGLE_PRETRAIN_CELL_LINES:
        t0_by_line[line] = {
            str(g)
            for g, c, s in zip(gene, ctx, split)
            if c == line and s == TRAIN_SPLIT and str(g) != CONTROL
        }
    common = sorted(held.intersection(*t0_by_line.values()))
    common_idx = {encoding.perturbation_to_idx[g] for g in common}
    print(
        f"held={len(held)} intersection={len(common)} "
        f"n_pert_cats={len(pert_of)}",
        flush=True,
    )
    if len(common) != 474:
        raise SystemExit(f"expected 474-gene intersection, got {len(common)}")

    per_run: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for metrics_path in sorted(args.runs_root.glob("*pca50*_st100000_ft50000_s*/final_metrics.json")):
        cfg_path = metrics_path.parent / "config.json"
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text())
        if int(cfg.get("pca_dim") or 0) not in (0, 50) and cfg.get("representation") != "pca":
            continue
        heldout = json.loads(metrics_path.read_text())["final"]["heldout"]
        rows = heldout.get("rows") or []
        sub = [r for r in rows if int(r["key"][1]) in common_idx]
        raw = [float(r["pearson_delta"]) for r in sub]
        resid = [float(r["pearson_delta_residual"]) for r in sub]
        rec = {
            "run_dir": str(metrics_path.parent),
            "label": run_label(cfg),
            "seed": cfg.get("seed"),
            "n_heldout_rows": len(rows),
            "n_intersection_rows": len(sub),
            "raw_945": float(heldout["pearson_delta_macro"]),
            "residual_945": float(heldout["pearson_delta_residual_macro"]),
            "raw_474": float(np.mean(raw)) if raw else float("nan"),
            "residual_474_from_945_pool": (
                float(np.mean(resid)) if resid else float("nan")
            ),
        }
        if rec["n_intersection_rows"] != 474:
            print(
                f"WARN {metrics_path.parent.name}: "
                f"{rec['n_intersection_rows']} intersection rows",
                flush=True,
            )
            continue
        per_run.append(rec)
        groups[rec["label"]].append(rec)

    summary = []
    for label, recs in sorted(groups.items()):
        row = {"label": label, "n_seeds": len(recs)}
        for key in (
            "raw_945",
            "residual_945",
            "raw_474",
            "residual_474_from_945_pool",
        ):
            m, se = mean_se([float(r[key]) for r in recs])
            row[f"{key}_mean"] = m
            row[f"{key}_se"] = se
        summary.append(row)
        print(
            f"{label:<36} raw945={row['raw_945_mean']:.3f} "
            f"raw474={row['raw_474_mean']:.3f} "
            f"res945={row['residual_945_mean']:.3f} "
            f"res474~={row['residual_474_from_945_pool_mean']:.3f}",
            flush=True,
        )

    payload = {
        "h5ad": str(path),
        "n_heldout": len(held),
        "n_intersection": len(common),
        "note": (
            "residual_474_from_945_pool averages stored residual scores "
            "computed against the 945-gene pool mean; it is not a "
            "re-residualization on the 474-gene pool."
        ),
        "oracles": {
            "B1_raw_945": 0.393,
            "B4_residual_945": 0.357,
            "B4_raw_474": 0.400,
            "B4_residual_474": 0.379,
            "O1_raw_474": 0.408,
            "O1_residual_474": 0.379,
        },
        "summary": summary,
        "runs": per_run,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
