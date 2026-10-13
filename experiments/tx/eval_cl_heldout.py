#!/usr/bin/env python3
"""Sequential heldout Pearson-Δ from saved Tx CL checkpoints.

Run this after ``train_cl.py`` has finished. It reopens each task checkpoint
and scores STATE-test strata on contexts seen by that step, injecting
same-context train/finetune controls. Writes ``heldout_curve.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import numpy as np
import torch

from experiments.tx.train_cl import (
    cl_representation,
    dump_json,
    eval_provenance,
    eval_seed,
    eval_track,
    load_cl_arrays,
    model_from_checkpoint,
    task_id_from_context,
)
from trajectory_replay.data.tx_cl_tasks import (
    assert_stream_test_never_trained,
    heldout_mask,
    seen_train_mask,
    stream_for,
)
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank


def load_run_args(out_dir: Path) -> Namespace:
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    args = Namespace(**cfg)
    args.h5ad_path = Path(args.h5ad_path)
    args.representation = str(getattr(args, "representation", "se") or "se")
    se_raw = getattr(args, "se_path", None)
    args.se_path = Path(se_raw) if se_raw else args.h5ad_path
    args.out_dir = Path(args.out_dir)
    cols = args.covariate_cols or []
    if isinstance(cols, str):
        cols = [x.strip() for x in cols.split(",") if x.strip()]
    args.covariate_cols = list(cols)
    return args


def reusable_final_heldout(
    out_dir: Path,
    cfg_args: Namespace,
    task_id: int,
) -> dict | None:
    """``train_cl.py``'s heldout track, when it is the identical computation.

    The last point of this curve and the ``heldout`` entry of
    ``final_metrics.json`` answer the same question on the same checkpoint.
    Reusing it keeps one number per ``(run, task)`` instead of two that can
    drift apart, and saves a redundant ODE pass.
    """
    path = out_dir / "final_metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    track = (payload.get("final") or {}).get("heldout")
    stored = payload.get("eval_provenance")
    if not track or not stored:
        return None
    want = eval_provenance(cfg_args, task_id, tracks=("heldout",))
    if any(stored.get(key) != value for key, value in want.items() if key != "track_seeds"):
        return None
    stored_seeds = stored.get("track_seeds") or {}
    if stored_seeds.get("heldout") != want["track_seeds"]["heldout"]:
        return None
    return track


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args_cli = p.parse_args()

    out_dir = args_cli.out_dir
    cfg_args = load_run_args(out_dir)
    device = torch.device(args_cli.device)
    stream = stream_for(cfg_args.dataset)
    n_tasks = stream.n_tasks
    if int(getattr(cfg_args, "max_tasks", 0) or 0) > 0:
        n_tasks = min(n_tasks, int(cfg_args.max_tasks))
    if getattr(cfg_args, "smoke", False):
        n_tasks = min(n_tasks, 2)

    covariates = tuple(cfg_args.covariate_cols)
    arrays = load_cl_arrays(
        cfg_args.h5ad_path,
        cfg_args.se_path,
        stream,
        covariate_cols=covariates,
        control_perturbation=cfg_args.control_perturbation,
        representation=cl_representation(cfg_args),
    )
    context_ids = task_id_from_context(stream, arrays.context)
    assert_stream_test_never_trained(
        stream,
        arrays.context,
        arrays.split,
        arrays.perturbation,
        n_tasks=n_tasks,
    )
    adata = ad.read_h5ad(cfg_args.h5ad_path, backed="r")
    decoder_dir = Path(getattr(cfg_args, "decoder_dir", out_dir / "decoders"))
    decoders = TaskDecoderBank.load_dir(decoder_dir)

    print("=== Sequential heldout curve ===", flush=True)
    curve: list[dict] = []
    for k in range(n_tasks):
        ckpt = out_dir / "tasks" / f"{k:02d}" / "checkpoint.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        test_idx = np.flatnonzero(heldout_mask(stream, k, arrays.context, arrays.split))
        reused = (
            reusable_final_heldout(out_dir, cfg_args, k)
            if k == n_tasks - 1
            else None
        )
        if test_idx.size == 0:
            track = {"n_cells": 0, "selection_score": float("nan"), "rows": []}
            track["source"] = "empty"
        elif reused is not None:
            track = dict(reused)
            track["source"] = "final_metrics.json"
        else:
            model, _ = model_from_checkpoint(ckpt, device)
            reference_idx = np.flatnonzero(
                seen_train_mask(stream, k, arrays.context, arrays.split)
            )
            track = eval_track(
                model,
                arrays,
                adata,
                decoders,
                test_idx,
                context_ids,
                cfg_args,
                device,
                seed=eval_seed(cfg_args.seed, "heldout", k),
                control_reference=reference_idx,
            )
            track["source"] = "recomputed"
        curve.append({"task_id": k, "heldout": track})
        print(
            f"  T{k}: heldout Pearson-Δ="
            f"{track.get('pearson_delta_macro', float('nan')):.5f} "
            f"n={track.get('n_cells', 0)} [{track['source']}]",
            flush=True,
        )

    payload = {
        "max_cells_per_stratum": int(cfg_args.eval_max_cells_per_stratum),
        "eval_provenance": eval_provenance(
            cfg_args,
            n_tasks - 1,
            tracks=("heldout",),
        ),
        "heldout_curve": curve,
    }
    dump_json(out_dir / "heldout_curve.json", payload)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Wrote {out_dir / 'heldout_curve.json'}", flush=True)


if __name__ == "__main__":
    main()
