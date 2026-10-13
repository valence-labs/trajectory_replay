#!/usr/bin/env python3
"""Run Cell-Eval ``minimal`` on sequential Tx CL final tracks or heldout curve."""

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
import pandas as pd
import torch

from experiments.tx.cell_eval_export import clip_gene_matrix_for_celleval
from experiments.tx.run_cell_eval import (
    _agg_to_jsonable,
    _import_cell_eval,
    _pert_results_to_jsonable,
)
from experiments.tx.train_cl import (
    cap_eval_indices,
    cl_representation,
    control_reference_indices,
    dump_json,
    eval_control_cap,
    eval_gen_per_stratum,
    eval_provenance,
    eval_seed,
    load_cl_arrays,
    model_from_checkpoint,
    task_id_from_context,
)
from experiments.tx.eval_cl_heldout import load_run_args
from trajectory_replay.data.tx_cl_data import (
    load_gene_rows,
    stratified_cap_indices,
)
from trajectory_replay.data.tx_cl_tasks import (
    assert_stream_test_never_trained,
    heldout_mask,
    seen_train_mask,
    step_masks,
    stream_for,
    train_task_ids,
)
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank
from trajectory_replay.eval.tx_cl_pearson import (
    apply_pred_control_genes,
    predicted_genes_for_indices,
    stratum_generation_plan,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument("--heldout-curve", action="store_true")
    p.add_argument("--save-h5ad", action="store_true")
    p.add_argument(
        "--tracks",
        nargs="+",
        choices=("fit", "forget", "stream", "heldout"),
        help="Evaluate only these final tracks (default: all).",
    )
    return p


def _pool_keys(arrays, indices: np.ndarray) -> np.ndarray:
    cols = [arrays.c[indices]]
    cols.extend(arrays.z[col][indices] for col in arrays.covariate_col_order)
    return np.stack(cols, axis=1).astype(np.int64, copy=False)


def evaluate_indices(
    model,
    arrays,
    adata,
    decoders,
    target_indices: np.ndarray,
    task_ids: np.ndarray,
    args,
    device: torch.device,
    *,
    seed: int,
    output_dir: Path,
    control_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    MetricsEvaluator = _import_cell_eval()
    pool_donors = bool(
        getattr(args, "pool_eval_donors_by_cell_type", False)
    )
    if pool_donors:
        if "cell_type" not in arrays.z:
            raise ValueError(
                "Pooling donors by cell type requires a cell_type covariate"
            )
        target_idx = np.asarray(target_indices, dtype=np.int64)
        control = arrays.encoding.control_pert_index
        is_control = (
            arrays.p[target_idx] == int(control)
            if control is not None
            else np.zeros(target_idx.size, dtype=bool)
        )

        def cap(subset: np.ndarray, maximum: int) -> np.ndarray:
            return stratified_cap_indices(
                subset,
                arrays.z["cell_type"],
                arrays.p,
                {},
                (),
                max_per_stratum=maximum,
                seed=seed,
            )

        target = np.sort(
            np.concatenate(
                (
                    cap(target_idx[~is_control], args.eval_max_cells_per_stratum),
                    cap(target_idx[is_control], eval_control_cap(args)),
                )
            )
        )
    else:
        target = cap_eval_indices(arrays, target_indices, args, seed=seed)
    controls = (
        control_reference_indices(
            arrays,
            target,
            control_reference,
            max_per_stratum=eval_control_cap(args),
            seed=seed + 1,
        )
        if control_reference is not None
        else np.empty(0, dtype=np.int64)
    )
    indices = np.concatenate([target, controls])
    if indices.size == 0:
        return {"n_cells": 0, "n_control_reference_cells": 0, "slices": {}}

    true_genes = load_gene_rows(adata, indices)
    pred_idx, pred_tids, pred_src = stratum_generation_plan(
        arrays,
        indices,
        task_ids[indices],
        samples_per_stratum=eval_gen_per_stratum(args),
        control_pert_index=arrays.encoding.control_pert_index,
    )
    source_mode = str(getattr(args, "flow_source", "noise"))
    source_indices = None
    if source_mode == "control":
        source_indices = (
            np.asarray(control_reference, dtype=np.int64)
            if control_reference is not None
            else target
        )
    pred_genes = predicted_genes_for_indices(
        model,
        arrays,
        decoders,
        pred_idx,
        pred_tids,
        device=device,
        batch_size=args.eval_batch_size,
        seed=seed,
        ode_method=args.ode_method,
        ode_step_size=args.ode_step_size,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        pred_control=str(getattr(args, "pred_control", "decoded")),
        flow_source=source_mode,
        source_indices=source_indices,
    )
    true_for_pred = np.zeros_like(pred_genes)
    carried = pred_src >= 0
    true_for_pred[carried] = true_genes[pred_src[carried]]
    pred_genes = apply_pred_control_genes(
        arrays,
        pred_idx,
        true_for_pred,
        pred_genes,
        mode=str(getattr(args, "pred_control", "decoded")),
    )
    true_genes = clip_gene_matrix_for_celleval(true_genes)
    pred_genes = clip_gene_matrix_for_celleval(pred_genes)

    def pool_keys_for(rows: np.ndarray) -> np.ndarray:
        if pool_donors:
            return np.asarray(arrays.z["cell_type"][rows], dtype=np.int64)[:, None]
        return _pool_keys(arrays, rows)

    keys = pool_keys_for(indices)
    pred_keys = pool_keys_for(pred_idx)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    slices: dict[str, Any] = {}
    for slice_id, key in enumerate(unique):
        take = inverse == slice_id
        take_pred = np.all(pred_keys == key, axis=1)
        perts = arrays.perturbation[indices[take]].astype(str)
        pred_perts = arrays.perturbation[pred_idx[take_pred]].astype(str)
        control_name = arrays.encoding.control_perturbation
        if control_name is None or not np.any(perts == control_name):
            continue
        if pool_donors:
            label_parts = [
                f"cell_type={arrays.encoding.covariate_categories['cell_type'][int(key[0])]}"
            ]
        else:
            label_parts = [arrays.encoding.context_categories[int(key[0])]]
            for col, value in zip(arrays.covariate_col_order, key[1:]):
                label_parts.append(
                    f"{col}={arrays.encoding.covariate_categories[col][int(value)]}"
                )
        label = "__".join(label_parts).replace("/", "_")
        obs = pd.DataFrame({"perturbation": pd.Categorical(perts)})
        pred_obs = pd.DataFrame({"perturbation": pd.Categorical(pred_perts)})
        real = ad.AnnData(X=true_genes[take], obs=obs.copy(), var=adata.var.copy())
        pred = ad.AnnData(
            X=pred_genes[take_pred], obs=pred_obs.copy(), var=adata.var.copy()
        )
        slice_dir = output_dir / label
        slice_dir.mkdir(parents=True, exist_ok=True)
        if args.save_h5ad:
            real.write_h5ad(slice_dir / "adata_real.h5ad")
            pred.write_h5ad(slice_dir / "adata_pred.h5ad")
        evaluator = MetricsEvaluator(
            adata_pred=pred,
            adata_real=real,
            control_pert=control_name,
            pert_col="perturbation",
            num_threads=args.num_threads,
            outdir=str(slice_dir),
        )
        results, agg = evaluator.compute(profile="minimal", write_csv=True)
        slices[label] = {
            "n_cells": int(take.sum()),
            "n_pred_cells": int(take_pred.sum()),
            "perturbation_results": _pert_results_to_jsonable(results),
            "agg_results": _agg_to_jsonable(agg),
        }
    return {
        "n_cells": int(target.size),
        "n_control_reference_cells": int(controls.size),
        "n_generated_cells": int(pred_idx.size),
        "gen_per_stratum": eval_gen_per_stratum(args),
        "profile": "minimal",
        "aggregate_covariate": "cell_type" if pool_donors else None,
        "pool_contexts": pool_donors,
        "slices": slices,
    }


def main() -> None:
    cli = make_parser().parse_args()
    out_dir = cli.out_dir
    args = load_run_args(out_dir)
    args.save_h5ad = cli.save_h5ad
    args.num_threads = cli.num_threads
    device = torch.device(cli.device)
    stream = stream_for(args.dataset)
    n_tasks = stream.n_tasks
    if int(getattr(args, "max_tasks", 0) or 0) > 0:
        n_tasks = min(n_tasks, int(args.max_tasks))
    if getattr(args, "smoke", False):
        n_tasks = min(n_tasks, 2)
    arrays = load_cl_arrays(
        args.h5ad_path,
        args.se_path,
        stream,
        covariate_cols=tuple(args.covariate_cols),
        control_perturbation=args.control_perturbation,
        representation=cl_representation(args),
    )
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    decoder_dir = Path(getattr(args, "decoder_dir", out_dir / "decoders"))
    decoders = TaskDecoderBank.load_dir(decoder_dir)
    train_ids = train_task_ids(stream, arrays.context, arrays.split)
    context_ids = task_id_from_context(stream, arrays.context)
    assert_stream_test_never_trained(
        stream,
        arrays.context,
        arrays.split,
        arrays.perturbation,
        n_tasks=n_tasks,
    )
    payload: dict[str, Any] = {
        "profile": "minimal",
        "max_cells_per_stratum": int(args.eval_max_cells_per_stratum),
    }

    if cli.heldout_curve:
        curve = []
        for k in range(n_tasks):
            model, _ = model_from_checkpoint(
                out_dir / "tasks" / f"{k:02d}" / "checkpoint.pt", device
            )
            target = np.flatnonzero(heldout_mask(stream, k, arrays.context, arrays.split))
            reference = np.flatnonzero(
                seen_train_mask(stream, k, arrays.context, arrays.split)
            )
            result = evaluate_indices(
                model,
                arrays,
                adata,
                decoders,
                target,
                context_ids,
                args,
                device,
                seed=eval_seed(args.seed, "heldout", k),
                output_dir=out_dir / "cell_eval_minimal" / "heldout_curve" / f"{k:02d}",
                control_reference=reference,
            )
            curve.append({"task_id": k, "heldout": result})
        payload["heldout_curve"] = curve
        payload["eval_provenance"] = eval_provenance(
            args, n_tasks - 1, tracks=("heldout",)
        )
        output = out_dir / "cell_eval_heldout_curve.json"
    else:
        k = n_tasks - 1
        model, _ = model_from_checkpoint(
            out_dir / "tasks" / f"{k:02d}" / "checkpoint.pt", device
        )
        masks = step_masks(stream, k, arrays.context, arrays.split, arrays.perturbation)
        stream_idx = np.flatnonzero(masks.seen_train)
        specs = {
            "fit": (np.flatnonzero(masks.new_train), train_ids, None),
            "forget": (np.flatnonzero(masks.old_train), train_ids, None),
            "stream": (stream_idx, train_ids, None),
            "heldout": (np.flatnonzero(masks.heldout), context_ids, stream_idx),
        }
        selected_tracks = tuple(cli.tracks or specs)
        specs = {name: specs[name] for name in selected_tracks}
        final = {}
        for name, (target, tids, reference) in specs.items():
            final[name] = evaluate_indices(
                model,
                arrays,
                adata,
                decoders,
                target,
                tids,
                args,
                device,
                seed=eval_seed(args.seed, name, k),
                output_dir=out_dir / "cell_eval_minimal" / "final" / name,
                control_reference=reference,
            )
        payload["final"] = final
        payload["eval_provenance"] = eval_provenance(
            args, k, tracks=tuple(specs)
        )
        output = (
            out_dir / "cell_eval_minimal.json"
            if cli.tracks is None
            else out_dir / f"cell_eval_minimal_{'_'.join(selected_tracks)}.json"
        )
    dump_json(output, payload)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
