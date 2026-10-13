#!/usr/bin/env python3
"""One-shot CFM on Replogle PCA, then HepG2-only Pearson-Δ / CellEval.

Same protocol and budget as ``train_gene_oneshot_hepg2.py`` /
``train_se_oneshot_hepg2.py``: all non-test cells, 100k Adam steps by default,
MLP CFM, HepG2 fit + heldout. The vector field lives in ``obsm['X_pca']``.
Decode with a frozen inverse PCA, not a fitted SE→gene map.

Two bases are selectable. The dump's ``varm['PCs']`` / ``uns['means_train']``
(default) is fit on ``train`` + ``finetune``, and since ``finetune`` is HepG2 it
encodes a future task -- fine for oneshot, not for sequential CL. Passing
``--pca-basis`` / ``--pca-embedding`` from ``scripts/build_replogle.py`` uses a
basis fit on task-0 contexts alone, which has no such leak.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from dataclasses import replace
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

from experiments.tx.eval_cl_celleval import evaluate_indices
from experiments.tx.train_cl import (
    DEFAULTS,
    capped_random,
    dump_json,
    epochs_for_target_steps,
    eval_seed,
    eval_track,
    fit_decoder_for_task,
    make_loader,
    make_model,
    model_from_checkpoint,
    save_checkpoint,
    steps_per_epoch,
    task_id_from_context,
)
from trajectory_replay.data.pca_decode import (
    calibrate_tied_affine,
    load_pca_gene_decoder,
    pca_gene_decoder_from_basis,
    pca_inverse_as_task_decoder,
)
from trajectory_replay.data.tx_cl_data import load_gene_rows, load_tx_cl_pca_arrays
from trajectory_replay.data.tx_dataset import build_encoding, encode_labels
from trajectory_replay.data.tx_preprocess import load_pca_basis
from trajectory_replay.data.tx_cl_tasks import (
    TEST_SPLIT,
    step_masks,
    stream_for,
    train_task_ids,
)
from trajectory_replay.data.tx_task_decoder import (
    DEFAULT_BATCH_SIZE as DECODER_BATCH_SIZE,
    DEFAULT_DEPTH as DECODER_DEPTH,
    DEFAULT_EPOCHS as DECODER_EPOCHS,
    DEFAULT_HIDDEN as DECODER_HIDDEN,
    DEFAULT_LR as DECODER_LR,
    TaskDecoderBank,
)
from trajectory_replay.training.flow_matching import (
    train_data_to_data_flow_matching,
    train_flow_matching,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("replogle", "pbmc"), default="replogle")
    p.add_argument("--h5ad-path", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--run-name", default="oneshot_pca_hepg2")
    p.add_argument(
        "--pca-basis",
        type=Path,
        help="Basis .npz from build_replogle.py. Default: dump varm['PCs'] (leaks HepG2).",
    )
    p.add_argument(
        "--pca-embedding",
        type=Path,
        help="Embedding .npy matching --pca-basis. Default: dump obsm['X_pca'].",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--target-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-cells", type=int, default=0)
    p.add_argument("--cond-emb-dim", type=int, default=128)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument(
        "--conditioning-mode",
        choices=("concat", "donor_film"),
        default="concat",
        help="Concatenate donor context, or use donor FiLM/offset on a shared trunk.",
    )
    p.add_argument("--ode-method", default="euler")
    p.add_argument(
        "--ode-step-size",
        type=float,
        default=0.05,
        help="Fixed step for Euler. Ignored for adaptive solvers (dopri5).",
    )
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--eval-max-cells-per-stratum", type=int, default=500)
    p.add_argument(
        "--eval-gen-per-stratum",
        type=int,
        default=500,
        help="Fixed generated cells per non-control evaluation stratum.",
    )
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--init-ckpt",
        type=Path,
        help="Load this CFM checkpoint (with --skip-train). Decoder is still built from --pca-decoder.",
    )
    p.add_argument(
        "--decoder-dir",
        type=Path,
        help="Load an existing TaskDecoderBank from this dir instead of fitting one. "
        "Use the CL campaign's bank to make this run decoder-matched to the CL table.",
    )
    p.add_argument(
        "--cl-tracks",
        action="store_true",
        help="Score the CL fit/forget/heldout tracks (stream masks, per-task decoder "
        "routing, CL eval seeds) instead of the HepG2-only fit/heldout pair.",
    )
    p.add_argument(
        "--pca-decoder",
        choices=("inverse", "tied_affine", "linear_hepg2", "linear_nontest"),
        default="inverse",
        help="inverse = tied P0.T; tied_affine = per-gene a_g,b_g on few-shot non-test; "
        "linear_hepg2 = full D1 on few-shot; linear_nontest = D on all non-test.",
    )
    p.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="Keep the first k PCs of --pca-embedding (0 = all).",
    )
    p.add_argument("--decoder-hidden", type=int, default=DECODER_HIDDEN)
    p.add_argument("--decoder-depth", type=int, default=DECODER_DEPTH)
    p.add_argument("--decoder-lr", type=float, default=DECODER_LR)
    p.add_argument("--decoder-batch-size", type=int, default=DECODER_BATCH_SIZE)
    p.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    p.add_argument("--decoder-max-fit-cells", type=int, default=0)
    p.add_argument("--skip-celleval", action="store_true")
    p.add_argument(
        "--conditioning",
        choices=("donor_cell_type", "cell_type"),
        default="donor_cell_type",
        help="PBMC conditioning: donor context + cell-type covariate, or cell type only.",
    )
    p.add_argument(
        "--cell-eval-grouping",
        choices=("condition", "cell_type"),
        default="condition",
        help="Cell-Eval grouping. cell_type pools PBMC donors before scoring.",
    )
    p.add_argument(
        "--pearson-grouping",
        choices=("condition", "cell_type"),
        default="condition",
        help="Pearson grouping. cell_type pools PBMC donors before scoring.",
    )
    p.add_argument(
        "--flow-type",
        choices=("n2d", "d2d"),
        default="n2d",
        help="Noise-to-data (default) or matched-control-to-target flow matching.",
    )
    p.add_argument(
        "--d2d-exclude-identity-targets",
        action="store_true",
        help="For D2D, omit control cells as targets (controls remain x0 sources).",
    )
    p.add_argument(
        "--pred-control",
        choices=("decoded", "observed", "generated"),
        default="decoded",
        help="Predicted control basal: inverse-PCA of observed z (default), "
        "real X, or ODE from noise under the control condition.",
    )
    p.add_argument("--save-h5ad", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def _strip_rows(block: dict[str, Any]) -> dict[str, Any]:
    out = dict(block)
    out.pop("rows", None)
    return out


def fewshot_eval_pools(arrays) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Oneshot train = all non-test; Fit/Heldout = few-shot contexts only."""
    stream = arrays.stream
    split = np.asarray(arrays.split).astype(str)
    context = np.asarray(arrays.context).astype(str)
    fewshot = tuple(c for task in stream.tasks[1:] for c in task.contexts)
    in_few = np.isin(context, list(fewshot))
    train_idx = np.flatnonzero(split != TEST_SPLIT)
    fit_idx = np.flatnonzero(in_few & (split != TEST_SPLIT))
    held_idx = np.flatnonzero(in_few & (split == TEST_SPLIT))
    return train_idx, fit_idx, held_idx, fewshot


def _celleval_mean_pearson_delta(result: dict[str, Any]) -> float:
    vals: list[float] = []
    for sl in result.get("slices", {}).values():
        for row in sl.get("perturbation_results", []):
            v = row.get("pearson_delta")
            if v is not None and np.isfinite(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def cl_track_specs(args, arrays) -> dict[str, tuple]:
    """CL fit / forget / heldout eval specs for a non-sequential checkpoint.

    Mirrors the final-task tracks in ``train_cl.py`` exactly -- same stream masks,
    same per-task decoder routing, same eval seeds -- so a jointly trained model is
    scored by the identical computation as the sequential runs it is compared to.
    """
    stream = stream_for(args.dataset)
    final_task = stream.n_tasks - 1
    masks = step_masks(
        stream, final_task, arrays.context, arrays.split, arrays.perturbation
    )
    train_ids = train_task_ids(stream, arrays.context, arrays.split)
    context_ids = task_id_from_context(stream, arrays.context)
    stream_idx = np.flatnonzero(masks.seen_train)
    return {
        "fit": (
            np.flatnonzero(masks.new_train),
            train_ids,
            None,
            eval_seed(args.seed, "fit", final_task),
        ),
        "forget": (
            np.flatnonzero(masks.old_train),
            train_ids,
            None,
            eval_seed(args.seed, "forget", final_task),
        ),
        "heldout": (
            np.flatnonzero(masks.heldout),
            context_ids,
            stream_idx,
            eval_seed(args.seed, "heldout", final_task),
        ),
    }


def _tied_inverse(args, arrays, n_fit: int):
    if args.pca_basis is not None:
        print(f"Using frozen inverse PCA from {args.pca_basis}", flush=True)
        pca_dec = pca_gene_decoder_from_basis(
            load_pca_basis(args.pca_basis), n_components=arrays.se_dim
        )
    else:
        print("Using frozen inverse PCA (dump PCs + means_train)", flush=True)
        pca_dec = load_pca_gene_decoder(args.h5ad_path)
    return pca_inverse_as_task_decoder(pca_dec, task_id=0, n_fit=int(n_fit))


def build_pca_decoder_bank(
    args: argparse.Namespace,
    arrays,
    adata: ad.AnnData,
    train_idx: np.ndarray,
    hepg2_fit: np.ndarray,
    device: torch.device,
) -> tuple[TaskDecoderBank, str, int, int]:
    kind = str(args.pca_decoder)
    if kind == "inverse":
        decoder = _tied_inverse(args, arrays, int(train_idx.size))
        manifest_kind = "oneshot_pca_inverse"
    elif kind == "tied_affine":
        tied = _tied_inverse(args, arrays, int(hepg2_fit.size))
        print(
            f"Calibrating per-gene affine on {hepg2_fit.size:,} few-shot non-test cells",
            flush=True,
        )
        decoder = calibrate_tied_affine(
            tied,
            np.asarray(arrays.se[hepg2_fit], dtype=np.float64),
            load_gene_rows(adata, hepg2_fit).astype(np.float64),
        )
        manifest_kind = "oneshot_pca_tied_affine"
    else:
        args.representation = "se"
        args.decoder_seed = int(args.seed)
        fit_idx = hepg2_fit if kind == "linear_hepg2" else train_idx
        print(f"Fitting linear PCA→gene D ({kind}) on {fit_idx.size:,} cells", flush=True)
        decoder = fit_decoder_for_task(arrays, adata, 0, fit_idx, args, device)
        manifest_kind = f"oneshot_pca_{kind}"
    if decoder.se_dim != arrays.se_dim:
        raise ValueError(
            f"PCA decoder se_dim={decoder.se_dim} != embedding width {arrays.se_dim}"
        )
    bank = TaskDecoderBank()
    bank.add(decoder)
    return bank, manifest_kind, int(decoder.n_fit), int(decoder.n_genes)


def main() -> None:
    args = make_parser().parse_args()
    defaults = DEFAULTS[args.dataset]
    args.h5ad_path = args.h5ad_path or defaults["h5ad"]
    args.control_perturbation = defaults["control"]
    args.covariate_cols = list(defaults["covariates"])
    out_dir = args.out_dir or (defaults["bucket"] / args.run_name)
    args.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(args.ode_method) != "euler":
        args.ode_step_size = None
    if args.flow_type == "d2d":
        if args.dataset != "replogle" or args.covariate_cols:
            raise ValueError(
                "The D2D pilot currently supports Replogle only; PBMC covariate-matched "
                "D2D training is not wired yet."
            )
        if not args.skip_celleval:
            raise ValueError("The D2D pilot currently requires --skip-celleval")
        if args.pred_control == "generated":
            raise ValueError("D2D requires decoded or observed controls, not generated controls")
        args.flow_source = "control"
    else:
        args.flow_source = "noise"
    if args.conditioning_mode == "donor_film" and args.conditioning != "donor_cell_type":
        raise ValueError("donor_film requires --conditioning donor_cell_type")
    tag_parts: list[str] = []
    if args.flow_type == "d2d":
        tag_parts.append("d2d")
    if args.cl_tracks:
        tag_parts.append("cltracks")
    if args.pearson_grouping == "cell_type":
        tag_parts.append("pearson_celltype")
    if str(args.ode_method) != "euler":
        tag_parts.append(str(args.ode_method))
    if str(args.pred_control) != "decoded":
        tag_parts.append(f"ctrl{args.pred_control}")
    eval_tag = ("_" + "_".join(tag_parts)) if tag_parts else ""
    device = torch.device(args.device)
    if args.smoke:
        args.target_steps = 0
        args.epochs = 1
        args.max_train_cells = min(args.max_train_cells or 4096, 4096)
        args.eval_max_cells_per_stratum = 32

    if (args.pca_basis is None) != (args.pca_embedding is None):
        raise ValueError("--pca-basis and --pca-embedding must be given together")
    pca_source = "task0_sidecar" if args.pca_basis is not None else "dump_obsm"

    stream = stream_for(args.dataset)
    arrays = load_tx_cl_pca_arrays(
        args.h5ad_path,
        stream,
        covariate_cols=tuple(args.covariate_cols),
        control_perturbation=args.control_perturbation,
        pca_path=args.pca_embedding,
    )
    if args.conditioning == "cell_type":
        if args.dataset != "pbmc":
            raise ValueError("--conditioning cell_type is currently PBMC-only")
        if "cell_type" not in arrays.z:
            raise ValueError("PBMC cell_type covariate is missing from loaded arrays")
        cell_type_categories = np.asarray(
            arrays.encoding.covariate_categories["cell_type"], dtype=object
        )
        cell_types = cell_type_categories[arrays.z["cell_type"]]
        encoding = build_encoding(
            cell_types,
            arrays.perturbation,
            control_perturbation=args.control_perturbation,
        )
        c, p, z = encode_labels(cell_types, arrays.perturbation, encoding)
        arrays = replace(arrays, encoding=encoding, c=c, p=p, z=z)
        args.covariate_cols = []
        print(
            f"Conditioning on cell_type only ({encoding.n_contexts} categories); "
            "donor remains available only for split selection.",
            flush=True,
        )
    args.pool_eval_donors_by_cell_type = (
        args.dataset == "pbmc"
        and args.conditioning == "donor_cell_type"
        and args.cell_eval_grouping == "cell_type"
    )
    if int(args.pca_dim) > 0:
        d = int(args.pca_dim)
        if d > arrays.se_dim:
            raise ValueError(f"--pca-dim={d} > embedding width {arrays.se_dim}")
        arrays = replace(
            arrays,
            se=np.ascontiguousarray(arrays.se[:, :d]),
            se_dim=d,
        )
    train_idx, hepg2_fit, hepg2_heldout, eval_contexts = fewshot_eval_pools(arrays)
    print(
        f"train cells (non-test, all contexts)={train_idx.size:,}  "
        f"fewshot fit={hepg2_fit.size:,}  fewshot test={hepg2_heldout.size:,}  "
        f"eval_ctx={eval_contexts}  n_pca={arrays.se_dim}",
        flush=True,
    )
    if train_idx.size == 0:
        raise ValueError("No non-test cells to train on")

    d2d_train_idx = capped_random(train_idx, int(args.max_train_cells), args.seed)
    n_loader = int(d2d_train_idx.size)
    if args.flow_type == "d2d" and args.d2d_exclude_identity_targets:
        control = arrays.encoding.control_pert_index
        if control is None:
            raise ValueError("D2D requires an encoded control perturbation")
        n_loader = int((arrays.p[d2d_train_idx] != int(control)).sum())
    spe = steps_per_epoch(n_loader, args.batch_size)
    epochs = (
        args.epochs
        if args.smoke or int(args.target_steps) <= 0
        else epochs_for_target_steps(n_loader, args.batch_size, args.target_steps)
    )
    print(
        f"steps/epoch={spe}  epochs={epochs}  ~{spe * epochs:,} Adam steps",
        flush=True,
    )

    cfg = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "representation": "pca",
        "pca_source": pca_source,
        "n_pca": int(arrays.se_dim),
        "n_train_cells": int(train_idx.size),
        "steps_per_epoch": spe,
        "epochs_used": epochs,
        "approx_adam_steps": int(spe * epochs),
        "eval_context": list(eval_contexts),
        "pca_decoder": str(args.pca_decoder),
        "flow_type": str(args.flow_type),
        "flow_source": str(args.flow_source),
        "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
    }
    if args.skip_train:
        dump_json(out_dir / f"eval_config{eval_tag}.json", cfg)
    else:
        dump_json(out_dir / "config.json", cfg)

    ckpt_path = out_dir / "checkpoint.pt"
    decoder_dir = out_dir / "decoders"
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    load_ckpt = args.init_ckpt if args.init_ckpt is not None else (
        ckpt_path if args.skip_train and ckpt_path.is_file() else None
    )

    if args.decoder_dir is not None:
        bank = TaskDecoderBank.load_dir(args.decoder_dir)
        print(
            f"Loaded decoder bank from {args.decoder_dir} "
            f"({len(bank)} tasks); not refitting",
            flush=True,
        )
    else:
        bank, manifest_kind, n_fit, n_genes = build_pca_decoder_bank(
            args, arrays, adata, train_idx, hepg2_fit, device
        )
        bank.save_dir(decoder_dir)
        dump_json(
            decoder_dir / "manifest.json",
            {
                "format_version": 1,
                "dataset": args.dataset,
                "kind": manifest_kind,
                "pca_decoder": str(args.pca_decoder),
                "n_fit": n_fit,
                "n_genes": n_genes,
                "se_dim": int(arrays.se_dim),
            },
        )

    if args.skip_train:
        if load_ckpt is None or not Path(load_ckpt).is_file():
            raise FileNotFoundError(
                "skip-train needs --init-ckpt or an existing checkpoint in --out-dir"
            )
        model, _ = model_from_checkpoint(Path(load_ckpt), device)
        print(f"Loaded CFM {load_ckpt}; decoder={args.pca_decoder}", flush=True)
    else:
        print(f"Training one-shot {args.flow_type.upper()} CFM on PCA", flush=True)
        model = make_model(arrays, args).to(device)
        if args.flow_type == "d2d":
            control = arrays.encoding.control_pert_index
            if control is None:
                raise ValueError("D2D requires an encoded control perturbation")
            x = torch.from_numpy(
                np.ascontiguousarray(arrays.se[d2d_train_idx], dtype=np.float32)
            )
            c = torch.from_numpy(
                np.ascontiguousarray(arrays.c[d2d_train_idx], dtype=np.int64)
            )
            p = torch.from_numpy(
                np.ascontiguousarray(arrays.p[d2d_train_idx], dtype=np.int64)
            )
            history = train_data_to_data_flow_matching(
                model,
                x,
                c,
                p,
                control_pert=int(control),
                device=device,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                train_seed=args.seed,
                exclude_identity_targets=bool(args.d2d_exclude_identity_targets),
            )
        else:
            loader = make_loader(arrays, train_idx, args, seed=args.seed)
            history = train_flow_matching(
                model,
                loader,
                data_dim=arrays.se_dim,
                device=device,
                epochs=epochs,
                lr=args.lr,
                train_seed=args.seed,
            )
        dump_json(out_dir / "history.json", history)
        save_checkpoint(
            ckpt_path,
            model,
            {
                "experiment": "tx_pca_oneshot_hepg2",
                "dataset": args.dataset,
                "representation": "pca",
                "flow_type": str(args.flow_type),
                "data_dim": model.data_dim,
                "n_contexts": model.n_contexts,
                "n_perturbations": model.n_perturbations,
                "cond_emb_dim": model.cond_emb_dim,
                "time_emb_dim": model.time_emb_dim,
                "hidden_dim": model.hidden_dim,
                "depth": model.depth,
                "conditioning_mode": model.conditioning_mode,
                "covariate_sizes": arrays.encoding.covariate_sizes(),
                "encoding": arrays.encoding.to_jsonable(),
            },
        )
        print(f"Wrote {ckpt_path}", flush=True)

    if args.cl_tracks:
        specs = cl_track_specs(args, arrays)
    else:
        flat_ids = np.zeros(arrays.n_cells, dtype=np.int16)
        specs = {
            "fit": (hepg2_fit, flat_ids, None, args.seed + 30_000),
            "heldout": (
                hepg2_heldout,
                flat_ids,
                hepg2_fit,
                args.seed + 30_001,
            ),
        }

    pearson = {
        name: _strip_rows(
            eval_track(
                model,
                arrays,
                adata,
                bank,
                indices,
                tids,
                args,
                device,
                seed=seed,
                control_reference=reference,
                generation_reference=(train_idx if args.flow_type == "d2d" else None),
            )
        )
        for name, (indices, tids, reference, seed) in specs.items()
    }
    pearson_stem = "hepg2_pearson" if args.dataset == "replogle" else "oneshot_pearson"
    pearson_path = out_dir / f"{pearson_stem}{eval_tag}.json"
    dump_json(pearson_path, pearson)
    print(
        "Pearson-Δ  "
        + "  ".join(
            f"{name}={block.get('pearson_delta_macro')}"
            for name, block in pearson.items()
        )
        + f"  ode={args.ode_method}  wrote {pearson_path.name}",
        flush=True,
    )

    if not args.skip_celleval:
        eval_ns = Namespace(**vars(args))
        celleval: dict[str, Any] = {
            "profile": "minimal",
            "eval_context": list(eval_contexts),
            "representation": "pca",
        }
        for name, (indices, tids, reference, seed) in specs.items():
            celleval[name] = evaluate_indices(
                model,
                arrays,
                adata,
                bank,
                indices,
                tids,
                eval_ns,
                device,
                seed=seed + 1_000,
                output_dir=out_dir / f"cell_eval_minimal{eval_tag}" / name,
                control_reference=reference,
            )
        ce_stem = "hepg2_celleval" if args.dataset == "replogle" else "oneshot_celleval"
        dump_json(out_dir / f"{ce_stem}{eval_tag}.json", celleval)
        print(
            "CellEval mean Pearson-Δ  "
            + "  ".join(
                f"{name}={_celleval_mean_pearson_delta(celleval[name]):.4f}"
                for name in specs
            ),
            flush=True,
        )

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Done. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
