#!/usr/bin/env python3
"""Sequential transcriptomics continual learning.

Default representation is STATE SE (``--representation se``) with a frozen
``nn.Linear`` SE→gene decoder per task. ``--representation scimilarity`` is the
same protocol on a SCimilarity sidecar. ``--representation genes`` trains the
CFM on h5ad ``X`` and uses an identity decoder.

The ODE and MeanFlow replay backends mirror ``experiments/image_cl/train_cl.py``:

* Replogle 2-task / PBMC 5-task stream from ``tx_cl_tasks``;
* one frozen decoder per task, precomputed once per dataset;
* task 0 CFM on real λ-train cells;
* later tasks support standard, GFT, real, generative, and trajectory objectives;
* later-task scope is embedding-only, full vector field, or merged LoRA;
* λ-grid {0.25, 0.5, 0.75} or equal λ=k/(k+1);
* λ-grid selection on capped seen train/finetune strata, never STATE test:
  old-task and new-task stratum macros are averaged with equal weight;
* resumable ``tasks/00``, ``tasks/01``, ... checkpoints;
* after the last task, capped fit / forget / stream / heldout on that checkpoint.
  Sequential heldout-over-tasks is ``experiments/tx/eval_cl_heldout.py``.

This trainer uses a lightweight decoded gene-space Pearson-Δ scorer for λ.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

from trajectory_replay.data.pca_decode import (
    calibrate_tied_affine,
    pca_gene_decoder_from_basis,
    pca_inverse_as_task_decoder,
)
from trajectory_replay.data.tx_cl_data import (
    IndexedSeDataset,
    TxClArrays,
    load_cl_representation_arrays,
    load_gene_rows,
    load_se_rows,
    stratified_cap_indices,
)
from trajectory_replay.data.tx_preprocess import (
    DEFAULT_PCA_DIM,
    load_pca_basis,
    pca_sidecar_names,
)
from trajectory_replay.data.tx_cl_tasks import (
    TxClStream,
    assert_stream_test_never_trained,
    assert_test_never_trained,
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
    fit_task_decoder,
    identity_task_decoder,
)
from trajectory_replay.eval.tx_cl_pearson import (
    evaluate_model_gene_space,
    old_new_balanced_selection,
)
from trajectory_replay.holdout.data import (
    build_stratum_cell_index,
    build_support_flat_index,
    build_support_stratum_distribution,
)
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import (
    make_frozen_model_copy,
    resolve_two_stage_1_epochs,
    train_flow_matching,
    train_flow_matching_generative_replay,
    train_flow_matching_gft,
    train_flow_matching_real_replay,
)
from trajectory_replay.training.mean_flow_cl import train_conditional_joint_mean_flow
from trajectory_replay.training.mean_flow_distill import (
    MEAN_FLOW_CKPT_NAME,
    load_mean_flow_checkpoint,
    make_mean_flow_from_teacher,
    save_mean_flow_checkpoint,
    train_mean_flow_distill,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAMBDA_GRID = (0.25, 0.5, 0.75)
REPLAY_STRATEGIES = ("real", "generative", "trajectory")
SYNTHETIC_REPLAY_STRATEGIES = ("generative", "trajectory")
DEFAULTS = {
    "pbmc": {
        "h5ad": REPO_ROOT / "data_dir/pbmc_parse_with_splits.h5ad",
        "se": REPO_ROOT / "results_dir/arc_state/se/pbmc_parse_with_splits_se.npy",
        "bucket": REPO_ROOT / "results_dir/tx_results/pbmc",
        "covariates": ("cell_type",),
        "control": "PBS",
        "scimilarity": REPO_ROOT / "results_dir/scimilarity/pbmc_parse_with_splits_sci.npy",
        "h5ad_pca": REPO_ROOT / "data_dir/pbmc/adata.h5ad",
        # Unlabeled pca_* is DEFAULT_PCA_DIM (256). Other widths are pca{d}_*.
        "pca": REPO_ROOT / "data_dir/pbmc/pca_embedding.npy",
        "pca_basis": REPO_ROOT / "data_dir/pbmc/pca_basis.npz",
    },
    "replogle": {
        "h5ad": REPO_ROOT / "data_dir/replogle_with_splits.h5ad",
        "se": REPO_ROOT / "results_dir/arc_state/se/replogle_with_splits_se.npy",
        "bucket": REPO_ROOT / "results_dir/tx_results/replogle",
        "covariates": (),
        "control": "non-targeting",
        "scimilarity": REPO_ROOT / "results_dir/scimilarity/replogle_with_splits_sci.npy",
        "h5ad_pca": REPO_ROOT / "data_dir/replogle/adata.h5ad",
        # Unlabeled pca_* is DEFAULT_PCA_DIM (256). Other widths are pca{d}_*.
        "pca": REPO_ROOT / "data_dir/replogle/pca_embedding.npy",
        "pca_basis": REPO_ROOT / "data_dir/replogle/pca_basis.npz",
    },
}


CL_REPRESENTATIONS = ("se", "genes", "scimilarity", "pca")
EMBEDDING_REPRESENTATIONS = ("se", "scimilarity", "pca")


def cl_representation(args: argparse.Namespace) -> str:
    rep = str(getattr(args, "representation", "se") or "se")
    if rep not in CL_REPRESENTATIONS:
        raise ValueError(f"representation must be one of {CL_REPRESENTATIONS}, got {rep!r}")
    return rep


def resolve_pca_artifacts(
    defaults: dict,
    *,
    pca_dim: int | None = None,
    se_path: Path | None = None,
    pca_basis: Path | None = None,
) -> tuple[Path, Path, int | None]:
    """Embedding, basis, and expected width for a PCA CL run.

    ``pca_dim`` selects the sidecar pair. The canonical 256-PC dump is
    unlabeled ``pca_embedding.npy`` / ``pca_basis.npz``; other widths are
    ``pca{d}_*`` next to it. Explicit ``se_path`` / ``pca_basis`` still win.
    When ``se_path`` is set and ``pca_dim`` is omitted, expected width is
    left as ``None`` and inferred from the loaded array.
    """
    if "pca" not in defaults or "pca_basis" not in defaults:
        raise ValueError("this dataset has no PCA sidecars")
    requested = int(pca_dim) if pca_dim else None
    dim = requested if requested is not None else DEFAULT_PCA_DIM
    root = Path(defaults["pca"]).parent
    basis_name, emb_name = pca_sidecar_names(dim)
    embedding = Path(se_path) if se_path is not None else root / emb_name
    basis = Path(pca_basis) if pca_basis is not None else root / basis_name
    if requested is None and se_path is not None:
        return embedding, basis, None
    return embedding, basis, dim


def embedding_sidecar_path(
    defaults: dict,
    representation: str,
    se_path: Path | None,
    *,
    pca_dim: int | None = None,
) -> Path | None:
    if representation == "genes":
        return None
    if se_path is not None:
        return se_path
    if representation == "scimilarity":
        return Path(defaults["scimilarity"])
    if representation == "pca":
        embedding, _, _ = resolve_pca_artifacts(defaults, pca_dim=pca_dim)
        return embedding
    return Path(defaults["se"])


def load_cl_arrays(
    h5ad_path: Path | str,
    se_path: Path | str | None,
    stream: TxClStream,
    *,
    covariate_cols: tuple[str, ...],
    control_perturbation: str,
    representation: str = "se",
) -> TxClArrays:
    return load_cl_representation_arrays(
        h5ad_path,
        stream,
        se_path=se_path,
        covariate_cols=covariate_cols,
        control_perturbation=control_perturbation,
        representation=representation,
    )


def shared_decoder_dirname(
    representation: str,
    hidden: int = DECODER_HIDDEN,
    depth: int = DECODER_DEPTH,
    pca_dim: int = 0,
) -> str:
    """Bank directory; deep banks get their own suffix so both can coexist."""
    if representation == "genes":
        return "decoders_genes"
    if representation == "pca":
        dim = int(pca_dim) if pca_dim else 0
        return f"decoders_pca{dim}_affine" if dim else "decoders_pca_affine"
    base = "decoders_scimilarity" if representation == "scimilarity" else "decoders"
    if int(hidden) > 0:
        return f"{base}_mlp_h{int(hidden)}_d{int(depth)}"
    return base


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def parse_float_list(spec: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in spec.split(",") if x.strip())
    if not values or any(not 0.0 < x < 1.0 for x in values):
        raise ValueError(f"λ grid must contain values in (0,1), got {values}")
    return values


def parse_covariates(spec: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if spec is None:
        return defaults
    return tuple(x.strip() for x in spec.split(",") if x.strip())


def equal_weight_lambda(task_id: int) -> float:
    """Old-data mixture weight after introducing task ``task_id``."""
    if task_id <= 0:
        return 0.0
    return float(task_id) / float(task_id + 1)


def steps_per_epoch(n_cells: int, batch_size: int) -> int:
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return max(1, (int(n_cells) + int(batch_size) - 1) // int(batch_size))


def target_steps_for_task(args: argparse.Namespace, task_id: int) -> int:
    """Step budget for one stream task. T0 uses ``target_steps``; later tasks
    use ``later_target_steps`` when that is positive.
    """
    if int(task_id) <= 0:
        return int(args.target_steps)
    later = int(getattr(args, "later_target_steps", 0) or 0)
    return later if later > 0 else int(args.target_steps)


def epochs_for_target_steps(n_cells: int, batch_size: int, target_steps: int) -> int:
    """Smallest epoch count whose loader visits at least ``target_steps`` batches."""
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    spe = steps_per_epoch(n_cells, batch_size)
    return max(1, (int(target_steps) + spe - 1) // spe)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("pbmc", "replogle"), required=True)
    p.add_argument(
        "--strategy",
        choices=("standard", "gft", *REPLAY_STRATEGIES),
        required=True,
        help="Sequential objective used from task 1 onward.",
    )
    p.add_argument("--replay-backend", choices=("ode", "mean_flow"), default="ode")
    p.add_argument(
        "--replay-condition-sampling",
        choices=("proportional", "observed_uniform"),
        default="proportional",
        help=(
            "How to draw old-task (context, perturbation[, covariates]) for replay. "
            "proportional (default): P(stratum) ∝ cell count. observed_uniform: one "
            "equal vote per observed T0 stratum, so a large control pool is not "
            "oversampled."
        ),
    )
    p.add_argument("--lambda-mode", choices=("none", "grid", "equal"), default=None)
    p.add_argument("--lambda-grid", default=",".join(map(str, LAMBDA_GRID)))
    p.add_argument(
        "--finetune-scope",
        choices=("embedding", "full", "lora"),
        default="full",
        help="Parameters updated after task 0: new context rows, full VF, or LoRA + new rows.",
    )
    p.add_argument(
        "--finetune-train-old-pert-embeddings",
        action="store_true",
        help="Also train perturbation rows seen on earlier tasks. Rows that first "
        "appear on the new task always train.",
    )
    p.add_argument("--finetune-train-old-context-embeddings", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-train-output-layer", action="store_true")
    p.add_argument("--two-stage-1-epochs", type=int, default=0)
    p.add_argument("--two-stage-1-epoch-fraction", type=float)
    p.add_argument("--gft-beta-start", type=float, default=1.0)
    p.add_argument("--gft-beta-end", type=float, default=0.0)
    p.add_argument(
        "--gft-beta-schedule",
        choices=("sigmoid", "linear", "constant"),
        default="sigmoid",
    )
    p.add_argument("--gft-sigmoid-midpoint", type=float, default=0.5)
    p.add_argument("--gft-sigmoid-steepness", type=float, default=10.0)
    p.add_argument(
        "--representation",
        choices=CL_REPRESENTATIONS,
        default="se",
        help="CFM ambient space: STATE SE, SCimilarity, task-0 PCA, or h5ad X.",
    )
    p.add_argument("--h5ad-path", type=Path)
    p.add_argument("--se-path", type=Path)
    p.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help=(
            "Task-0 PCA width. 0 (default) uses the unlabeled sidecar at "
            f"{DEFAULT_PCA_DIM} components. Other values load "
            "pca{d}_embedding.npy and pca{d}_basis.npz next to it, unless "
            "--se-path / --pca-basis override."
        ),
    )
    p.add_argument(
        "--pca-basis",
        type=Path,
        help="Task-0 PCA basis .npz. Defaults with --pca-dim via pca_sidecar_names.",
    )
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--run-name")
    p.add_argument(
        "--decoder-dir",
        type=Path,
        help=(
            "Shared dataset-level decoder bank. Defaults to <dataset bucket>/decoders "
            "or decoders_genes for canonical inputs, or <out-dir>/decoders otherwise."
        ),
    )
    p.add_argument(
        "--require-existing-decoders",
        action="store_true",
        help="Load a complete compatible bank; never fit or overwrite decoders.",
    )
    p.add_argument("--covariate-cols", default=None)
    p.add_argument("--control-perturbation")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--target-steps",
        type=int,
        default=100_000,
        help=(
            "Target Adam steps for task 0 (default: 100000), implemented by choosing "
            "the smallest whole-epoch count that reaches the target. Overrides "
            "--epochs. MeanFlow T0 distill uses the same epoch count."
        ),
    )
    p.add_argument(
        "--later-target-steps",
        type=int,
        default=0,
        help=(
            "Target Adam steps for tasks after T0. 0 (default) uses --target-steps. "
            "Replogle few-shot T1 should be smaller than T0 so HepG2 is not a "
            "second full pretrain."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--later-lr",
        type=float,
        default=0.0,
        help="Learning rate after T0; 0 uses --lr.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Keep 0: the training tensor lives in the parent process.",
    )
    p.add_argument("--max-train-cells", type=int, default=0)

    p.add_argument("--cond-emb-dim", type=int, default=128)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument(
        "--backbone-type",
        choices=("mlp", "residual_mlp"),
        default="mlp",
    )

    p.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    p.add_argument("--decoder-lr", type=float, default=DECODER_LR)
    p.add_argument("--decoder-batch-size", type=int, default=DECODER_BATCH_SIZE)
    p.add_argument("--decoder-seed", type=int, default=0)
    p.add_argument(
        "--decoder-hidden",
        type=int,
        default=DECODER_HIDDEN,
        help="Hidden width of the GELU MLP decoder; 0 (default) keeps it linear.",
    )
    p.add_argument(
        "--decoder-depth",
        type=int,
        default=DECODER_DEPTH,
        help="Number of hidden layers when --decoder-hidden > 0.",
    )
    p.add_argument(
        "--decoder-max-fit-cells",
        type=int,
        default=0,
        help="Optional random cap per task; 0 (default) fits every trainable task cell.",
    )

    p.add_argument("--ode-method", default="euler")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument(
        "--pred-control",
        choices=("decoded", "observed", "generated"),
        default="decoded",
        help="Predicted control basal: decode observed latents (default), "
        "copy observed X, or ODE the control condition from noise.",
    )
    p.add_argument(
        "--flow-source",
        choices=("noise", "control"),
        default="noise",
        help="ODE start: Gaussian noise (default) or matched observed controls.",
    )
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument(
        "--eval-max-cells-per-stratum",
        type=int,
        default=0,
        help=(
            "Cap on *observed* perturbed cells used for the true mean. "
            "0 (default) keeps every real cell. Predicted cells are a "
            "separate budget, --eval-gen-per-stratum."
        ),
    )
    p.add_argument(
        "--eval-max-control-cells",
        type=int,
        default=0,
        help=(
            "Separate cap for control strata, which form the Pearson-Δ basal. "
            "0 (default) keeps every available control so the basal does not "
            "depend on a random draw."
        ),
    )
    p.add_argument(
        "--eval-gen-per-stratum",
        type=int,
        default=500,
        help=(
            "Model samples per scored stratum. Fixed K makes the predicted "
            "mean's Monte Carlo error the same for every perturbation instead "
            "of tracking how many cells the assay happened to capture. 0 "
            "reverts to one sample per observed cell."
        ),
    )
    p.add_argument(
        "--lambda-gen-per-stratum",
        type=int,
        default=100,
        help=(
            "Same fixed-K idea as --eval-gen-per-stratum, but for λ-grid "
            "selection only. 0 reverts to one sample per observed cell."
        ),
    )
    p.add_argument(
        "--eval-final-tracks",
        default="+".join(FINAL_EVAL_TRACKS),
        help=(
            "Final Pearson tracks, joined by + (sbatch --export cannot carry "
            "commas). Default is all four. PBMC table only needs stream+heldout."
        ),
    )
    p.add_argument("--traj-kappa", default="none")
    p.add_argument(
        "--trajectory-batch-sampling",
        choices=("equal", "lambda"),
        default="equal",
        help="Allocate trajectory/new Monte Carlo samples 50/50 or according to lambda.",
    )
    p.add_argument(
        "--trajectory-replay-batch-multiplier",
        type=float,
        default=1.0,
        help="Multiplier on old-trajectory samples without changing new-task samples.",
    )
    p.add_argument(
        "--trajectory-replay-loss-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplier on the old trajectory preservation loss.",
    )
    p.add_argument(
        "--trajectory-time-sampling",
        choices=("batch", "per_example_euler_grid"),
        default="batch",
        help="Use one shared time or independently selected cached Euler-grid times.",
    )
    p.add_argument("--mean-flow-steps", type=int, choices=(1, 4), default=1)
    p.add_argument("--mean-flow-distill-epochs", type=int, default=100)
    p.add_argument("--mean-flow-lr", type=float, default=None)
    p.add_argument("--mean-flow-diag-fraction", type=float, default=0.25)
    p.add_argument("--mean-flow-adaptive-p", type=float, default=0.75)
    p.add_argument("--mean-flow-time-sampler", choices=("uniform", "logit_normal"), default="logit_normal")
    p.add_argument("--mean-flow-time-mu", type=float, default=0.4)
    p.add_argument("--mean-flow-time-sigma", type=float, default=1.0)
    p.add_argument("--mean-flow-max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--mean-flow-use-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--mean-flow-adam-beta2", type=float, default=0.95)
    p.add_argument("--mean-flow-weight-decay", type=float, default=0.0)

    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def make_model(arrays: TxClArrays, args: argparse.Namespace) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=arrays.se_dim,
        n_contexts=arrays.encoding.n_contexts,
        n_perturbations=arrays.encoding.n_perturbations,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        backbone_type=args.backbone_type,
        covariate_sizes=arrays.encoding.covariate_sizes() or None,
        conditioning_mode=getattr(args, "conditioning_mode", "concat"),
    )


def checkpoint_config(
    arrays: TxClArrays,
    model: ConditionalFlowMatching,
    args: argparse.Namespace,
    *,
    task_id: int,
    chosen_lambda: float | None,
) -> dict[str, Any]:
    return {
        "experiment": "tx_cl",
        "dataset": args.dataset,
        "representation": cl_representation(args),
        "strategy": args.strategy,
        "finetune_scope": args.finetune_scope,
        "replay_backend": args.replay_backend,
        "replay_condition_sampling": str(
            getattr(args, "replay_condition_sampling", "proportional")
        ),
        "mean_flow_steps": int(args.mean_flow_steps),
        "task_id": task_id,
        "chosen_lambda": chosen_lambda,
        "data_dim": model.data_dim,
        "n_contexts": model.n_contexts,
        "n_perturbations": model.n_perturbations,
        "cond_emb_dim": model.cond_emb_dim,
        "time_emb_dim": model.time_emb_dim,
        "hidden_dim": model.hidden_dim,
        "depth": model.depth,
        "backbone_type": model.backbone_type,
        "conditioning_mode": model.conditioning_mode,
        "covariate_sizes": arrays.encoding.covariate_sizes(),
        "encoding": arrays.encoding.to_jsonable(),
    }


def save_checkpoint(path: Path, model: ConditionalFlowMatching, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def model_from_checkpoint(path: Path, device: torch.device) -> tuple[ConditionalFlowMatching, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = ConditionalFlowMatching(
        data_dim=int(cfg["data_dim"]),
        n_contexts=int(cfg["n_contexts"]),
        n_perturbations=int(cfg["n_perturbations"]),
        cond_emb_dim=int(cfg["cond_emb_dim"]),
        time_emb_dim=int(cfg["time_emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        backbone_type=str(cfg.get("backbone_type", "mlp")),
        covariate_sizes=cfg.get("covariate_sizes") or None,
        conditioning_mode=str(cfg.get("conditioning_mode", "concat")),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device), cfg


def find_resume(out_dir: Path, n_tasks: int, device: torch.device):
    done = 0
    last: Path | None = None
    metrics: list[dict] = []
    for k in range(n_tasks):
        task_dir = out_dir / "tasks" / f"{k:02d}"
        ckpt = task_dir / "checkpoint.pt"
        metric = task_dir / "metrics.json"
        if not ckpt.is_file() or not metric.is_file():
            break
        done = k + 1
        last = ckpt
        metrics.append(json.loads(metric.read_text(encoding="utf-8")))
    if last is None:
        return None, 0, metrics
    model, _ = model_from_checkpoint(last, device)
    return model, done, metrics


def capped_random(indices: np.ndarray, cap: int, seed: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if cap <= 0 or idx.size <= cap:
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=cap, replace=False))


def make_loader(
    arrays: TxClArrays,
    indices: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> DataLoader:
    idx = capped_random(indices, int(args.max_train_cells), seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        IndexedSeDataset(arrays, idx),
        batch_size=args.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def fit_decoder_for_task(
    arrays: TxClArrays,
    adata: ad.AnnData,
    task_id: int,
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    fit_idx = capped_random(
        indices,
        int(args.decoder_max_fit_cells),
        int(args.decoder_seed) + 71 * task_id,
    )
    if cl_representation(args) == "genes":
        print(
            f"  decoder D_{task_id}: identity on {fit_idx.size:,} cells "
            f"({arrays.se_dim} genes)",
            flush=True,
        )
        return identity_task_decoder(
            task_id=task_id, n_genes=arrays.se_dim, n_fit=int(fit_idx.size)
        )
    if cl_representation(args) == "pca":
        basis_path = getattr(args, "pca_basis", None)
        if basis_path is None:
            raise ValueError("--pca-basis is required for representation=pca")
        print(
            f"  decoder D_{task_id}: tied inverse PCA + per-gene affine "
            f"on {fit_idx.size:,} task cells",
            flush=True,
        )
        tied = pca_inverse_as_task_decoder(
            pca_gene_decoder_from_basis(
                load_pca_basis(basis_path), n_components=arrays.se_dim
            ),
            task_id=task_id,
            n_fit=int(fit_idx.size),
        )
        z = load_se_rows(arrays.se, fit_idx)
        x = load_gene_rows(adata, fit_idx)
        return calibrate_tied_affine(tied, z, x)
    hidden = int(getattr(args, "decoder_hidden", DECODER_HIDDEN))
    depth = int(getattr(args, "decoder_depth", DECODER_DEPTH))
    arch = "linear" if hidden <= 0 else f"mlp h={hidden} d={depth}"
    print(
        f"  decoder D_{task_id}: fitting {arch} on {fit_idx.size:,} task cells",
        flush=True,
    )
    z = load_se_rows(arrays.se, fit_idx)
    x = load_gene_rows(adata, fit_idx)
    return fit_task_decoder(
        z,
        x,
        task_id=task_id,
        lr=args.decoder_lr,
        batch_size=args.decoder_batch_size,
        epochs=args.decoder_epochs,
        hidden=hidden,
        depth=depth,
        device=str(device),
        seed=args.decoder_seed + task_id,
    )


def decoder_manifest(
    arrays: TxClArrays,
    stream: TxClStream,
    h5ad_path: Path,
    se_path: Path | None,
    args: argparse.Namespace,
    *,
    n_tasks: int,
) -> dict[str, Any]:
    """Identity and fit settings for a reusable decoder bank."""
    return {
        "format_version": 1,
        "dataset": args.dataset,
        "representation": cl_representation(args),
        "h5ad_path": str(h5ad_path.resolve()),
        "se_path": str(se_path.resolve()) if se_path is not None else str(h5ad_path.resolve()),
        "n_cells": int(arrays.n_cells),
        "se_dim": int(arrays.se_dim),
        "n_tasks": int(n_tasks),
        "decoder_lr": float(args.decoder_lr),
        "decoder_batch_size": int(args.decoder_batch_size),
        "decoder_epochs": int(args.decoder_epochs),
        "decoder_hidden": int(getattr(args, "decoder_hidden", DECODER_HIDDEN)),
        "decoder_depth": int(getattr(args, "decoder_depth", DECODER_DEPTH)),
        "decoder_max_fit_cells": int(args.decoder_max_fit_cells),
        "decoder_seed": int(args.decoder_seed),
        "pca_basis": (
            str(Path(args.pca_basis).resolve())
            if cl_representation(args) == "pca" and getattr(args, "pca_basis", None)
            else None
        ),
        "decoder_kind": "tied_affine" if cl_representation(args) == "pca" else None,
        "pca_dim": (
            int(arrays.se_dim) if cl_representation(args) == "pca" else None
        ),
    }


def prepare_decoder_bank(
    decoder_dir: Path,
    arrays: TxClArrays,
    adata: ad.AnnData,
    stream: TxClStream,
    h5ad_path: Path,
    se_path: Path | None,
    args: argparse.Namespace,
    device: torch.device,
    *,
    n_tasks: int | None = None,
    require_existing: bool | None = None,
) -> TaskDecoderBank:
    """Load a complete compatible bank, fitting missing tasks before CL starts.

    ``require_existing`` (or ``args.require_existing_decoders``) refuses to fit
    anything: the bank must already contain every task and match the manifest.
    """
    target_tasks = stream.n_tasks if n_tasks is None else int(n_tasks)
    assert_stream_test_never_trained(
        stream,
        arrays.context,
        arrays.split,
        arrays.perturbation,
        n_tasks=target_tasks,
    )
    expected = decoder_manifest(
        arrays, stream, h5ad_path, se_path, args, n_tasks=target_tasks
    )
    manifest_path = decoder_dir / "manifest.json"
    decoder_paths = sorted(decoder_dir.glob("decoder_task*.npz"))
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual.setdefault("representation", "se")
        actual.setdefault("decoder_hidden", DECODER_HIDDEN)
        actual.setdefault("decoder_depth", DECODER_DEPTH)
        actual.setdefault("pca_dim", expected.get("pca_dim"))
        if actual != expected:
            differing = sorted(
                key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                f"Decoder bank {decoder_dir} is incompatible; differing fields: {differing}"
            )
    elif decoder_paths:
        raise ValueError(
            f"Decoder bank {decoder_dir} has no manifest.json; use an empty directory "
            "or explicitly select a compatible bank."
        )

    bank = TaskDecoderBank.load_dir(decoder_dir) if decoder_paths else TaskDecoderBank()
    missing = [tid for tid in range(target_tasks) if tid not in bank]
    if not missing:
        return bank
    reuse_only = (
        bool(require_existing)
        if require_existing is not None
        else bool(getattr(args, "require_existing_decoders", False))
    )
    if reuse_only:
        raise ValueError(
            f"Decoder bank {decoder_dir} is missing tasks {missing}; "
            "refusing to fit because --require-existing-decoders is set"
        )
    train_ids = train_task_ids(stream, arrays.context, arrays.split)
    for task_id in missing:
        indices = np.flatnonzero(train_ids == task_id)
        if indices.size == 0:
            raise ValueError(f"Task {task_id} has no cells for decoder fitting")
        decoder = fit_decoder_for_task(arrays, adata, task_id, indices, args, device)
        bank.add(decoder)
        bank.save_dir(decoder_dir)
    dump_json(manifest_path, expected)
    return bank


def task_id_from_context(stream: TxClStream, context: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(context).shape[0], -1, dtype=np.int16)
    ctx = np.asarray(context).astype(str)
    for task in stream.tasks:
        out[np.isin(ctx, task.contexts)] = task.task_id
    return out


EVAL_TRACK_SEED_OFFSETS = {
    "lambda_grid": 10_000,
    "fit": 20_000,
    "forget": 21_000,
    "stream": 22_000,
    "heldout": 23_000,
}
FINAL_EVAL_TRACKS = ("fit", "forget", "stream", "heldout")


def parse_final_eval_tracks(spec: str | None) -> tuple[str, ...]:
    """Comma-, plus-, or space-separated subset of ``FINAL_EVAL_TRACKS``.

    Slurm ``--export`` is comma-separated, so sbatch must pass ``stream+heldout``
    rather than ``stream,heldout``.
    """
    raw = str(spec or "+".join(FINAL_EVAL_TRACKS))
    for sep in (",", "+"):
        raw = raw.replace(sep, " ")
    tracks = tuple(part.strip() for part in raw.split() if part.strip())
    unknown = [name for name in tracks if name not in FINAL_EVAL_TRACKS]
    if unknown:
        raise ValueError(
            f"unknown eval track(s) {unknown}; expected a subset of "
            f"{FINAL_EVAL_TRACKS}"
        )
    if not tracks:
        raise ValueError("at least one final eval track is required")
    return tracks
EVAL_TASK_SEED_STRIDE = 101
# Bump when the scored metric set or residual definition changes so
# ``reusable_final_heldout`` cannot reuse a JSON that lacks the new fields.
EVAL_METRIC_VERSION = 2


def eval_seed(base_seed: int, track: str, task_id: int) -> int:
    """Canonical eval seed for one ``(track, task)`` pair.

    Every entry point derives eval seeds here, so scoring the same track at
    the same task from the same checkpoint is the same computation whether it
    runs inside ``train_cl.py`` or in a standalone eval script.
    """
    if track not in EVAL_TRACK_SEED_OFFSETS:
        raise ValueError(
            f"unknown eval track {track!r}; expected one of "
            f"{sorted(EVAL_TRACK_SEED_OFFSETS)}"
        )
    if int(task_id) < 0:
        raise ValueError(f"task_id must be >= 0, got {task_id}")
    return (
        int(base_seed)
        + EVAL_TRACK_SEED_OFFSETS[track]
        + EVAL_TASK_SEED_STRIDE * int(task_id)
    )


def eval_control_cap(args: argparse.Namespace) -> int:
    """Per-stratum control cap for the basal; ``<= 0`` keeps every control."""
    return int(getattr(args, "eval_max_control_cells", 0) or 0)


def eval_gen_per_stratum(args: argparse.Namespace) -> int:
    """Fixed model samples per scored stratum; ``<= 0`` is one per cell."""
    return int(getattr(args, "eval_gen_per_stratum", 0) or 0)


def lambda_gen_per_stratum(args: argparse.Namespace) -> int:
    """Fixed model samples per stratum for λ-grid selection."""
    return int(getattr(args, "lambda_gen_per_stratum", 0) or 0)


def eval_provenance(
    args: argparse.Namespace,
    task_id: int,
    *,
    tracks: tuple[str, ...],
) -> dict[str, Any]:
    """Everything a score depends on apart from the checkpoint itself."""
    return {
        "seed": int(args.seed),
        "task_id": int(task_id),
        "max_cells_per_stratum": int(args.eval_max_cells_per_stratum),
        "max_control_cells": eval_control_cap(args),
        "gen_per_stratum": eval_gen_per_stratum(args),
        "ode_method": str(args.ode_method),
        "ode_step_size": (
            None if args.ode_step_size is None else float(args.ode_step_size)
        ),
        "ode_rtol": float(args.ode_rtol),
        "ode_atol": float(args.ode_atol),
        "pred_control": str(getattr(args, "pred_control", "decoded")),
        "flow_source": str(getattr(args, "flow_source", "noise")),
        "metric_version": int(EVAL_METRIC_VERSION),
        "track_seeds": {
            track: eval_seed(args.seed, track, task_id) for track in tracks
        },
    }


def cap_eval_indices(
    arrays: TxClArrays,
    indices: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> np.ndarray:
    """Cap scored strata, and cap control strata separately.

    Controls are the Pearson-Δ basal rather than a scored stratum, and every
    stratum in a pool subtracts the same control mean. Subsampling controls
    therefore shifts all deltas in that pool coherently instead of averaging
    out, so the basal is kept whole unless ``--eval-max-control-cells`` asks
    otherwise.
    """
    idx = np.asarray(indices, dtype=np.int64)
    control = arrays.encoding.control_pert_index

    def cap(subset: np.ndarray, max_per_stratum: int) -> np.ndarray:
        return stratified_cap_indices(
            subset,
            arrays.c,
            arrays.p,
            arrays.z,
            arrays.covariate_col_order,
            max_per_stratum=max_per_stratum,
            seed=seed,
        )

    if control is None or idx.size == 0:
        return cap(idx, args.eval_max_cells_per_stratum)
    is_control = arrays.p[idx] == int(control)
    return np.sort(
        np.concatenate(
            (
                cap(idx[~is_control], args.eval_max_cells_per_stratum),
                cap(idx[is_control], eval_control_cap(args)),
            )
        )
    )


def prepare_eval(
    arrays: TxClArrays,
    adata: ad.AnnData,
    indices: np.ndarray,
    task_ids: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = cap_eval_indices(arrays, indices, args, seed=seed)
    return idx, load_gene_rows(adata, idx), task_ids[idx]


def control_reference_indices(
    arrays: TxClArrays,
    target_indices: np.ndarray,
    trainable_indices: np.ndarray,
    *,
    max_per_stratum: int,
    seed: int,
) -> np.ndarray:
    """Same-context/covariate controls to append to a heldout eval slice."""
    control = arrays.encoding.control_pert_index
    target = np.asarray(target_indices, dtype=np.int64)
    reference = np.asarray(trainable_indices, dtype=np.int64)
    if control is None or target.size == 0 or reference.size == 0:
        return np.empty(0, dtype=np.int64)

    def pool_codes(indices: np.ndarray) -> np.ndarray:
        codes = arrays.c[indices].astype(np.int64, copy=True)
        for col in arrays.covariate_col_order:
            values = arrays.z[col]
            codes = codes * (int(values.max()) + 1) + values[indices]
        return codes

    target_pools = np.unique(pool_codes(target))
    keep_pool = np.isin(pool_codes(reference), target_pools)
    candidates = reference[
        keep_pool & (arrays.p[reference] == int(control))
    ]
    return stratified_cap_indices(
        candidates,
        arrays.c,
        arrays.p,
        arrays.z,
        arrays.covariate_col_order,
        max_per_stratum=max_per_stratum,
        seed=seed,
    )


def eval_track(
    model,
    arrays,
    adata,
    decoders,
    indices,
    all_task_ids,
    args,
    device,
    *,
    seed,
    control_reference=None,
    generation_reference=None,
):
    target_idx = cap_eval_indices(arrays, indices, args, seed=seed)
    if target_idx.size == 0:
        return {"n_cells": 0, "selection_score": float("nan"), "rows": []}
    control_idx = (
        control_reference_indices(
            arrays,
            target_idx,
            np.asarray(control_reference, dtype=np.int64),
            max_per_stratum=eval_control_cap(args),
            seed=seed + 1,
        )
        if control_reference is not None
        else np.empty(0, dtype=np.int64)
    )
    idx = np.concatenate((target_idx, control_idx))
    genes = load_gene_rows(adata, idx)
    tids = all_task_ids[idx]
    source_mode = str(getattr(args, "flow_source", "noise"))
    if generation_reference is None and source_mode == "control":
        generation_reference = (
            control_reference if control_reference is not None else target_idx
        )
    score_arrays = None
    score_task_ids = None
    if str(getattr(args, "pearson_grouping", "condition")) == "cell_type":
        if "cell_type" not in arrays.z:
            raise ValueError("Pearson cell_type grouping requires a cell_type covariate")
        cell_types = arrays.encoding.covariate_categories["cell_type"]
        score_encoding = replace(
            arrays.encoding,
            context_categories=cell_types,
            context_to_idx={name: i for i, name in enumerate(cell_types)},
            covariate_col_order=(),
            covariate_categories={},
            covariate_to_idx={},
        )
        score_arrays = replace(
            arrays,
            encoding=score_encoding,
            c=arrays.z["cell_type"],
            z={},
        )
        score_task_ids = np.zeros(idx.size, dtype=np.int16)
    result = evaluate_model_gene_space(
        model,
        arrays,
        decoders,
        idx,
        genes,
        tids,
        device=device,
        batch_size=args.eval_batch_size,
        seed=seed,
        ode_method=args.ode_method,
        ode_step_size=args.ode_step_size,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        pred_control=str(getattr(args, "pred_control", "decoded")),
        flow_source=str(getattr(args, "flow_source", "noise")),
        source_indices=(
            np.asarray(generation_reference, dtype=np.int64)
            if generation_reference is not None
            else None
        ),
        samples_per_stratum=eval_gen_per_stratum(args),
        score_arrays=score_arrays,
        score_task_ids=score_task_ids,
    )
    control_pert = arrays.encoding.control_pert_index
    result["n_cells"] = int(target_idx.size)
    result["n_control_reference_cells"] = int(control_idx.size)
    result["n_basal_cells"] = (
        int((arrays.p[idx] == int(control_pert)).sum())
        if control_pert is not None
        else 0
    )
    return result


def validate_method_args(args: argparse.Namespace) -> None:
    if args.target_steps <= 0:
        raise ValueError(
            f"--target-steps must be positive for CL runs; got {args.target_steps}"
        )
    if int(getattr(args, "later_target_steps", 0) or 0) < 0:
        raise ValueError(
            f"--later-target-steps must be >= 0; got {args.later_target_steps}"
        )
    if float(args.later_lr) < 0:
        raise ValueError(f"--later-lr must be >= 0; got {args.later_lr}")
    if float(args.trajectory_replay_batch_multiplier) <= 0:
        raise ValueError("--trajectory-replay-batch-multiplier must be > 0")
    if float(args.trajectory_replay_loss_multiplier) <= 0:
        raise ValueError("--trajectory-replay-loss-multiplier must be > 0")
    uses_lambda = args.strategy in REPLAY_STRATEGIES
    if args.lambda_mode is None:
        args.lambda_mode = "grid" if uses_lambda else "none"
    if uses_lambda and args.lambda_mode == "none":
        raise ValueError(f"--strategy {args.strategy} requires --lambda-mode grid or equal")
    if not uses_lambda and args.lambda_mode != "none":
        raise ValueError(f"--strategy {args.strategy} requires --lambda-mode none")
    if args.replay_backend == "mean_flow" and args.strategy not in SYNTHETIC_REPLAY_STRATEGIES:
        raise ValueError("--replay-backend mean_flow requires generative or trajectory")
    if args.lora_rank < 1 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    resolve_two_stage_1_epochs(
        args.epochs,
        two_stage_1_epochs=args.two_stage_1_epochs,
        two_stage_1_epoch_fraction=args.two_stage_1_epoch_fraction,
    )


def task_finetune_params(
    model: ConditionalFlowMatching,
    arrays: TxClArrays,
    new_indices: np.ndarray,
    args: argparse.Namespace,
    *,
    stage: int | None,
    old_indices: np.ndarray | None = None,
) -> tuple[list[torch.nn.Parameter], frozenset[int] | None, frozenset[int] | None]:
    """Configure task-k trainables; return params plus optional embedding row masks.

    Context and perturbation tables use the same rule: ``None`` means every row
    trains; a frozenset restricts Adam to those rows. New-task-only perturbation
    rows always train (they have never seen a T0 gradient). ``pert=0`` still
    freezes perturbation rows that already appeared in the old-task pool.
    """
    context_rows = frozenset(int(x) for x in np.unique(arrays.c[new_indices]))
    if not context_rows:
        raise ValueError("Task has no context rows")
    new_perts = {int(x) for x in np.unique(arrays.p[new_indices])}
    old = np.asarray([] if old_indices is None else old_indices, dtype=np.int64)
    old_perts = {int(x) for x in np.unique(arrays.p[old])} if old.size else set()
    novel_perts = new_perts - old_perts
    scope = "embedding" if stage == 1 else args.finetune_scope
    # Replay supplies old-task supervision, so all conditioning rows must be
    # trainable. In particular, perturbations unique to the new context have
    # never received a T0 gradient and cannot remain frozen at initialization.
    full_replay_update = stage != 1 and args.strategy in REPLAY_STRATEGIES
    train_all_perts = full_replay_update or (
        stage != 1 and args.finetune_train_old_pert_embeddings
    )
    train_all_contexts = full_replay_update or (
        stage != 1 and args.finetune_train_old_context_embeddings
    )
    train_pert_table = train_all_perts or bool(novel_perts)
    params, _, _ = model.configure_finetune(
        holdout_mode="context",
        train_vector_field=scope == "full",
        train_lora=scope == "lora",
        train_old_pert_embeddings=train_pert_table,
        train_old_context_embeddings=train_all_contexts,
        train_covariate_embeddings=full_replay_update,
        held_out_pert=0,
        held_out_context=min(context_rows),
    )
    pert_rows = None if train_all_perts else (frozenset(novel_perts) if novel_perts else None)
    return params, None if train_all_contexts else context_rows, pert_rows


def train_candidate(
    base_model: ConditionalFlowMatching,
    base_mean_flow,
    arrays: TxClArrays,
    new_indices: np.ndarray,
    old_indices: np.ndarray,
    task_id: int,
    lam: float | None,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epochs: int,
):
    student = copy.deepcopy(base_model).to(device)
    loader = make_loader(
        arrays,
        new_indices,
        args,
        seed=args.seed + 1000 * task_id + int(100 * (lam or 0)),
    )
    if task_id == 0:
        history = train_flow_matching(
            student,
            loader,
            data_dim=arrays.se_dim,
            device=device,
            epochs=epochs,
            lr=args.lr,
            train_seed=args.seed + 1000 * task_id,
        )
        mean_flow = None
        if args.replay_backend == "mean_flow":
            teacher = make_frozen_model_copy(student).to(device)
            mean_flow = make_mean_flow_from_teacher(
                teacher, freeze_condition_embeddings=False
            ).to(device)
            mf_history = train_mean_flow_distill(
                mean_flow,
                teacher,
                loader,
                data_dim=arrays.se_dim,
                device=device,
                epochs=epochs if args.target_steps > 0 else args.mean_flow_distill_epochs,
                lr=args.mean_flow_lr or args.lr,
                train_seed=args.seed + 50_000,
                diag_fraction=args.mean_flow_diag_fraction,
                adaptive_p=args.mean_flow_adaptive_p,
                time_sampler=args.mean_flow_time_sampler,
                time_mu=args.mean_flow_time_mu,
                time_sigma=args.mean_flow_time_sigma,
                covariate_col_order=arrays.covariate_col_order,
                max_grad_norm=args.mean_flow_max_grad_norm,
                use_adamw=args.mean_flow_use_adamw,
                betas=(0.9, args.mean_flow_adam_beta2),
                weight_decay=args.mean_flow_weight_decay,
            )
            for row, mf_row in zip(history[-len(mf_history) :], mf_history):
                row["mf_loss_mean"] = mf_row["loss_mean"]
        return student, mean_flow, history

    if args.strategy in REPLAY_STRATEGIES:
        assert lam is not None
    else:
        assert lam is None

    teacher = (
        make_frozen_model_copy(base_model).to(device)
        if args.strategy in ("gft", *SYNTHETIC_REPLAY_STRATEGIES)
        else None
    )
    old = np.asarray(old_indices, dtype=np.int64)
    support_contexts: list[int] = []
    c_old = p_old = None
    z_old = None
    dist = None
    if args.strategy in REPLAY_STRATEGIES:
        c_old = torch.from_numpy(arrays.c[old]).long()
        p_old = torch.from_numpy(arrays.p[old]).long()
        z_old = (
            {
                col: torch.from_numpy(arrays.z[col][old]).long()
                for col in arrays.covariate_col_order
            }
            if arrays.covariate_col_order
            else None
        )
        support_contexts = sorted(set(int(x) for x in arrays.c[old]))
        dist = build_support_stratum_distribution(
            c_old,
            p_old,
            support_contexts,
            z=z_old,
            covariate_col_order=arrays.covariate_col_order,
        )

    replay_mean_flow = None
    mean_flow = None
    if args.replay_backend == "mean_flow":
        if base_mean_flow is None:
            raise ValueError("MeanFlow replay requested without the previous task checkpoint")
        replay_mean_flow = copy.deepcopy(base_mean_flow).to(device).eval()
        mean_flow = copy.deepcopy(base_mean_flow).to(device).train()
        for parameter in mean_flow.parameters():
            parameter.requires_grad_(True)

    if args.finetune_scope == "lora":
        student.apply_mlp_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            train_output_layer=args.lora_train_output_layer,
        )

    stage_1 = resolve_two_stage_1_epochs(
        epochs,
        two_stage_1_epochs=args.two_stage_1_epochs,
        two_stage_1_epoch_fraction=args.two_stage_1_epoch_fraction,
    )
    stages = (
        [(stage_1, 1), (epochs - stage_1, 2)]
        if stage_1 is not None
        else [(epochs, None)]
    )
    history: list[dict[str, float]] = []
    epoch_offset = 0
    seed_base = args.seed + 1000 * task_id + int(100 * (lam or 0.0))

    real_replay: dict[str, Any] | None = None
    if args.strategy == "real":
        assert c_old is not None and p_old is not None and dist is not None
        old_global = torch.from_numpy(old).long()
        local_cells = build_stratum_cell_index(
            c_old,
            p_old,
            support_contexts,
            z=z_old,
            covariate_col_order=arrays.covariate_col_order,
        )
        stratum_cells = {
            key: old_global[local_idx] for key, local_idx in local_cells.items()
        }
        support_local = build_support_flat_index(c_old, support_contexts)
        real_replay = {
            "x_support": torch.from_numpy(arrays.se),
            "c_support": torch.from_numpy(arrays.c).long(),
            "p_support": torch.from_numpy(arrays.p).long(),
            "z_support": (
                {
                    col: torch.from_numpy(arrays.z[col]).long()
                    for col in arrays.covariate_col_order
                }
                if arrays.covariate_col_order
                else None
            ),
            "covariate_col_order": arrays.covariate_col_order,
            "stratum_cells": stratum_cells,
            "support_flat_idx": old_global[support_local],
        }

    for stage_epochs, stage in stages:
        params, context_rows, pert_rows = task_finetune_params(
            student, arrays, new_indices, args, stage=stage, old_indices=old
        )
        common = {
            "data_dim": arrays.se_dim,
            "device": device,
            "epochs": int(stage_epochs),
            "lr": args.later_lr or args.lr,
            "train_seed": seed_base + epoch_offset,
            "params": params,
            "context_emb_train_rows": context_rows,
            "pert_emb_train_rows": pert_rows,
            "epoch_offset": epoch_offset,
            "log_total_epochs": epochs,
            "finetune_stage": stage,
        }
        if args.strategy == "standard":
            part = train_flow_matching(student, loader, **common)
        elif args.strategy == "gft":
            assert teacher is not None
            part = train_flow_matching_gft(
                student,
                teacher,
                loader,
                beta_start=args.gft_beta_start,
                beta_end=args.gft_beta_end,
                beta_schedule=args.gft_beta_schedule,
                sigmoid_midpoint=args.gft_sigmoid_midpoint,
                sigmoid_steepness=args.gft_sigmoid_steepness,
                global_step_offset=epoch_offset * len(loader),
                gft_total_steps=max(1, epochs * len(loader)),
                **common,
            )
        elif args.strategy == "real":
            assert lam is not None and dist is not None and real_replay is not None
            part = train_flow_matching_real_replay(
                student,
                loader,
                n_perturbations=arrays.encoding.n_perturbations,
                replay_fraction=float(lam),
                support_contexts=support_contexts,
                replay_condition_sampling=args.replay_condition_sampling,
                replay_stratum_dist=dist,
                **real_replay,
                **common,
            )
        elif args.replay_backend == "mean_flow":
            assert (
                lam is not None
                and teacher is not None
                and mean_flow is not None
                and dist is not None
            )
            part = train_conditional_joint_mean_flow(
                student,
                mean_flow,
                teacher,
                loader,
                strategy=args.strategy,
                lambda_mix=float(lam),
                support_contexts=support_contexts,
                n_perturbations=arrays.encoding.n_perturbations,
                replay_stratum_dist=dist,
                replay_condition_sampling=args.replay_condition_sampling,
                replay_mean_flow=replay_mean_flow,
                mean_flow_n_steps=args.mean_flow_steps,
                ode_step_size=args.ode_step_size,
                traj_kappa=args.traj_kappa,
                mf_lr=args.mean_flow_lr,
                diag_fraction=args.mean_flow_diag_fraction,
                time_sampler=args.mean_flow_time_sampler,
                time_mu=args.mean_flow_time_mu,
                time_sigma=args.mean_flow_time_sigma,
                mf_max_grad_norm=args.mean_flow_max_grad_norm,
                mf_use_adamw=args.mean_flow_use_adamw,
                mf_betas=(0.9, args.mean_flow_adam_beta2),
                mf_weight_decay=args.mean_flow_weight_decay,
                **common,
            )
        else:
            assert lam is not None and teacher is not None and dist is not None
            part = train_flow_matching_generative_replay(
                student,
                teacher,
                loader,
                support_contexts=support_contexts,
                n_perturbations=arrays.encoding.n_perturbations,
                replay_fraction=float(lam),
                replay_mode="endpoint" if args.strategy == "generative" else "field",
                trajectory_batch_sampling=args.trajectory_batch_sampling,
                trajectory_replay_batch_multiplier=args.trajectory_replay_batch_multiplier,
                trajectory_replay_loss_multiplier=args.trajectory_replay_loss_multiplier,
                trajectory_time_sampling=args.trajectory_time_sampling,
                replay_ode_method=args.ode_method,
                replay_ode_rtol=args.ode_rtol,
                replay_ode_atol=args.ode_atol,
                replay_ode_step_size=args.ode_step_size,
                replay_condition_sampling=args.replay_condition_sampling,
                replay_stratum_dist=dist,
                traj_kappa=args.traj_kappa,
                **common,
            )
        history.extend(part)
        epoch_offset += int(stage_epochs)

    if args.finetune_scope == "lora":
        student.merge_mlp_lora()
    return student, mean_flow, history


def evaluate_final_checkpoint(
    out_dir: Path,
    arrays: TxClArrays,
    adata: ad.AnnData,
    decoders: TaskDecoderBank,
    stream: TxClStream,
    n_tasks: int,
    train_ids: np.ndarray,
    context_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Capped fit / forget / stream / heldout on the last task checkpoint."""
    print("\n=== Final evaluation ===", flush=True)
    assert_stream_test_never_trained(
        stream,
        arrays.context,
        arrays.split,
        arrays.perturbation,
        n_tasks=n_tasks,
    )
    final_model, _ = model_from_checkpoint(
        out_dir / "tasks" / f"{n_tasks - 1:02d}" / "checkpoint.pt",
        device,
    )
    final_masks = step_masks(
        stream,
        n_tasks - 1,
        arrays.context,
        arrays.split,
        arrays.perturbation,
    )
    fit_idx = np.flatnonzero(final_masks.new_train)
    forget_idx = np.flatnonzero(final_masks.old_train)
    stream_idx = np.flatnonzero(final_masks.seen_train)
    heldout_idx = np.flatnonzero(final_masks.heldout)
    final_task = n_tasks - 1
    wanted = parse_final_eval_tracks(getattr(args, "eval_final_tracks", None))
    specs: dict[str, tuple] = {
        "fit": (fit_idx, train_ids, None),
        "forget": (forget_idx, train_ids, None),
        "stream": (stream_idx, train_ids, None),
        "heldout": (heldout_idx, context_ids, stream_idx),
    }
    final_tracks: dict[str, Any] = {}
    for name in wanted:
        target, tids, reference = specs[name]
        print(f"  track={name} n={int(np.asarray(target).size):,}", flush=True)
        final_tracks[name] = eval_track(
            final_model,
            arrays,
            adata,
            decoders,
            target,
            tids,
            args,
            device,
            seed=eval_seed(args.seed, name, final_task),
            control_reference=reference,
        )
    result = {
        "max_cells_per_stratum": int(args.eval_max_cells_per_stratum),
        "eval_provenance": eval_provenance(
            args,
            final_task,
            tracks=wanted,
        ),
        "final": final_tracks,
    }
    dump_json(out_dir / "final_metrics.json", result)
    return result


def main() -> None:
    args = make_parser().parse_args()
    defaults = DEFAULTS[args.dataset]
    stream = stream_for(args.dataset)
    n_tasks = stream.n_tasks
    if args.max_tasks > 0:
        n_tasks = min(n_tasks, args.max_tasks)
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.mean_flow_distill_epochs = min(args.mean_flow_distill_epochs, 1)
        args.decoder_epochs = min(args.decoder_epochs, 2)
        args.max_train_cells = 2048
        args.decoder_max_fit_cells = 2048
        args.eval_max_cells_per_stratum = 2
        args.eval_max_control_cells = 4
        args.eval_gen_per_stratum = min(max(eval_gen_per_stratum(args), 1), 2)
        args.lambda_gen_per_stratum = min(max(lambda_gen_per_stratum(args), 1), 2)
        n_tasks = min(n_tasks, 2)
    validate_method_args(args)

    h5ad_path = args.h5ad_path or defaults["h5ad"]
    representation = cl_representation(args)
    expected_pca_dim: int | None = None
    if representation == "pca":
        h5ad_path = args.h5ad_path or defaults.get("h5ad_pca") or defaults["h5ad"]
        se_path, args.pca_basis, expected_pca_dim = resolve_pca_artifacts(
            defaults,
            pca_dim=int(getattr(args, "pca_dim", 0) or 0) or None,
            se_path=args.se_path,
            pca_basis=getattr(args, "pca_basis", None),
        )
    else:
        se_path = embedding_sidecar_path(defaults, representation, args.se_path)
    covariates = parse_covariates(args.covariate_cols, defaults["covariates"])
    control = args.control_perturbation or defaults["control"]
    arrays = load_cl_arrays(
        h5ad_path,
        se_path,
        stream,
        covariate_cols=covariates,
        control_perturbation=control,
        representation=representation,
    )
    if representation == "pca":
        if expected_pca_dim is not None and int(arrays.se_dim) != int(expected_pca_dim):
            raise ValueError(
                f"PCA sidecar {se_path} has dim {arrays.se_dim}, expected "
                f"{expected_pca_dim} (--pca-dim / DEFAULT_PCA_DIM={DEFAULT_PCA_DIM})"
            )
        args.pca_dim = int(arrays.se_dim)
    name_parts = [args.strategy]
    if representation == "genes":
        name_parts.append("gene")
    elif representation == "scimilarity":
        name_parts.append("sci")
    elif representation == "pca":
        name_parts.append(f"pca{arrays.se_dim}")
    if args.strategy in SYNTHETIC_REPLAY_STRATEGIES:
        name_parts.append(
            f"mf_n{args.mean_flow_steps}"
            if args.replay_backend == "mean_flow"
            else "ode"
        )
    if args.strategy in REPLAY_STRATEGIES:
        name_parts.append(args.lambda_mode)
    if args.finetune_scope != "full":
        name_parts.append(args.finetune_scope)
    if args.finetune_scope == "lora":
        name_parts.append(f"r{int(args.lora_rank)}")
    if args.strategy == "gft":
        name_parts.append(str(args.gft_beta_schedule))
    if args.two_stage_1_epochs > 0:
        name_parts.append(f"ts{args.two_stage_1_epochs}")
    elif args.two_stage_1_epoch_fraction is not None:
        name_parts.append(f"tsf{args.two_stage_1_epoch_fraction:g}")
    if args.strategy not in REPLAY_STRATEGIES:
        pert_tag = 1 if args.finetune_train_old_pert_embeddings else 0
        ctx_tag = 1 if args.finetune_train_old_context_embeddings else 0
        name_parts.append(f"p{pert_tag}c{ctx_tag}")
    if args.target_steps > 0:
        name_parts.append(f"st{args.target_steps}")
        later = int(getattr(args, "later_target_steps", 0) or 0)
        if later > 0 and later != int(args.target_steps):
            name_parts.append(f"ft{later}")
    run_name = args.run_name or "_".join((*name_parts, f"s{args.seed}"))
    out_dir = args.out_dir or (defaults["bucket"] / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_inputs = args.h5ad_path is None and (
        representation == "genes" or args.se_path is None
    )
    use_shared_decoders = canonical_inputs and not args.smoke and args.max_tasks <= 0
    decoder_dir = args.decoder_dir or (
        defaults["bucket"]
        / shared_decoder_dirname(
            representation,
            args.decoder_hidden,
            args.decoder_depth,
            pca_dim=int(arrays.se_dim) if representation == "pca" else 0,
        )
        if use_shared_decoders
        else out_dir / "decoders"
    )
    device = torch.device(args.device)
    lambda_grid = parse_float_list(args.lambda_grid)

    cfg = vars(args).copy()
    cfg.update(
        {
            "h5ad_path": str(h5ad_path),
            "se_path": str(se_path) if se_path is not None else str(h5ad_path),
            "representation": representation,
            "out_dir": str(out_dir),
            "decoder_dir": str(decoder_dir),
            "covariate_cols": list(covariates),
            "control_perturbation": control,
            "lambda_grid": list(lambda_grid),
            "stream": stream.to_jsonable(),
            "encoding": arrays.encoding.to_jsonable(),
        }
    )
    for key, value in list(cfg.items()):
        if isinstance(value, Path):
            cfg[key] = str(value)
    dump_json(out_dir / "config.json", cfg)

    train_ids = train_task_ids(stream, arrays.context, arrays.split)
    context_ids = task_id_from_context(stream, arrays.context)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    decoders = prepare_decoder_bank(
        decoder_dir,
        arrays,
        adata,
        stream,
        Path(h5ad_path),
        None if se_path is None else Path(se_path),
        args,
        device,
        n_tasks=n_tasks,
    )

    model = None
    mean_flow = None
    start_task = 0
    metrics_by_task: list[dict] = []
    if not args.no_resume:
        model, start_task, metrics_by_task = find_resume(out_dir, n_tasks, device)
        if start_task and args.replay_backend == "mean_flow":
            mf_path = out_dir / "tasks" / f"{start_task - 1:02d}" / MEAN_FLOW_CKPT_NAME
            if not mf_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume MeanFlow run: missing {mf_path}"
                )
            mean_flow, _ = load_mean_flow_checkpoint(mf_path, device)
    if model is None:
        # Pair initial weights across methods and make each named seed reproducible.
        # Training, replay, and evaluation use their own derived generators.
        torch.manual_seed(args.seed)
        model = make_model(arrays, args).to(device)
    if start_task:
        print(f"Resume: tasks 0..{start_task - 1} complete", flush=True)

    t_run = time.time()
    for k in range(start_task, n_tasks):
        task = stream.task(k)
        print(f"\n=== Task {k}/{n_tasks - 1}: {task.name} {task.contexts} ===", flush=True)
        task_dir = out_dir / "tasks" / f"{k:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        masks = step_masks(stream, k, arrays.context, arrays.split, arrays.perturbation)
        assert_test_never_trained(masks, arrays.split)
        new_train = np.flatnonzero(masks.new_train)
        old_train = np.flatnonzero(masks.old_train)
        if new_train.size == 0:
            raise ValueError(f"Task {k} has no training cells")
        n_loader = capped_random(
            new_train, int(args.max_train_cells), args.seed + 1000 * k
        ).size
        spe = steps_per_epoch(n_loader, args.batch_size)
        task_steps = target_steps_for_task(args, k)
        task_epochs = (
            args.epochs
            if args.smoke or int(task_steps) <= 0
            else epochs_for_target_steps(n_loader, args.batch_size, task_steps)
        )
        print(
            f"  train cells={n_loader:,}  steps/epoch={spe}  epochs={task_epochs}  "
            f"target_steps={task_steps:,}  ~{spe * task_epochs:,} Adam steps",
            flush=True,
        )

        candidates = (
            (equal_weight_lambda(k),)
            if k > 0
            and args.strategy in REPLAY_STRATEGIES
            and args.lambda_mode == "equal"
            else (
                lambda_grid
                if k > 0
                and args.strategy in REPLAY_STRATEGIES
                and args.lambda_mode == "grid"
                else (None,)
            )
        )
        use_grid_selection = (
            k > 0
            and args.strategy in REPLAY_STRATEGIES
            and args.lambda_mode == "grid"
        )
        if use_grid_selection:
            seen_train = np.flatnonzero(masks.seen_train)
            eval_idx, eval_genes, eval_tids = prepare_eval(
                arrays,
                adata,
                seen_train,
                train_ids,
                args,
                seed=eval_seed(args.seed, "lambda_grid", k),
            )
        candidate_rows: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_model = None
        best_mean_flow = None
        best_history = None
        chosen_lambda = None
        for lam in candidates:
            print(f"  candidate λ={lam}", flush=True)
            t0 = time.time()
            student, candidate_mean_flow, history = train_candidate(
                model,
                mean_flow,
                arrays,
                new_train,
                old_train,
                k,
                lam,
                args,
                device,
                epochs=task_epochs,
            )
            train_seconds = time.time() - t0
            row = {
                "lambda": lam,
                "last_loss": history[-1]["loss_mean"] if history else None,
                "train_seconds": train_seconds,
            }
            if use_grid_selection:
                t0 = time.time()
                val = evaluate_model_gene_space(
                    student,
                    arrays,
                    decoders,
                    eval_idx,
                    eval_genes,
                    eval_tids,
                    device=device,
                    batch_size=args.eval_batch_size,
                    seed=eval_seed(args.seed, "lambda_grid", k),
                    ode_method=args.ode_method,
                    ode_step_size=args.ode_step_size,
                    ode_rtol=args.ode_rtol,
                    ode_atol=args.ode_atol,
                    pred_control=str(getattr(args, "pred_control", "decoded")),
                    flow_source=str(getattr(args, "flow_source", "noise")),
                    samples_per_stratum=lambda_gen_per_stratum(args),
                )
                val_seconds = time.time() - t0
                balanced = old_new_balanced_selection(val, new_task_id=k)
                score = float(balanced["selection_score"])
                row.update(
                    {
                        "selection_metric": balanced["selection_metric"],
                        "selection_score": score,
                        "old_task_macro": balanced["old_task_macro"],
                        "new_task_macro": balanced["new_task_macro"],
                        "n_old_strata": balanced["n_old_strata"],
                        "n_new_strata": balanced["n_new_strata"],
                        "pearson_delta_macro": val["pearson_delta_macro"],
                        "pearson_macro": val["pearson_macro"],
                        "n_strata": val["n_strata"],
                        "n_delta_strata": val["n_delta_strata"],
                        "n_pearson_fallback_strata": val[
                            "n_pearson_fallback_strata"
                        ],
                        "per_task": val["per_task"],
                        "eval_seconds": val_seconds,
                        "max_cells_per_stratum": int(
                            args.eval_max_cells_per_stratum
                        ),
                        "gen_per_stratum": lambda_gen_per_stratum(args),
                    }
                )
                print(
                    f"    {balanced['selection_metric']}={score:.5f} "
                    f"old={balanced['old_task_macro']:.5f} "
                    f"new={balanced['new_task_macro']:.5f} "
                    f"train={train_seconds:.1f}s eval={val_seconds:.1f}s",
                    flush=True,
                )
            else:
                score = float("nan")
                row["evaluation"] = "skipped"
                print(f"    train={train_seconds:.1f}s (evaluation skipped)", flush=True)
            candidate_rows.append(row)
            if (
                best_model is None
                or not use_grid_selection
                or (
                    np.isfinite(score)
                    and (not np.isfinite(best_score) or score > best_score)
                )
            ):
                best_score = score
                best_model = student
                best_mean_flow = candidate_mean_flow
                best_history = history
                chosen_lambda = lam

        assert best_model is not None
        model = best_model
        mean_flow = best_mean_flow
        dump_json(
            task_dir / "lambda_sweep.json",
            {
                "lambda_mode": (
                    args.lambda_mode
                    if k > 0 and args.strategy in REPLAY_STRATEGIES
                    else None
                ),
                "selection_evaluated": use_grid_selection,
                "gen_per_stratum": (
                    lambda_gen_per_stratum(args) if use_grid_selection else None
                ),
                "candidates": candidate_rows,
                "chosen_lambda": chosen_lambda,
            },
        )
        dump_json(task_dir / "history.json", {"lambda": chosen_lambda, "history": best_history})
        ckpt_cfg = checkpoint_config(
            arrays, model, args, task_id=k, chosen_lambda=chosen_lambda
        )
        save_checkpoint(task_dir / "checkpoint.pt", model, ckpt_cfg)
        save_checkpoint(out_dir / "checkpoint.pt", model, ckpt_cfg)
        if args.replay_backend == "mean_flow":
            if mean_flow is None:
                raise RuntimeError("MeanFlow backend did not produce a MeanFlow model")
            save_mean_flow_checkpoint(
                task_dir / MEAN_FLOW_CKPT_NAME,
                mean_flow,
                ckpt_cfg,
                parent_checkpoint="checkpoint.pt",
                args=args,
                extra={
                    "joint_mean_flow": k > 0,
                    "mean_flow_steps": int(args.mean_flow_steps),
                    "task_id": k,
                },
            )
            save_mean_flow_checkpoint(
                out_dir / MEAN_FLOW_CKPT_NAME,
                mean_flow,
                ckpt_cfg,
                parent_checkpoint="checkpoint.pt",
                args=args,
                extra={
                    "joint_mean_flow": k > 0,
                    "mean_flow_steps": int(args.mean_flow_steps),
                    "task_id": k,
                },
            )

        metrics = {
            "task_id": k,
            "task_name": task.name,
            "contexts": list(task.contexts),
            "n_train_cells": int(n_loader),
            "steps_per_epoch": int(spe),
            "epochs": int(task_epochs),
            "target_steps": int(task_steps) or None,
            "approx_adam_steps": int(spe * task_epochs),
            "chosen_lambda": chosen_lambda,
            "selection_evaluated": use_grid_selection,
            "selection_score": best_score if use_grid_selection else None,
        }
        dump_json(task_dir / "metrics.json", metrics)
        metrics_by_task.append(metrics)
        dump_json(
            out_dir / "summary.json",
            {
                "config": cfg,
                "metrics_by_task": metrics_by_task,
                "final": None,
                "elapsed_seconds": time.time() - t_run,
            },
        )

    final = evaluate_final_checkpoint(
        out_dir,
        arrays,
        adata,
        decoders,
        stream,
        n_tasks,
        train_ids,
        context_ids,
        args,
        device,
    )
    dump_json(
        out_dir / "summary.json",
        {
            "config": cfg,
            "metrics_by_task": metrics_by_task,
            "final": final,
            "elapsed_seconds": time.time() - t_run,
        },
    )
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
