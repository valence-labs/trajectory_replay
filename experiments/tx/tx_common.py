"""Shared helpers for transcriptomics pretrain / finetune scripts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from trajectory_replay.utils.cli import resolve_device
from trajectory_replay.io.results import resolve_tx_path
from trajectory_replay.data.tx_dataset import (
    TxEncoding,
    context_index,
    default_h5ad_path,
    load_tx_data,
    make_tx_tensor_dataset,
    parse_covariate_cols,
    parse_phases,
)
from trajectory_replay.model.flow_matching import (
    ConditionalFlowMatching,
    build_conditional_flow_matching_from_config,
)
from trajectory_replay.data.tx_strata import (
    control_indices_by_prefix,
    control_pool_key_for_stratum,
    iter_observed_strata,
)
from trajectory_replay.holdout.data import stratum_seed
from trajectory_replay.training.flow_matching import control_indices_by_context, eval_ode_strata


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--h5ad-path",
        type=Path,
        default=None,
        help="Preprocessed AnnData (default: data_dir/pbmc_parse_with_splits.h5ad).",
    )
    p.add_argument("--split-col", default="split1", help="obs column with split labels.")
    p.add_argument(
        "--context-col",
        default="donor",
        help="obs column for context c (default donor for PBMC).",
    )
    p.add_argument(
        "--perturbation-col",
        default="cytokine",
        help="obs column for perturbation p (default cytokine for PBMC).",
    )
    p.add_argument("--obsm-key", default="X_pca", help="obsm matrix used as x.")
    p.add_argument(
        "--control-perturbation",
        default="PBS",
        help="Control perturbation label stored in encoding metadata.",
    )
    p.add_argument(
        "--covariate-cols",
        default="",
        help=(
            "Comma-separated obs columns for covariate embeddings (e.g. cell_type). "
            "Empty = none (Replogle default)."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stratified subsample cap per load (per (c,p)).",
    )
    p.add_argument("--subsample-seed", type=int, default=0)


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cond-emb-dim", type=int, default=128)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cpu or cuda")


def add_ode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ode-method", default="dopri5")
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument("--ode-step-size", type=float, default=None)
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help=(
            "Optional global stratified cap on ODE eval cells when "
            "--eval-max-samples-per-stratum is 0. Default 0 = use all cells."
        ),
    )
    p.add_argument(
        "--eval-max-samples-per-stratum",
        type=int,
        default=0,
        help=(
            "Cap ODE/cell-eval cells per (context, perturbation) stratum. "
            "Default 0 = unlimited (all cells in each stratum). Uses --subsample-seed "
            "per stratum when capped."
        ),
    )


def context_axis_label_from_args(args: argparse.Namespace | dict[str, object] | None) -> str:
    """Human-readable x-axis label from ``context_col`` (e.g. donor → Donor)."""
    col = None
    if isinstance(args, dict):
        col = args.get("context_col")
    elif args is not None:
        col = getattr(args, "context_col", None)
    if not col:
        return "Context"
    return str(col).replace("_", " ").strip().title()


def add_vcb_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--vcb-eval",
        action="store_true",
        help=(
            "Decode ODE predictions to gene space: VCB PEP metrics per (c,p) stratum "
            "and retrieval_mae per donor (VCB retrieval suite)."
        ),
    )
    p.add_argument(
        "--no-vcb-distributional",
        action="store_true",
        help="Skip per-stratum edistance in gene space (still compute mse/pearson/cosine/deltas).",
    )
    p.add_argument(
        "--vcb-control-phases",
        default="auto",
        help=(
            "Split phase(s) for observed control cells when the eval phase has none "
            "(comma-separated, or 'auto' / 'none'). Auto uses --finetune-phase or --train-phase."
        ),
    )
    p.add_argument(
        "--vcb-control-max-cells",
        type=int,
        default=256,
        help=(
            "Max observed control cells per context used for VCB basal/retrieval (default 256). "
            "Set 0 to use all controls in the reference phase(s)."
        ),
    )
    p.add_argument(
        "--vcb-retrieval-subset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Subsample perturbation groups for retrieval_mae (default: off). "
            "When enabled, caps the N×N gene-space distance matrix (use --vcb-retrieval-subset-size)."
        ),
    )
    p.add_argument(
        "--vcb-retrieval-subset-size",
        type=int,
        default=500,
        help="Max perturbation groups per context for retrieval when --vcb-retrieval-subset (default 500).",
    )
    p.add_argument(
        "--vcb-retrieval-subset-seed",
        type=int,
        default=42,
        help="Base seed for retrieval group subsampling (default 42; combined per context).",
    )


def add_val_monitor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--val-eval-every",
        type=int,
        default=0,
        help=(
            "If > 0, run ODE eval on --val-monitor-phase every N finetune epochs during training "
            "and append scalars to val_monitor_history (mean/median energy and mse_to_true "
            "on held-out context strata)."
        ),
    )
    p.add_argument(
        "--val-monitor-phase",
        default="val",
        help="OOD split label for in-training monitors (default val).",
    )


def resolve_h5ad(args: argparse.Namespace) -> Path:
    return args.h5ad_path if args.h5ad_path is not None else default_h5ad_path()


def ode_options_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    if args.ode_step_size is None:
        return None
    return {"step_size": float(args.ode_step_size)}


def eval_load_kwargs(
    args: argparse.Namespace,
    *,
    label: str = "eval",
) -> dict[str, Any]:
    """``load_tx_data`` subsampling for ODE/VCB eval (default: all cells)."""
    per_stratum = int(getattr(args, "eval_max_samples_per_stratum", 0) or 0)
    if per_stratum > 0:
        return {
            "max_samples": None,
            "max_samples_per_stratum": per_stratum,
            "subsample_seed": args.subsample_seed,
        }
    max_samples = int(getattr(args, "eval_max_samples", 0) or 0)
    if max_samples > 0:
        return {
            "max_samples": max_samples,
            "max_samples_per_stratum": None,
            "subsample_seed": args.subsample_seed + hash(label) % 10_000,
        }
    return {
        "max_samples": None,
        "max_samples_per_stratum": None,
        "subsample_seed": args.subsample_seed,
    }


def covariate_cols_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    return parse_covariate_cols(getattr(args, "covariate_cols", None))


def make_model(
    data_dim: int,
    encoding: TxEncoding,
    args: argparse.Namespace,
    *,
    n_contexts: int | None = None,
) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=data_dim,
        n_contexts=n_contexts if n_contexts is not None else encoding.n_contexts,
        n_perturbations=encoding.n_perturbations,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        covariate_sizes=encoding.covariate_sizes() or None,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ConditionalFlowMatching, dict[str, Any], TxEncoding]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    enc = TxEncoding.from_jsonable(cfg["encoding"])
    model = build_conditional_flow_matching_from_config(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, cfg, enc


def checkpoint_config(
    model: ConditionalFlowMatching,
    encoding: TxEncoding,
    args: argparse.Namespace,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "data_dim": model.data_dim,
        "n_contexts": model.n_contexts,
        "n_perturbations": model.n_perturbations,
        "cond_emb_dim": model.cond_emb_dim,
        "time_emb_dim": model.time_emb_dim,
        "hidden_dim": model.hidden_dim,
        "depth": model.depth,
        "backbone_type": model.backbone_type,
        "covariate_sizes": {
            col: model.covariate_embs[col].num_embeddings for col in model.covariate_col_order
        }
        if model.covariate_col_order
        else {},
        "experiment": "tx",
        "obsm_key": args.obsm_key,
        "split_col": args.split_col,
        "covariate_cols": ",".join(covariate_cols_from_args(args)),
        "encoding": encoding.to_jsonable(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def sync_finetune_model_args_from_checkpoint(
    args: argparse.Namespace,
    model: ConditionalFlowMatching,
) -> None:
    """Align CLI model flags with the loaded checkpoint (weights win over argparse)."""
    for key in ("cond_emb_dim", "time_emb_dim", "hidden_dim", "depth"):
        cli_val = int(getattr(args, key))
        model_val = int(getattr(model, key))
        if cli_val != model_val:
            print(
                f"  note: ignoring --{key.replace('_', '-')}={cli_val}; "
                f"checkpoint uses {model_val}"
            )
        setattr(args, key, model_val)


def sync_encoding_args(args: argparse.Namespace, encoding: TxEncoding) -> None:
    """Mirror checkpoint encoding metadata onto args for metrics / downstream eval."""
    if encoding.covariate_col_order and not covariate_cols_from_args(args):
        args.covariate_cols = ",".join(encoding.covariate_col_order)


# ODE eval protocol shared by pretrain and finetune.
ODE_EVAL_SEED_OFFSET = 999
PRETRAIN_ODE_EVAL_SEED_OFFSET = ODE_EVAL_SEED_OFFSET  # alias
PRETRAIN_ODE_EVAL_LABEL = "pretrain_ode_train_slice"
FINETUNE_ODE_EVAL_LABEL = "finetune_ode_finetune_slice"
FINETUNE_COMBINED_ODE_EVAL_LABEL = "finetune_ode_train_plus_finetune"
HOLDOUT_ODE_EVAL_SEED_OFFSET = 2999
HOLDOUT_ODE_EVAL_LABEL = "finetune_ode_holdout"
OOD_PHASE_SEED_OFFSETS: dict[str, int] = {"val": 2999, "test": 3999}
PRETRAIN_GRAD_PHASES = frozenset({"train"})
FINETUNE_GRAD_PHASES = frozenset({"finetune"})
FORBIDDEN_IN_PRETRAIN_GRAD = frozenset({"finetune", "val", "test", "hold_out"})
FORBIDDEN_IN_FINETUNE_GRAD = frozenset({"train", "val", "test", "hold_out"})


def assert_grad_safe_phases(
    stage: str,
    phases: tuple[str, ...],
    *,
    allowed: frozenset[str],
    forbidden_extra: frozenset[str],
) -> None:
    """Ensure split labels used for gradient updates are training-only."""
    leaked = set(phases) & forbidden_extra
    if leaked:
        raise ValueError(
            f"{stage}: must not train on {sorted(leaked)!r} (gradients forbidden)"
        )
    unknown = set(phases) - allowed
    if unknown:
        raise ValueError(
            f"{stage}: phase(s) {sorted(unknown)!r} not in allowed {sorted(allowed)!r}"
        )


def split_phases_present(h5ad: Path, split_col: str) -> frozenset[str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad, backed="r")
    return frozenset(adata.obs[split_col].astype(str).unique())


def resolve_eval_ood_phases(
    h5ad: Path,
    split_col: str,
    *,
    eval_ood_phases: str | None,
    no_eval_ood: bool,
    legacy_holdout_phase: str | None,
) -> tuple[str, ...]:
    """OOD eval phases (val/test): eval-only, never used for training."""
    if no_eval_ood:
        return ()
    out: list[str] = []
    if legacy_holdout_phase:
        out.extend(parse_phases(legacy_holdout_phase))

    def _append_auto() -> None:
        present = split_phases_present(h5ad, split_col)
        for label in ("val", "test"):
            if label in present:
                out.append(label)

    if eval_ood_phases is not None:
        spec = eval_ood_phases.strip().lower()
        if spec in ("", "none"):
            pass
        elif spec == "auto":
            _append_auto()
        else:
            out.extend(parse_phases(eval_ood_phases))
    else:
        _append_auto()
    return tuple(dict.fromkeys(p for p in out if p))


def resolve_cell_eval_phases(
    metrics: dict[str, Any],
    h5ad: Path,
    split_col: str,
    *,
    cli_eval_phases: str | None,
) -> tuple[str, ...]:
    """Phases to score in post-hoc cell-eval (no retraining).

    Priority: CLI ``--eval-phases`` (``auto`` = fit/forget + OOD phases) →
    ``eval_ood_phases`` (finetune) → auto-detect val/test in h5ad →
    ``eval_phases`` (saved run) → saved ``train_phase``.
    """
    if cli_eval_phases and str(cli_eval_phases).strip().lower() == "auto":
        return resolve_cell_eval_phases_auto(metrics, h5ad, split_col)
    if cli_eval_phases:
        return parse_phases(cli_eval_phases)
    ood = metrics.get("eval_ood_phases")
    if ood:
        return tuple(ood)
    auto = resolve_eval_ood_phases(
        h5ad,
        split_col,
        eval_ood_phases=None,
        no_eval_ood=False,
        legacy_holdout_phase=None,
    )
    if auto:
        return auto
    saved_eval = metrics.get("eval_phases")
    if saved_eval:
        return tuple(saved_eval)
    train_phase = str((metrics.get("args") or {}).get("train_phase") or "train")
    return (train_phase,)


def resolve_cell_eval_phases_auto(
    metrics: dict[str, Any],
    h5ad: Path,
    split_col: str,
) -> tuple[str, ...]:
    """Phases for bucket CellEval: ID strata for fit/forget + OOD val/test.

    - **Pretrain:** ``train`` (forget baseline on seen contexts).
    - **Finetune:** ``train`` (+ ``finetune`` when present) plus OOD ``val``/``test``.
    """
    present = split_phases_present(h5ad, split_col)
    stage = str(metrics.get("stage") or "finetune")
    out: list[str] = []
    train_phase = str((metrics.get("args") or {}).get("train_phase") or "train")

    if stage == "pretrain":
        if train_phase in present:
            out.append(train_phase)
    else:
        for phase in (train_phase, "finetune"):
            if phase in present and phase not in out:
                out.append(phase)
        ood = resolve_eval_ood_phases(
            h5ad,
            split_col,
            eval_ood_phases=None,
            no_eval_ood=False,
            legacy_holdout_phase=None,
        )
        for phase in ood:
            if phase not in out:
                out.append(phase)
        saved_ood = metrics.get("eval_ood_phases") or ()
        for phase in saved_ood:
            if phase in present and phase not in out:
                out.append(str(phase))

    if out:
        return tuple(out)
    return resolve_cell_eval_phases(metrics, h5ad, split_col, cli_eval_phases=None)


def cell_eval_phase_block_usable(
    existing: Any,
    *,
    profile: str,
    use_observed_controls_only: bool,
    use_raw_x: bool,
    export_protocol_min: int = 10,
    aggregate_covariate: str | None = None,
) -> bool:
    if not isinstance(existing, dict):
        return False
    if str(existing.get("profile") or "") != profile:
        return False
    if bool(existing.get("use_observed_controls_only", False)) != use_observed_controls_only:
        return False
    if bool(existing.get("use_raw_x", False)) != use_raw_x:
        return False
    if aggregate_covariate is not None:
        if (existing.get("aggregate_covariate") or None) != aggregate_covariate:
            return False
    return int(existing.get("export_protocol", 0) or 0) >= export_protocol_min


def cell_eval_run_is_complete(
    metrics: dict[str, Any],
    h5ad: Path,
    split_col: str,
    *,
    profile: str = "full",
    export_protocol_min: int = 10,
    use_observed_controls_only: bool = True,
    use_raw_x: bool = True,
    eval_phases: str = "auto",
    aggregate_covariate: str | None = None,
) -> bool:
    """True when every required ``cell_eval_{phase}`` block matches export settings."""
    mode = str(eval_phases or "auto").strip().lower()
    if mode == "auto":
        required = resolve_cell_eval_phases_auto(metrics, h5ad, split_col)
    elif eval_phases:
        required = parse_phases(eval_phases)
    else:
        required = resolve_cell_eval_phases(metrics, h5ad, split_col, cli_eval_phases=None)
    if not required:
        return False
    for phase in required:
        block = metrics.get(f"cell_eval_{phase}")
        if not cell_eval_phase_block_usable(
            block,
            profile=profile,
            use_observed_controls_only=use_observed_controls_only,
            use_raw_x=use_raw_x,
            export_protocol_min=export_protocol_min,
            aggregate_covariate=aggregate_covariate,
        ):
            return False
    return True


def ood_eval_seed(args: argparse.Namespace, phase: str) -> int:
    offset = OOD_PHASE_SEED_OFFSETS.get(phase, HOLDOUT_ODE_EVAL_SEED_OFFSET)
    return int(args.train_seed) + offset


def finetune_ode_label_for_ood_phase(phase: str) -> str:
    return f"finetune_ode_{phase}"


def ode_eval_seed(args: argparse.Namespace) -> int:
    """Base ODE eval seed (per-stratum noise uses ``stratum_seed`` on top of this)."""
    return int(args.train_seed) + ODE_EVAL_SEED_OFFSET


def resolve_context_label_indices(
    encoding: TxEncoding,
    context_spec: str | None,
) -> tuple[frozenset[int], tuple[str, ...]]:
    """Parse ``--held-out-context`` (one label or comma-separated) to row indices."""
    if not context_spec:
        return frozenset(), ()
    labels = parse_phases(context_spec)
    indices = frozenset(context_index(encoding, label) for label in labels)
    return indices, labels


def assert_tx_checkpoint_matches_encoding(
    model: ConditionalFlowMatching,
    encoding: TxEncoding,
    *,
    stage: str,
) -> None:
    """Require full context table (same layout as h5ad encoding)."""
    if model.n_contexts == encoding.n_contexts:
        return
    if model.n_contexts == encoding.n_contexts - 1:
        raise ValueError(
            f"Legacy {stage} checkpoint with n_contexts={model.n_contexts} "
            f"(encoding has {encoding.n_contexts}). Re-run pretrain with the current "
            "code (full context embeddings, no expand-at-finetune)."
        )
    raise ValueError(
        f"{stage} checkpoint n_contexts={model.n_contexts} does not match "
        f"encoding.n_contexts={encoding.n_contexts}"
    )


def pretrain_matching_ode_eval_kwargs(
    args: argparse.Namespace,
    *,
    skip_contexts: frozenset[int] | None,
) -> dict[str, Any]:
    """Return deterministic train-slice ODE evaluation options."""
    return {
        "eval_seed": ode_eval_seed(args),
        "skip_contexts": skip_contexts,
        "label": PRETRAIN_ODE_EVAL_LABEL,
    }


def merge_strata_eval_rows(
    *row_lists: list[dict[str, float | int]] | None,
) -> list[dict[str, float | int]]:
    merged: list[dict[str, float | int]] = []
    for rows in row_lists:
        if rows:
            merged.extend(rows)
    return merged


def mean_paired_mse_from_ode_rows(rows: list[dict[str, float | int]]) -> float:
    mses = [float(r["mse_to_true"]) for r in rows if "mse_to_true" in r]
    return float(np.mean(mses)) if mses else float("nan")


def release_eval_memory() -> None:
    """Return cached GPU/CPU memory after caller drops large eval/training tensors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ControlPoolKey = tuple[int, ...]


def missing_control_pool_keys_in_phase(
    c: torch.Tensor,
    p: torch.Tensor,
    control_pert: int,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
) -> frozenset[ControlPoolKey]:
    """Control pool keys (c [, *cov]) for observed strata lacking in-phase controls."""
    pools = control_indices_by_prefix(
        c,
        p,
        control_pert,
        z=z,
        covariate_col_order=covariate_col_order,
    )
    missing: set[ControlPoolKey] = set()
    for key in iter_observed_strata(
        c,
        p,
        z=z,
        covariate_col_order=covariate_col_order,
    ):
        pool_key = control_pool_key_for_stratum(key, covariate_col_order)
        if pool_key not in pools:
            missing.add(pool_key)
    return frozenset(missing)


def contexts_missing_in_phase_controls(
    c: torch.Tensor,
    p: torch.Tensor,
    control_pert: int,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
) -> frozenset[int]:
    """Context indices present in eval data but without in-phase control cells."""
    if covariate_col_order:
        missing_pools = missing_control_pool_keys_in_phase(
            c,
            p,
            control_pert,
            z=z,
            covariate_col_order=covariate_col_order,
        )
        return frozenset(pk[0] for pk in missing_pools)
    pools = control_indices_by_context(c, p, control_pert)
    contexts = {int(c[i].item()) for i in range(c.shape[0])}
    return frozenset(ci for ci in contexts if ci not in pools)


def eval_phase_for_control_reference(phases: tuple[str, ...]) -> str:
    """Best-effort eval split label for control-borrow candidate ordering."""
    for label in reversed(phases):
        if label in ("val", "test", "finetune"):
            return str(label)
    return str(phases[0]) if phases else "val"


def resolve_control_reference_phases(
    args: argparse.Namespace,
    *,
    eval_phase: str,
    missing_contexts: frozenset[int],
    present_phases: frozenset[str] | None = None,
) -> tuple[str, ...] | None:
    """Reference split label(s) for borrowing controls when eval phase has none."""
    if not missing_contexts:
        return None
    spec = str(getattr(args, "vcb_control_phases", "auto") or "auto").strip().lower()
    if spec in ("none", "off", "no", ""):
        return None

    finetune_phase = str(getattr(args, "finetune_phase", None) or "finetune")
    train_phase = str(getattr(args, "train_phase", None) or "train")
    eval_phase = str(eval_phase)

    if spec != "auto":
        candidates = parse_phases(spec)
    elif eval_phase in ("val", "test"):
        candidates = (finetune_phase, "test", train_phase)
    elif eval_phase in (finetune_phase, "finetune"):
        candidates = ("test", "val")
    else:
        candidates = (finetune_phase, "test", "val", train_phase)

    present = present_phases or frozenset()
    out: list[str] = []
    for phase in candidates:
        phase = str(phase).strip()
        if not phase or phase == eval_phase:
            continue
        if present and phase not in present:
            continue
        if phase not in out:
            out.append(phase)
    return tuple(out) if out else None


def resolve_vcb_control_reference_phases(
    args: argparse.Namespace,
    *,
    missing_contexts: frozenset[int],
    eval_phase: str = "val",
) -> tuple[str, ...] | None:
    """Reference split label(s) for VCB when eval phase has no in-phase controls."""
    return resolve_control_reference_phases(
        args,
        eval_phase=eval_phase,
        missing_contexts=missing_contexts,
    )


def _subsample_control_indices(
    idx: torch.Tensor,
    max_cells: int,
    seed: int,
) -> torch.Tensor:
    n = int(idx.numel())
    if max_cells <= 0 or n <= max_cells:
        return idx
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)[:max_cells]
    return idx[perm]


def load_vcb_control_basal_by_pool(
    h5ad: Path,
    args: argparse.Namespace,
    encoding: TxEncoding,
    reference_phases: tuple[str, ...],
    pool_keys: frozenset[ControlPoolKey],
    decoder: Any,
    control_pert: int,
) -> tuple[dict[ControlPoolKey, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Decode observed control cells from reference phase(s) for control pool keys."""
    from trajectory_replay.data.pca_decode import PcaGeneDecoder

    if not isinstance(decoder, PcaGeneDecoder):
        raise TypeError(f"decoder must be PcaGeneDecoder, got {type(decoder)!r}")

    cov_cols = encoding.covariate_col_order or covariate_cols_from_args(args)
    max_cells = int(getattr(args, "vcb_control_max_cells", 256) or 0)
    ref_bundle = load_tx_data(
        h5ad,
        split_col=args.split_col,
        phases=reference_phases,
        context_col=args.context_col,
        perturbation_col=args.perturbation_col,
        covariate_cols=cov_cols,
        obsm_key=args.obsm_key,
        encoding=encoding,
        only_control_cells=True,
        max_samples_per_stratum=max_cells if max_cells > 0 else None,
        subsample_seed=int(args.subsample_seed) + 91_001,
    )
    pools = control_indices_by_prefix(
        ref_bundle.c,
        ref_bundle.p,
        control_pert,
        z=ref_bundle.z or None,
        covariate_col_order=cov_cols,
    )
    base_seed = int(args.subsample_seed) + 91_002

    out: dict[ControlPoolKey, tuple[np.ndarray, np.ndarray]] = {}
    n_cells_used: dict[str, int] = {}
    still_missing: list[ControlPoolKey] = []

    try:
        for pool_key in sorted(pool_keys):
            if pool_key not in pools:
                still_missing.append(pool_key)
                continue
            idx = _subsample_control_indices(
                pools[pool_key],
                max_cells,
                seed=stratum_seed(base_seed, *pool_key),
            )
            x_np = ref_bundle.x[idx].cpu().numpy()
            y_decoded = decoder.decode(x_np)
            out[pool_key] = (y_decoded.mean(axis=0), y_decoded)
            n_cells_used[str(pool_key)] = int(y_decoded.shape[0])
            del x_np
    finally:
        del ref_bundle

    meta: dict[str, Any] = {
        "reference_phases": list(reference_phases),
        "pool_keys": [list(pk) for pk in sorted(pool_keys)],
        "context_indices": sorted({pk[0] for pk in pool_keys}),
        "max_cells_per_context": max_cells,
        "n_control_cells": n_cells_used,
    }
    if still_missing:
        meta["pools_without_reference_controls"] = [list(pk) for pk in still_missing]
    return out, meta


def load_vcb_control_basal_by_context(
    h5ad: Path,
    args: argparse.Namespace,
    encoding: TxEncoding,
    reference_phases: tuple[str, ...],
    context_indices: frozenset[int],
    decoder: Any,
    control_pert: int,
) -> tuple[dict[ControlPoolKey, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Decode reference controls for contexts (``(ci,)`` pool keys; no covariates)."""
    cov_cols = encoding.covariate_col_order or covariate_cols_from_args(args)
    if cov_cols:
        raise ValueError(
            "load_vcb_control_basal_by_context is for encodings without covariates; "
            "use load_vcb_control_basal_by_pool"
        )
    pool_keys = frozenset((ci,) for ci in context_indices)
    return load_vcb_control_basal_by_pool(
        h5ad,
        args,
        encoding,
        reference_phases,
        pool_keys,
        decoder,
        control_pert,
    )


def run_ode_eval(
    model: nn.Module,
    h5ad: Path,
    args: argparse.Namespace,
    encoding: TxEncoding,
    phases: tuple[str, ...],
    *,
    eval_seed: int,
    skip_contexts: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
    label: str = "eval",
    decoder: Any | None = None,
) -> tuple[
    list[dict[str, float | int]],
    float,
    list[dict[str, float | int]] | None,
    dict[str, Any] | None,
]:
    load_kw = eval_load_kwargs(args, label=label)
    cov_cols = encoding.covariate_col_order or covariate_cols_from_args(args)
    bundle = load_tx_data(
        h5ad,
        split_col=args.split_col,
        phases=phases,
        context_col=args.context_col,
        perturbation_col=args.perturbation_col,
        covariate_cols=cov_cols,
        obsm_key=args.obsm_key,
        encoding=encoding,
        **load_kw,
    )
    n_cells = bundle.n_cells
    device = resolve_device(args.device)
    ode_kw = dict(
        n_contexts=encoding.n_contexts,
        n_perturbations=encoding.n_perturbations,
        device=device,
        eval_seed=eval_seed,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options_from_args(args),
        skip_contexts=skip_contexts,
        skip_combos=skip_combos,
        z=bundle.z or None,
        covariate_col_order=encoding.covariate_col_order,
        encoding=encoding,
    )

    vcb_rows: list[dict[str, float | int]] | None = None
    vcb_control_ref: dict[str, Any] | None = None
    owned_decoder: Any | None = None
    per_combo: list[dict[str, float | int]]
    mean_mse: float
    try:
        if getattr(args, "vcb_eval", False):
            raise ValueError("VCB evaluation was removed with the retired TX pipeline")
        per_combo, mean_mse = eval_ode_strata(
            model,
            bundle.x,
            bundle.c,
            bundle.p,
            **ode_kw,
        )
    finally:
        del bundle
        if owned_decoder is not None:
            del owned_decoder
        release_eval_memory()

    print(f"{label}: phases={phases} n_cells={n_cells} mean_mse={mean_mse:.6f}")
    if vcb_rows is not None:
        n_retrieval = sum(
            1
            for r in vcb_rows
            if "retrieval_mae" in r
            and np.isfinite(float(r["retrieval_mae"]))
        )
        n_retrieval_ctx = len(
            {
                int(r["c"])
                for r in vcb_rows
                if "retrieval_mae" in r
                and np.isfinite(float(r["retrieval_mae"]))
            }
        )
        print(
            f"{label}: vcb strata n={len(vcb_rows)} "
            f"retrieval_finite_rows={n_retrieval} retrieval_contexts={n_retrieval_ctx}"
        )
    return per_combo, mean_mse, vcb_rows, vcb_control_ref


def method_display_label(label: str) -> str:
    """User-facing method name for plots and summary tables.

    Manifest keys stay short (``replay_r25``); display names mirror ``real_replay_*``::
    ``generative_replay_r25``, ``generative_field_replay_r25``.
    """
    if label.startswith(("generative_", "real_replay", "pretrain")) or label.startswith("vf"):
        return label
    if label.startswith("field_replay"):
        return f"generative_{label}"
    if label.startswith("replay_"):
        return f"generative_{label}"
    return label


def resolve_checkpoint(path: Path) -> Path:
    """Resolve ``checkpoint.pt`` from a run directory or file path."""
    if path.is_dir():
        ckpt = path / "checkpoint.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"No checkpoint.pt in {path}")
        return ckpt
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_flow_map_checkpoint_path(
    repo_root: Path,
    pretrain_ckpt: Path,
    flow_map_arg: Path | None,
) -> Path:
    """Resolve ``flow_map.pt`` from an explicit path or the pretrain run directory."""
    from trajectory_replay.training.flow_map_distill import resolve_flow_map_checkpoint

    if flow_map_arg is not None:
        resolved = resolve_tx_path(repo_root, str(flow_map_arg))
        if resolved.is_dir():
            return resolve_flow_map_checkpoint(resolved)
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(resolved)
    return resolve_flow_map_checkpoint(pretrain_ckpt.parent)


def resolve_mean_flow_checkpoint_path(
    repo_root: Path,
    pretrain_ckpt: Path,
    mean_flow_arg: Path | None,
) -> Path:
    """Resolve ``mean_flow.pt`` from an explicit path or the pretrain run directory."""
    from trajectory_replay.training.mean_flow_distill import resolve_mean_flow_checkpoint

    if mean_flow_arg is not None:
        resolved = resolve_tx_path(repo_root, str(mean_flow_arg))
        if resolved.is_dir():
            return resolve_mean_flow_checkpoint(resolved)
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(resolved)
    return resolve_mean_flow_checkpoint(pretrain_ckpt.parent)
