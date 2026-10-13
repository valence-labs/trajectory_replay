#!/usr/bin/env python3
"""Post-hoc Cell-Eval on a saved tx run (no retraining).

Builds gene-space ``adata_real`` / ``adata_pred`` per split phase and context,
runs ``cell_eval.MetricsEvaluator``, and merges summaries into ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
from experiments.tx.cell_eval_export import (
    CELL_EVAL_EXPORT_PROTOCOL,
    CELL_EVAL_GENE_MAX,
    build_cell_eval_pair,
    context_indices_in_adata,
    contexts_missing_in_phase_adata,
    default_aggregate_covariate,
    load_control_reference_adata,
    load_phase_adata,
    merge_control_reference_adata,
    export_slice_has_controls,
)
from experiments.tx.cell_eval_metrics import DEFAULT_CELL_EVAL_PROFILE
from trajectory_replay.data.pca_decode import load_pca_gene_decoder
from trajectory_replay.data.tx_strata import iter_aggregate_covariate_slices
from trajectory_replay.data.tx_dataset import parse_phases
from trajectory_replay.io.results import dump_metrics, resolve_tx_path, tx_path_rel

from experiments.tx.tx_common import (
    add_data_args,
    add_ode_args,
    load_checkpoint,
    ode_options_from_args,
    ood_eval_seed,
    resolve_cell_eval_phases,
    resolve_control_reference_phases,
    resolve_device,
    resolve_h5ad,
    split_phases_present,
)


def _import_cell_eval():
    try:
        from cell_eval import MetricsEvaluator
    except ImportError as e:
        raise ImportError(
            "Cell-Eval requires the cell-eval extra. Install with: uv sync --extra cell-eval"
        ) from e
    return MetricsEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_ode_args(p)
    p.add_argument(
        "run_dir",
        type=Path,
        help="Run directory with checkpoint.pt and metrics.json.",
    )
    p.add_argument(
        "--eval-phases",
        default=None,
        help=(
            "Comma-separated split phases (default: eval_ood_phases / eval_phases "
            "from metrics.json, else auto-detect val+test in h5ad)."
        ),
    )
    p.add_argument(
        "--context",
        default=None,
        help="Comma-separated context labels to score (default: all present in phase).",
    )
    p.add_argument(
        "--pert-col",
        default=None,
        help="obs column for Cell-Eval perturbation labels (default: --perturbation-col).",
    )
    p.add_argument(
        "--profile",
        default=DEFAULT_CELL_EVAL_PROFILE,
        choices=("full", "vcc", "minimal", "de", "anndata"),
        help=f"Cell-Eval metric profile (default {DEFAULT_CELL_EVAL_PROFILE}).",
    )
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument(
        "--out-subdir",
        default="cell_eval",
        help="Subdirectory under run_dir for Cell-Eval outputs (default cell_eval).",
    )
    p.add_argument(
        "--save-h5ad",
        action="store_true",
        help="Write adata_real/pred.h5ad under each context outdir.",
    )
    p.add_argument(
        "--no-metrics-update",
        action="store_true",
        help="Skip patching metrics.json.",
    )
    p.add_argument(
        "--use-observed-controls-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy observed control expression into adata_pred for control strata "
            "(STATE-style; default: enabled)."
        ),
    )
    p.add_argument(
        "--use-raw-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use stored adata.X for observed/real gene matrices instead of PCA decode "
            "(model predictions are still ODE + decode; default: enabled)."
        ),
    )
    p.add_argument(
        "--aggregate-covariate",
        default=None,
        help=(
            "obs column to pool contexts/donors within for Cell-Eval export "
            "(default: cell_type for PBMC, cell_line for Replogle). "
            "Use 'none' for legacy donor×covariate slices."
        ),
    )
    p.add_argument(
        "--skip-existing-phases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep cell_eval_{phase} blocks already in metrics.json (same profile/protocol).",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _phase_block_usable(
    existing: Any,
    *,
    profile: str,
    use_observed_controls_only: bool,
    use_raw_x: bool,
    aggregate_covariate: str | None,
) -> bool:
    if not isinstance(existing, dict):
        return False
    if str(existing.get("profile") or "") != profile:
        return False
    if bool(existing.get("use_observed_controls_only", False)) != use_observed_controls_only:
        return False
    if bool(existing.get("use_raw_x", False)) != use_raw_x:
        return False
    if (existing.get("aggregate_covariate") or None) != aggregate_covariate:
        return False
    return int(existing.get("export_protocol", 0) or 0) >= CELL_EVAL_EXPORT_PROTOCOL


def _resolve_aggregate_covariate(
    cli_val: str | None,
    *,
    context_col: str,
    covariate_col_order: tuple[str, ...],
) -> str | None:
    if cli_val is not None:
        raw = str(cli_val).strip()
        if raw.lower() in ("none", "null", ""):
            return None
        return raw
    return default_aggregate_covariate(
        context_col=context_col,
        covariate_col_order=covariate_col_order,
    )


def _args_from_metrics(metrics: dict[str, Any]) -> argparse.Namespace:
    margs = metrics.get("args") or {}
    return argparse.Namespace(**margs)


def _merge_run_args(cli: argparse.Namespace, metrics: dict[str, Any]) -> argparse.Namespace:
    """Merge saved run args with CLI; metrics win for training/data, CLI for cell-eval knobs."""
    base = _args_from_metrics(metrics)
    merged = vars(base).copy()
    cli_override_keys = frozenset(
        {
            "context",
            "profile",
            "out_subdir",
            "num_threads",
            "save_h5ad",
            "no_metrics_update",
            "device",
            "eval_phases",
            "pert_col",
            "use_observed_controls_only",
            "use_raw_x",
            "skip_existing_phases",
            "aggregate_covariate",
        }
    )
    for key, val in vars(cli).items():
        if key == "run_dir":
            continue
        if key in cli_override_keys:
            if val is not None and val != "":
                merged[key] = val
            else:
                merged.setdefault(key, val)
        elif key not in merged and val is not None and val != "":
            merged[key] = val
    if merged.get("h5ad_path") is not None:
        merged["h5ad_path"] = Path(merged["h5ad_path"])
    if merged.get("pert_col") is None:
        merged["pert_col"] = merged.get("perturbation_col", "target")
    if merged.get("use_observed_controls_only") is None:
        merged["use_observed_controls_only"] = True
    if merged.get("use_raw_x") is None:
        merged["use_raw_x"] = True
    return argparse.Namespace(**merged)


def _agg_to_jsonable(agg) -> list[dict[str, Any]]:
    return [{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()} for row in agg.to_dicts()]


def _pert_results_to_jsonable(results) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results.to_dicts():
        out: dict[str, Any] = {}
        for key, val in row.items():
            if key == "perturbation":
                out[key] = str(val)
            elif isinstance(val, (int, float)):
                out[key] = float(val)
            else:
                out[key] = val
        rows.append(out)
    return rows


def main() -> None:
    cli = parse_args()
    MetricsEvaluator = _import_cell_eval()

    run_dir = resolve_tx_path(_REPO_ROOT, str(cli.run_dir))
    ckpt = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.json"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    args = _merge_run_args(cli, metrics)
    h5ad = resolve_h5ad(args)
    phases = resolve_cell_eval_phases(
        metrics,
        h5ad,
        args.split_col,
        cli_eval_phases=cli.eval_phases,
    )
    context_filter = parse_phases(args.context) if args.context else None

    device = resolve_device(args.device)
    model, _cfg, encoding = load_checkpoint(ckpt, device)
    model.eval()

    aggregate_covariate = _resolve_aggregate_covariate(
        cli.aggregate_covariate,
        context_col=args.context_col,
        covariate_col_order=encoding.covariate_col_order,
    )
    cov_order = encoding.covariate_col_order
    pool_by_covariate = (
        aggregate_covariate is not None
        and aggregate_covariate in cov_order
    )
    print(
        f"cell-eval aggregate_covariate={aggregate_covariate!r} "
        f"pool_contexts={pool_by_covariate}"
    )

    decoder = load_pca_gene_decoder(h5ad, obsm_key=args.obsm_key)
    control_pert = args.control_perturbation
    if encoding.control_perturbation:
        control_pert = encoding.control_perturbation

    per_stratum = int(getattr(args, "eval_max_samples_per_stratum", 0) or 0)

    ode_kw = dict(
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options_from_args(args),
    )

    out_root = run_dir / args.out_subdir
    cell_eval_metrics: dict[str, Any] = {}
    present_phases = split_phases_present(h5ad, args.split_col)
    control_pert_idx = encoding.control_pert_index
    if control_pert is not None and control_pert in encoding.perturbation_to_idx:
        control_pert_idx = encoding.perturbation_to_idx[control_pert]

    for phase in phases:
        existing = metrics.get(f"cell_eval_{phase}")
        if cli.skip_existing_phases and _phase_block_usable(
            existing,
            profile=args.profile,
            use_observed_controls_only=bool(args.use_observed_controls_only),
            use_raw_x=bool(args.use_raw_x),
            aggregate_covariate=aggregate_covariate,
        ):
            cell_eval_metrics[f"cell_eval_{phase}"] = existing
            print(f"cell-eval {phase}: skip existing block in metrics.json")
            continue
        adata_phase = load_phase_adata(h5ad, split_col=args.split_col, phase=phase)
        reference_control_adata = None
        if control_pert_idx is not None:
            missing_ctx = contexts_missing_in_phase_adata(
                adata_phase,
                encoding,
                control_pert_idx,
                context_col=args.context_col,
                perturbation_col=args.perturbation_col,
                context_labels=context_filter,
                aggregate_covariate=aggregate_covariate,
            )
            if missing_ctx:
                ref_phases = resolve_control_reference_phases(
                    args,
                    eval_phase=phase,
                    missing_contexts=missing_ctx,
                    present_phases=present_phases,
                )
                if ref_phases:
                    reference_control_adata = load_control_reference_adata(
                        h5ad,
                        split_col=args.split_col,
                        reference_phases=ref_phases,
                    )
                    if reference_control_adata is not None:
                        print(
                            f"cell-eval {phase}: control reference phases={ref_phases} "
                            f"contexts={sorted(missing_ctx)}"
                        )
        control_phase_adata = merge_control_reference_adata(
            adata_phase,
            reference_control_adata,
        )
        ctx_indices = context_indices_in_adata(
            adata_phase,
            encoding,
            args.context_col,
            context_labels=context_filter,
        )
        if aggregate_covariate is not None:
            export_slices = iter_aggregate_covariate_slices(
                adata_phase,
                encoding,
                aggregate_covariate=aggregate_covariate,
                context_col=args.context_col,
                context_labels=context_filter or encoding.context_categories,
            )
        else:
            from trajectory_replay.data.tx_strata import iter_export_slices

            export_slices = iter_export_slices(
                adata_phase,
                encoding,
                context_col=args.context_col,
                covariate_col_order=encoding.covariate_col_order,
                context_labels=context_filter or encoding.context_categories,
            )
            if not export_slices and ctx_indices:
                export_slices = [
                    (ci, {}, encoding.context_categories[ci]) for ci in ctx_indices
                ]
        eval_seed = ood_eval_seed(args, phase)
        phase_out: dict[str, Any] = {
            "profile": args.profile,
            "export_protocol": CELL_EVAL_EXPORT_PROTOCOL,
            "aggregate_covariate": aggregate_covariate,
            "gene_max_cap": CELL_EVAL_GENE_MAX,
            "use_observed_controls_only": bool(args.use_observed_controls_only),
            "use_raw_x": bool(args.use_raw_x),
            "eval_seed": eval_seed,
            "contexts": {},
        }

        for ci, cov_map, slice_suffix in export_slices:
            pool_contexts = pool_by_covariate
            if pool_contexts:
                context_label = slice_suffix
            else:
                context_label = encoding.context_categories[int(ci)]  # type: ignore[arg-type]
            if control_pert and not export_slice_has_controls(
                control_phase_adata,
                context_label=context_label,
                control_perturbation=control_pert,
                context_col=args.context_col,
                perturbation_col=args.perturbation_col,
                encoding=encoding,
                covariate_indices=cov_map or None,
                pool_contexts=pool_contexts,
            ):
                print(
                    f"cell-eval {phase} {slice_suffix}: skip (no in-slice {control_pert!r}; "
                    "STATE/cell-eval requires same aggregation category)"
                )
                continue
            pair = build_cell_eval_pair(
                model,
                adata_phase,
                encoding=encoding,
                context_idx=ci,
                covariate_indices=cov_map or None,
                decoder=decoder,
                device=device,
                eval_seed=eval_seed,
                context_col=args.context_col,
                perturbation_col=args.perturbation_col,
                pert_col=args.pert_col,
                max_per_stratum=per_stratum,
                subsample_seed=int(args.subsample_seed),
                obsm_key=args.obsm_key,
                control_perturbation=control_pert,
                control_phase_adata=control_phase_adata,
                use_observed_controls_only=bool(args.use_observed_controls_only),
                use_raw_x=bool(args.use_raw_x),
                aggregate_covariate=aggregate_covariate,
                pool_contexts=pool_contexts,
                **ode_kw,
            )
            ctx_outdir = out_root / phase / pair.slice_label
            ctx_outdir.mkdir(parents=True, exist_ok=True)

            if cli.save_h5ad:
                pair.adata_real.write_h5ad(ctx_outdir / "adata_real.h5ad")
                pair.adata_pred.write_h5ad(ctx_outdir / "adata_pred.h5ad")

            evaluator = MetricsEvaluator(
                adata_pred=pair.adata_pred,
                adata_real=pair.adata_real,
                control_pert=control_pert,
                pert_col=args.pert_col,
                num_threads=int(args.num_threads),
                outdir=str(ctx_outdir),
            )
            _results, agg = evaluator.compute(profile=args.profile, write_csv=True)

            phase_out["contexts"][pair.slice_label] = {
                "outdir": tx_path_rel(_REPO_ROOT, ctx_outdir),
                "context_label": pair.context_label,
                "aggregate_covariate": aggregate_covariate,
                "pool_contexts": pool_contexts,
                "n_cells_real": int(pair.adata_real.n_obs),
                "n_cells_pred": int(pair.adata_pred.n_obs),
                "n_strata": pair.n_strata,
                "perturbation_results": _pert_results_to_jsonable(_results),
                "agg_results": _agg_to_jsonable(agg),
            }
            print(
                f"cell-eval {phase} {pair.slice_label}: "
                f"n_real={pair.adata_real.n_obs} n_pred={pair.adata_pred.n_obs} "
                f"-> {ctx_outdir}"
            )

        cell_eval_metrics[f"cell_eval_{phase}"] = phase_out

    if not cli.no_metrics_update:
        metrics.update(cell_eval_metrics)
        dump_metrics(metrics_path, metrics)
        print(f"updated {metrics_path}")

    print(f"done phases={phases}")


if __name__ == "__main__":
    main()
