"""Cell-Eval metric profile constants shared by run/collect/render/plot scripts."""

from __future__ import annotations

from typing import Any

DEFAULT_CELL_EVAL_PROFILE = "full"

# STATE paper table metrics (subset of Cell-Eval ``full`` profile).
CELL_EVAL_METRICS: tuple[str, ...] = (
    "pearson_delta",
    "overlap_at_N",
    "de_direction_match",
    "de_spearman_lfc_sig",
    "clustering_agreement",
    "mae",
)

CELL_EVAL_LOWER_IS_BETTER: frozenset[str] = frozenset({"mae"})

CELL_EVAL_NO_BEST: frozenset[str] = frozenset()

OOD_PHASES: tuple[str, ...] = ("val", "test")

# Legacy suffix on old summary CSVs (macro-mean over context×perturbation pairs).
PER_CONDITION_SUFFIX = "_per_condition"

OOD_CSV_META_COLUMNS: frozenset[str] = frozenset(
    {
        "label",
        "held_out_context",
        "n_strata_val",
        "n_strata_test",
    }
)

# Present in legacy CSVs but omitted from tables (long paths hurt readability).
OOD_CSV_HIDDEN_COLUMNS: frozenset[str] = frozenset({"finetune_path", "pretrain_path"})


def cell_eval_fitting_phases(metrics: dict[str, Any]) -> tuple[str, ...]:
    """Phases for held-out fitting on finetune-set perturbations.

    Uses the ``finetune`` Cell-Eval block when present (FT context, FT perts only).
    Falls back to ``train`` when no finetune block exists. Never uses ``val`` / ``test``
    (those belong in OOD fitting).
    """
    if metrics.get("cell_eval_finetune"):
        return ("finetune",)
    if metrics.get("cell_eval_train"):
        return ("train",)
    return ("finetune",)


def cell_eval_fitting_higher_is_better(col: str) -> bool | None:
    if col in OOD_CSV_META_COLUMNS or col == "n_strata_fitting":
        return None
    if col.startswith("fitting_"):
        return cell_eval_higher_is_better(col[len("fitting_") :])
    return None


def cell_eval_forgetting_higher_is_better(col: str) -> bool | None:
    if col in OOD_CSV_META_COLUMNS or col == "n_strata_forget":
        return None
    if col.startswith("forget_"):
        metric = col[len("forget_") :]
        if metric in CELL_EVAL_LOWER_IS_BETTER:
            # Delta on error metrics: less increase is better.
            return False
        if metric in CELL_EVAL_NO_BEST:
            return None
        return True
    return None


def cell_eval_higher_is_better(metric: str) -> bool | None:
    if metric in CELL_EVAL_NO_BEST:
        return None
    return metric not in CELL_EVAL_LOWER_IS_BETTER


def ood_csv_fieldnames() -> list[str]:
    """Canonical OOD summary columns (STATE-style macro-mean over conditions)."""
    base = [
        "label",
        "held_out_context",
        "n_strata_val",
        "n_strata_test",
    ]
    metric_cols: list[str] = []
    for phase in OOD_PHASES:
        for metric in CELL_EVAL_METRICS:
            metric_cols.append(f"{phase}_{metric}")
    return base + metric_cols


def _metric_display(metric: str) -> str:
    labels = {
        "pearson_delta": "Pearson Δ",
        "overlap_at_N": "Overlap@N",
        "de_direction_match": "DE direction match",
        "de_spearman_lfc_sig": "DE Spearman LFC",
        "clustering_agreement": "Centroid accuracy",
        "mae": "MAE",
    }
    return labels.get(metric, metric.replace("_", " "))


def celleval_ood_column_header(col: str) -> str:
    if col == "label":
        return "Method"
    if col == "held_out_context":
        return "Held-out"
    if col == "finetune_path":
        return "Run"
    if col == "n_strata_val":
        return "N strata (val)"
    if col == "n_strata_test":
        return "N strata (test)"
    for phase in OOD_PHASES:
        prefix = f"{phase}_"
        if col.startswith(prefix):
            metric_part = col[len(prefix) :]
            if metric_part.endswith(PER_CONDITION_SUFFIX):
                metric_part = metric_part[: -len(PER_CONDITION_SUFFIX)]
            return f"{phase.capitalize()} {_metric_display(metric_part)}"
    return col.replace("_", " ")


def celleval_ood_higher_is_better(col: str) -> bool | None:
    if col in OOD_CSV_META_COLUMNS:
        return None
    for phase in OOD_PHASES:
        prefix = f"{phase}_"
        if col.startswith(prefix):
            metric_part = col[len(prefix) :]
            if metric_part.endswith(PER_CONDITION_SUFFIX):
                metric_part = metric_part[: -len(PER_CONDITION_SUFFIX)]
            return cell_eval_higher_is_better(metric_part)
    return None


def plot_metric_specs_from_columns(columns: list[str]) -> list[tuple[str, str, bool]]:
    """Build bar-plot specs: test metrics first, then val, for each metric name."""
    specs: list[tuple[str, str, bool]] = []
    for metric in CELL_EVAL_METRICS:
        for phase in ("test", "val"):
            col = f"{phase}_{metric}"
            legacy = f"{phase}_{metric}{PER_CONDITION_SUFFIX}"
            if col in columns:
                pick = col
            elif legacy in columns:
                pick = legacy
            else:
                continue
            higher = cell_eval_higher_is_better(metric)
            title = f"{phase.capitalize()} {_metric_display(metric)} (held-out)"
            specs.append((pick, title, higher if higher is not None else True))
    return specs
