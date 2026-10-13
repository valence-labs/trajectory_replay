"""Shared helpers for fit/forget and OOD metric summaries."""

from __future__ import annotations

from typing import Any


def median_or_nan(values: list[float]) -> float:
    import numpy as np

    if not values:
        return float("nan")
    return float(np.median(values))


def mean_or_nan(values: list[float]) -> float:
    import numpy as np

    if not values:
        return float("nan")
    return float(np.mean(values))


CELL_EVAL_SLICE_SEP = "__"


def covariate_col_order_from_metrics(metrics: dict[str, Any]) -> tuple[str, ...]:
    enc = metrics.get("encoding") or {}
    return tuple(enc.get("covariate_col_order") or ())


def stratum_key_from_row(
    row: dict[str, Any],
    covariate_col_order: tuple[str, ...] = (),
) -> tuple[int, ...]:
    """Full stratum identity ``(c, p[, z...])`` for pairing pre/post metrics."""
    key: tuple[int, ...] = (int(row["c"]), int(row["p"]))
    for col in covariate_col_order:
        key = key + (int(row[f"z_{col}"]),)
    return key


def strata_map(
    rows: list[dict[str, Any]],
    metric: str,
    covariate_col_order: tuple[str, ...] = (),
) -> dict[tuple[int, ...], float]:
    out: dict[tuple[int, ...], float] = {}
    for r in rows:
        if metric not in r:
            continue
        val = float(r[metric])
        if not _finite(val):
            continue
        out[stratum_key_from_row(r, covariate_col_order)] = val
    return out


def cell_eval_slice_matches_context(slice_label: str, context_label: str) -> bool:
    """True when ``slice_label`` is a cell-eval export slice for ``context_label``."""
    if slice_label == context_label:
        return True
    prefix = f"{context_label}{CELL_EVAL_SLICE_SEP}"
    return slice_label.startswith(prefix)


def cell_eval_slice_labels_for_contexts(
    contexts: dict[str, Any],
    context_labels: list[str],
) -> list[str]:
    """All export slice keys belonging to any of ``context_labels``."""
    labels: list[str] = []
    seen: set[str] = set()
    for held in context_labels:
        for key in contexts:
            if not cell_eval_slice_matches_context(key, held):
                continue
            if key in seen:
                continue
            seen.add(key)
            labels.append(key)
    return sorted(labels)


def cell_eval_slice_labels_for_summary(
    metrics: dict[str, Any],
    phase: str,
    context_labels: list[str],
) -> list[str]:
    """Export slice keys to include when macro-averaging Cell-Eval perturbation rows."""
    phase_out = metrics.get(f"cell_eval_{phase}") or {}
    contexts = phase_out.get("contexts") or {}
    if phase_out.get("aggregate_covariate"):
        return sorted(contexts.keys())
    return cell_eval_slice_labels_for_contexts(contexts, context_labels)


def cell_eval_forgetting_slice_labels(
    metrics: dict[str, Any],
    phase: str,
    held_context_labels: frozenset[str],
) -> list[str]:
    """Slice keys for forgetting deltas (exclude held-out contexts when legacy slices)."""
    phase_out = metrics.get(f"cell_eval_{phase}") or {}
    contexts = phase_out.get("contexts") or {}
    if phase_out.get("aggregate_covariate"):
        return sorted(contexts.keys())
    return [
        label
        for label in sorted(contexts.keys())
        if not cell_eval_slice_is_held_out(label, held_context_labels)
    ]


def cell_eval_slice_is_held_out(slice_label: str, held_context_labels: frozenset[str]) -> bool:
    return any(
        cell_eval_slice_matches_context(slice_label, held)
        for held in held_context_labels
    )


def context_metric_map(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in rows:
        if metric not in r:
            continue
        val = float(r[metric])
        if not _finite(val):
            continue
        out[int(r["c"])] = val
    return out


def _finite(x: float) -> bool:
    import numpy as np

    return bool(np.isfinite(x))


def held_out_context_indices(metrics: dict[str, Any]) -> frozenset[int]:
    raw = metrics.get("held_out_context_indices")
    if raw:
        return frozenset(int(i) for i in raw)
    idx = metrics.get("held_out_context_idx")
    if idx is not None:
        return frozenset({int(idx)})
    enc = metrics.get("encoding") or {}
    ctx_to_idx = enc.get("context_to_idx") or {}
    held = metrics.get("held_out_context")
    if held and ctx_to_idx:
        from trajectory_replay.data.tx_dataset import parse_phases

        return frozenset(int(ctx_to_idx[lab]) for lab in parse_phases(str(held)))
    args = metrics.get("args") or {}
    held = args.get("held_out_context")
    if held and ctx_to_idx:
        from trajectory_replay.data.tx_dataset import parse_phases

        return frozenset(int(ctx_to_idx[lab]) for lab in parse_phases(str(held)))
    raise KeyError("held_out_context_indices missing in finetune metrics")


def held_out_stratum_values(
    rows: list[dict[str, Any]],
    metric: str,
    held_out: frozenset[int],
) -> list[float]:
    vals = [float(r[metric]) for r in rows if metric in r and int(r["c"]) in held_out]
    return [v for v in vals if _finite(v)]


def median_on_held_out_strata(
    rows: list[dict[str, Any]],
    metric: str,
    held_out: frozenset[int],
) -> float:
    return median_or_nan(held_out_stratum_values(rows, metric, held_out))


def mean_on_held_out_strata(
    rows: list[dict[str, Any]],
    metric: str,
    held_out: frozenset[int],
) -> float:
    return mean_or_nan(held_out_stratum_values(rows, metric, held_out))


def median_retrieval_on_held_out(
    rows: list[dict[str, Any]],
    held_out: frozenset[int],
) -> float:
    cmap = context_metric_map(rows, "retrieval_mae")
    vals = [cmap[c] for c in held_out if c in cmap]
    return median_or_nan(vals)


def mean_retrieval_on_held_out(
    rows: list[dict[str, Any]],
    held_out: frozenset[int],
) -> float:
    cmap = context_metric_map(rows, "retrieval_mae")
    vals = [cmap[c] for c in held_out if c in cmap]
    return mean_or_nan(vals)


VAL_OOD_COUNT_FIELDS: frozenset[str] = frozenset({"n_strata_val"})


def val_phase_absent(*counts: int) -> bool:
    """True when every supplied val stratum/context count is zero."""
    return all(int(c or 0) == 0 for c in counts)


def val_phase_absent_from_rows(rows: list[Any], attr: str = "n_strata_val") -> bool:
    return val_phase_absent(*(int(getattr(r, attr, 0) or 0) for r in rows))


def filter_val_ood_fieldnames(fieldnames: list[str], *, drop_val: bool) -> list[str]:
    if not drop_val:
        return list(fieldnames)
    return [
        f
        for f in fieldnames
        if f not in VAL_OOD_COUNT_FIELDS and not f.startswith("val_")
    ]


def fitting_eval_keys(_metrics: dict[str, Any]) -> tuple[str, str]:
    """Return ``(vcb_key, ode_key)`` for held-out fitting metrics.

    Always uses in-distribution finetune eval (``train`` + ``finetune`` phases on
    the     held-out context). OOD ``val`` / ``test`` blocks are for OOD fitting only.
    """
    return "vcb_eval_train_finetune", "ode_eval_train_finetune"


# Columns kept in metrics CSVs for provenance but omitted from table figures.
TABLE_HIDDEN_COLUMNS: frozenset[str] = frozenset(
    {"finetune_path", "pretrain_path"}
)

TABLE_META_COLUMNS: frozenset[str] = frozenset(
    {
        "label",
        "held_out_context",
        "held_out_context_idx",
    }
    | TABLE_HIDDEN_COLUMNS
    | VAL_OOD_COUNT_FIELDS
)


def is_table_score_column(col: str) -> bool:
    if col in TABLE_META_COLUMNS:
        return False
    if col.startswith("n_"):
        return False
    if col.startswith("rank_"):
        return False
    return True


def metric_higher_is_better(col: str) -> bool | None:
    """Whether a larger value is better for highlighting / ranks.

    ``retrieval_mae`` is a higher-is-better score despite the ``_mae`` suffix.
    ``forget_*`` deltas are higher-is-better (less forgetting).
    """
    if not is_table_score_column(col):
        return None
    if "de_nsig" in col:
        return None
    if "retrieval" in col:
        return True
    if col.endswith("_pca_energy") or col.endswith("_energy"):
        return False
    if "_mse" in col:
        return False
    if col.endswith("_mae") and not col.startswith("forget_"):
        return False
    return True


def filter_table_display_columns(columns: list[str]) -> list[str]:
    """Drop provenance paths from rendered tables."""
    return [c for c in columns if c not in TABLE_HIDDEN_COLUMNS]


def best_row_indices_by_column(
    rows: list[list[str]],
    columns: list[str],
) -> dict[int, set[int]]:
    """Row indices tied for best value in each score column (0-based column index)."""
    best: dict[int, set[int]] = {}
    for col_idx, col_name in enumerate(columns):
        higher = metric_higher_is_better(col_name)
        if higher is None:
            continue
        parsed: list[tuple[float, int]] = []
        for row_idx, row in enumerate(rows):
            try:
                parsed.append((float(row[col_idx]), row_idx))
            except ValueError:
                continue
        if not parsed:
            continue
        target = max(v for v, _ in parsed) if higher else min(v for v, _ in parsed)
        best[col_idx] = {row_idx for v, row_idx in parsed if v == target}
    return best


def summarize_val_ode_monitor(
    ode_rows: list[dict[str, Any]],
    held_out: frozenset[int],
) -> dict[str, float | int]:
    """Scalars from a val-phase ODE eval on held-out context strata only."""
    return {
        "n_val_strata_held_out": sum(1 for r in ode_rows if int(r["c"]) in held_out),
        "val_energy_median": median_on_held_out_strata(ode_rows, "energy", held_out),
        "val_energy_mean": mean_on_held_out_strata(ode_rows, "energy", held_out),
        "val_mse_to_true_median": median_on_held_out_strata(ode_rows, "mse_to_true", held_out),
        "val_mse_to_true_mean": mean_on_held_out_strata(ode_rows, "mse_to_true", held_out),
        "val_mse_mean_vectors_median": median_on_held_out_strata(
            ode_rows, "mse_mean_vectors", held_out
        ),
        "val_mse_mean_vectors_mean": mean_on_held_out_strata(
            ode_rows, "mse_mean_vectors", held_out
        ),
    }
