#!/usr/bin/env python3
"""Paper-ready TX heldout rankings and fit-retention summaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from experiments.tx.audit_final_cl_results import (
    METHODS,
    ROOT,
    SEEDS,
    SPECS,
    DatasetSpec,
    run_path,
)


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    color: str
    marker: str
    matches: Callable[[str], bool]


@dataclass(frozen=True)
class Summary:
    category: Category
    run_key: str
    table_label: str
    values: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.values.mean(axis=0)

    @property
    def se(self) -> np.ndarray:
        return self.values.std(axis=0, ddof=1) / np.sqrt(self.values.shape[0])


# Core replay colors match the sequential simulation figures.
CATEGORIES = (
    Category(
        "naive_freeze",
        "Naive freeze",
        "#6C757D",
        "s",
        lambda key: key.startswith("standard_embedding_"),
    ),
    Category(
        "naive_full",
        "Naive full",
        "#003049",
        "X",
        lambda key: key.startswith("standard_p"),
    ),
    Category(
        "lora",
        "LoRA",
        "#8C6D31",
        "D",
        lambda key: key.startswith("standard_lora_"),
    ),
    Category("gft", "GFT", "#7A5195", "^", lambda key: key.startswith("gft_")),
    Category(
        "two_stage",
        "Two-stage",
        "#B279A2",
        "v",
        lambda key: key.startswith("standard_tsf"),
    ),
    Category(
        "real",
        "Real replay",
        "#2A9D8F",
        "*",
        lambda key: key.startswith("real_"),
    ),
    Category(
        "generative",
        "Generative replay",
        "#C1121F",
        "P",
        lambda key: key.startswith("generative_"),
    ),
    Category(
        "trajectory",
        "Trajectory replay",
        "#457B9D",
        "o",
        lambda key: key.startswith("trajectory_"),
    ),
)

BASELINES = {
    # Pearson: target-context finetune mean (B1). Replogle L1 is the directly
    # scored B1 predictor; PBMC L1 uses CellEval's chance level because the
    # centroid B1 and donor-pooled CellEval protocols are not comparable.
    "pbmc": {2: (0.203, "Mean baseline"), 5: (0.500, "Chance")},
    "replogle": {2: (0.393, "Mean baseline"), 5: (0.502, "Mean baseline")},
}

AXIS_LIMITS = {
    ("pbmc", 2): (0.10, 0.39),
    ("pbmc", 5): (0.48, 0.69),
    ("replogle", 2): (0.12, 0.43),
    ("replogle", 5): (0.48, 0.74),
}

DATASET_ORDER = ("replogle", "pbmc")

HELDOUT_CMAP = LinearSegmentedColormap.from_list(
    "simulation_sequential_blue",
    ("#E8EEF2", "#7FA9C0", "#457B9D", "#003049"),
)

SHORT_CATEGORY_LABELS = {
    "Naive freeze": "Naive fr.",
    "Naive full": "Naive full",
    "LoRA": "LoRA",
    "GFT": "GFT",
    "Two-stage": "Two-stage",
    "Real replay": "Real (stored)",
    "Generative replay": "Generative",
    "Trajectory replay": "Trajectory",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
        }
    )


def _all_summaries(spec: DatasetSpec) -> list[Summary]:
    summaries: list[Summary] = []
    for table_label, run_key in METHODS:
        rows = np.asarray(
            [
                _extended_run_metrics(spec, run_path(spec, run_key, seed))
                for seed in SEEDS
            ],
            dtype=np.float64,
        )
        category = next(
            (category for category in CATEGORIES if category.matches(run_key)),
            None,
        )
        if category is None:
            raise ValueError(f"No category defined for {run_key}")
        summaries.append(
            Summary(
                category=category,
                run_key=run_key,
                table_label=table_label,
                values=rows,
            )
        )
    return summaries


def _mean_cell_metric(
    track: dict,
    *,
    metric: str,
    include: Callable[[str], bool],
) -> float:
    values = [
        float(result[metric])
        for key, payload in track["slices"].items()
        if include(key)
        for result in payload["perturbation_results"]
    ]
    if not values or not np.isfinite(values).all():
        raise ValueError(f"Missing or non-finite CellEval metric {metric}")
    return float(np.mean(values))


def _extended_run_metrics(spec: DatasetSpec, run: Path) -> tuple[float, ...]:
    """Final fit/retention/heldout metrics from the canonical K=500 outputs."""
    pearson_payload = json.loads((run / "final_metrics.json").read_text())
    cell_payload = json.loads(
        (run / "cell_eval_minimal_stream_heldout.json").read_text()
    )
    if pearson_payload["eval_provenance"].get("pred_control") != "decoded":
        raise ValueError(f"{run.name}: Pearson controls are not decoded")
    if int(pearson_payload["eval_provenance"].get("gen_per_stratum", -1)) != 500:
        raise ValueError(f"{run.name}: Pearson K is not 500")
    if cell_payload["eval_provenance"].get("pred_control") != "decoded":
        raise ValueError(f"{run.name}: CellEval controls are not decoded")
    if int(cell_payload["eval_provenance"].get("gen_per_stratum", -1)) != 500:
        raise ValueError(f"{run.name}: CellEval K is not 500")

    pearson = pearson_payload["final"]
    cell = cell_payload["final"]
    if spec.name == "pbmc":
        later_donors = {"Donor1", "Donor4", "Donor9", "Donor12"}
        include_fit = lambda key: key.split("__", 1)[0] in later_donors
        include_retention = lambda key: key.split("__", 1)[0] not in later_donors
        stream = pearson["stream"]["per_task"]
        heldout = pearson["heldout"]["per_task"]

        def p_triplet(metric: str) -> tuple[float, float, float]:
            return (
                float(np.mean([float(stream[str(i)][metric]) for i in range(1, 5)])),
                float(stream["0"][metric]),
                float(np.mean([float(heldout[str(i)][metric]) for i in range(1, 5)])),
            )
    else:
        include_fit = lambda key: "hepg2" in key.lower()
        include_retention = lambda key: "hepg2" not in key.lower()
        stream = pearson["stream"]["per_task"]
        heldout = pearson["heldout"]["per_task"]

        def p_triplet(metric: str) -> tuple[float, float, float]:
            return (
                float(stream["1"][metric]),
                float(stream["0"][metric]),
                float(heldout["1"][metric]),
            )

    def c_triplet(metric: str) -> tuple[float, float, float]:
        return (
            _mean_cell_metric(
                cell["stream"], metric=metric, include=include_fit
            ),
            _mean_cell_metric(
                cell["stream"], metric=metric, include=include_retention
            ),
            _mean_cell_metric(
                cell["heldout"], metric=metric, include=include_fit
            ),
        )

    pearson_delta = p_triplet("pearson_delta")
    l1 = c_triplet("discrimination_score_l1")
    residual = p_triplet("pearson_delta_residual")
    overlap = c_triplet("overlap_at_N")
    mae = c_triplet("mae")
    return (*pearson_delta, *l1, *residual, *overlap, *mae)


def _select_representatives(summaries: list[Summary]) -> list[Summary]:
    """Select each family's highest mean heldout Pearson Delta configuration."""
    selected: list[Summary] = []
    for category in CATEGORIES:
        candidates = [
            summary for summary in summaries if summary.category.key == category.key
        ]
        selected.append(max(candidates, key=lambda summary: float(summary.mean[2])))
    return selected


def _short_selected_label(summary: Summary) -> str:
    key = summary.run_key
    if key.startswith("standard_embedding_"):
        return f"Naive freeze ({key.rsplit('_', 1)[-1]})"
    if key.startswith("standard_lora_"):
        rank = key.split("_")[2].removeprefix("r")
        return f"LoRA ($r={rank}$)"
    if key.startswith("standard_tsf"):
        fraction = float(key.split("_", 2)[1].removeprefix("tsf"))
        return f"Two-stage ({fraction:.0%})"
    if key.startswith("standard_p"):
        return f"Naive full ({key.rsplit('_', 1)[-1]})"
    if key.startswith("gft_"):
        return f"GFT ({key.split('_')[1]})"
    if key.startswith("real_"):
        return f"Real replay ({key.rsplit('_', 1)[-1]})"
    if key.startswith(("generative_", "trajectory_")):
        family = "Generative" if key.startswith("generative_") else "Trajectory"
        backend = (
            "ODE"
            if "_ode_" in key
            else "MF-1"
            if "_mf_n1_" in key
            else "MF-4"
        )
        mode = key.rsplit("_", 1)[-1]
        return f"{family} ({backend}, {mode})"
    return summary.category.label


def _load() -> tuple[
    dict[str, list[Summary]],
    dict[str, list[Summary]],
]:
    all_by_dataset: dict[str, list[Summary]] = {}
    selected_by_dataset: dict[str, list[Summary]] = {}
    for spec in SPECS:
        summaries = _all_summaries(spec)
        all_by_dataset[spec.name] = summaries
        selected_by_dataset[spec.name] = _select_representatives(summaries)
    return all_by_dataset, selected_by_dataset


def _draw_lollipop(
    ax: plt.Axes,
    *,
    dataset: str,
    selected: list[Summary],
    metric_idx: int,
    show_labels: bool,
) -> None:
    real_reference = next(
        summary for summary in selected if summary.category.key == "real"
    )
    displayed = [
        summary for summary in selected if summary.category.key != "real"
    ]
    y = np.arange(len(displayed))
    means = np.asarray([summary.mean[metric_idx] for summary in displayed])
    ses = np.asarray([summary.se[metric_idx] for summary in displayed])
    xmin, xmax = AXIS_LIMITS[(dataset, metric_idx)]

    for yi, summary, value, error in zip(
        y, displayed, means, ses, strict=True
    ):
        ax.hlines(
            yi,
            xmin,
            value,
            color=summary.category.color,
            linewidth=2.4,
            alpha=0.20,
            zorder=1,
        )
        ax.errorbar(
            value,
            yi,
            xerr=error,
            fmt="none",
            ecolor="#37444D",
            elinewidth=1.15,
            capsize=2.7,
            capthick=1.0,
            zorder=4,
        )
        ax.scatter(
            value,
            yi,
            color=summary.category.color,
            marker=summary.category.marker,
            edgecolor="white",
            linewidth=0.55,
            s=46 if summary.category.key != "real" else 72,
            zorder=5,
        )
        ax.annotate(
            f"{value:.3f}",
            (value, yi),
            xytext=(5, -1),
            textcoords="offset points",
            color=summary.category.color,
            fontsize=7.2,
            va="center",
        )

    baseline, baseline_label = BASELINES[dataset][metric_idx]
    baseline_style = (0, (1.5, 1.5)) if baseline_label == "Chance" else (0, (3, 2))
    ax.axvline(
        baseline,
        color="#8DA3B5",
        linestyle=baseline_style,
        linewidth=1.2,
        zorder=0,
    )
    real_mean = float(real_reference.mean[metric_idx])
    real_se = float(real_reference.se[metric_idx])
    ax.axvspan(
        real_mean - real_se,
        real_mean + real_se,
        color="#2A9D8F",
        alpha=0.10,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(
        real_mean,
        color="#2A9D8F",
        linestyle=(0, (5, 2, 1.5, 2)),
        linewidth=1.25,
        zorder=1,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.65, len(displayed) - 0.35)
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels(
            [_short_selected_label(summary) for summary in displayed]
        )
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.20, linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Heldout score $\\uparrow$")


def plot_heldout(
    selected_by_dataset: dict[str, list[Summary]],
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65))
    for row, dataset in enumerate(DATASET_ORDER):
        _draw_lollipop(
            axes[row, 0],
            dataset=dataset,
            selected=selected_by_dataset[dataset],
            metric_idx=2,
            show_labels=True,
        )
        _draw_lollipop(
            axes[row, 1],
            dataset=dataset,
            selected=selected_by_dataset[dataset],
            metric_idx=5,
            show_labels=False,
        )
        axes[row, 0].set_title(f"{dataset.upper()} — Pearson $\\Delta$")
        axes[row, 1].set_title(f"{dataset.upper()} — L1 discrimination")
    reference_handles = [
        Line2D(
            [0],
            [0],
            color="#8DA3B5",
            linestyle=(0, (3, 2)),
            linewidth=1.2,
            label="Mean-response baseline",
        ),
        Line2D(
            [0],
            [0],
            color="#8DA3B5",
            linestyle=(0, (1.5, 1.5)),
            linewidth=1.2,
            label="Chance (PBMC L1 only)",
        ),
        Line2D(
            [0],
            [0],
            color="#2A9D8F",
            linestyle=(0, (5, 2, 1.5, 2)),
            linewidth=1.25,
            label="Real-replay reference",
        ),
    ]
    fig.legend(
        handles=reference_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1), pad=0.35, h_pad=0.75, w_pad=0.7)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_heldout_sensitivity(
    selected_by_dataset: dict[str, list[Summary]],
    out: Path,
) -> None:
    """Compare alternative heldout metrics on the same representatives."""
    metrics = (
        (8, "Residual Pearson $\\Delta$", True),
        (11, "DE Overlap@N", True),
        (14, "MAE", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.6))
    for row, dataset in enumerate(DATASET_ORDER):
        selected = selected_by_dataset[dataset]
        real_reference = next(
            summary for summary in selected if summary.category.key == "real"
        )
        displayed = [
            summary for summary in selected if summary.category.key != "real"
        ]
        y = np.arange(len(displayed))
        for col, (metric_idx, title, higher_is_better) in enumerate(metrics):
            ax = axes[row, col]
            means = np.asarray(
                [summary.mean[metric_idx] for summary in displayed]
            )
            ses = np.asarray([summary.se[metric_idx] for summary in displayed])
            real_mean = float(real_reference.mean[metric_idx])
            real_se = float(real_reference.se[metric_idx])
            all_values = np.append(means, real_mean)
            span = max(0.01, float(np.ptp(all_values)))
            xmin = float(all_values.min()) - 0.12 * span
            xmax = float(all_values.max()) + 0.18 * span

            for yi, summary, value, error in zip(
                y, displayed, means, ses, strict=True
            ):
                start, end = (
                    (xmin, value) if higher_is_better else (value, xmax)
                )
                ax.hlines(
                    yi,
                    start,
                    end,
                    color=summary.category.color,
                    linewidth=2.2,
                    alpha=0.18,
                    zorder=1,
                )
                ax.errorbar(
                    value,
                    yi,
                    xerr=error,
                    fmt="none",
                    ecolor="#37444D",
                    elinewidth=1.1,
                    capsize=2.5,
                    zorder=4,
                )
                ax.scatter(
                    value,
                    yi,
                    color=summary.category.color,
                    marker=summary.category.marker,
                    edgecolor="white",
                    linewidth=0.55,
                    s=43,
                    zorder=5,
                )

            ax.axvspan(
                real_mean - real_se,
                real_mean + real_se,
                color="#2A9D8F",
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )
            ax.axvline(
                real_mean,
                color="#2A9D8F",
                linestyle=(0, (5, 2, 1.5, 2)),
                linewidth=1.2,
                zorder=1,
            )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.65, len(displayed) - 0.35)
            ax.set_yticks(y)
            if col == 0:
                ax.set_yticklabels(
                    [_short_selected_label(summary) for summary in displayed]
                )
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.18, linewidth=0.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            direction = "$\\uparrow$" if higher_is_better else "$\\downarrow$"
            ax.set_xlabel(f"Heldout {direction}")
            ax.set_title(f"{dataset.upper()}\n{title}", linespacing=0.9)

    real_handle = Line2D(
        [0],
        [0],
        color="#2A9D8F",
        linestyle=(0, (5, 2, 1.5, 2)),
        linewidth=1.25,
        label="Real-replay reference",
    )
    fig.legend(
        handles=[real_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    fig.tight_layout(
        rect=(0, 0.045, 1, 1),
        pad=0.30,
        h_pad=0.70,
        w_pad=0.55,
    )
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_fit_retention(
    all_by_dataset: dict[str, list[Summary]],
    selected_by_dataset: dict[str, list[Summary]],
    out: Path,
    *,
    fit_idx: int,
    retention_idx: int,
    metric_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.05, 2.35))
    for panel_idx, (ax, dataset) in enumerate(
        zip(axes, DATASET_ORDER, strict=True)
    ):
        all_summaries = all_by_dataset[dataset]
        selected = selected_by_dataset[dataset]
        all_x = np.asarray([summary.mean[fit_idx] for summary in all_summaries])
        all_y = np.asarray([summary.mean[retention_idx] for summary in all_summaries])
        x_pad = max(0.010, 0.075 * float(np.ptp(all_x)))
        y_pad = max(0.010, 0.075 * float(np.ptp(all_y)))
        ax.scatter(
            all_x,
            all_y,
            s=11,
            color="0.78",
            alpha=0.28,
            linewidths=0,
            zorder=1,
            label="_nolegend_",
        )
        for summary in selected:
            x, y = float(summary.mean[fit_idx]), float(summary.mean[retention_idx])
            xerr = float(summary.se[fit_idx])
            yerr = float(summary.se[retention_idx])
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt=summary.category.marker,
                color=summary.category.color,
                markerfacecolor=(
                    "none"
                    if summary.category.key == "real"
                    else summary.category.color
                ),
                markeredgecolor=(
                    summary.category.color
                    if summary.category.key == "real"
                    else "white"
                ),
                markeredgewidth=1.25 if summary.category.key == "real" else 0.6,
                markersize=8.5 if summary.category.key != "real" else 9.5,
                elinewidth=1.1,
                capsize=2.2,
                label=summary.category.label,
                zorder=5 if summary.category.key == "real" else 3,
            )
        ax.set_title(dataset.upper())
        ax.set_xlabel(f"Fit ({metric_label})")
        ax.set_ylabel(f"Retention ({metric_label})" if panel_idx == 0 else "")
        ax.set_xlim(float(all_x.min()) - x_pad, float(all_x.max()) + x_pad)
        ax.set_ylim(float(all_y.min()) - y_pad, float(all_y.max()) + y_pad)
        ax.grid(alpha=0.20, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.76), pad=0.2, w_pad=0.55)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_fit_retention_by_heldout(
    all_by_dataset: dict[str, list[Summary]],
    out: Path,
    *,
    fit_idx: int,
    retention_idx: int,
    heldout_idx: int,
    metric_label: str,
    higher_is_better: bool = True,
) -> None:
    """Modular square panels with all configurations shown equally."""
    cmap = HELDOUT_CMAP if higher_is_better else HELDOUT_CMAP.reversed()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))
    panel_mappables: list[plt.cm.ScalarMappable] = []
    for ax, dataset in zip(axes, DATASET_ORDER, strict=True):
        all_summaries = all_by_dataset[dataset]
        all_x = np.asarray([summary.mean[fit_idx] for summary in all_summaries])
        all_y = np.asarray(
            [summary.mean[retention_idx] for summary in all_summaries]
        )
        all_heldout = np.asarray(
            [summary.mean[heldout_idx] for summary in all_summaries]
        )
        norm = Normalize(
            vmin=float(all_heldout.min()),
            vmax=float(all_heldout.max()),
        )
        panel_mappables.append(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
        x_pad = max(0.010, 0.075 * float(np.ptp(all_x)))
        y_pad = max(0.010, 0.075 * float(np.ptp(all_y)))

        for category in CATEGORIES:
            category_rows = [
                summary
                for summary in all_summaries
                if summary.category.key == category.key
            ]
            x_values = [
                summary.mean[fit_idx] for summary in category_rows
            ]
            y_values = [
                summary.mean[retention_idx] for summary in category_rows
            ]
            if category.key == "real":
                ax.scatter(
                    x_values,
                    y_values,
                    facecolors="none",
                    edgecolors="#2A9D8F",
                    marker=category.marker,
                    s=78,
                    linewidth=1.15,
                    zorder=2,
                )
            else:
                ax.scatter(
                    x_values,
                    y_values,
                    c=[
                        summary.mean[heldout_idx]
                        for summary in category_rows
                    ],
                    cmap=cmap,
                    norm=norm,
                    marker=category.marker,
                    s=72,
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.92,
                    zorder=3,
                )

        ax.set_title(dataset.upper(), fontsize=14)
        ax.set_xlabel(f"Fit ({metric_label})", fontsize=13.5)
        ax.set_ylabel(f"Retention ({metric_label})", fontsize=13.5)
        ax.set_xlim(float(all_x.min()) - x_pad, float(all_x.max()) + x_pad)
        ax.set_ylim(float(all_y.min()) - y_pad, float(all_y.max()) + y_pad)
        ax.set_box_aspect(1.38)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=category.marker,
            markerfacecolor="white" if category.key == "real" else "0.48",
            markeredgecolor="#2A9D8F" if category.key == "real" else "white",
            markeredgewidth=1.4 if category.key == "real" else 0.7,
            markersize=7.5 if category.key != "real" else 9.5,
            label=SHORT_CATEGORY_LABELS[category.label],
        )
        for category in CATEGORIES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=8,
        frameon=False,
        fontsize=9.4,
        handletextpad=0.20,
        columnspacing=0.55,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.93,
        bottom=0.14,
        top=0.79,
        wspace=0.50,
    )
    for ax, mappable in zip(axes, panel_mappables, strict=True):
        position = ax.get_position()
        colorbar_ax = fig.add_axes(
            [position.x1 + 0.008, position.y0, 0.012, position.height]
        )
        colorbar = fig.colorbar(mappable, cax=colorbar_ax)
        direction = "$\\uparrow$" if higher_is_better else "$\\downarrow$"
        colorbar.set_label(
            f"Heldout {metric_label} {direction}",
            fontsize=11,
            labelpad=2,
        )
        colorbar.ax.tick_params(labelsize=10, length=2, pad=1)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_all_metric_grid(
    all_by_dataset: dict[str, list[Summary]],
    out: Path,
    *,
    metrics: tuple[tuple[int, int, int, str], ...] | None = None,
    figsize: tuple[float, float] = (6.55, 4.05),
) -> None:
    """Dataset-by-metric grid with every configuration shown equally."""
    if metrics is None:
        metrics = (
            (0, 1, 2, "Pearson $\\Delta$"),
            (6, 7, 8, "Residual $\\Delta$"),
            (9, 10, 11, "DE Overlap@N"),
            (3, 4, 5, "L1 discrimination"),
        )
    ncols = len(metrics)
    fig, axes = plt.subplots(2, ncols, figsize=figsize, squeeze=False)
    cmap = HELDOUT_CMAP
    relative_norm = Normalize(vmin=0.0, vmax=1.0)
    for col, (fit_idx, retention_idx, heldout_idx, title) in enumerate(metrics):
        for row, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, col]
            summaries = all_by_dataset[dataset]
            all_x = np.asarray([summary.mean[fit_idx] for summary in summaries])
            all_y = np.asarray(
                [summary.mean[retention_idx] for summary in summaries]
            )
            all_heldout = np.asarray(
                [summary.mean[heldout_idx] for summary in summaries]
            )
            heldout_min = float(all_heldout.min())
            heldout_span = max(
                1e-12,
                float(all_heldout.max()) - heldout_min,
            )
            x_pad = max(0.008, 0.07 * float(np.ptp(all_x)))
            y_pad = max(0.008, 0.07 * float(np.ptp(all_y)))
            for category in CATEGORIES:
                category_rows = [
                    summary
                    for summary in summaries
                    if summary.category.key == category.key
                ]
                x_values = [
                    summary.mean[fit_idx] for summary in category_rows
                ]
                y_values = [
                    summary.mean[retention_idx]
                    for summary in category_rows
                ]
                if category.key == "real":
                    ax.scatter(
                        x_values,
                        y_values,
                        facecolors="none",
                        edgecolors="#2A9D8F",
                        marker=category.marker,
                        s=38,
                        linewidth=1.0,
                        zorder=2,
                    )
                else:
                    ax.scatter(
                        x_values,
                        y_values,
                        c=[
                            (
                                float(summary.mean[heldout_idx])
                                - heldout_min
                            )
                            / heldout_span
                            for summary in category_rows
                        ],
                        cmap=cmap,
                        norm=relative_norm,
                        marker=category.marker,
                        s=25,
                        edgecolor="white",
                        linewidth=0.45,
                        alpha=0.88,
                        zorder=3,
                    )
            ax.set_xlim(
                float(all_x.min()) - x_pad,
                float(all_x.max()) + x_pad,
            )
            ax.set_ylim(
                float(all_y.min()) - y_pad,
                float(all_y.max()) + y_pad,
            )
            ax.grid(alpha=0.16, linewidth=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.set_title(title, fontsize=10.5)
            if row == 1:
                ax.set_xlabel("Fit", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"{dataset.upper()}\nRetention",
                    fontsize=10,
                    multialignment="center",
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=category.marker,
            markerfacecolor="white" if category.key == "real" else "0.48",
            markeredgecolor="#2A9D8F" if category.key == "real" else "white",
            markeredgewidth=1.3 if category.key == "real" else 0.6,
            markersize=6.5 if category.key != "real" else 8.5,
            label=(
                "Real replay (stored data)"
                if category.key == "real"
                else category.label
            ),
        )
        for category in CATEGORIES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=8.3,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    fig.subplots_adjust(
        left=0.095 if ncols == 2 else 0.075,
        right=0.995,
        bottom=0.22,
        top=0.76,
        wspace=0.30,
        hspace=0.38,
    )
    colorbar_ax = fig.add_axes([0.35, 0.055, 0.30, 0.012])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=relative_norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=(0.0, 1.0),
    )
    colorbar.ax.set_xticklabels(("Lower", "Higher"))
    colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
    colorbar.set_label(
        "Heldout performance (normalized within dataset and metric)",
        fontsize=7.5,
        labelpad=1,
    )
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_main_one_by_four(
    all_by_dataset: dict[str, list[Summary]],
    out: Path,
) -> None:
    """Compact square 1x4 main-paper view for Pearson and DE overlap."""
    metrics = (
        (0, 1, 2, "Pearson $\\Delta$"),
        (9, 10, 11, "DE Overlap@N"),
    )
    panels = [
        (dataset, metric)
        for dataset in DATASET_ORDER
        for metric in metrics
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 3.05))
    relative_norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, (dataset, metric) in zip(axes, panels, strict=True):
        fit_idx, retention_idx, heldout_idx, title = metric
        summaries = all_by_dataset[dataset]
        all_x = np.asarray([summary.mean[fit_idx] for summary in summaries])
        all_y = np.asarray(
            [summary.mean[retention_idx] for summary in summaries]
        )
        all_heldout = np.asarray(
            [summary.mean[heldout_idx] for summary in summaries]
        )
        heldout_min = float(all_heldout.min())
        heldout_span = max(1e-12, float(all_heldout.max()) - heldout_min)
        x_pad = max(0.007, 0.055 * float(np.ptp(all_x)))
        y_pad = max(0.007, 0.055 * float(np.ptp(all_y)))

        for category in CATEGORIES:
            category_rows = [
                summary
                for summary in summaries
                if summary.category.key == category.key
            ]
            x_values = [
                summary.mean[fit_idx] for summary in category_rows
            ]
            y_values = [
                summary.mean[retention_idx] for summary in category_rows
            ]
            if category.key == "real":
                ax.scatter(
                    x_values,
                    y_values,
                    facecolors="none",
                    edgecolors="#2A9D8F",
                    marker=category.marker,
                    s=60,
                    linewidth=1.05,
                    zorder=2,
                )
            else:
                ax.scatter(
                    x_values,
                    y_values,
                    c=[
                        (
                            float(summary.mean[heldout_idx])
                            - heldout_min
                        )
                        / heldout_span
                        for summary in category_rows
                    ],
                    cmap=HELDOUT_CMAP,
                    norm=relative_norm,
                    marker=category.marker,
                    s=50,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.92,
                    zorder=3,
                )
        ax.set_xlim(float(all_x.min()) - x_pad, float(all_x.max()) + x_pad)
        ax.set_ylim(float(all_y.min()) - y_pad, float(all_y.max()) + y_pad)
        ax.set_box_aspect(1.28)
        ax.set_title(title, fontsize=12.5, pad=4)
        ax.set_xlabel("Fit", fontsize=12, labelpad=2)
        ax.set_ylabel("Retention", fontsize=12, labelpad=2)
        ax.tick_params(labelsize=11, pad=2)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(alpha=0.16, linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=category.marker,
            markerfacecolor="white" if category.key == "real" else "0.48",
            markeredgecolor="#2A9D8F" if category.key == "real" else "white",
            markeredgewidth=1.3 if category.key == "real" else 0.6,
            markersize=6.5 if category.key != "real" else 8.5,
            label=SHORT_CATEGORY_LABELS[category.label],
        )
        for category in CATEGORIES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=8,
        frameon=False,
        fontsize=8.8,
        handletextpad=0.2,
        columnspacing=0.55,
    )
    fig.text(0.275, 0.83, "REPLOGLE", ha="center", fontsize=13)
    fig.text(0.755, 0.83, "PBMC", ha="center", fontsize=13)
    fig.add_artist(
        Line2D(
            [0.515, 0.515],
        [0.22, 0.78],
            transform=fig.transFigure,
            color="0.86",
            linewidth=0.7,
        )
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.24,
        top=0.72,
        wspace=0.38,
    )
    colorbar_ax = fig.add_axes([0.35, 0.075, 0.30, 0.014])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=relative_norm, cmap=HELDOUT_CMAP),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=(0.0, 1.0),
    )
    colorbar.ax.set_xticklabels(("Lower", "Higher"))
    colorbar.ax.tick_params(labelsize=7.5, length=2, pad=1)
    colorbar.set_label(
        "Heldout performance (within dataset and metric)",
        fontsize=8.5,
        labelpad=1,
    )
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_main_two_by_two(
    all_by_dataset: dict[str, list[Summary]],
    out: Path,
) -> None:
    """Horizontal 2x2 main-paper candidate with numeric panel colour bars."""
    metrics = (
        (0, 1, 2, "Pearson $\\Delta$"),
        (9, 10, 11, "DE Overlap@N"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.65))
    panel_mappables: list[tuple[plt.Axes, plt.cm.ScalarMappable, str]] = []

    for row, (fit_idx, retention_idx, heldout_idx, metric_label) in enumerate(
        metrics
    ):
        for col, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, col]
            summaries = all_by_dataset[dataset]
            all_x = np.asarray(
                [summary.mean[fit_idx] for summary in summaries]
            )
            all_y = np.asarray(
                [summary.mean[retention_idx] for summary in summaries]
            )
            all_heldout = np.asarray(
                [summary.mean[heldout_idx] for summary in summaries]
            )
            norm = Normalize(
                vmin=float(all_heldout.min()),
                vmax=float(all_heldout.max()),
            )
            panel_mappables.append(
                (
                    ax,
                    plt.cm.ScalarMappable(norm=norm, cmap=HELDOUT_CMAP),
                    metric_label,
                )
            )
            x_pad = max(0.009, 0.065 * float(np.ptp(all_x)))
            y_pad = max(0.009, 0.065 * float(np.ptp(all_y)))

            for category in CATEGORIES:
                category_rows = [
                    summary
                    for summary in summaries
                    if summary.category.key == category.key
                ]
                x_values = [
                    summary.mean[fit_idx] for summary in category_rows
                ]
                y_values = [
                    summary.mean[retention_idx] for summary in category_rows
                ]
                if category.key == "real":
                    ax.scatter(
                        x_values,
                        y_values,
                        facecolors="none",
                        edgecolors="#2A9D8F",
                        marker=category.marker,
                        s=82,
                        linewidth=1.25,
                        zorder=5,
                    )
                else:
                    ax.scatter(
                        x_values,
                        y_values,
                        c=[
                            summary.mean[heldout_idx]
                            for summary in category_rows
                        ],
                        cmap=HELDOUT_CMAP,
                        norm=norm,
                        marker=category.marker,
                        s=104 if category.key == "trajectory" else 72,
                        edgecolor="white",
                        linewidth=0.6,
                        alpha=0.94,
                        zorder=3,
                    )

            ax.set_xlim(
                float(all_x.min()) - x_pad,
                float(all_x.max()) + x_pad,
            )
            ax.set_ylim(
                float(all_y.min()) - y_pad,
                float(all_y.max()) + y_pad,
            )
            ax.set_box_aspect(0.46)
            if row == 0:
                ax.set_title(dataset.upper(), fontsize=13)
            ax.set_xlabel(f"Fit ({metric_label})", fontsize=11.5, labelpad=2)
            ax.set_ylabel(
                f"Retention ({metric_label})",
                fontsize=11.5,
                labelpad=2,
            )
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=10.5, pad=2)
            ax.grid(alpha=0.16, linewidth=0.45)
            ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=category.marker,
            markerfacecolor="white" if category.key == "real" else "0.48",
            markeredgecolor=(
                "#2A9D8F"
                if category.key == "real"
                else "white"
            ),
            markeredgewidth=(
                1.3
                if category.key == "real"
                else 0.6
            ),
            markersize=(
                8.5
                if category.key == "real"
                else 7.0
            ),
            label=(
                "Trajectory replay"
                if category.key == "trajectory"
                else SHORT_CATEGORY_LABELS[category.label]
            ),
        )
        for category in CATEGORIES
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=8,
        frameon=False,
        fontsize=8.8,
        handletextpad=0.2,
        columnspacing=0.55,
    )
    for text in legend.get_texts():
        if text.get_text() == "Trajectory replay":
            text.set_fontweight("bold")
    fig.subplots_adjust(
        left=0.09,
        right=0.91,
        bottom=0.11,
        top=0.84,
        wspace=0.55,
        hspace=0.78,
    )
    for ax, mappable, metric_label in panel_mappables:
        position = ax.get_position()
        colorbar_ax = fig.add_axes(
            [position.x1 + 0.008, position.y0, 0.012, position.height]
        )
        colorbar = fig.colorbar(mappable, cax=colorbar_ax)
        colorbar.set_label(
            f"Heldout {metric_label} $\\uparrow$",
            fontsize=8.8,
            labelpad=1,
        )
        colorbar.ax.tick_params(labelsize=8, length=2, pad=1)

    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _write_selection(
    selected_by_dataset: dict[str, list[Summary]],
    out: Path,
) -> None:
    payload = {
        "selection_rule": (
            "Within each predefined model family, select the configuration with "
            "the highest three-seed mean heldout Pearson Delta; use that same "
            "configuration for every displayed metric."
        ),
        "seeds": list(SEEDS),
        "evaluation": {
            "generated_cells_per_non_control_stratum": 500,
            "error_bars": "standard error across seeds",
        },
        "datasets": {
            dataset: [
                {
                    "category": summary.category.label,
                    "run_key": summary.run_key,
                    "table_label": summary.table_label,
                    "mean": summary.mean.tolist(),
                    "se": summary.se.tolist(),
                }
                for summary in selected
            ]
            for dataset, selected in selected_by_dataset.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_metric_best(
    all_by_dataset: dict[str, list[Summary]],
    out: Path,
) -> None:
    metrics = (
        ("pearson_delta", 2, True),
        ("l1_discrimination", 5, True),
        ("residual_pearson_delta", 8, True),
        ("de_overlap_at_n", 11, True),
        ("mae", 14, False),
    )
    payload: dict[str, object] = {
        "selection": (
            "Metric-specific best three-seed mean within each predefined model "
            "family. Higher is better except MAE."
        ),
        "datasets": {},
    }
    datasets: dict[str, object] = {}
    for dataset, summaries in all_by_dataset.items():
        categories: dict[str, object] = {}
        for category in CATEGORIES:
            candidates = [
                summary
                for summary in summaries
                if summary.category.key == category.key
            ]
            best_by_metric: dict[str, object] = {}
            for metric, idx, higher_is_better in metrics:
                pick = (
                    max(candidates, key=lambda summary: float(summary.mean[idx]))
                    if higher_is_better
                    else min(
                        candidates,
                        key=lambda summary: float(summary.mean[idx]),
                    )
                )
                best_by_metric[metric] = {
                    "run_key": pick.run_key,
                    "table_label": pick.table_label,
                    "mean": float(pick.mean[idx]),
                    "se": float(pick.se[idx]),
                }
            categories[category.label] = best_by_metric
        datasets[dataset] = categories
    payload["datasets"] = datasets
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results_dir" / "tx_results" / "figures",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _set_style()
    all_by_dataset, selected_by_dataset = _load()
    heldout = args.out_dir / "tx_heldout_category_representatives.png"
    pearson_tradeoff = (
        args.out_dir / "tx_fit_retention_category_representatives.png"
    )
    l1_tradeoff = (
        args.out_dir / "tx_fit_retention_l1_category_representatives.png"
    )
    pearson_heldout_color = (
        args.out_dir / "tx_fit_retention_heldout_color.png"
    )
    l1_heldout_color = (
        args.out_dir / "tx_fit_retention_l1_heldout_color.png"
    )
    residual_heldout_color = (
        args.out_dir / "tx_fit_retention_residual_heldout_color.png"
    )
    overlap_heldout_color = (
        args.out_dir / "tx_fit_retention_overlap_heldout_color.png"
    )
    mae_heldout_color = (
        args.out_dir / "tx_fit_retention_mae_heldout_color.png"
    )
    all_metric_grid = args.out_dir / "tx_fit_retention_all_metrics_grid.png"
    main_metric_grid = (
        args.out_dir / "tx_main_fit_retention_pearson_overlap.png"
    )
    main_one_by_four = (
        args.out_dir / "tx_main_fit_retention_pearson_overlap_1x4.png"
    )
    main_two_by_two = (
        args.out_dir / "tx_main_fit_retention_pearson_overlap_2x2.png"
    )
    sensitivity = args.out_dir / "tx_heldout_sensitive_metrics.png"
    selection = args.out_dir / "tx_category_representatives.json"
    metric_best = args.out_dir / "tx_category_metric_best.json"
    plot_heldout(selected_by_dataset, heldout)
    plot_heldout_sensitivity(selected_by_dataset, sensitivity)
    plot_fit_retention(
        all_by_dataset,
        selected_by_dataset,
        pearson_tradeoff,
        fit_idx=0,
        retention_idx=1,
        metric_label="Pearson $\\Delta$",
    )
    plot_fit_retention(
        all_by_dataset,
        selected_by_dataset,
        l1_tradeoff,
        fit_idx=3,
        retention_idx=4,
        metric_label="L1 discrimination",
    )
    plot_fit_retention_by_heldout(
        all_by_dataset,
        pearson_heldout_color,
        fit_idx=0,
        retention_idx=1,
        heldout_idx=2,
        metric_label="Pearson $\\Delta$",
    )
    plot_fit_retention_by_heldout(
        all_by_dataset,
        l1_heldout_color,
        fit_idx=3,
        retention_idx=4,
        heldout_idx=5,
        metric_label="L1 discrimination",
    )
    plot_fit_retention_by_heldout(
        all_by_dataset,
        residual_heldout_color,
        fit_idx=6,
        retention_idx=7,
        heldout_idx=8,
        metric_label="Residual Pearson $\\Delta$",
    )
    plot_fit_retention_by_heldout(
        all_by_dataset,
        overlap_heldout_color,
        fit_idx=9,
        retention_idx=10,
        heldout_idx=11,
        metric_label="DE Overlap@N",
    )
    plot_fit_retention_by_heldout(
        all_by_dataset,
        mae_heldout_color,
        fit_idx=12,
        retention_idx=13,
        heldout_idx=14,
        metric_label="MAE",
        higher_is_better=False,
    )
    plot_all_metric_grid(all_by_dataset, all_metric_grid)
    plot_all_metric_grid(
        all_by_dataset,
        main_metric_grid,
        metrics=(
            (0, 1, 2, "Pearson $\\Delta$"),
            (9, 10, 11, "DE Overlap@N"),
        ),
        figsize=(5.65, 3.8),
    )
    plot_main_one_by_four(all_by_dataset, main_one_by_four)
    plot_main_two_by_two(all_by_dataset, main_two_by_two)
    _write_selection(selected_by_dataset, selection)
    _write_metric_best(all_by_dataset, metric_best)
    print(heldout)
    print(sensitivity)
    print(pearson_tradeoff)
    print(l1_tradeoff)
    print(pearson_heldout_color)
    print(l1_heldout_color)
    print(residual_heldout_color)
    print(overlap_heldout_color)
    print(mae_heldout_color)
    print(all_metric_grid)
    print(main_metric_grid)
    print(main_one_by_four)
    print(main_two_by_two)
    print(selection)
    print(metric_best)


if __name__ == "__main__":
    main()
