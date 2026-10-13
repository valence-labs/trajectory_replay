#!/usr/bin/env python3
"""Plot PX stream/task FID, including adaptive lambda and Masip when complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
EXPERIMENTS = ("U2OS-01", "U2OS-02", "U2OS-03")
METHODS = {
    "Joint training": {
        "stem": "cumulative_ode_equal_s{seed}",
        "color": "#8C857B",
        "marker": "D",
        "zorder": 2,
    },
    "Real replay": {
        "stem": "real_ode_equal_s{seed}",
        "color": "#2A9D8F",
        "marker": "*",
        "zorder": 3,
    },
    "Naive fine-tuning": {
        "stem": "naive_ode_equal_s{seed}",
        "color": "#4A4A4A",
        "marker": "X",
        "zorder": 2,
    },
    "Gen. replay": {
        "stem": "generative_ode_equal_s{seed}",
        "color": "#C1121F",
        "marker": "s",
        "zorder": 4,
    },
    "Gen. replay (adapt.)": {
        "stem": "generative_ode_grid_s{seed}",
        "color": "#C1121F",
        "marker": "s",
        "open": True,
        "zorder": 3,
    },
    "Traj. replay": {
        "stem": "trajectory_ode_equal_s{seed}",
        "color": "#457B9D",
        "marker": "o",
        "zorder": 5,
    },
    "Traj. replay (adapt.)": {
        "stem": "trajectory_ode_grid_s{seed}",
        "color": "#457B9D",
        "marker": "o",
        "open": True,
        "zorder": 4,
    },
    "Traj. MF-4": {
        "stem": "trajectory_mean_flow_n4_equal_s{seed}",
        "color": "#5B8FB3",
        "marker": "p",
        "zorder": 4,
    },
    "Traj. MF-1": {
        "stem": "trajectory_mean_flow_n1_equal_s{seed}",
        "color": "#7BA3C2",
        "marker": "h",
        "zorder": 3,
    },
    "Gen. distillation (diffusion)": {
        "family": "masip",
        "stem": "generative_distillation_s{seed}",
        "color": "#F4A261",
        "marker": "v",
        "zorder": 1,
    },
}
TASK_LINESTYLE = (0, (2.0, 2.0))
STREAM_LINESTYLE = "solid"
# Both keep historical data, so they bound what a no-storage method can reach.
REFERENCE_METHODS = ("Joint training", "Real replay")
COMPACT_METHODS = (
    "Naive fine-tuning",
    "Gen. distillation (diffusion)",
    "Gen. replay (adapt.)",
    "Gen. replay",
    "Traj. replay (adapt.)",
    "Traj. replay",
)
COMPACT_STYLES = {
    "Naive fine-tuning": {
        "label": "Naive fine-tuning",
        "color": "#4A4A4A",
        "marker": "X",
        "linestyle": "solid",
        "open": False,
        "zorder": 2,
    },
    "Gen. distillation (diffusion)": {
        "label": "Gen. distill. (diffusion)",
        "color": "#F4A261",
        "marker": "v",
        "linestyle": "solid",
        "open": False,
        "zorder": 2,
    },
    "Gen. replay (adapt.)": {
        "label": r"Gen. replay, adaptive $\lambda$",
        "color": "#C1121F",
        "marker": "s",
        "linestyle": "solid",
        "open": False,
        "zorder": 3,
    },
    "Gen. replay": {
        "label": "Gen. replay, equal-task",
        "color": "#C1121F",
        "marker": "s",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 2,
    },
    "Traj. replay (adapt.)": {
        "label": r"Traj. replay, adaptive $\lambda$",
        "color": "#457B9D",
        "marker": "o",
        "linestyle": "solid",
        "open": False,
        "zorder": 5,
    },
    "Traj. replay": {
        "label": "Traj. replay, equal-task",
        "color": "#457B9D",
        "marker": "o",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "px" / "cl_final",
    )
    parser.add_argument(
        "--masip-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "px" / "masip_final",
        help="Root for matched Masip diffusion runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "px"
            / "figures"
            / "px_final_experiment_fid_equal"
        ),
        help="Output stem; writes .pdf, .png, and .json.",
    )
    parser.add_argument(
        "--stream-out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "px"
            / "figures"
            / "px_stream_fid_equal"
        ),
        help="Output stem for the pooled stream-FID figure.",
    )
    parser.add_argument(
        "--rank-out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "px"
            / "figures"
            / "px_stream_fid_rank_equal"
        ),
        help="Output stem for within-stage stream-FID ranks.",
    )
    parser.add_argument(
        "--stage-out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "px"
            / "figures"
            / "px_stage_retention_equal"
        ),
        help="Output stem for the per-experiment retention figure.",
    )
    parser.add_argument(
        "--retention-out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "px"
            / "figures"
            / "px_task0_retention_equal"
        ),
        help="Output stem for the compact U2OS-01 retention figure.",
    )
    return parser.parse_args()


def latest_complete_metrics(run_root: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    if not run_root.exists():
        return None
    for path in run_root.rglob("metrics.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            blocks = payload.get("metrics_by_task", [])
            final = blocks[-1] if blocks else {}
            rows = final.get("test_by_experiment", [])
            if (
                final.get("task_id") == 2
                and final.get("test_stream") is not None
                and len(rows) == len(EXPERIMENTS)
            ):
                candidates.append((path.stat().st_mtime, path, payload))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def collect_task0(results_root: Path) -> dict[str, float] | None:
    """Stage-0 FID of the shared task-0 checkpoint, if it has been evaluated."""
    values: list[float] = []
    for seed in SEEDS:
        path = results_root / f"task0_s{seed}" / "eval_task0.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values.append(float(payload["test_by_experiment"][0]["fid"]))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "se": float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
    }


def _summarize(values: np.ndarray) -> tuple[list[float], list[float]]:
    mean = np.full(values.shape[1], np.nan, dtype=float)
    se = np.full(values.shape[1], np.nan, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        mean[j] = float(col.mean())
        if col.size > 1:
            se[j] = float(col.std(ddof=1) / np.sqrt(col.size))
        else:
            se[j] = 0.0
    return mean.tolist(), se.tolist()


def collect(results_root: Path, masip_root: Path) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    for method, style in METHODS.items():
        seed_rows: list[dict[str, Any]] = []
        family_root = masip_root if style.get("family") == "masip" else results_root
        for seed in SEEDS:
            run_root = family_root / style["stem"].format(seed=seed)
            found = latest_complete_metrics(run_root)
            if found is None:
                continue
            path, payload = found
            blocks = payload["metrics_by_task"]
            final = blocks[-1]
            by_name = {
                row["batch_name"]: float(row["fid"])
                for row in final["test_by_experiment"]
            }
            task_fid = [by_name[name] for name in EXPERIMENTS]
            stream_fid = [np.nan] * len(EXPERIMENTS)
            # stage x evaluated experiment; NaN where the pair was not evaluated.
            stage_fid = np.full(
                (len(EXPERIMENTS), len(EXPERIMENTS)), np.nan, dtype=float
            )
            for block in blocks:
                task_id = block.get("task_id")
                if not isinstance(task_id, int) or not 0 <= task_id < len(EXPERIMENTS):
                    continue
                stream = block.get("test_stream")
                if stream is not None and stream.get("fid") is not None:
                    stream_fid[task_id] = float(stream["fid"])
                for row in block.get("test_by_experiment") or []:
                    if row["batch_name"] in EXPERIMENTS:
                        column = EXPERIMENTS.index(row["batch_name"])
                        stage_fid[task_id, column] = float(row["fid"])
            seed_rows.append(
                {
                    "seed": seed,
                    "task_fid": task_fid,
                    "stream_fid": stream_fid,
                    "stage_fid": stage_fid,
                    "source": str(path),
                }
            )
        if not seed_rows:
            continue
        task_values = np.asarray([row["task_fid"] for row in seed_rows], dtype=float)
        stream_values = np.asarray(
            [row["stream_fid"] for row in seed_rows], dtype=float
        )
        stage_values = np.stack([row["stage_fid"] for row in seed_rows])
        task_mean, task_se = _summarize(task_values)
        stream_mean, stream_se = _summarize(stream_values)
        by_experiment: dict[str, dict[str, list[float | None]]] = {}
        for column, name in enumerate(EXPERIMENTS):
            mean, se = _summarize(stage_values[:, :, column])
            by_experiment[name] = {
                "mean": [None if not np.isfinite(v) else v for v in mean],
                "se": [None if not np.isfinite(v) else v for v in se],
            }
        collected[method] = {
            "n": int(task_values.shape[0]),
            "seeds": [row["seed"] for row in seed_rows],
            "task": {"mean": task_mean, "se": task_se},
            "stream": {
                "mean": [None if not np.isfinite(v) else v for v in stream_mean],
                "se": [None if not np.isfinite(v) else v for v in stream_se],
            },
            "by_stage": by_experiment,
            "sources": [row["source"] for row in seed_rows],
        }
    return collected


def _errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    series: dict[str, list[float]],
    *,
    color: str,
    linestyle: str | tuple,
    marker: str,
    markersize: float,
    markerfacecolor: str,
    markeredgewidth: float,
    linewidth: float,
    alpha: float,
    zorder: int,
    label: str | None,
) -> None:
    mean = np.asarray(series["mean"], dtype=float)
    se = np.asarray(series["se"], dtype=float)
    mask = np.isfinite(mean) & np.isfinite(se)
    if not mask.any():
        return
    ax.errorbar(
        x[mask],
        mean[mask],
        yerr=se[mask],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        markerfacecolor=markerfacecolor,
        markeredgecolor=color,
        markeredgewidth=markeredgewidth,
        capsize=5.0,
        capthick=1.5,
        elinewidth=1.5,
        barsabove=True,
        label=label,
        alpha=alpha,
        zorder=zorder,
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )


def _method_style(method: str, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    style = METHODS[method]
    is_trajectory = method.startswith("Traj.")
    is_real = method == "Real replay"
    is_open = bool(style.get("open")) or is_real
    kwargs = dict(
        color=style["color"],
        marker=style["marker"],
        markersize=4.8 if is_trajectory else (5.2 if is_real else 3.8),
        markerfacecolor="white" if is_open else style["color"],
        markeredgewidth=1.2 if is_real or is_trajectory or is_open else 0.7,
        linewidth=2.1 if method == "Traj. replay" else 1.25,
        alpha=1.0 if is_trajectory else 0.82,
        zorder=style["zorder"],
    )
    label = method if row["n"] == 3 else rf"{method} ($n={row['n']}$)"
    return kwargs, label


def _rank_ytick(method: str, n: int) -> str:
    label = method.replace(" (adapt.)", "\n(adapt.)")
    label = label.replace(" (diffusion)", "\n(diffusion)")
    if n != 3:
        label = rf"{label} ($n={n}$)"
    return label


def _compact_series(
    ax: plt.Axes,
    x: np.ndarray,
    series: dict[str, Any],
    method: str,
) -> None:
    style = COMPACT_STYLES[method]
    values = np.asarray(series["mean"], dtype=float)
    errors = np.asarray(series["se"], dtype=float)
    mask = np.isfinite(values) & np.isfinite(errors)
    x_valid = x[mask]
    values = values[mask]
    errors = errors[mask]
    ax.fill_between(
        x_valid,
        values - errors,
        values + errors,
        color=style["color"],
        alpha=0.11,
        linewidth=0,
        zorder=style["zorder"] - 1,
    )
    ax.plot(
        x_valid,
        values,
        color=style["color"],
        marker=style["marker"],
        markersize=6.4,
        markerfacecolor="white" if style["open"] else style["color"],
        markeredgecolor=style["color"],
        markeredgewidth=1.2 if style["open"] else 0.7,
        linestyle=style["linestyle"],
        linewidth=2.0 if method.startswith("Traj.") else 1.7,
        label=style["label"],
        zorder=style["zorder"],
    )


def _finish(
    fig: plt.Figure,
    ax: plt.Axes,
    out: Path,
    *,
    ncol: int,
    note: str | None,
) -> None:
    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    caption = (
        None
        if note is None
        else fig.text(
            0.5,
            -0.03,
            note,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(
            out.with_suffix(suffix),
            bbox_inches="tight",
            bbox_extra_artists=(
                (legend,) if caption is None else (legend, caption)
            ),
            **kwargs,
        )
    plt.close(fig)


def render_task_fid(data: dict[str, Any], out: Path) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    x = np.arange(len(EXPERIMENTS), dtype=float)
    present = [method for method in METHODS if method in data]
    offsets = dict(zip(present, np.linspace(-0.08, 0.08, max(len(present), 1))))

    for method in METHODS:
        if method not in data:
            continue
        row = data[method]
        kwargs, label = _method_style(method, row)
        _errorbar(
            ax,
            x + offsets[method],
            row["task"],
            linestyle=TASK_LINESTYLE,
            label=label,
            **kwargs,
        )

    ax.set_xticks(x, EXPERIMENTS)
    ax.set_xlabel("Evaluation experiment at final checkpoint")
    ax.set_ylabel(r"FID $\downarrow$")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _finish(
        fig,
        ax,
        out,
        ncol=3,
        note="Mean $\\pm$ SE across seeds; dotted lines guide the eye.",
    )


def render_stream_fid(
    data: dict[str, Any],
    task0: dict[str, float] | None,
    out: Path,
) -> None:
    """Pooled FID over every experiment seen so far, after each task."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    x = np.arange(len(EXPERIMENTS), dtype=float)
    for method in COMPACT_METHODS:
        if method not in data:
            continue
        row = data[method]
        series = row["stream"]
        # Stage 0 pools a single experiment and is shared by every method.
        if task0 is not None:
            series = {
                "mean": [task0["mean"], *series["mean"][1:]],
                "se": [task0["se"], *series["se"][1:]],
            }
        _compact_series(ax, x, series, method)

    if task0 is not None:
        ax.plot(
            x[0],
            task0["mean"],
            marker="P",
            markersize=6.5,
            color="#111111",
            zorder=10,
            linestyle="none",
        )
    else:
        ax.annotate(
            "stage 0\nnot evaluated",
            xy=(x[0], 0.06),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=7.0,
            style="italic",
            color="#999999",
        )

    seen = ["{01}", "{01,02}", "{01,02,03}"]
    ax.set_xticks(x, [f"T{i}\n{label}" for i, label in enumerate(seen)])
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    ax.set_title("Pooled stream", fontsize=12, pad=5)
    ax.set_xlabel("Tasks completed (experiments pooled)")
    ax.set_ylabel(r"Stream FID $\downarrow$")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _finish(
        fig,
        ax,
        out,
        ncol=3,
        note=None,
    )


def render_task0_retention(
    data: dict[str, Any],
    task0: dict[str, float] | None,
    out: Path,
) -> None:
    """Compact retention curve for the oldest experiment, U2OS-01."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    x = np.arange(len(EXPERIMENTS), dtype=float)
    for method in COMPACT_METHODS:
        if method not in data:
            continue
        row = data[method]
        series = row["by_stage"]["U2OS-01"]
        if task0 is not None:
            series = {
                "mean": [task0["mean"], *series["mean"][1:]],
                "se": [task0["se"], *series["se"][1:]],
            }
        _compact_series(ax, x, series, method)

    if task0 is not None:
        ax.plot(
            x[0],
            task0["mean"],
            marker="P",
            markersize=6.5,
            color="#111111",
            zorder=10,
            linestyle="none",
        )

    ax.set_xticks(x, [f"T{i}" for i in range(len(EXPERIMENTS))])
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    ax.set_title("U2OS-01 retention", fontsize=12, pad=5)
    ax.set_xlabel("Tasks completed")
    ax.set_ylabel(r"U2OS-01 FID $\downarrow$")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _finish(
        fig,
        ax,
        out,
        ncol=3,
        note=None,
    )


def render_stream_rank(
    data: dict[str, Any],
    out: Path,
    *,
    figsize: tuple[float, float] | None = None,
) -> list[Path]:
    """Ranked dot plots of pooled stream FID within Tasks 2 and 3."""
    stages = (1, 2)
    stage_rows: dict[int, list[tuple[str, float, float]]] = {}
    for stage in stages:
        stage_rows[stage] = sorted(
            (
                (
                    method,
                    float(row["stream"]["mean"][stage]),
                    float(row["stream"]["se"][stage]),
                )
                for method, row in data.items()
                if row["stream"]["mean"][stage] is not None
            ),
            key=lambda item: item[1],
        )

    # Keep methods on the same row in both figures; order by the final stage.
    method_order = [method for method, _, _ in stage_rows[stages[-1]]]
    y_by_method = {
        method: position for position, method in enumerate(method_order, start=1)
    }

    if figsize is None:
        figsize = (8.0, 1.25 + 0.42 * max(len(method_order), 5))

    _apply_style()
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 19,
            "axes.titlesize": 18,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, stage in zip(axes, stages):
        for method in REFERENCE_METHODS:
            if method in y_by_method:
                ax.axhspan(
                    y_by_method[method] - 0.5,
                    y_by_method[method] + 0.5,
                    color="#EFEBE4",
                    zorder=0,
                )
        ordered = stage_rows[stage]
        for method, mean, se in ordered:
            kwargs, _ = _method_style(method, data[method])
            y = y_by_method[method]
            ax.plot(
                mean,
                y,
                color=kwargs["color"],
                marker=kwargs["marker"],
                markersize=kwargs["markersize"] * 2.9,
                markerfacecolor=kwargs["markerfacecolor"],
                markeredgecolor=kwargs["color"],
                markeredgewidth=kwargs["markeredgewidth"] * 1.3,
                linestyle="none",
                zorder=kwargs["zorder"],
            )
            # Compose mean and SE so the SE can sit inline at a smaller size.
            ax.add_artist(
                AnnotationBbox(
                    HPacker(
                        children=[
                            TextArea(
                                f"{mean:.1f}",
                                textprops=dict(
                                    fontsize=16, color=kwargs["color"]
                                ),
                            ),
                            TextArea(
                                f"$\\pm${se:.2f}",
                                textprops=dict(fontsize=13, color="#7A7A7A"),
                            ),
                        ],
                        align="baseline",
                        pad=0,
                        sep=3,
                    ),
                    xy=(mean, y),
                    xybox=(12, 0),
                    boxcoords="offset points",
                    box_alignment=(0.0, 0.5),
                    frameon=False,
                    annotation_clip=False,
                    zorder=kwargs["zorder"],
                )
            )
        ax.set_yticks(np.arange(1, len(method_order) + 1))
        ax.set_yticklabels(
            [_rank_ytick(method, data[method]["n"]) for method in method_order]
        )
        for label, method in zip(ax.get_yticklabels(), method_order):
            label.set_linespacing(0.85)
            if method in REFERENCE_METHODS:
                label.set_style("italic")
                label.set_color("#6B6257")
        ax.set_ylim(len(method_order) + 0.5, 0.5)
        values = [mean for _, mean, _ in ordered]
        span = max(values) - min(values)
        ax.set_xlim(min(values) - 0.12 * span, max(values) + 1.15 * span)
        ax.set_xlabel(r"Stream FID $\downarrow$")
        ax.set_title(
            f"After Task {stage + 1}\n"
            + ("{01,02}" if stage == 1 else "{01,02,03}"),
            pad=8,
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    caption = fig.text(
        0.5,
        -0.02,
        "Shaded rows keep historical data and bound what no-storage replay can reach.",
        ha="center",
        va="top",
        fontsize=14,
        color="#555555",
    )
    fig.tight_layout(w_pad=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(
            out.with_suffix(suffix),
            bbox_inches="tight",
            bbox_extra_artists=(caption,),
            **kwargs,
        )
        written.append(out.with_suffix(suffix))
    plt.close(fig)
    return written


def render_stage_panels(
    data: dict[str, Any],
    task0: dict[str, float] | None,
    out: Path,
) -> None:
    """One panel per evaluation experiment: FID against a fixed reference set."""
    _apply_style()
    fig, axes = plt.subplots(
        1,
        len(EXPERIMENTS),
        figsize=(6.9, 2.5),
        sharex=True,
    )
    x = np.arange(len(EXPERIMENTS), dtype=float)
    present = [method for method in METHODS if method in data]
    offsets = dict(zip(present, np.linspace(-0.07, 0.07, max(len(present), 1))))

    for column, (ax, experiment) in enumerate(zip(axes, EXPERIMENTS)):
        for method in METHODS:
            if method not in data:
                continue
            row = data[method]
            series = row["by_stage"][experiment]
            kwargs, label = _method_style(method, row)
            # All methods branch from the shared task-0 checkpoint at stage 0.
            if column == 0 and task0 is not None:
                series = {
                    "mean": [task0["mean"], *series["mean"][1:]],
                    "se": [task0["se"], *series["se"][1:]],
                }
            _errorbar(
                ax,
                x + offsets[method],
                series,
                linestyle=STREAM_LINESTYLE,
                label=label if column == 0 else None,
                **kwargs,
            )
        if column == 0:
            if task0 is not None:
                ax.plot(
                    x[0],
                    task0["mean"],
                    marker="P",
                    markersize=6.0,
                    color="#111111",
                    zorder=10,
                    linestyle="none",
                )
                ax.annotate(
                    "shared\ntask-0 model",
                    xy=(x[0], task0["mean"]),
                    xytext=(4, 8),
                    textcoords="offset points",
                    fontsize=6.8,
                    color="#333333",
                )
            else:
                ax.annotate(
                    "stage 0\nnot evaluated",
                    xy=(x[0], 0.06),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    style="italic",
                    color="#999999",
                )
        ax.set_title(f"Evaluated on {experiment}", fontsize=9, pad=5)
        ax.set_xticks(x, [f"T{i}" for i in range(len(EXPERIMENTS))])
        ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"FID $\downarrow$")
    fig.supxlabel("Tasks completed", fontsize=10, y=0.02)
    _finish(
        fig,
        ax=axes[0],
        out=out,
        ncol=3,
        note=(
            "Each panel fixes the reference set, so changes reflect the model, "
            "not the evaluation composition."
        ),
    )


def main() -> None:
    args = parse_args()
    data = collect(args.results_root, args.masip_root)
    task0 = collect_task0(args.results_root)
    if task0 is None:
        print(
            "No task-0 evaluation found; run experiments/px/eval_task0_checkpoint.py "
            "to populate stage 0."
        )
    render_task_fid(data, args.out)
    render_stream_fid(data, task0, args.stream_out)
    rank_paths = render_stream_rank(data, args.rank_out)
    render_task0_retention(data, task0, args.retention_out)
    render_stage_panels(data, task0, args.stage_out)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "experiments": list(EXPERIMENTS),
                "task0": task0,
                "methods": data,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for stem in (args.out, args.stream_out, args.stage_out):
        print(f"Wrote {stem.with_suffix('.pdf')}")
        print(f"Wrote {stem.with_suffix('.png')}")
    for path in rank_paths:
        print(f"Wrote {path}")
    print(f"Wrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
