#!/usr/bin/env python3
"""Plot PBMC sequential held-out performance and adaptive-lambda schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.tx.audit_final_cl_results import ROOT, SEEDS, SPECS, run_path


PBMC_SPEC = next(spec for spec in SPECS if spec.name == "pbmc")
TASK_IDS = (1, 2, 3, 4)
TASK_DONORS = ("Donor1", "Donor4", "Donor9", "Donor12")

METHODS = {
    "naive": {
        "run_key": "standard_p1c0",
        "label": "Naive fine-tuning",
        "color": "#4A4A4A",
        "marker": "X",
        "linestyle": "solid",
        "open": False,
        "zorder": 2,
    },
    "real_adaptive": {
        "run_key": "real_grid",
        "label": r"Real replay, adaptive $\lambda$",
        "color": "#2A9D8F",
        "marker": "*",
        "linestyle": "solid",
        "open": False,
        "zorder": 5,
    },
    "real_equal": {
        "run_key": "real_equal",
        "label": "Real replay, equal-task",
        "color": "#2A9D8F",
        "marker": "*",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 4,
    },
    "gen_adaptive": {
        "run_key": "generative_ode_grid",
        "label": r"Gen. replay (ODE), adaptive $\lambda$",
        "color": "#C1121F",
        "marker": "s",
        "linestyle": "solid",
        "open": False,
        "zorder": 4,
    },
    "gen_equal": {
        "run_key": "generative_ode_equal",
        "label": "Gen. replay (ODE), equal-task",
        "color": "#C1121F",
        "marker": "s",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 3,
    },
    "traj_adaptive": {
        "run_key": "trajectory_ode_grid",
        "label": r"Traj. replay (ODE), adaptive $\lambda$",
        "color": "#457B9D",
        "marker": "o",
        "linestyle": "solid",
        "open": False,
        "zorder": 6,
    },
    "traj_equal": {
        "run_key": "trajectory_ode_equal",
        "label": "Traj. replay (ODE), equal-task",
        "color": "#457B9D",
        "marker": "o",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 5,
    },
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
        }
    )


def _summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        values.mean(axis=0),
        values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]),
    )


def _load_heldout_curve(run: Path) -> list[float]:
    path = run / "heldout_curve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("eval_provenance") or {}
    if int(provenance.get("gen_per_stratum", -1)) != 500:
        raise ValueError(f"{path}: expected K=500 held-out evaluation")
    by_task = {
        int(row["task_id"]): row["heldout"]
        for row in payload["heldout_curve"]
    }
    values = [
        float(by_task[task_id]["pearson_delta_macro"])
        for task_id in TASK_IDS
    ]
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite held-out Pearson Delta")
    return values


def _load_lambda_schedule(run: Path) -> list[float]:
    values: list[float] = []
    for task_id in TASK_IDS:
        path = run / "tasks" / f"{task_id:02d}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("chosen_lambda")
        if value is None or not payload.get("selection_evaluated"):
            raise ValueError(f"{path}: missing adaptively selected lambda")
        values.append(float(value))
    return values


def collect() -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for method, spec in METHODS.items():
        seed_values = np.asarray(
            [
                _load_heldout_curve(run_path(PBMC_SPEC, spec["run_key"], seed))
                for seed in SEEDS
            ],
            dtype=float,
        )
        curves[method] = {
            "values": seed_values,
            "sources": [
                str(
                    run_path(PBMC_SPEC, spec["run_key"], seed)
                    / "heldout_curve.json"
                )
                for seed in SEEDS
            ],
        }

    schedules: dict[str, Any] = {}
    for method in ("real_adaptive", "gen_adaptive", "traj_adaptive"):
        spec = METHODS[method]
        seed_values = np.asarray(
            [
                _load_lambda_schedule(run_path(PBMC_SPEC, spec["run_key"], seed))
                for seed in SEEDS
            ],
            dtype=float,
        )
        schedules[method] = {
            "values": seed_values,
            "sources": [
                str(run_path(PBMC_SPEC, spec["run_key"], seed) / "tasks")
                for seed in SEEDS
            ],
        }
    return {"curves": curves, "schedules": schedules}


def _plot_method(
    ax: plt.Axes,
    values: np.ndarray,
    method: str,
) -> None:
    spec = METHODS[method]
    mean, se = _summary(values)
    x = np.arange(1, len(mean) + 1, dtype=float)
    ax.fill_between(
        x,
        mean - se,
        mean + se,
        color=spec["color"],
        alpha=0.11,
        linewidth=0,
        zorder=spec["zorder"] - 1,
    )
    ax.plot(
        x,
        mean,
        color=spec["color"],
        marker=spec["marker"],
        markersize=8.5 if method.startswith("real") else 6.8,
        markerfacecolor="white" if spec["open"] else spec["color"],
        markeredgecolor=spec["color"],
        markeredgewidth=1.25 if spec["open"] else 0.7,
        linestyle=spec["linestyle"],
        linewidth=2.0 if method.startswith(("real", "traj")) else 1.7,
        label=spec["label"],
        zorder=spec["zorder"],
    )


def render_performance(data: dict[str, Any], out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    for method in METHODS:
        _plot_method(ax, data["curves"][method]["values"], method)

    x = np.arange(1, len(TASK_IDS) + 1)
    ax.set_xticks(
        x,
        [rf"$T_{task}$" + f"\n{donor}" for task, donor in zip(TASK_IDS, TASK_DONORS)],
    )
    ax.set_xlim(0.72, len(TASK_IDS) + 0.28)
    ax.set_title("Parse-PBMC donor stream", pad=7)
    ax.set_xlabel("Tasks completed")
    ax.set_ylabel(r"Sequential held-out Pearson $\Delta$ $\uparrow$")
    ax.grid(axis="y", color="0.87", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHODS[method]["color"],
            marker=METHODS[method]["marker"],
            markerfacecolor=METHODS[method]["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.7,
            markersize=6.5 if method == "real_adaptive" else 5.3,
            label=label,
        )
        for method, label in (
            ("naive", "Naive FT"),
            ("real_adaptive", "Real replay"),
            ("gen_adaptive", "Generative (ODE)"),
            ("traj_adaptive", "Trajectory (ODE)"),
        )
    ]
    policy_handles = [
        Line2D(
            [0],
            [0],
            color="0.38",
            marker="o",
            markerfacecolor="0.38" if adaptive else "white",
            markeredgecolor="0.38",
            linewidth=1.6,
            linestyle="solid" if adaptive else (0, (3.0, 2.0)),
            markersize=5.0,
            label=label,
        )
        for adaptive, label in (
            (True, r"Adaptive $\lambda$"),
            (False, "Equal-task"),
        )
    ]
    legend = fig.legend(
        handles=method_handles + policy_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        handlelength=1.6,
        columnspacing=0.65,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.68)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=400,
        bbox_extra_artists=(legend,),
    )
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)


def render_lambda_schedule(data: dict[str, Any], out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    x = np.arange(1, len(TASK_IDS) + 1, dtype=float)
    equal = x / (x + 1.0)
    ax.plot(
        x,
        equal,
        color="#555555",
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.8,
        marker="*",
        markersize=8.5,
        markerfacecolor="white",
        markeredgecolor="#555555",
        markeredgewidth=1.0,
        label=r"Equal-task $\lambda_k=k/(k+1)$",
        zorder=2,
    )

    offsets = {
        "real_adaptive": -0.045,
        "gen_adaptive": 0.0,
        "traj_adaptive": 0.045,
    }
    labels = {
        "real_adaptive": "Real replay",
        "gen_adaptive": "Generative replay (ODE)",
        "traj_adaptive": "Trajectory replay (ODE)",
    }
    for method, offset in offsets.items():
        spec = METHODS[method]
        values = data["schedules"][method]["values"]
        mean, se = _summary(values)
        ax.errorbar(
            x + offset,
            mean,
            yerr=se,
            color=spec["color"],
            marker=spec["marker"],
            markersize=8.5 if method.startswith("real") else 7.2,
            markerfacecolor=spec["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            elinewidth=1.1,
            capsize=2.8,
            capthick=1.0,
            label=labels[method],
            zorder=4,
        )

    ax.set_xticks(
        x,
        [
            rf"after $T_{task}$" + f"\n{donor}"
            for task, donor in zip(TASK_IDS, TASK_DONORS)
        ],
    )
    ax.set_xlim(0.7, len(TASK_IDS) + 0.3)
    ax.set_ylim(0.19, 0.86)
    ax.set_yticks(
        [0.25, 0.5, 2 / 3, 0.75, 0.8],
        ["0.25", "0.50", "0.67", "0.75", "0.80"],
    )
    ax.set_title("Parse-PBMC donor stream", pad=7)
    ax.set_xlabel("Sequential update")
    ax.set_ylabel(r"Mean adaptive $\lambda_k$")
    ax.grid(axis="y", color="0.87", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    handles, legend_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        handlelength=1.7,
        columnspacing=0.75,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.68)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=400,
        bbox_extra_artists=(legend,),
    )
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)


def write_summary(data: dict[str, Any], out: Path) -> None:
    payload: dict[str, Any] = {
        "dataset": "pbmc",
        "tasks": [
            {"task_id": task, "donor": donor}
            for task, donor in zip(TASK_IDS, TASK_DONORS)
        ],
        "seeds": list(SEEDS),
        "evaluation": {
            "metric": "heldout pearson_delta_macro",
            "generated_cells_per_non_control_stratum": 500,
            "error_bars": "standard error across seeds",
        },
        "methods": {},
        "adaptive_lambda": {},
    }
    for method, row in data["curves"].items():
        mean, se = _summary(row["values"])
        payload["methods"][method] = {
            "run_key": METHODS[method]["run_key"],
            "mean": mean.tolist(),
            "se": se.tolist(),
            "sources": row["sources"],
        }
    for method, row in data["schedules"].items():
        mean, se = _summary(row["values"])
        payload["adaptive_lambda"][method] = {
            "run_key": METHODS[method]["run_key"],
            "mean": mean.tolist(),
            "se": se.tolist(),
            "sources": row["sources"],
        }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results_dir" / "tx_results" / "figures",
    )
    args = parser.parse_args()

    data = collect()
    performance_out = args.out_dir / "pbmc_stream_pearson_by_task.png"
    lambda_out = args.out_dir / "pbmc_adaptive_lambda_by_task.png"
    render_performance(data, performance_out)
    render_lambda_schedule(data, lambda_out)
    write_summary(data, args.out_dir / "pbmc_stream_metrics_by_task.json")
    print(performance_out)
    print(lambda_out)


if __name__ == "__main__":
    main()
