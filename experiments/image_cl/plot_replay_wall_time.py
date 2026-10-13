#!/usr/bin/env python3
"""Accounted training time for generative vs trajectory replay (equal-task ODE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.image_cl.plot_backend_tradeoff_all import (
    DATASETS,
    collect,
    parse_args,
)


STRATEGIES = {
    "generative": ("Generative replay", "#C1121F"),
    "trajectory": ("Trajectory replay", "#457B9D"),
}
POLICY = "equal"
BACKEND = "ode"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.85,
        }
    )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / np.sqrt(array.size)),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        entry: dict[str, Any] = {}
        for strategy in STRATEGIES:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["strategy"] == strategy
                and row["policy"] == POLICY
                and row["backend"] == BACKEND
            ]
            if len(selected) != 3:
                raise ValueError(
                    f"{dataset}/{strategy}: expected three seeds, got {len(selected)}"
                )
            mean, se = _mean_se(
                [float(row["accounted_training_hours"]) for row in selected]
            )
            entry[strategy] = {
                "mean_hours": mean,
                "se_hours": se,
                "seeds": [int(row["seed"]) for row in selected],
                "sources": [row["timing_source"] for row in selected],
            }
        entry["speedup"] = (
            entry["generative"]["mean_hours"] / entry["trajectory"]["mean_hours"]
        )
        entry["hours_saved"] = (
            entry["generative"]["mean_hours"] - entry["trajectory"]["mean_hours"]
        )
        out[dataset] = entry
    return out


def render(summary: dict[str, dict[str, Any]], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.4, 2.7))
    x = np.arange(len(STRATEGIES), dtype=float)

    for ax, dataset in zip(axes, DATASETS, strict=True):
        entry = summary[dataset]
        means = [entry[s]["mean_hours"] for s in STRATEGIES]
        errors = [entry[s]["se_hours"] for s in STRATEGIES]
        colors = [STRATEGIES[s][1] for s in STRATEGIES]
        ax.bar(
            x,
            means,
            yerr=errors,
            width=0.62,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            capsize=3.0,
            error_kw={"elinewidth": 1.0},
            zorder=3,
        )
        for xi, value in zip(x, means):
            ax.text(
                xi,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
            )
        headroom = 1.32 if dataset == "imagenet100" else 1.26
        ax.set_ylim(0.0, max(means) * headroom)
        ax.annotate(
            rf"$\times${entry['speedup']:.2f} faster",
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=9,
            color="#457B9D",
        )
        ax.set_xticks(x, ["Gen.", "Traj."])
        ax.set_title(DATASETS[dataset][0], pad=5)
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accounted training time (h)")
    fig.tight_layout(w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect(args.results_root)
    summary = summarize(rows)
    out = args.out_dir / "replay_wall_time_equal.png"
    render(summary, out)
    (args.out_dir / "replay_wall_time_equal.json").write_text(
        json.dumps(
            {
                "policy": POLICY,
                "backend": BACKEND,
                "metric": "accounted training hours (all training calls + MeanFlow distill)",
                "datasets": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
