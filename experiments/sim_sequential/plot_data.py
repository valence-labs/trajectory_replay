#!/usr/bin/env python3
"""Plot the sequential Gaussian-ring tasks and their arrival order."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from experiments.sim_sequential.common import (
    DEFAULT_TASK_COLORS,
    REPO_ROOT,
    load_dataset_bundle,
    sequential_data_root,
    sequential_results_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--points-per-task", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/figures/task_ring.png",
    )
    return parser.parse_args()


def plot_ring(
    x: np.ndarray,
    task: np.ndarray,
    means: np.ndarray,
    *,
    out: Path,
    points_per_task: int,
    seed: int,
) -> None:
    if means.shape[0] != len(DEFAULT_TASK_COLORS):
        colors = tuple(plt.cm.tab10(i) for i in range(means.shape[0]))
    else:
        colors = DEFAULT_TASK_COLORS

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(2.8, 2.65))
    rng = np.random.default_rng(seed)

    for task_id, (mean, color) in enumerate(zip(means, colors, strict=True)):
        indices = np.flatnonzero(task == task_id)
        if indices.size > points_per_task:
            indices = rng.choice(indices, size=points_per_task, replace=False)
        ax.scatter(
            x[indices, 0],
            x[indices, 1],
            s=10,
            color=color,
            edgecolors="none",
            linewidths=0,
            alpha=0.72,
            rasterized=True,
            zorder=2,
        )
        ax.text(
            mean[0],
            mean[1],
            rf"$T_{{{task_id}}}$",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black" if task_id == 2 else "white",
            zorder=4,
        )

    # Curved arrows outside the clouds indicate top-first clockwise arrival.
    arrow_radius = np.linalg.norm(means[0, :2]) * 1.25
    unit = means[:, :2] / np.linalg.norm(means[:, :2], axis=1, keepdims=True)
    arrow_points = arrow_radius * unit
    for task_id in range(means.shape[0] - 1):
        arrow = FancyArrowPatch(
            arrow_points[task_id],
            arrow_points[task_id + 1],
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="0.35",
            zorder=1,
        )
        ax.add_patch(arrow)

    extent = float(np.max(np.abs(x[:, :2]))) + 0.35
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="none",
            label=rf"Task {i}",
        )
        for i, color in enumerate(colors)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=means.shape[0],
        frameon=False,
        handlelength=0.8,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or sequential_data_root(REPO_ROOT, dim=args.dim)
    splits, _ = load_dataset_bundle(data_dir)
    dataset = splits[args.split]
    out = args.out or sequential_results_root(REPO_ROOT) / "figures" / "task_ring.png"
    plot_ring(
        dataset.x,
        dataset.task,
        dataset.means,
        out=out,
        points_per_task=args.points_per_task,
        seed=args.seed,
    )
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
