#!/usr/bin/env python3
"""Plot final generated task masses across sequential replay policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from experiments.sim_sequential.common import REPO_ROOT, sequential_results_root

METHODS = (
    ("generative", "Generative replay"),
    ("trajectory", "Trajectory replay"),
)
POLICIES = (
    ("target", None, "Target"),
    ("constant", 0.25, r"$\lambda=.25$"),
    ("constant", 0.5, r"$\lambda=.50$"),
    ("constant", 0.75, r"$\lambda=.75$"),
    ("constant", 0.9, r"$\lambda=.90$"),
    ("equal", None, "Equal-task"),
    ("adaptive", None, "Adaptive"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/runs_lr3e4.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--required-seeds", type=int, default=3)
    return parser.parse_args()


def _load_policy_mass(
    runs_dir: Path,
    strategy: str,
    mode: str,
    constant: float | None,
    *,
    required_seeds: int,
) -> np.ndarray | None:
    values: dict[int, np.ndarray] = {}
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        config_path = run_dir / "config.json"
        stream_path = run_dir / "stream_metrics.json"
        if not (config_path.is_file() and stream_path.is_file()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            config.get("strategy") != strategy
            or config.get("lambda_mode") != mode
            or config.get("lambda_constant") != constant
            or config.get("smoke")
        ):
            continue
        stream = json.loads(stream_path.read_text(encoding="utf-8"))
        if len(stream) != 5:
            continue
        mass = np.asarray(stream[-1]["metrics"]["mass"], dtype=np.float64)
        if mass.shape == (5,) and np.isfinite(mass).all():
            values[int(config["seed"])] = mass
    if len(values) < required_seeds:
        return None
    return np.stack([values[seed] for seed in sorted(values)]).mean(axis=0)


def build_matrices(
    runs_dir: Path, *, required_seeds: int
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for strategy, _ in METHODS:
        rows: list[np.ndarray] = []
        for mode, constant, _ in POLICIES:
            if mode == "target":
                rows.append(np.full(5, 0.2, dtype=np.float64))
                continue
            mass = _load_policy_mass(
                runs_dir,
                strategy,
                mode,
                constant,
                required_seeds=required_seeds,
            )
            rows.append(
                mass if mass is not None else np.full(5, np.nan, dtype=np.float64)
            )
        matrices[strategy] = np.stack(rows)
    return matrices


def plot_heatmap(
    matrices: dict[str, np.ndarray], out: Path, *, annotate: bool = True
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.35), sharey=True)
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad("0.92")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.2, vmax=0.75)
    image = None

    for ax, (strategy, title) in zip(axes, METHODS, strict=True):
        matrix = matrices[strategy]
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(5), [rf"$T_{{{task}}}$" for task in range(5)])
        ax.set_yticks(
            np.arange(len(POLICIES)),
            [label for _, _, label in POLICIES],
        )
        ax.tick_params(length=0)
        if annotate:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix[row, col]
                    if np.isfinite(value):
                        color = (
                            "white" if value > 0.43 or value < 0.055 else "black"
                        )
                        text = f"{100.0 * value:.0f}%"
                    else:
                        color = "0.45"
                        text = "—"
                    ax.text(
                        col,
                        row,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert image is not None
    fig.subplots_adjust(left=0.21, right=0.85, bottom=0.15, top=0.86, wspace=0.08)
    colorbar_ax = fig.add_axes((0.89, 0.14, 0.022, 0.74))
    colorbar = fig.colorbar(
        image,
        cax=colorbar_ax,
        ticks=(0.0, 0.2, 0.4, 0.6),
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    colorbar.set_label("Generated proportion", fontsize=9)
    colorbar.ax.tick_params(labelsize=8, length=2)
    fig.text(0.5, 0.01, "Task", ha="center", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_stacked_bars(matrices: dict[str, np.ndarray], out: Path) -> None:
    task_colors = ("#8F0000", "#C1121F", "#FDECC8", "#68A0C0", "#003049")
    row_order = (0, 6, 5, 4, 3, 2, 1)
    row_labels = [POLICIES[index][2] for index in row_order]
    y = np.arange(len(row_order))
    legend_handles = [
        Patch(facecolor=color, label=rf"$T_{{{task}}}$")
        for task, color in enumerate(task_colors)
    ]
    for strategy, _ in (*METHODS, METHODS[0]):
        method_out = (
            out
            if strategy == "trajectory"
            else out.with_name(f"{out.stem}_{strategy}{out.suffix}")
        )
        matrix = matrices[strategy][list(row_order)]
        fig = plt.figure(figsize=(2.05, 1.9))
        ax = fig.add_axes((0.34, 0.26, 0.57, 0.56))
        legend_ax = fig.add_axes((0.04, 0.87, 0.92, 0.09))
        legend_ax.axis("off")
        left = np.zeros(matrix.shape[0])
        for task, color in enumerate(task_colors):
            values = matrix[:, task]
            ax.barh(
                y,
                values,
                left=left,
                height=0.72,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=rf"$T_{{{task}}}$",
            )
            left += values
        ax.axhline(0.5, color="0.25", linewidth=0.65)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_yticks(y, row_labels)
        ax.tick_params(axis="y", length=0, labelleft=True)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.invert_yaxis()
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=5,
            frameon=False,
            handlelength=0.75,
            handletextpad=0.2,
            columnspacing=0.4,
            fontsize=7.5,
        )
        ax.set_xlabel("Generated proportion")
        method_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(method_out.with_suffix(".pdf"))
        fig.set_dpi(600)
        fig.canvas.draw()
        fig.canvas.print_png(method_out)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    root = sequential_results_root(REPO_ROOT)
    runs_dir = args.runs_dir or root / "runs_lr3e4"
    out = args.out or root / "figures" / "lambda_task_mass_heatmap.png"
    matrices = build_matrices(runs_dir, required_seeds=args.required_seeds)
    plot_heatmap(matrices, out)
    clean_out = out.with_name(f"{out.stem}_clean{out.suffix}")
    stacked_out = out.with_name("lambda_task_mass_stacked_bars.png")
    plot_heatmap(matrices, clean_out, annotate=False)
    plot_stacked_bars(matrices, stacked_out)
    generative_stacked_out = stacked_out.with_name(
        f"{stacked_out.stem}_generative{stacked_out.suffix}"
    )
    print(
        f"Wrote {out}, {clean_out}, {stacked_out}, {generative_stacked_out}, "
        "and corresponding PDFs"
    )


if __name__ == "__main__":
    main()
