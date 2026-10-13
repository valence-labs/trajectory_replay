#!/usr/bin/env python3
"""Plot final stability-plasticity trade-offs across replay policies."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from trajectory_replay.eval.metrics import energy_distance


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "generative": ("Generative replay", "#c1121f"),
    "trajectory": ("Trajectory replay", "#457b9d"),
}
POLICY_MARKERS = {"equal": "*", "adaptive": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "sim_sequential" / "runs_lr3e4",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "sim_sequential" / "data" / "dim2",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "sim_sequential"
            / "figures"
            / "lambda_fit_retention.png"
        ),
    )
    parser.add_argument("--required-seeds", type=int, default=3)
    return parser.parse_args()


def load_points(
    runs_root: Path,
    *,
    required_seeds: int,
) -> dict[str, dict[tuple[str, float | None], np.ndarray]]:
    grouped: dict[
        tuple[str, str, float | None], list[tuple[int, float, float]]
    ] = defaultdict(list)
    for run_dir in sorted(runs_root.iterdir()):
        config_path = run_dir / "config.json"
        stream_path = run_dir / "stream_metrics.json"
        if not (config_path.is_file() and stream_path.is_file()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        strategy = config.get("strategy")
        if strategy not in METHODS or config.get("smoke"):
            continue
        stream = json.loads(stream_path.read_text(encoding="utf-8"))
        if len(stream) != 5 or int(stream[-1]["task_id"]) != 4:
            continue
        final = stream[-1]["metrics"]
        retention = float(final["retention_distribution_energy"])
        fit = float(final["fit_distribution_energy"])
        if not (math.isfinite(retention) and math.isfinite(fit)):
            continue
        grouped[
            (
                strategy,
                str(config["lambda_mode"]),
                config.get("lambda_constant"),
            )
        ].append((int(config["seed"]), retention, fit))

    points: dict[str, dict[tuple[str, float | None], np.ndarray]] = {
        strategy: {} for strategy in METHODS
    }
    for (strategy, mode, constant), values in grouped.items():
        by_seed = {seed: (retention, fit) for seed, retention, fit in values}
        if len(by_seed) < required_seeds:
            continue
        selected = [by_seed[seed] for seed in sorted(by_seed)[:required_seeds]]
        points[strategy][(mode, constant)] = np.asarray(selected, dtype=np.float64)
    return points


def oracle_point(data_root: Path) -> tuple[float, float]:
    validation = np.load(data_root / "val.npz")
    test = np.load(data_root / "test.npz")
    generated = validation["x"]
    historical = test["x"][test["task"] < 4]
    newest = test["x"][test["task"] == 4]
    return (
        energy_distance(historical, generated),
        energy_distance(newest, generated),
    )


def render(
    points: dict[str, dict[tuple[str, float | None], np.ndarray]],
    *,
    oracle: tuple[float, float],
    out: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(4.15, 2.75))
    constant_means: dict[float, list[np.ndarray]] = defaultdict(list)

    for strategy, (method_label, color) in METHODS.items():
        method_points = points[strategy]
        constants = sorted(
            (
                (float(constant), values)
                for (mode, constant), values in method_points.items()
                if mode == "constant" and constant is not None
            ),
            key=lambda item: item[0],
        )
        if constants:
            means = np.asarray([values.mean(axis=0) for _, values in constants])
            ax.plot(
                means[:, 0],
                means[:, 1],
                color=color,
                linewidth=2.0,
                alpha=0.82,
                zorder=2,
            )
            for (lam, values), mean in zip(constants, means, strict=True):
                std = values.std(axis=0, ddof=1)
                constant_means[lam].append(mean)
                ax.scatter(
                    values[:, 0],
                    values[:, 1],
                    color=color,
                    s=25,
                    alpha=0.25,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
                ax.errorbar(
                    mean[0],
                    mean[1],
                    xerr=std[0],
                    yerr=std[1],
                    fmt="o",
                    color=color,
                    markersize=8.0,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    elinewidth=1.2,
                    capsize=2.8,
                    zorder=3,
                )

        for mode, marker in POLICY_MARKERS.items():
            values = method_points.get((mode, None))
            if values is None:
                continue
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            ax.scatter(
                values[:, 0],
                values[:, 1],
                color=color,
                s=25,
                alpha=0.25,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            ax.errorbar(
                mean[0],
                mean[1],
                xerr=std[0],
                yerr=std[1],
                fmt=marker,
                color=color,
                markerfacecolor=("white" if mode == "adaptive" else color),
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=12 if mode == "equal" else 8.0,
                elinewidth=1.2,
                capsize=2.8,
                zorder=5 if mode == "adaptive" else 4,
            )

    label_offsets = {
        0.25: (8, 4),
        0.5: (8, 4),
        0.75: (8, 4),
        0.9: (8, 2),
    }
    for lam, means in constant_means.items():
        center = np.mean(np.stack(means), axis=0)
        ax.annotate(
            f"{lam:g}",
            center,
            xytext=label_offsets[lam],
            textcoords="offset points",
            fontsize=10.5,
            color="#555555",
        )

    ax.scatter(
        oracle[0],
        oracle[1],
        marker="o",
        s=185,
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.6,
        zorder=3,
    )
    ax.annotate(
        "Uniform target",
        xy=oracle,
        xytext=(36, 28),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.7),
        fontsize=10.5,
        ha="left",
        va="center",
        color="#333333",
    )

    ax.set_xlabel(r"Historical-distribution energy $\downarrow$")
    ax.set_ylabel(r"Newest-task energy $\downarrow$")
    ax.set_xlim(0.0, 3.65)
    ax.set_ylim(0.0, 4.25)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=color, marker="o", markersize=6.5, label=label)
        for label, color in METHODS.values()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="o",
                markersize=6.5,
                label=r"Constant $\lambda$",
            ),
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="*",
                linestyle="none",
                markersize=10,
                label="Equal-task",
            ),
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="D",
                markerfacecolor="white",
                linestyle="none",
                markersize=6.5,
                label="Adaptive",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        ncol=1,
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.18,
    )
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    points = load_points(args.runs_root, required_seeds=args.required_seeds)
    render(points, oracle=oracle_point(args.data_root), out=args.out)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
