#!/usr/bin/env python3
"""Plot the final stability-plasticity trade-off across replay policies."""

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

from experiments.sim_sequential.common import (
    REPO_ROOT,
    load_dataset_bundle,
    sequential_data_root,
    sequential_results_root,
)
from experiments.sim_sequential.metrics import evaluate_sequential_samples

METHODS = {
    "generative": ("Generative replay", "#C1121F"),
    "trajectory": ("Trajectory replay", "#457B9D"),
}
POLICY_MARKERS = {"equal": "*", "adaptive": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/runs_lr3e4.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--required-seeds",
        type=int,
        default=3,
        help="Only show policies with this many complete finite seeds.",
    )
    return parser.parse_args()


def load_points(
    runs_dir: Path, *, required_seeds: int
) -> dict[str, dict[tuple[str, float | None], np.ndarray]]:
    grouped: dict[
        tuple[str, str, float | None], list[tuple[int, float, float]]
    ] = defaultdict(list)
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        config_path = run_dir / "config.json"
        stream_path = run_dir / "stream_metrics.json"
        if not (config_path.is_file() and stream_path.is_file()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        strategy = config.get("strategy")
        if strategy not in METHODS or config.get("smoke"):
            continue
        stream = json.loads(stream_path.read_text(encoding="utf-8"))
        if len(stream) != 5:
            continue
        final = stream[-1]["metrics"]
        retention = float(final["retention_distribution_energy"])
        fit = float(final["fit_distribution_energy"])
        if not (math.isfinite(retention) and math.isfinite(fit)):
            continue
        key = (
            strategy,
            config["lambda_mode"],
            config.get("lambda_constant"),
        )
        grouped[key].append((int(config["seed"]), retention, fit))

    points: dict[str, dict[tuple[str, float | None], np.ndarray]] = {
        strategy: {} for strategy in METHODS
    }
    for (strategy, mode, constant), values in grouped.items():
        # Deduplicate by seed defensively if a result root contains reruns.
        by_seed = {seed: (retention, fit) for seed, retention, fit in values}
        if len(by_seed) < required_seeds:
            continue
        selected = [by_seed[seed] for seed in sorted(by_seed)]
        points[strategy][(mode, constant)] = np.asarray(selected, dtype=np.float64)
    return points


def oracle_point(data_dir: Path) -> tuple[float, float]:
    splits, _ = load_dataset_bundle(data_dir)
    val, test = splits["val"], splits["test"]
    policy = np.full(test.n_tasks, 1.0 / test.n_tasks)
    metrics = evaluate_sequential_samples(
        val.x,
        test,
        n_seen=test.n_tasks,
        policy_weights=policy,
    )
    return (
        float(metrics["retention_distribution_energy"]),
        float(metrics["fit_distribution_energy"]),
    )


def plot_tradeoff(
    points: dict[str, dict[tuple[str, float | None], np.ndarray]],
    *,
    oracle: tuple[float, float],
    out: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(3.05, 2.55))

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
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )
            for (lam, values), mean in zip(constants, means, strict=True):
                std = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
                ax.errorbar(
                    mean[0],
                    mean[1],
                    xerr=std[0],
                    yerr=std[1],
                    fmt="o",
                    color=color,
                    markersize=4.5,
                    elinewidth=0.7,
                    capsize=1.8,
                    zorder=3,
                )
                ax.scatter(
                    values[:, 0],
                    values[:, 1],
                    color=color,
                    s=8,
                    alpha=0.28,
                    zorder=2,
                )
                if strategy == "trajectory":
                    ax.annotate(
                        f"{lam:g}",
                        mean,
                        xytext=(4, 3),
                        textcoords="offset points",
                        color=color,
                        fontsize=7.5,
                    )

        for mode, marker in POLICY_MARKERS.items():
            values = method_points.get((mode, None))
            if values is None:
                continue
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            ax.errorbar(
                mean[0],
                mean[1],
                xerr=std[0],
                yerr=std[1],
                fmt=marker,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=8 if mode == "equal" else 5.5,
                elinewidth=0.7,
                capsize=1.8,
                zorder=5,
            )
            ax.scatter(
                values[:, 0],
                values[:, 1],
                color=color,
                s=8,
                alpha=0.28,
                zorder=2,
            )

    ax.scatter(
        oracle[0],
        oracle[1],
        marker="o",
        s=90,
        facecolor="none",
        edgecolor="black",
        linewidth=1.0,
        zorder=4,
    )
    ax.annotate(
        "Uniform oracle",
        oracle,
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=7.5,
        va="center",
    )

    ax.set_xlabel(r"Historical-distribution energy $\downarrow$")
    ax.set_ylabel(r"Newest-task energy $\downarrow$")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=4,
            label=label.replace(" replay", ""),
        )
        for label, color in METHODS.values()
    ]
    available_modes = {
        mode for method_points in points.values() for mode, _ in method_points
    }
    policy_handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            marker="o",
            linestyle="-",
            markersize=4,
            label=r"constant $\lambda$",
        )
    ]
    if "equal" in available_modes:
        policy_handles.append(
            Line2D(
            [0],
            [0],
            color="0.35",
            marker="*",
            linestyle="none",
            markersize=7,
            label="equal-task",
            )
        )
    if "adaptive" in available_modes:
        policy_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="D",
                linestyle="none",
                markersize=4,
                label="adaptive",
            )
        )
    ax.legend(
        handles=method_handles + policy_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.tight_layout(pad=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = sequential_results_root(REPO_ROOT)
    runs_dir = args.runs_dir or root / "runs_lr3e4"
    data_dir = args.data_dir or sequential_data_root(REPO_ROOT, dim=2)
    out = args.out or root / "figures" / "lambda_fit_retention.png"
    points = load_points(runs_dir, required_seeds=args.required_seeds)
    plot_tradeoff(points, oracle=oracle_point(data_dir), out=out)
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
