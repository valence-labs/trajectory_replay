#!/usr/bin/env python3
"""Summarize sequential Gaussian-ring CL runs across seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_sequential.common import REPO_ROOT, sequential_results_root

METHOD_STYLE = {
    "naive": ("Naive", "#003049", "X"),
    "generative": ("Generative replay", "#C1121F", "s"),
    "trajectory": ("Trajectory replay", "#457B9D", "o"),
}
POLICY_LINESTYLE = {
    "none": ":",
    "constant": "--",
    "equal": "-",
    "adaptive": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/runs/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/figures/cl_summary.png",
    )
    return parser.parse_args()


def collect_runs(runs_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "stream_metrics.json"
        if not (config_path.is_file() and metrics_path.is_file()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("smoke"):
            continue
        stream = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(stream) != int(config["data_meta"]["n_tasks"]):
            continue
        runs.append({"run_dir": run_dir, "config": config, "stream": stream})
    return runs


def _policy_label(config: dict) -> str:
    mode = config["lambda_mode"]
    if mode == "constant":
        return rf"constant $\lambda={float(config['lambda_constant']):g}$"
    if mode == "equal":
        return "equal-task"
    if mode == "adaptive":
        return "validation-selected"
    return "none"


def _aggregate_curves(runs: list[dict], metric: str) -> dict:
    buckets: dict[tuple[str, str, float | None], list[np.ndarray]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        key = (
            config["strategy"],
            config["lambda_mode"],
            config.get("lambda_constant"),
        )
        values = np.asarray(
            [record["metrics"][metric] for record in run["stream"]],
            dtype=np.float64,
        )
        buckets[key].append(values)
    return {
        key: (
            np.mean(np.stack(values), axis=0),
            np.std(np.stack(values), axis=0, ddof=1) / np.sqrt(len(values))
            if len(values) > 1
            else np.zeros_like(values[0]),
        )
        for key, values in buckets.items()
    }


def plot_summary(runs: list[dict], out: Path) -> None:
    if not runs:
        raise ValueError("No complete non-smoke runs found")
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
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.0))

    for ax, metric, ylabel in (
        (axes[0], "mass_kl_uniform", r"Mass $D_{\mathrm{KL}}$ $\downarrow$"),
        (axes[1], "energy_uniform_mixture", r"Mixture energy $\downarrow$"),
    ):
        curves = _aggregate_curves(runs, metric)
        for (strategy, mode, constant), (mean, std) in sorted(curves.items()):
            label, color, marker = METHOD_STYLE[strategy]
            policy = (
                rf"$\lambda={float(constant):g}$"
                if mode == "constant"
                else _policy_label({"lambda_mode": mode, "lambda_constant": constant})
            )
            x = np.arange(1, len(mean) + 1)
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.2,
                linestyle=POLICY_LINESTYLE[mode],
                label=f"{label}, {policy}",
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)
        ax.set_xlabel("Tasks seen")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(1, len(mean) + 1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Final fit-retention view. Constant schedules form each method's trade-off curve;
    # equal/adaptive policies are highlighted as individually labelled points.
    ax = axes[2]
    grouped: dict[tuple[str, str, float | None], list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        final = run["stream"][-1]["metrics"]
        grouped[
            (
                config["strategy"],
                config["lambda_mode"],
                config.get("lambda_constant"),
            )
        ].append(
            (
                final["retention_distribution_energy"],
                final["fit_distribution_energy"],
            )
        )
    for (strategy, mode, constant), values in sorted(grouped.items()):
        label, color, marker = METHOD_STYLE[strategy]
        mean = np.asarray(values).mean(axis=0)
        display = (
            rf"{label}, $\lambda={float(constant):g}$"
            if mode == "constant"
            else f"{label}, {_policy_label({'lambda_mode': mode, 'lambda_constant': constant})}"
        )
        ax.scatter(
            mean[0],
            mean[1],
            color=color,
            marker=marker,
            s=26,
            label=display,
            zorder=3,
        )
    ax.set_xlabel(r"Historical energy $\downarrow$")
    ax.set_ylabel(r"Latest-task energy $\downarrow$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=min(4, len(labels)),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
    )
    fig.tight_layout(pad=0.25, w_pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_forgetting(runs: list[dict], out: Path) -> None:
    """Appendix view: error increase relative to each task's acquisition point."""
    fig, ax = plt.subplots(figsize=(2.8, 2.1))
    curves = _aggregate_curves(runs, "forgetting_distribution_mean")
    for (strategy, mode, constant), (mean, std) in sorted(curves.items()):
        label, color, marker = METHOD_STYLE[strategy]
        policy = (
            rf"$\lambda={float(constant):g}$"
            if mode == "constant"
            else _policy_label({"lambda_mode": mode, "lambda_constant": constant})
        )
        x = np.arange(1, len(mean) + 1)
        ax.plot(
            x[1:],
            mean[1:],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.2,
            linestyle=POLICY_LINESTYLE[mode],
            label=f"{label}, {policy}",
        )
        ax.fill_between(
            x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color, alpha=0.1
        )
    ax.axhline(0.0, color="0.5", linewidth=0.7)
    ax.set_xlabel("Tasks seen")
    ax.set_ylabel(r"Historical energy increase $\downarrow$")
    ax.set_xticks(np.arange(2, len(mean) + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.25)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = sequential_results_root(REPO_ROOT)
    runs_dir = args.runs_dir or root / "runs"
    out = args.out or root / "figures" / "cl_summary.png"
    runs = collect_runs(runs_dir)
    plot_summary(runs, out)
    forgetting_out = out.with_name(f"{out.stem}_forgetting{out.suffix}")
    plot_forgetting(runs, forgetting_out)
    print(
        f"Wrote {out}, {forgetting_out}, and PDFs from {len(runs)} complete runs"
    )


if __name__ == "__main__":
    main()
