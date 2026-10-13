#!/usr/bin/env python3
"""Plot FMNIST replay-backend quality versus model evaluations and wall time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
STRATEGIES = {
    "generative": ("Generative replay", "#c1121f"),
    "trajectory": ("Trajectory replay", "#457b9d"),
}
POLICIES = {
    "selected": ("Adaptive $\\lambda$", "solid", False),
    "equal": ("Equal-task", (0, (3.0, 2.0)), True),
}
BACKENDS = {
    "mf1": ("MF-1", 1),
    "mf4": ("MF-4", 4),
    "ode": ("ODE", 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "fmnist",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "figures",
    )
    return parser.parse_args()


def _run_name(
    strategy: str,
    policy: str,
    backend: str,
    seed: int,
) -> str:
    if backend == "ode":
        suffix = "_eq" if policy == "equal" else ""
        return f"{strategy}{suffix}_s{seed}"
    n = 1 if backend == "mf1" else 4
    equal = "_eq" if policy == "equal" else ""
    return f"{strategy}_mf_n{n}{equal}_tfix_s{seed}"


def _replay_evals(strategy: str, backend: str) -> float:
    """Replay-network evaluations per update, excluding MF optimization overhead."""
    n = BACKENDS[backend][1]
    if strategy == "generative":
        return float(n)
    # Trajectory replay adds one frozen-CFM velocity evaluation at the anchor.
    if backend != "ode":
        return float(n + 1)
    # With r~Uniform(0,1), E[ceil(10r)] = 5.5 rollout evaluations, plus v(r).
    return 6.5


def collect(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for policy in POLICIES:
            for backend in BACKENDS:
                for seed in SEEDS:
                    run = _run_name(strategy, policy, backend, seed)
                    run_dir = results_root / run
                    metrics_path = run_dir / "metrics.json"
                    timing_path = run_dir / "timing.json"
                    if not metrics_path.exists() or not timing_path.exists():
                        raise FileNotFoundError(
                            metrics_path if not metrics_path.exists() else timing_path
                        )
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    timing = json.loads(timing_path.read_text(encoding="utf-8"))
                    final = metrics["metrics_by_task"][-1]
                    train_calls = [
                        float(value)
                        for value in timing["single_ft_train_wall_seconds_all"]
                    ]
                    distill = [
                        float(value)
                        for value in timing.get(
                            "mean_flow_distill_wall_seconds_all", []
                        )
                    ]
                    rows.append(
                        {
                            "strategy": strategy,
                            "policy": policy,
                            "backend": backend,
                            "backend_label": BACKENDS[backend][0],
                            "seed": seed,
                            "replay_model_evals": _replay_evals(strategy, backend),
                            "final_fid": float(final["fid_stream_test"]),
                            "final_kld": float(final["stream_kld"]),
                            "total_wall_hours": (
                                float(timing["total_wall_seconds"]) / 3600.0
                            ),
                            "post_task_candidate_minutes": (
                                float(np.mean(train_calls[1:])) / 60.0
                            ),
                            "task0_meanflow_overhead_minutes": (
                                float(np.sum(distill)) / 60.0 if distill else 0.0
                            ),
                            "n_training_calls": len(train_calls),
                            "metrics_source": str(metrics_path),
                            "timing_source": str(timing_path),
                        }
                    )
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / np.sqrt(array.size)),
    )


def render(
    rows: list[dict[str, Any]],
    *,
    x_metric: str,
    out: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex="col")
    metric_rows = (
        ("final_fid", r"Final stream FID $\downarrow$", False),
        ("final_kld", r"Final stream KLD $\downarrow$", True),
    )

    for column, strategy in enumerate(STRATEGIES):
        method_label, color = STRATEGIES[strategy]
        axes[0, column].set_title(method_label, pad=6)
        for row_index, (quality, ylabel, log_y) in enumerate(metric_rows):
            ax = axes[row_index, column]
            for policy, (policy_label, linestyle, open_marker) in POLICIES.items():
                points: list[tuple[float, float, float, float, str]] = []
                for backend in ("mf1", "mf4", "ode"):
                    selected = [
                        row
                        for row in rows
                        if row["strategy"] == strategy
                        and row["policy"] == policy
                        and row["backend"] == backend
                    ]
                    x_mean, x_se = _mean_se(
                        [float(row[x_metric]) for row in selected]
                    )
                    y_mean, y_se = _mean_se(
                        [float(row[quality]) for row in selected]
                    )
                    points.append(
                        (
                            x_mean,
                            x_se,
                            y_mean,
                            y_se,
                            str(selected[0]["backend_label"]),
                        )
                    )
                points.sort(key=lambda item: item[0])
                ax.plot(
                    [point[0] for point in points],
                    [point[2] for point in points],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    zorder=2,
                )
                for x_mean, x_se, y_mean, y_se, backend_label in points:
                    ax.errorbar(
                        x_mean,
                        y_mean,
                        xerr=(x_se if x_metric == "total_wall_hours" else None),
                        yerr=y_se,
                        color=color,
                        marker="o",
                        markersize=6.5,
                        markerfacecolor="white" if open_marker else color,
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        linestyle="none",
                        elinewidth=0.9,
                        capsize=2.3,
                        zorder=3,
                    )
                    if policy == "selected":
                        ax.annotate(
                            backend_label,
                            (x_mean, y_mean),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=7.5,
                            color="#555555",
                        )
            if column == 0:
                ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(axis="y", color="0.87", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if x_metric == "replay_model_evals":
        xlabel = "Replay-model evaluations per update"
        for ax in axes[-1]:
            ax.set_xscale("log", base=2)
    else:
        xlabel = "Measured end-to-end wall time (h)"
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)

    policy_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            markerfacecolor=("white" if open_marker else "#555555"),
            linestyle=linestyle,
            markersize=6,
            label=label,
        )
        for label, linestyle, open_marker in POLICIES.values()
    ]
    legend = fig.legend(
        handles=policy_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.1,
    )
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=400,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.03,
    )
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.03,
    )
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "strategy",
        "policy",
        "backend",
        "backend_label",
        "seed",
        "replay_model_evals",
        "final_fid",
        "final_kld",
        "total_wall_hours",
        "post_task_candidate_minutes",
        "task0_meanflow_overhead_minutes",
        "n_training_calls",
        "metrics_source",
        "timing_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.results_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nfe_out = args.out_dir / "backend_quality_vs_replay_evals.png"
    wall_out = args.out_dir / "backend_quality_vs_wall_time.png"
    render(rows, x_metric="replay_model_evals", out=nfe_out)
    render(rows, x_metric="total_wall_hours", out=wall_out)
    csv_path = args.out_dir / "backend_quality_compute.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {nfe_out} and {nfe_out.with_suffix('.pdf')}")
    print(f"Wrote {wall_out} and {wall_out.with_suffix('.pdf')}")
    print(f"Wrote {csv_path} ({len(rows)} seed-level rows)")


if __name__ == "__main__":
    main()
