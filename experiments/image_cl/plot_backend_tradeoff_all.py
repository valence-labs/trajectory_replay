#!/usr/bin/env python3
"""Image-CL backend quality versus replay evaluations and normalized wall time."""

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
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
DATASETS = {
    "fmnist": ("Fashion-MNIST", 4),
    "cifar10": ("CIFAR-10", 4),
    "imagenet100": ("ImageNet-100", 2),
}
STRATEGIES = {
    "generative": ("Generative replay", "#c1121f"),
    "trajectory": ("Trajectory replay", "#457b9d"),
}
POLICIES = {
    "adaptive": ("Adaptive $\\lambda$", "solid", False),
    "equal": ("Equal-task", (0, (3.0, 2.0)), True),
}
BACKENDS = {
    "mf1": ("MF-1", "^"),
    "mf4": ("MF-4", "s"),
    "ode": ("ODE", "o"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "figures",
    )
    return parser.parse_args()


def _run_name(
    dataset: str,
    strategy: str,
    policy: str,
    backend: str,
    seed: int,
) -> str:
    equal = policy == "equal"
    if backend == "ode":
        if equal:
            suffix = "_eq_eq" if dataset == "imagenet100" else "_eq"
        else:
            suffix = "" if dataset == "fmnist" else "_g3"
        return f"{strategy}{suffix}_s{seed}"

    n = 1 if backend == "mf1" else 4
    if equal:
        mode = "_eq"
    else:
        mode = "" if dataset == "fmnist" else "_g3"
    corrected = "_tfix" if dataset == "fmnist" else ""
    return f"{strategy}_mf_n{n}{mode}{corrected}_s{seed}"


def _replay_evals(
    strategy: str,
    backend: str,
    replay_step_size: float,
) -> float:
    if backend == "mf1":
        rollout = 1.0
    elif backend == "mf4":
        rollout = 4.0
    else:
        rollout = float(round(1.0 / replay_step_size))
    if strategy == "generative":
        return rollout
    if backend != "ode":
        return rollout + 1.0
    # r~U(0,1): E[ceil(Mr)] = (M+1)/2, plus one frozen-field v(r) call.
    return (rollout + 3.0) / 2.0


def collect(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, (_, n_updates) in DATASETS.items():
        for strategy in STRATEGIES:
            for policy in POLICIES:
                for backend in BACKENDS:
                    for seed in SEEDS:
                        run = _run_name(
                            dataset, strategy, policy, backend, seed
                        )
                        run_dir = results_root / dataset / run
                        metrics_path = run_dir / "metrics.json"
                        timing_path = run_dir / "timing.json"
                        if not metrics_path.exists() or not timing_path.exists():
                            raise FileNotFoundError(
                                metrics_path
                                if not metrics_path.exists()
                                else timing_path
                            )
                        metrics = json.loads(
                            metrics_path.read_text(encoding="utf-8")
                        )
                        timing = json.loads(
                            timing_path.read_text(encoding="utf-8")
                        )
                        final = metrics["metrics_by_task"][-1]
                        train_calls = np.asarray(
                            timing["single_ft_train_wall_seconds_all"],
                            dtype=float,
                        )
                        mf_setup = np.asarray(
                            timing.get(
                                "mean_flow_distill_wall_seconds_all", []
                            ),
                            dtype=float,
                        )
                        replay_step = float(
                            metrics.get(
                                "replay_ode_step_size",
                                metrics["ode_step_size"],
                            )
                        )
                        post_update_hours = float(train_calls[1:].mean() / 3600.0)
                        setup_hours = (
                            float(mf_setup.sum() / 3600.0)
                            if mf_setup.size
                            else 0.0
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "strategy": strategy,
                                "policy": policy,
                                "backend": backend,
                                "seed": seed,
                                "replay_model_evals": _replay_evals(
                                    strategy, backend, replay_step
                                ),
                                "final_fid": float(final["fid_stream_test"]),
                                "final_kld": (
                                    None
                                    if final.get("stream_kld") is None
                                    else float(final["stream_kld"])
                                ),
                                "post_update_hours": post_update_hours,
                                "meanflow_setup_hours": setup_hours,
                                "amortized_setup_hours": setup_hours / n_updates,
                                "amortized_update_hours": (
                                    post_update_hours + setup_hours / n_updates
                                ),
                                "accounted_training_hours": (
                                    float(train_calls.sum() + mf_setup.sum())
                                    / 3600.0
                                ),
                                "total_wall_hours": (
                                    float(timing["total_wall_seconds"]) / 3600.0
                                ),
                                "n_training_calls": int(train_calls.size),
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
            "font.size": 10.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
        }
    )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / np.sqrt(array.size)),
    )


def _group(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    strategy: str,
    policy: str,
    backend: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["strategy"] == strategy
        and row["policy"] == policy
        and row["backend"] == backend
    ]


def _performance_legend(fig: plt.Figure) -> Any:
    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=label)
        for label, color in STRATEGIES.values()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=linestyle,
                marker="o",
                markerfacecolor="white" if open_marker else "#555555",
                label=label,
            )
            for label, linestyle, open_marker in POLICIES.values()
        ]
    )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="none",
                marker=marker,
                label=label,
            )
            for label, marker in BACKENDS.values()
        ]
    )
    return fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.35,
    )


def render_quality(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    datasets: tuple[str, ...],
    out: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(2.9 * len(datasets), 2.9),
        squeeze=False,
    )
    for panel, dataset in enumerate(datasets):
        ax = axes[0, panel]
        for strategy, (_, color) in STRATEGIES.items():
            for policy, (_, linestyle, open_marker) in POLICIES.items():
                points: list[tuple[float, float, float, str, str]] = []
                for backend in ("mf1", "mf4", "ode"):
                    selected = _group(
                        rows,
                        dataset=dataset,
                        strategy=strategy,
                        policy=policy,
                        backend=backend,
                    )
                    x = float(selected[0]["replay_model_evals"])
                    values = [
                        float(row[metric])
                        for row in selected
                        if row[metric] is not None
                    ]
                    mean, se = _mean_se(values)
                    points.append(
                        (x, mean, se, BACKENDS[backend][0], BACKENDS[backend][1])
                    )
                points.sort(key=lambda item: item[0])
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    zorder=2,
                )
                for x, mean, se, _backend_label, marker in points:
                    ax.errorbar(
                        x,
                        mean,
                        yerr=se,
                        color=color,
                        marker=marker,
                        markersize=7,
                        markerfacecolor="white" if open_marker else color,
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        linestyle="none",
                        elinewidth=1.0,
                        capsize=2.5,
                        zorder=3,
                    )
        ax.set_xscale("log", base=2)
        ax.set_title(DATASETS[dataset][0], pad=6)
        ax.set_xlabel("Replay-field evaluations per update")
        if panel == 0:
            ax.set_ylabel(
                r"Final stream FID $\downarrow$"
                if metric == "final_fid"
                else r"Final stream KLD $\downarrow$"
            )
        if metric == "final_kld":
            ax.set_yscale("log")
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend = _performance_legend(fig)
    fig.tight_layout(w_pad=1.0)
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


def render_wall(rows: list[dict[str, Any]], out: Path) -> None:
    """Equal-task only: post-T0 update time plus amortized MF setup."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=False)
    backends = ("ode", "mf1", "mf4")
    x = np.arange(len(backends), dtype=float)
    width = 0.34

    for panel, (ax, dataset) in enumerate(
        zip(axes, DATASETS, strict=True)
    ):
        for strategy_index, (strategy, (label, color)) in enumerate(
            STRATEGIES.items()
        ):
            positions = x + (strategy_index - 0.5) * width
            update_means: list[float] = []
            update_se: list[float] = []
            setup_means: list[float] = []
            for backend in backends:
                selected = _group(
                    rows,
                    dataset=dataset,
                    strategy=strategy,
                    policy="equal",
                    backend=backend,
                )
                mean, se = _mean_se(
                    [float(row["post_update_hours"]) for row in selected]
                )
                setup, _ = _mean_se(
                    [float(row["amortized_setup_hours"]) for row in selected]
                )
                update_means.append(mean)
                update_se.append(se)
                setup_means.append(setup)
            ax.bar(
                positions,
                update_means,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                yerr=update_se,
                error_kw=dict(elinewidth=1.0, capsize=2.5),
                label=label,
                zorder=3,
            )
            ax.bar(
                positions,
                setup_means,
                bottom=update_means,
                width=width,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                hatch="///",
                zorder=3,
            )
        ax.set_xticks(x, [BACKENDS[backend][0] for backend in backends])
        ax.set_title(DATASETS[dataset][0], pad=6)
        ax.set_xlabel("Replay backend")
        if panel == 0:
            ax.set_ylabel("Amortized wall time per update (h)")
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in STRATEGIES.values()
    ]
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch="///",
            label="Task-0 MF setup (amortized)",
        )
    )
    legend = fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        handlelength=1.3,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.tight_layout(w_pad=1.0)
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


def render_relative_fid(rows: list[dict[str, Any]], out: Path) -> None:
    """Paired seed-level FID ratios relative to the matched ODE policy."""
    _style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.4, 2.9),
        sharey=True,
    )
    for panel, (ax, dataset) in enumerate(
        zip(axes, DATASETS, strict=True)
    ):
        for strategy, (_, color) in STRATEGIES.items():
            for policy, (_, linestyle, open_marker) in POLICIES.items():
                ode_rows = _group(
                    rows,
                    dataset=dataset,
                    strategy=strategy,
                    policy=policy,
                    backend="ode",
                )
                ode_by_seed = {
                    int(row["seed"]): float(row["final_fid"])
                    for row in ode_rows
                }
                points: list[tuple[float, float, float, str]] = []
                for backend in ("mf1", "mf4", "ode"):
                    selected = _group(
                        rows,
                        dataset=dataset,
                        strategy=strategy,
                        policy=policy,
                        backend=backend,
                    )
                    ratios = [
                        float(row["final_fid"]) / ode_by_seed[int(row["seed"])]
                        for row in selected
                    ]
                    mean, se = _mean_se(ratios)
                    points.append(
                        (
                            float(selected[0]["replay_model_evals"]),
                            mean,
                            se,
                            BACKENDS[backend][1],
                        )
                    )
                points.sort(key=lambda item: item[0])
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    zorder=2,
                )
                for x, mean, se, marker in points:
                    ax.errorbar(
                        x,
                        mean,
                        yerr=se,
                        color=color,
                        marker=marker,
                        markersize=7,
                        markerfacecolor="white" if open_marker else color,
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        linestyle="none",
                        elinewidth=1.0,
                        capsize=2.5,
                        zorder=3,
                    )
        ax.axhline(
            1.0,
            color="#666666",
            linestyle=(0, (2.0, 2.0)),
            linewidth=1.0,
            zorder=1,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([0.5, 1.0, 2.0, 4.0], ["0.5×", "1×", "2×", "4×"])
        ax.set_title(DATASETS[dataset][0], pad=6)
        ax.set_xlabel("Replay-field evaluations per update")
        if panel == 0:
            ax.set_ylabel(r"FID ratio to matched ODE $\downarrow$")
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend = _performance_legend(fig)
    fig.tight_layout(w_pad=1.0)
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


def render_adaptive_time(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    datasets: tuple[str, ...],
    out: Path,
) -> None:
    """Adaptive versus equal-task quality using fully accounted training time."""
    _style()
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(2.9 * len(datasets), 2.9),
        squeeze=False,
    )
    for panel, dataset in enumerate(datasets):
        ax = axes[0, panel]
        for strategy, (_, color) in STRATEGIES.items():
            for backend, (_backend_label, marker) in BACKENDS.items():
                pair: list[tuple[float, float, float, float, str]] = []
                for policy in ("equal", "adaptive"):
                    selected = _group(
                        rows,
                        dataset=dataset,
                        strategy=strategy,
                        policy=policy,
                        backend=backend,
                    )
                    x_mean, x_se = _mean_se(
                        [
                            float(row["accounted_training_hours"])
                            for row in selected
                        ]
                    )
                    y_mean, y_se = _mean_se(
                        [float(row[metric]) for row in selected]
                    )
                    pair.append((x_mean, x_se, y_mean, y_se, policy))
                ax.plot(
                    [point[0] for point in pair],
                    [point[2] for point in pair],
                    color=color,
                    linewidth=1.2,
                    alpha=0.65,
                    zorder=2,
                )
                for x_mean, x_se, y_mean, y_se, policy in pair:
                    open_marker = policy == "equal"
                    ax.errorbar(
                        x_mean,
                        y_mean,
                        xerr=x_se,
                        yerr=y_se,
                        color=color,
                        marker=marker,
                        markersize=7,
                        markerfacecolor="white" if open_marker else color,
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        linestyle="none",
                        elinewidth=1.0,
                        capsize=2.5,
                        zorder=3,
                    )
        ax.set_xscale("log", base=2)
        if metric == "final_kld":
            ax.set_yscale("log")
        ax.set_title(DATASETS[dataset][0], pad=6)
        ax.set_xlabel("Accounted training time (h)")
        if panel == 0:
            ax.set_ylabel(
                r"Final stream FID $\downarrow$"
                if metric == "final_fid"
                else r"Final stream KLD $\downarrow$"
            )
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=label)
        for label, color in STRATEGIES.values()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="o",
                linestyle="none",
                markerfacecolor=("white" if policy == "equal" else "#555555"),
                label=POLICIES[policy][0],
            )
            for policy in ("adaptive", "equal")
        ]
    )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker=marker,
                linestyle="none",
                label=label,
            )
            for label, marker in BACKENDS.values()
        ]
    )
    legend = fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        handlelength=1.7,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.tight_layout(w_pad=1.0)
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
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.results_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fid_out = args.out_dir / "backend_fid_vs_replay_evals.png"
    kld_out = args.out_dir / "backend_kld_vs_replay_evals.png"
    relative_fid_out = args.out_dir / "backend_relative_fid_vs_replay_evals.png"
    wall_out = args.out_dir / "backend_amortized_update_wall_time.png"
    adaptive_fid_out = args.out_dir / "adaptive_vs_equal_fid_time.png"
    adaptive_kld_out = args.out_dir / "adaptive_vs_equal_kld_time.png"
    render_quality(
        rows,
        metric="final_fid",
        datasets=("fmnist", "cifar10", "imagenet100"),
        out=fid_out,
    )
    render_quality(
        rows,
        metric="final_kld",
        datasets=("fmnist", "cifar10"),
        out=kld_out,
    )
    render_relative_fid(rows, relative_fid_out)
    render_wall(rows, wall_out)
    render_adaptive_time(
        rows,
        metric="final_fid",
        datasets=("fmnist", "cifar10", "imagenet100"),
        out=adaptive_fid_out,
    )
    render_adaptive_time(
        rows,
        metric="final_kld",
        datasets=("fmnist", "cifar10"),
        out=adaptive_kld_out,
    )
    csv_path = args.out_dir / "backend_quality_compute_all.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {fid_out} and {fid_out.with_suffix('.pdf')}")
    print(f"Wrote {kld_out} and {kld_out.with_suffix('.pdf')}")
    print(
        f"Wrote {relative_fid_out} and "
        f"{relative_fid_out.with_suffix('.pdf')}"
    )
    print(f"Wrote {wall_out} and {wall_out.with_suffix('.pdf')}")
    print(f"Wrote {adaptive_fid_out} and {adaptive_fid_out.with_suffix('.pdf')}")
    print(f"Wrote {adaptive_kld_out} and {adaptive_kld_out.with_suffix('.pdf')}")
    print(f"Wrote {csv_path} ({len(rows)} seed-level rows)")


if __name__ == "__main__":
    main()
