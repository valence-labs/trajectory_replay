#!/usr/bin/env python3
"""Plot PX lambda-policy and trajectory replay-backend ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.px.plot_cl_experiment_fid import latest_complete_metrics


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_dir" / "px" / "cl_final"
OUT_DIR = ROOT / "results_dir" / "px" / "figures"
SEEDS = (42, 69, 1714)
STRATEGIES = {
    "generative": ("Generative replay", "#C1121F", "s"),
    "trajectory": ("Trajectory replay", "#457B9D", "o"),
}
POLICIES = {
    "adaptive": ("Adaptive $\\lambda$", "grid"),
    "equal": ("Equal-task", "equal"),
}
BACKENDS = {
    "mf1": ("MF-1", "trajectory_mean_flow_n1_equal_s{seed}", "^", 2.0),
    "mf4": ("MF-4", "trajectory_mean_flow_n4_equal_s{seed}", "s", 5.0),
    "ode": ("ODE", "trajectory_ode_equal_s{seed}", "o", 11.5),
}


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
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.85,
        }
    )


def _load(stem: str, seed: int) -> tuple[Path, dict[str, Any]]:
    found = latest_complete_metrics(
        RESULTS_ROOT / stem.format(seed=seed)
    )
    if found is None:
        raise FileNotFoundError(stem.format(seed=seed))
    return found


def _final_metrics(payload: dict[str, Any]) -> tuple[float, float]:
    final = payload["metrics_by_task"][-1]["test_stream"]
    return float(final["fid"]), float(final["fdd_dinov2"])


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / np.sqrt(array.size)),
    )


def collect_lambda() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in STRATEGIES:
        result[strategy] = {}
        for policy, (_, suffix) in POLICIES.items():
            stem = f"{strategy}_ode_{suffix}_s{{seed}}"
            final_fid: list[float] = []
            schedules: list[list[float]] = []
            sources: list[str] = []
            for seed in SEEDS:
                path, payload = _load(stem, seed)
                fid, _ = _final_metrics(payload)
                final_fid.append(fid)
                sources.append(str(path))
                chosen = [
                    row.get("chosen_lambda")
                    for row in payload["metrics_by_task"]
                    if int(row["task_id"]) > 0
                ]
                if len(chosen) != 2 or any(value is None for value in chosen):
                    raise ValueError(f"{path}: missing T1/T2 lambda schedule")
                schedules.append([float(value) for value in chosen])
            fid_mean, fid_se = _mean_se(final_fid)
            schedule = np.asarray(schedules, dtype=float)
            result[strategy][policy] = {
                "final_stream_fid": {
                    "mean": fid_mean,
                    "se": fid_se,
                    "seed_values": final_fid,
                },
                "lambda": {
                    "mean": schedule.mean(axis=0).tolist(),
                    "se": (
                        schedule.std(axis=0, ddof=1)
                        / np.sqrt(schedule.shape[0])
                    ).tolist(),
                    "seed_values": schedule.tolist(),
                },
                "sources": sources,
            }
    return result


def collect_backend() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for backend, (label, stem, marker, evaluations) in BACKENDS.items():
        fid: list[float] = []
        fdd: list[float] = []
        sources: list[str] = []
        for seed in SEEDS:
            path, payload = _load(stem, seed)
            seed_fid, seed_fdd = _final_metrics(payload)
            fid.append(seed_fid)
            fdd.append(seed_fdd)
            sources.append(str(path))
        fid_mean, fid_se = _mean_se(fid)
        fdd_mean, fdd_se = _mean_se(fdd)
        result[backend] = {
            "label": label,
            "marker": marker,
            "replay_field_evaluations": evaluations,
            "stream_fid": {
                "mean": fid_mean,
                "se": fid_se,
                "seed_values": fid,
            },
            "stream_fdd_dinov2": {
                "mean": fdd_mean,
                "se": fdd_se,
                "seed_values": fdd,
            },
            "sources": sources,
        }
    return result


def render_lambda(data: dict[str, Any], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.35, 3.0))

    policy_x = np.asarray([0.0, 1.0])
    strategy_offsets = {"generative": -0.035, "trajectory": 0.035}
    for strategy, (label, color, marker) in STRATEGIES.items():
        means = np.asarray(
            [
                data[strategy][policy]["final_stream_fid"]["mean"]
                for policy in POLICIES
            ],
            dtype=float,
        )
        errors = np.asarray(
            [
                data[strategy][policy]["final_stream_fid"]["se"]
                for policy in POLICIES
            ],
            dtype=float,
        )
        axes[0].plot(
            policy_x + strategy_offsets[strategy],
            means,
            color=color,
            linewidth=2.0,
            label=label,
            zorder=3,
        )
        for policy_index, policy in enumerate(POLICIES):
            is_equal = policy == "equal"
            axes[0].errorbar(
                policy_x[policy_index] + strategy_offsets[strategy],
                means[policy_index],
                yerr=errors[policy_index],
                color=color,
                marker=marker,
                markersize=7.5,
                markerfacecolor="white" if is_equal else color,
                markeredgecolor=color,
                markeredgewidth=1.2 if is_equal else 0.7,
                linestyle="none",
                elinewidth=1.1,
                capsize=2.8,
                zorder=4,
            )

    axes[0].set_xticks(
        policy_x,
        [POLICIES[policy][0] for policy in POLICIES],
    )
    axes[0].set_xlim(-0.25, 1.25)
    axes[0].set_ylabel(r"Final stream FID $\downarrow$")
    axes[0].set_title("Final pooled quality", pad=5)

    update_x = np.asarray([1.0, 2.0])
    equal = update_x / (update_x + 1.0)
    axes[1].plot(
        update_x,
        equal,
        color="#555555",
        linestyle=(0, (2.0, 2.0)),
        marker="*",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="#555555",
        linewidth=1.7,
        label=r"Equal-task $k/(k+1)$",
        zorder=2,
    )
    for strategy, (label, color, marker) in STRATEGIES.items():
        schedule = data[strategy]["adaptive"]["lambda"]
        means = np.asarray(schedule["mean"], dtype=float)
        errors = np.asarray(schedule["se"], dtype=float)
        axes[1].errorbar(
            update_x + strategy_offsets[strategy],
            means,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            elinewidth=1.1,
            capsize=2.8,
            label=label,
            zorder=4,
        )
    axes[1].set_xticks(update_x, [r"after $T_1$", r"after $T_2$"])
    axes[1].set_xlim(0.75, 2.25)
    axes[1].set_ylim(0.19, 0.81)
    axes[1].set_yticks([0.25, 0.5, 2 / 3, 0.75], ["0.25", "0.50", "0.67", "0.75"])
    axes[1].set_ylabel(r"Selected $\lambda_k$")
    axes[1].set_title(r"Selected replay weight", pad=5)

    for ax in axes:
        ax.set_box_aspect(0.78)
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        handlelength=1.9,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85), w_pad=1.0)
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


def render_backend(data: dict[str, Any], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.35, 3.0))
    order = ("mf1", "mf4", "ode")
    x = np.asarray(
        [data[backend]["replay_field_evaluations"] for backend in order],
        dtype=float,
    )
    for ax, metric, title, ylabel in (
        (axes[0], "stream_fid", "Inception features", r"Final stream FID $\downarrow$"),
        (
            axes[1],
            "stream_fdd_dinov2",
            "DINOv2 features",
            r"Final stream FDD $\downarrow$",
        ),
    ):
        means = np.asarray(
            [data[backend][metric]["mean"] for backend in order],
            dtype=float,
        )
        errors = np.asarray(
            [data[backend][metric]["se"] for backend in order],
            dtype=float,
        )
        ax.plot(x, means, color="#457B9D", linewidth=1.8, zorder=2)
        for index, backend in enumerate(order):
            ax.errorbar(
                x[index],
                means[index],
                yerr=errors[index],
                color="#457B9D",
                marker=data[backend]["marker"],
                markersize=7.5,
                markerfacecolor="#457B9D",
                markeredgecolor="white",
                markeredgewidth=0.7,
                linestyle="none",
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            x,
            [
                "MF-1\n2 evals",
                "MF-4\n5 evals",
                "ODE\n11.5 evals",
            ],
        )
        ax.set_xlabel("Replay backend")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=5)
        ax.set_box_aspect(0.78)
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    lambda_data = collect_lambda()
    backend_data = collect_backend()
    lambda_out = OUT_DIR / "px_adaptive_lambda_ablation.png"
    backend_out = OUT_DIR / "px_trajectory_backend_ablation.png"
    render_lambda(lambda_data, lambda_out)
    render_backend(backend_data, backend_out)
    (OUT_DIR / "px_lambda_backend_ablation.json").write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "lambda_ablation": lambda_data,
                "trajectory_backend_ablation": backend_data,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(lambda_out)
    print(backend_out)


if __name__ == "__main__":
    main()
