#!/usr/bin/env python3
"""Plot image-CL stream FID and class-balance KLD through the task stream."""

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
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
DATASETS = {
    "fmnist": {
        "label": "Fashion-MNIST",
        "runs": {
            "naive": "naive_s{seed}",
            "gen_selected": "generative_s{seed}",
            "gen_equal": "generative_eq_s{seed}",
            "traj_selected": "trajectory_s{seed}",
            "traj_equal": "trajectory_eq_s{seed}",
        },
    },
    "cifar10": {
        "label": "CIFAR-10",
        "runs": {
            "naive": "naive_s{seed}",
            "gen_selected": "generative_g3_s{seed}",
            "gen_equal": "generative_eq_s{seed}",
            "traj_selected": "trajectory_g3_s{seed}",
            "traj_equal": "trajectory_eq_s{seed}",
        },
    },
    "imagenet100": {
        "label": "ImageNet-100",
        "runs": {
            "naive": "naive_s{seed}",
            "gen_selected": "generative_g3_s{seed}",
            "gen_equal": "generative_eq_eq_s{seed}",
            "traj_selected": "trajectory_g3_s{seed}",
            "traj_equal": "trajectory_eq_eq_s{seed}",
        },
    },
}
METHODS = {
    "naive": {
        "label": "Naive fine-tuning",
        "color": "#4A4A4A",
        "marker": "X",
        "linestyle": "solid",
        "open": False,
        "zorder": 2,
    },
    "diff_distill": {
        "label": "Gen. distill. (diffusion)",
        "color": "#F4A261",
        "marker": "v",
        "linestyle": "solid",
        "open": False,
        "zorder": 2,
    },
    "gen_selected": {
        "label": r"Gen. replay, adaptive $\lambda$",
        "color": "#c1121f",
        "marker": "s",
        "linestyle": "solid",
        "open": False,
        "zorder": 3,
    },
    "gen_equal": {
        "label": r"Gen. replay, equal-task",
        "color": "#c1121f",
        "marker": "s",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 2,
    },
    "traj_selected": {
        "label": r"Traj. replay, adaptive $\lambda$",
        "color": "#457b9d",
        "marker": "o",
        "linestyle": "solid",
        "open": False,
        "zorder": 5,
    },
    "traj_equal": {
        "label": r"Traj. replay, equal-task",
        "color": "#457b9d",
        "marker": "o",
        "linestyle": (0, (3.0, 2.0)),
        "open": True,
        "zorder": 4,
    },
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
    parser.add_argument(
        "--masip-root",
        type=Path,
        default=(
            Path.home()
            / "sandbox"
            / "trajectory_replay"
            / "baselines"
            / "masip"
        ),
    )
    parser.add_argument(
        "--masip-imagenet-root",
        type=Path,
        default=(
            Path.home()
            / "sandbox"
            / "trajectory_replay"
            / "baselines"
            / "masip_fm_budget"
        ),
    )
    return parser.parse_args()


def _read_masip_metric(path: Path, metric: str) -> list[float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not str(row["metric_name"]).startswith(
                f"{metric}/eval_phase/test_stream/Task000"
            ):
                continue
            values[int(row["training_exp"])] = float(row["value"])
    if not values:
        raise ValueError(f"No {metric} rows in {path}")
    return [values[index] for index in sorted(values)]


def _read_masip_hist(path: Path, metric: str) -> list[np.ndarray]:
    values: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not str(row["metric_name"]).startswith(
                f"{metric}/eval_phase/test_stream/Task000"
            ):
                continue
            text = str(row["value"]).strip().strip("[]")
            values[int(row["training_exp"])] = np.fromstring(text, sep=" ")
    if not values:
        raise ValueError(f"No {metric} rows in {path}")
    return [values[index] for index in sorted(values)]


def _masip_path(
    dataset: str,
    seed: int,
    *,
    masip_root: Path,
    masip_imagenet_root: Path,
) -> Path:
    if dataset == "fmnist":
        stem = (
            "gr_diffusion_full_generation_distillation_"
            "steps_10_lambd_0.75_cnn"
        )
        root = masip_root / "split_fmnist"
    elif dataset == "cifar10":
        stem = (
            "gr_diffusion_full_generation_distillation_"
            "steps_20_lambd_0.75_cnn"
        )
        root = masip_root / "split_cifar10"
    else:
        stem = (
            "gr_diffusion_full_generation_distillation_"
            "steps_50_lambd_0.75_cc_e3_gs1.5"
        )
        root = masip_imagenet_root / "split_imagenet100"
    return root / stem / str(seed) / "logs" / "eval_results.csv"


def collect(
    results_root: Path,
    *,
    masip_root: Path,
    masip_imagenet_root: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset, dataset_spec in DATASETS.items():
        methods: dict[str, Any] = {}
        for method, stem in dataset_spec["runs"].items():
            seed_values: list[list[float]] = []
            seed_kld: list[list[float | None]] = []
            seed_hist: list[list[float]] = []
            seed_lambda: list[list[float]] = []
            sources: list[str] = []
            for seed in SEEDS:
                path = results_root / dataset / stem.format(seed=seed) / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload["metrics_by_task"]
                seed_values.append([float(row["fid_stream_test"]) for row in rows])
                seed_kld.append(
                    [
                        None
                        if row.get("stream_kld") is None
                        else float(row["stream_kld"])
                        for row in rows
                    ]
                )
                if rows[-1].get("hist_pred") is not None:
                    seed_hist.append(
                        [float(value) for value in rows[-1]["hist_pred"]]
                    )
                chosen = [row.get("chosen_lambda") for row in rows[1:]]
                if chosen and all(value is not None for value in chosen):
                    seed_lambda.append([float(value) for value in chosen])
                sources.append(str(path))
            fid = np.asarray(seed_values, dtype=float)
            methods[method] = {
                "fid": fid,
                "kld": seed_kld,
                "hist": (
                    None
                    if not seed_hist
                    else np.asarray(seed_hist, dtype=float)
                ),
                "lambda": (
                    None
                    if not seed_lambda
                    else np.asarray(seed_lambda, dtype=float)
                ),
                "sources": sources,
            }
        diffusion_fid: list[list[float]] = []
        diffusion_kld: list[list[float | None]] = []
        diffusion_hist: list[list[float]] = []
        diffusion_sources: list[str] = []
        for seed in SEEDS:
            path = _masip_path(
                dataset,
                seed,
                masip_root=masip_root,
                masip_imagenet_root=masip_imagenet_root,
            )
            if not path.exists():
                raise FileNotFoundError(path)
            diffusion_fid.append(_read_masip_metric(path, "stream_fid"))
            diffusion_kld.append(
                (
                    [None] * len(diffusion_fid[-1])
                    if dataset == "imagenet100"
                    else _read_masip_metric(path, "stream_kld")
                )
            )
            if dataset != "imagenet100":
                diffusion_hist.append(
                    _read_masip_hist(path, "stream_hist_pred")[-1].tolist()
                )
            diffusion_sources.append(str(path))
        methods["diff_distill"] = {
            "fid": np.asarray(diffusion_fid, dtype=float),
            "kld": diffusion_kld,
            "hist": (
                None
                if dataset == "imagenet100"
                else np.asarray(diffusion_hist, dtype=float)
            ),
            "lambda": None,
            "sources": diffusion_sources,
        }
        result[dataset] = {
            "label": dataset_spec["label"],
            "methods": methods,
        }
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
        }
    )


def _summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, se


def _plot_series(ax: plt.Axes, values: np.ndarray, method: str) -> None:
    spec = METHODS[method]
    mean, se = _summary(values)
    x = np.arange(1, len(mean) + 1)
    ax.fill_between(
        x,
        mean - se,
        mean + se,
        color=spec["color"],
        alpha=0.12,
        linewidth=0,
        zorder=spec["zorder"] - 1,
    )
    ax.plot(
        x,
        mean,
        color=spec["color"],
        marker=spec["marker"],
        markersize=6.8,
        markerfacecolor="white" if spec["open"] else spec["color"],
        markeredgecolor=spec["color"],
        markeredgewidth=1.2 if spec["open"] else 0.7,
        linestyle=spec["linestyle"],
        linewidth=2.0 if method.startswith("traj") else 1.7,
        label=spec["label"],
        zorder=spec["zorder"],
    )


def _finish_panel(ax: plt.Axes, n_tasks: int, *, first_task: int = 0) -> None:
    x = np.arange(1, n_tasks + 1)
    ax.set_xticks(x, [rf"$T_{i}$" for i in range(first_task, first_task + n_tasks)])
    ax.set_xlim(0.72, n_tasks + 0.28)
    ax.grid(axis="y", color="0.87", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_fid(data: dict[str, dict[str, Any]], out: Path) -> None:
    _style()
    datasets = ("fmnist", "cifar10", "imagenet100")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=False)
    for index, (ax, dataset) in enumerate(zip(axes, datasets, strict=True)):
        row = data[dataset]
        for method in METHODS:
            _plot_series(ax, row["methods"][method]["fid"][:, 1:], method)
        n_tasks = row["methods"]["naive"]["fid"].shape[1] - 1
        _finish_panel(ax, n_tasks, first_task=1)
        ax.set_title(row["label"], pad=6)
        ax.set_xlabel("Tasks completed")
        if index == 0:
            ax.set_ylabel(r"Stream FID $\downarrow$")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.tight_layout(w_pad=1.1)
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


def render_kld(data: dict[str, dict[str, Any]], out: Path) -> None:
    _style()
    datasets = ("fmnist", "cifar10")
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.9), sharey=False)
    for index, (ax, dataset) in enumerate(zip(axes, datasets, strict=True)):
        row = data[dataset]
        for method in METHODS:
            raw = row["methods"][method]["kld"]
            values = np.asarray(raw, dtype=float)
            if not np.isfinite(values).all():
                continue
            _plot_series(ax, values[:, 1:], method)
        n_tasks = len(row["methods"]["naive"]["kld"][0]) - 1
        _finish_panel(ax, n_tasks, first_task=1)
        ax.set_yscale("log")
        ax.set_title(row["label"], pad=6)
        ax.set_xlabel("Tasks completed")
        if index == 0:
            ax.set_ylabel(r"Stream KLD $\downarrow$")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.45,
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


def render_class_mass(data: dict[str, dict[str, Any]], out: Path) -> None:
    _style()
    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    datasets = ("fmnist", "cifar10")
    order = (
        "naive",
        "diff_distill",
        "gen_equal",
        "gen_selected",
        "traj_equal",
        "traj_selected",
        "target",
    )
    labels = {
        "target": "Target",
        "naive": "Naive",
        "diff_distill": "Diff. distill.",
        "gen_selected": r"Gen., adaptive $\lambda$",
        "gen_equal": "Gen., equal-task",
        "traj_selected": r"Traj., adaptive $\lambda$",
        "traj_equal": "Traj., equal-task",
    }
    colors = list(plt.get_cmap("tab10").colors)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 4.05),
        sharey=True,
    )

    for ax, dataset in zip(axes, datasets, strict=True):
        row = data[dataset]
        y = np.arange(len(order), dtype=float)
        for row_index, method in enumerate(order):
            if method == "target":
                proportions = np.full(10, 0.1, dtype=float)
                kld_text = "Target"
            else:
                hist = row["methods"][method]["hist"]
                if hist is None:
                    continue
                proportions = np.asarray(hist, dtype=float).mean(axis=0)
                kld = np.asarray(row["methods"][method]["kld"], dtype=float)
                final_kld = kld[:, -1]
                mean = float(final_kld.mean())
                se = float(final_kld.std(ddof=1) / np.sqrt(final_kld.size))
                kld_text = rf"{mean:.2f}$\pm${se:.2f}"
            left = 0.0
            for class_id, proportion in enumerate(proportions):
                ax.barh(
                    row_index,
                    proportion,
                    left=left,
                    height=0.74,
                    color=colors[class_id],
                    edgecolor="white",
                    linewidth=0.65,
                )
                left += float(proportion)
            ax.text(
                1.025,
                row_index,
                kld_text,
                ha="left",
                va="center",
                fontsize=9.5,
                color="#555555",
            )

        ax.set_xlim(0.0, 1.30)
        ax.set_xticks([0.0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.set_xlabel("Generated proportion")
        ax.set_title(row["label"], pad=8)
        ax.set_yticks(y, [labels[method] for method in order])
        ax.invert_yaxis()
        ax.axhline(
            len(order) - 1.5,
            color="#555555",
            linewidth=1.0,
            zorder=5,
        )
        ax.grid(axis="x", color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.10,
            1.08,
            "a" if dataset == "fmnist" else "b",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            0.80,
            1.015,
            r"KLD $\downarrow$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#444444",
        )
    handles = [
        Patch(facecolor=colors[class_id], edgecolor="none", label=str(class_id))
        for class_id in range(10)
    ]
    legend = fig.legend(
        handles=handles,
        title="Classifier-predicted class",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=10,
        handlelength=0.9,
        columnspacing=0.65,
        handletextpad=0.25,
        fontsize=11,
        title_fontsize=11.5,
    )
    fig.tight_layout(w_pad=1.35)
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


def render_lambda_schedule(data: dict[str, dict[str, Any]], out: Path) -> None:
    _style()
    plt.rcParams.update(
        {
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    datasets = ("fmnist", "cifar10", "imagenet100")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=True)
    styles = {
        "gen_selected": ("Generative replay", "#c1121f", "s", -0.035),
        "traj_selected": ("Trajectory replay", "#457b9d", "o", 0.035),
    }

    for panel, (ax, dataset) in enumerate(zip(axes, datasets, strict=True)):
        row = data[dataset]
        n_updates = row["methods"]["gen_selected"]["lambda"].shape[1]
        x = np.arange(1, n_updates + 1, dtype=float)
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
        for method, (label, color, marker, offset) in styles.items():
            values = row["methods"][method]["lambda"]
            values = np.asarray(values, dtype=float)
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            ax.errorbar(
                x + offset,
                mean,
                yerr=se,
                color=color,
                marker=marker,
                markersize=7.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=2.0,
                elinewidth=1.1,
                capsize=2.8,
                capthick=1.0,
                label=label,
                zorder=4,
            )
        ax.set_xticks(
            x,
            [rf"after $T_{task}$" for task in range(1, n_updates + 1)],
        )
        ax.set_xlim(0.7, n_updates + 0.3)
        ax.set_ylim(0.19, 0.94)
        ax.set_yticks(
            [0.25, 0.5, 2 / 3, 0.75, 0.8, 0.9],
            ["0.25", "0.50", "0.67", "0.75", "0.80", "0.90"],
        )
        ax.set_title(row["label"], pad=6)
        ax.set_xlabel("Sequential update")
        if panel == 0:
            ax.set_ylabel(r"Mean adaptive $\lambda_k$")
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.45,
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


def write_summary(data: dict[str, dict[str, Any]], path: Path) -> None:
    serializable: dict[str, Any] = {}
    for dataset, row in data.items():
        serializable[dataset] = {"label": row["label"], "methods": {}}
        for method, values in row["methods"].items():
            fid_mean, fid_se = _summary(values["fid"])
            kld = np.asarray(values["kld"], dtype=float)
            kld_summary = None
            if np.isfinite(kld).all():
                kld_mean, kld_se = _summary(kld)
                kld_summary = {
                    "mean": kld_mean.tolist(),
                    "se": kld_se.tolist(),
                }
            serializable[dataset]["methods"][method] = {
                "fid": {"mean": fid_mean.tolist(), "se": fid_se.tolist()},
                "kld": kld_summary,
                "hist_mean": (
                    None
                    if values["hist"] is None
                    else np.asarray(values["hist"], dtype=float).mean(axis=0).tolist()
                ),
                "lambda_mean": (
                    None
                    if values["lambda"] is None
                    else np.asarray(values["lambda"], dtype=float).mean(axis=0).tolist()
                ),
                "sources": values["sources"],
            }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = collect(
        args.results_root,
        masip_root=args.masip_root,
        masip_imagenet_root=args.masip_imagenet_root,
    )
    fid_out = args.out_dir / "stream_fid_by_task.png"
    kld_out = args.out_dir / "stream_kld_by_task.png"
    class_mass_out = args.out_dir / "final_class_mass.png"
    lambda_out = args.out_dir / "adaptive_lambda_by_task.png"
    render_fid(data, fid_out)
    render_kld(data, kld_out)
    render_class_mass(data, class_mass_out)
    render_lambda_schedule(data, lambda_out)
    write_summary(data, args.out_dir / "stream_metrics_by_task.json")
    print(f"Wrote {fid_out} and {fid_out.with_suffix('.pdf')}")
    print(f"Wrote {kld_out} and {kld_out.with_suffix('.pdf')}")
    print(f"Wrote {class_mass_out} and {class_mass_out.with_suffix('.pdf')}")
    print(f"Wrote {lambda_out} and {lambda_out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
