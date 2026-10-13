#!/usr/bin/env python3
"""Cross-method plots for the matched equal-task sequential experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_sequential.common import REPO_ROOT, sequential_results_root

METHODS = (
    ("naive_none", "Naive", "#003049", "X"),
    ("generative_equal", "Generative replay", "#C1121F", "s"),
    ("trajectory_equal", "Trajectory replay", "#457B9D", "o"),
)
MASS_METHODS = (
    ("naive_none", "Naive", "#003049", "X"),
    ("generative_equal", "Generative equal-task", "#C1121F", "s"),
    ("generative_adaptive", "Generative adaptive", "#E56B6F", "D"),
    ("trajectory_equal", "Trajectory equal-task", "#457B9D", "o"),
    ("trajectory_adaptive", "Trajectory adaptive", "#7FA9C0", "D"),
)
METRICS = (
    ("energy_uniform_mixture", "Mixture energy"),
    ("mass_kl_uniform", r"Mass $D_{\mathrm{KL}}$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/runs_lr3e4.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results_dir/sim_sequential/figures.",
    )
    parser.add_argument("--required-seeds", type=int, default=3)
    return parser.parse_args()


def load_curves(
    runs_dir: Path, *, required_seeds: int
) -> dict[str, dict[str, np.ndarray]]:
    curves: dict[str, dict[str, np.ndarray]] = {}
    for run_prefix, _, _, _ in MASS_METHODS:
        per_seed: dict[str, list[np.ndarray]] = {metric: [] for metric, _ in METRICS}
        for seed in (0, 1, 2):
            path = runs_dir / f"{run_prefix}_s{seed}" / "stream_metrics.json"
            if not path.is_file():
                continue
            stream = json.loads(path.read_text(encoding="utf-8"))
            if len(stream) != 5:
                continue
            seed_values: dict[str, np.ndarray] = {}
            for metric, _ in METRICS:
                values = np.asarray(
                    [record["metrics"][metric] for record in stream],
                    dtype=np.float64,
                )
                if not np.isfinite(values).all():
                    break
                seed_values[metric] = values
            else:
                for metric, values in seed_values.items():
                    per_seed[metric].append(values)
        n_valid = len(per_seed[METRICS[0][0]])
        if n_valid < required_seeds:
            raise ValueError(
                f"{run_prefix} has {n_valid} valid seeds; required {required_seeds}"
            )
        curves[run_prefix] = {
            metric: np.stack(values) for metric, values in per_seed.items()
        }
    return curves


def set_style() -> None:
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


def plot_curves(curves: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 1.9))
    x = np.arange(2, 6)
    for ax, (metric, ylabel) in zip(axes, METRICS, strict=True):
        for run_prefix, label, color, marker in METHODS:
            values = curves[run_prefix][metric][:, 1:]
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            for seed_values in values:
                ax.plot(x, seed_values, color=color, linewidth=0.7, alpha=0.22)
            ax.fill_between(
                x,
                np.maximum(mean - se, mean / 5.0),
                mean + se,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                label=label,
            )
            ax.annotate(
                f"{mean[-1]:.3f}",
                (x[-1], mean[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=color,
                fontsize=7.5,
                va="center",
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xlabel("Tasks seen")
        ax.set_ylabel(f"{ylabel} $\\downarrow$")
        ax.grid(True, which="major", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86), pad=0.25, w_pad=0.65)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_final_bars(
    curves: dict[str, dict[str, np.ndarray]],
    out: Path,
    *,
    metrics: tuple[tuple[str, str], ...] = METRICS,
    methods: tuple[tuple[str, str, str, str], ...] = METHODS,
    xticklabels: tuple[str, ...] | None = None,
    off_scale_reference: tuple[str, float] | None = None,
) -> None:
    if len(metrics) == 1 and len(methods) > 3:
        metric, xlabel = metrics[0]
        final = [curves[prefix][metric][:, -1] for prefix, _, _, _ in methods]
        means = np.asarray([values.mean() for values in final])
        ses = np.asarray(
            [values.std(ddof=1) / np.sqrt(len(values)) for values in final]
        )
        colors = [color for _, _, color, _ in methods]
        labels = xticklabels or tuple(label for _, label, _, _ in methods)
        y = np.arange(len(methods))
        rng = np.random.default_rng(0)

        fig, ax = plt.subplots(figsize=(2.05, 1.85))
        ax.barh(
            y,
            means,
            xerr=ses,
            color=colors,
            height=0.64,
            alpha=0.78,
            edgecolor="none",
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
        for method_id, values in enumerate(final):
            jitter = rng.uniform(-0.075, 0.075, size=len(values))
            ax.scatter(
                values,
                method_id + jitter,
                s=18,
                color=colors[method_id],
                marker=methods[method_id][3],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            label_x = 1.18 * max(
                means[method_id] + ses[method_id], float(values.max())
            )
            ax.text(
                label_x,
                method_id,
                f"{means[method_id]:.3f}",
                ha="left",
                va="center",
                fontsize=8.5,
            )

        lower = min(float(values.min()) for values in final) / 2.0
        upper = max(
            1.18 * max(mean + se, float(values.max()))
            for mean, se, values in zip(means, ses, final, strict=True)
        )
        ax.set_xscale("log")
        ax.set_xlim(lower, upper * 1.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"Final {xlabel.lower()} $\\downarrow$", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        if off_scale_reference is not None:
            reference_label, reference_value = off_scale_reference
            ax.set_title(
                f"{reference_label} = {reference_value:.3f} (off-scale)",
                fontsize=9,
                loc="left",
                pad=3,
            )
        ax.grid(True, axis="x", which="major", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.2)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return

    width = 3.9 if len(metrics) == 2 else (1.85 if len(methods) > 3 else 2.15)
    height = 1.7 if len(methods) > 3 else 1.55
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, height), squeeze=False)
    rng = np.random.default_rng(0)
    x = np.arange(len(methods))
    for ax, (metric, ylabel) in zip(axes[0], metrics, strict=True):
        final = [curves[prefix][metric][:, -1] for prefix, _, _, _ in methods]
        means = np.asarray([values.mean() for values in final])
        ses = np.asarray(
            [values.std(ddof=1) / np.sqrt(len(values)) for values in final]
        )
        colors = [color for _, _, color, _ in methods]
        ax.bar(
            x,
            means,
            yerr=ses,
            color=colors,
            width=0.68,
            alpha=0.78,
            edgecolor="none",
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
        for method_id, values in enumerate(final):
            jitter = rng.uniform(-0.075, 0.075, size=len(values))
            ax.scatter(
                method_id + jitter,
                values,
                s=18,
                color=colors[method_id],
                marker=methods[method_id][3],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            ax.text(
                method_id,
                1.35 * (means[method_id] + ses[method_id]),
                f"{means[method_id]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_yscale("log")
        labels = xticklabels or tuple(label for _, label, _, _ in methods)
        ax.set_xticks(x, labels)
        ax.set_ylabel(f"Final {ylabel.lower()} $\\downarrow$", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, axis="y", which="major", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = sequential_results_root(REPO_ROOT)
    runs_dir = args.runs_dir or root / "runs_lr3e4"
    out_dir = args.out_dir or root / "figures"
    curves = load_curves(runs_dir, required_seeds=args.required_seeds)
    set_style()
    line_out = out_dir / "cross_method_equal_curves.png"
    bar_out = out_dir / "cross_method_equal_final_bars.png"
    plot_curves(curves, line_out)
    plot_final_bars(curves, bar_out)
    separate_outs = []
    energy_metric = METRICS[0]
    for metric in METRICS:
        suffix = "energy" if metric[0] == "energy_uniform_mixture" else "mass"
        separate_out = out_dir / f"cross_method_equal_final_{suffix}_bars.png"
        plot_final_bars(
            curves,
            separate_out,
            metrics=(metric,),
            methods=MASS_METHODS,
            xticklabels=(
                "Naive",
                "G equal",
                "G adapt.",
                "T equal",
                "T adapt.",
            ),
        )
        separate_outs.append(separate_out)
    off_scale_out = (
        out_dir / "cross_method_equal_final_energy_bars_off_scale.png"
    )
    naive_energy = float(curves["naive_none"][energy_metric[0]][:, -1].mean())
    plot_final_bars(
        curves,
        off_scale_out,
        metrics=(energy_metric,),
        methods=MASS_METHODS[1:],
        xticklabels=("G equal", "G adapt.", "T equal", "T adapt."),
        off_scale_reference=("Naive", naive_energy),
    )
    print(
        f"Wrote {line_out}, {bar_out}, {', '.join(map(str, separate_outs))}, "
        f"{off_scale_out}, and PDF versions"
    )


if __name__ == "__main__":
    main()
