#!/usr/bin/env python3
"""Top-panel figures: stylized mixture thumbnails + π̂₀ stripplot + loglik boxes.

Saves separate PNGs under results_dir/sim_mixture/dim<d>/figures/:

  mixture_traj_lam0p1.png … lam0p9.png   (trajectory, one seed)
  pi0_stripplot.png
  energy_boxplot.png

Example:
  uv run python experiments/sim_mixture/plot_top_panel.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    DEFAULT_LAMBDAS,
    REPO_ROOT,
    load_checkpoint,
    load_mixture,
    load_tasks,
    mixture_data_root,
    mixture_runs_root,
    sample_unconditional,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--panel-seed", type=int, default=0, help="Seed used for mixture thumbnails.")
    p.add_argument("--n-gen", type=int, default=3000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["trajectory", "generative", "oracle"],
        help="Methods included in the strip/loglik plots (order = legend order).",
    )
    p.add_argument(
        "--skip-mixture-panels",
        action="store_true",
        help="Only write π̂₀ / loglik summary figures (no sampling thumbnails).",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--mixture-dpi",
        type=int,
        default=600,
        help="PNG dpi for mixture thumbnails (scatter is rasterized; PDF is not written).",
    )
    p.add_argument(
        "--stage-panels",
        action="store_true",
        help=(
            "Render the one-update sequence (task-0 model, trajectory replay, naive) "
            "at --stage-lambda instead of one thumbnail per λ."
        ),
    )
    p.add_argument(
        "--stage-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="λ values for the stage panels (default: every λ in the dataset meta).",
    )
    return p.parse_args()


def _lam_tag(lam: float) -> str:
    return f"lam{lam:.1f}".replace(".", "p")


def _ft_dir(runs: Path, strategy: str, lam: float | None, epochs: int, seed: int) -> Path:
    if strategy == "naive":
        return runs / "finetune" / f"ft_naive_e{epochs}_seed{seed}"
    return runs / "finetune" / f"ft_{strategy}_lam{lam}_e{epochs}_seed{seed}"


# Trajectory keeps the blue it has in the strip / NFE panels; naive takes the
# remaining dark tone so it never reads as a replay method.
STAGE_COLORS = {
    "task0": "#1f6aa5",
    "trajectory": "#1f6aa5",
    "naive": "#003049",
}


def _naive_dir(runs: Path, seed: int) -> Path:
    """Naive runs carry their own epoch budget, so match on the seed alone."""
    candidates = sorted((runs / "finetune").glob(f"ft_naive_e*_seed{seed}"))
    for c in candidates:
        if (c / "checkpoint.pt").exists():
            return c
    return runs / "finetune" / f"ft_naive_seed{seed}"


def _oracle_dir(runs: Path, lam: float, seed: int, *, epochs: int | None = None) -> Path:
    tag = _lam_tag(lam)
    candidates: list[Path] = []
    if epochs is not None:
        candidates.append(runs / "oracle_mixture" / f"oracle_mixture_{tag}_e{epochs}_seed{seed}")
    candidates.extend(
        [
            runs / "oracle_mixture" / f"oracle_mixture_{tag}_seed{seed}",
            runs / "oracle_mixture" / f"oracle_mixture_lam{lam}_seed{seed}",
        ]
    )
    for c in candidates:
        if (c / "metrics.json").exists():
            return c
    return candidates[0]


def plot_mixture_panel(
    path: Path,
    *,
    x_gen: np.ndarray,
    x_true: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    with_kde: bool = True,
    scatter_h: float = 1.85,
    legend_fontsize: float = 6.5,
    dpi: int = 600,
    gen_label: str = "Trajectory replay",
    gen_color: str = "#1f6aa5",
    stage_label: str | None = None,
) -> None:
    """Scatter overlay; optionally a top marginal KDE on the x-coordinate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "legend.fontsize": legend_fontsize,
        }
    )

    # Warm taupe (Real) vs the model colour: hue + luminance split for legend/KDE.
    traj_color = gen_color
    real_color = "#8a8178"
    x_true = np.asarray(x_true, dtype=np.float64)
    x_gen = np.asarray(x_gen, dtype=np.float64)

    # Ensure plotted points are never clipped by the frame.
    pts = np.vstack([x_true, x_gen])
    margin = 0.22
    xlim = (
        float(min(xlim[0], pts[:, 0].min() - margin)),
        float(max(xlim[1], pts[:, 0].max() + margin)),
    )
    ylim = (
        float(min(ylim[0], pts[:, 1].min() - margin)),
        float(max(ylim[1], pts[:, 1].max() + margin)),
    )

    rng = np.random.default_rng(0)
    x_true_pool = (
        x_true
        if x_true.shape[0] <= 4000
        else x_true[rng.choice(x_true.shape[0], size=4000, replace=False)]
    )
    x_gen_pool = (
        x_gen
        if x_gen.shape[0] <= 4000
        else x_gen[rng.choice(x_gen.shape[0], size=4000, replace=False)]
    )
    x_true_sc = (
        x_true_pool
        if x_true_pool.shape[0] <= 2000
        else x_true_pool[rng.choice(x_true_pool.shape[0], size=2000, replace=False)]
    )
    x_gen_sc = (
        x_gen_pool
        if x_gen_pool.shape[0] <= 1500
        else x_gen_pool[rng.choice(x_gen_pool.shape[0], size=1500, replace=False)]
    )

    dx = float(xlim[1] - xlim[0])
    dy = float(ylim[1] - ylim[0])
    scatter_w = scatter_h * (dx / dy)
    kde_in = 0.38 if with_kde else 0.0
    gap_in = 0.05 if with_kde else 0.0
    # Keep legend band roughly constant so smaller panels read as more stylized.
    legend_in = 0.26
    side_pad_in = 0.06
    fig_w = scatter_w + 2 * side_pad_in
    fig_h = legend_in + kde_in + gap_in + scatter_h + 0.06
    fig = plt.figure(figsize=(fig_w, fig_h))

    left = side_pad_in / fig_w
    width = scatter_w / fig_w
    scatter_bottom = 0.05 / fig_h
    scatter_height = scatter_h / fig_h

    ax = fig.add_axes([left, scatter_bottom, width, scatter_height])

    if with_kde:
        from scipy.stats import gaussian_kde

        kde_bottom = scatter_bottom + scatter_height + gap_in / fig_h
        kde_height = kde_in / fig_h
        ax_top = fig.add_axes([left, kde_bottom, width, kde_height], sharex=ax)

        xs = np.linspace(xlim[0], xlim[1], 320)
        dens_list = []
        for data, color in (
            (x_true_pool, real_color),
            (x_gen_pool, traj_color),
        ):
            dens = gaussian_kde(data[:, 0], bw_method=0.25)(xs)
            dens_list.append(dens)
            ax_top.plot(xs, dens, color=color, lw=1.35)
            ax_top.fill_between(xs, dens, color=color, alpha=0.20, linewidth=0)

        ax_top.set_xlim(*xlim)
        ax_top.set_ylim(0.0, 1.15 * float(max(d.max() for d in dens_list)))
        ax_top.set_yticks([])
        ax_top.tick_params(axis="x", labelbottom=False, length=0)
        for side in ("top", "right", "left"):
            ax_top.spines[side].set_visible(False)
        ax_top.spines["bottom"].set_linewidth(0.55)
        ax_top.spines["bottom"].set_color("0.55")

    # Legend uses scatter-like markers (not thin KDE lines) so colors read clearly.
    marker_size = 7.5 if scatter_h < 1.6 else 6.5
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=marker_size,
            markerfacecolor=real_color,
            markeredgecolor="none",
            label="Real",
        ),
        plt.Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=marker_size,
            markerfacecolor=traj_color,
            markeredgecolor="none",
            label=gen_label,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.0,
        columnspacing=1.0,
        borderpad=0.0,
        handletextpad=0.35,
    )

    ax.scatter(
        x_true_sc[:, 0],
        x_true_sc[:, 1],
        s=5.0,
        c=real_color,
        alpha=0.48,
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )
    ax.scatter(
        x_gen_sc[:, 0],
        x_gen_sc[:, 1],
        s=6.5,
        c=traj_color,
        alpha=0.82,
        edgecolors="none",
        zorder=3,
        rasterized=True,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.4")

    if stage_label is not None:
        ax.text(
            0.035,
            0.965,
            stage_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            color="0.25",
        )

    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)



def collect_pi0_rows(
    runs: Path,
    *,
    lambdas: tuple[float, ...],
    epochs: int,
    methods: list[str],
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    rows: list[dict] = []
    for method in methods:
        for seed in seeds:
            if method == "naive":
                d = _ft_dir(runs, "naive", None, epochs, seed)
                mpath = d / "metrics.json"
                if not mpath.exists():
                    continue
                ev = json.loads(mpath.read_text())["eval"]
                # naive has no λ; plot once per λ slot for visual reference
                for lam in lambdas:
                    row = {
                        "method": "naive",
                        "lambda": float(lam),
                        "seed": int(seed),
                        "pi0_hat": float(ev["pi0_hat"]),
                    }
                    if "mean_loglik" in ev:
                        row["mean_loglik"] = float(ev["mean_loglik"])
                    rows.append(row)
                continue
            for lam in lambdas:
                if method == "oracle":
                    d = _oracle_dir(runs, lam, seed, epochs=epochs)
                else:
                    d = _ft_dir(runs, method, lam, epochs, seed)
                mpath = d / "metrics.json"
                if not mpath.exists():
                    continue
                ev = json.loads(mpath.read_text())["eval"]
                row = {
                    "method": method,
                    "lambda": float(lam),
                    "seed": int(seed),
                    "pi0_hat": float(ev["pi0_hat"]),
                }
                if "mean_loglik" in ev:
                    row["mean_loglik"] = float(ev["mean_loglik"])
                if "energy_mixture" in ev:
                    row["energy_mixture"] = float(ev["energy_mixture"])
                rows.append(row)
    return rows


def plot_pi0_strip(
    path: Path,
    rows: list[dict],
    *,
    lambdas: tuple[float, ...],
) -> None:
    """π̂₀ vs λ for trajectory: mean ± 1 s.d. interval on the ideal diagonal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    traj = [r for r in rows if r["method"] == "trajectory"]
    oracle = [r for r in rows if r["method"] == "oracle"]
    if not traj:
        print(f"No trajectory rows for stripplot; skip {path}")
        return

    # LaTeX-like math + compact serif labels
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7.2,
        }
    )

    traj_color = "#457b9d"
    joint_color = "#780000"

    # Square panel sized for one third of a text column.
    # Axes are cropped to the λ range so the diagonal fills the frame.
    lo, hi = 0.02, 0.98
    fig, ax = plt.subplots(figsize=(1.8, 1.7))
    ax.plot(
        [lo, hi],
        [lo, hi],
        ls="--",
        lw=1.3,
        c="0.55",
        zorder=1,
        label=r"Ideal $\hat{\pi}_0=\lambda$",
    )

    if oracle:
        ox, oy = [], []
        for lam in lambdas:
            pts = [r["pi0_hat"] for r in oracle if abs(r["lambda"] - lam) < 1e-9]
            if not pts:
                continue
            ox.append(lam)
            oy.append(float(np.mean(pts)))
        if ox:
            ox_a = np.asarray(ox, dtype=np.float64)
            oy_a = np.asarray(oy, dtype=np.float64)
            half = 0.035
            ax.hlines(
                oy_a,
                ox_a - half,
                ox_a + half,
                colors=joint_color,
                lw=2.0,
                zorder=5,
                label="Joint training",
            )

    xs, means, stds = [], [], []
    for lam in lambdas:
        pts = [r["pi0_hat"] for r in traj if abs(r["lambda"] - lam) < 1e-9]
        if not pts:
            continue
        xs.append(lam)
        means.append(float(np.mean(pts)))
        stds.append(float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0)

    ax.errorbar(
        xs,
        means,
        yerr=stds,
        fmt="o",
        color=traj_color,
        ecolor=traj_color,
        elinewidth=1.3,
        capsize=2.7,
        capthick=1.3,
        markersize=6.8,
        markerfacecolor=traj_color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linestyle="none",
        zorder=4,
        label="Trajectory replay",
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(list(lambdas))
    ax.set_yticks(list(lambdas))
    ax.set_xlabel(r"Mixture weight $\lambda$", labelpad=2)
    ax.set_ylabel(r"Generated mass $\hat{\pi}_0$", labelpad=2)
    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.0,
        handletextpad=0.25,
        borderpad=0.0,
        borderaxespad=0.15,
        labelspacing=0.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.9, length=3.2, pad=2)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Wrote {path} and {path.with_suffix('.pdf')}")


METHOD_STYLES: list[tuple[str, str, str]] = [
    ("oracle", "Joint training", "#780000"),
    ("generative", "Generative replay", "#c1121f"),
    ("trajectory", "Trajectory replay", "#457b9d"),
]


def plot_metric_box(
    path: Path,
    rows: list[dict],
    *,
    lambdas: tuple[float, ...],
    metric: str,
    ylabel: str,
    legend_loc: str = "best",
) -> None:
    """Grouped boxplot of ``metric`` vs λ, hue = method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    present = [
        (key, label, color)
        for key, label, color in METHOD_STYLES
        if any(r["method"] == key and metric in r for r in rows)
    ]
    if not present:
        print(f"No {metric} rows yet; skip {path}")
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
        }
    )

    n_h = len(present)
    width = 0.18
    offsets = (np.arange(n_h) - (n_h - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(3.6, 2.55))

    for j, (key, label, color) in enumerate(present):
        data, positions = [], []
        for i, lam in enumerate(lambdas):
            pts = [
                r[metric]
                for r in rows
                if r["method"] == key and abs(r["lambda"] - lam) < 1e-9 and metric in r
            ]
            if not pts:
                continue
            data.append(pts)
            positions.append(i + offsets[j])
            jitter = (np.arange(len(pts)) - (len(pts) - 1) / 2.0) * 0.012
            ax.scatter(
                np.full(len(pts), i + offsets[j]) + jitter,
                pts,
                s=18,
                c=color,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )
        if not data:
            continue
        ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="0.15", linewidth=1.0),
            whiskerprops=dict(color=color, linewidth=0.9),
            capprops=dict(color=color, linewidth=0.9),
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=0.9),
            zorder=3,
        )
        ax.plot([], [], color=color, lw=6, alpha=0.55, solid_capstyle="butt", label=label)

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{lam:g}" for lam in lambdas])
    ax.set_xlabel(r"Training mixture weight $\lambda$")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc=legend_loc, handlelength=1.4, borderpad=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {path} and {path.with_suffix('.pdf')}")


def plot_metric_summary(
    path: Path,
    rows: list[dict],
    *,
    lambdas: tuple[float, ...],
    metric: str,
    ylabel: str,
) -> None:
    """Seed points and mean ± SD across lambda for each method."""
    present = [
        (key, label, color)
        for key, label, color in METHOD_STYLES
        if any(r["method"] == key and metric in r for r in rows)
    ]
    if not present:
        print(f"No {metric} rows yet; skip {path}")
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.8,
        }
    )
    markers = {"oracle": "D", "generative": "s", "trajectory": "o"}
    offsets = np.linspace(-0.018, 0.018, len(present))
    fig, ax = plt.subplots(figsize=(3.45, 3.2))

    for offset, (key, label, color) in zip(offsets, present):
        means: list[float] = []
        stds: list[float] = []
        xs: list[float] = []
        for lam in lambdas:
            values = np.asarray(
                [
                    float(r[metric])
                    for r in rows
                    if r["method"] == key
                    and abs(float(r["lambda"]) - lam) < 1e-9
                    and metric in r
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            x = float(lam + offset)
            xs.append(x)
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
            jitter = (
                np.arange(values.size, dtype=np.float64)
                - (values.size - 1) / 2.0
            ) * 0.003
            ax.scatter(
                np.full(values.size, x) + jitter,
                values,
                s=28,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.42,
                zorder=3,
            )

        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=color,
            marker=markers[key],
            markersize=8.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="none",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            label=label,
            zorder=4,
        )

    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{lam:g}" for lam in lambdas])
    ax.set_xlim(min(lambdas) - 0.06, max(lambdas) + 0.06)
    ax.set_xlabel(r"Training mixture weight $\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.88", linewidth=0.6, alpha=0.8)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=1,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {path} and {path.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = mixture_data_root(REPO_ROOT, dim=args.dim)
    runs = mixture_runs_root(REPO_ROOT, dim=args.dim)
    out_dir = args.out_dir or (runs / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, meta = load_tasks(data_dir)
    lambdas = tuple(float(x) for x in meta.get("lambdas", DEFAULT_LAMBDAS))
    sigma = float(meta["sigma"])

    # Tight crop around the two modes (no forced square → less empty vertical space).
    # Square window with enough pad that ~99.9% of true points are inside.
    # Data-driven limits (no clipping) with mild y-padding so the panel is less wide.
    x_all = np.vstack([ds.subset(0).x, ds.subset(1).x])
    pad = 0.55 * sigma
    x_lo, x_hi = np.quantile(x_all[:, 0], [0.001, 0.999])
    y_lo, y_hi = np.quantile(x_all[:, 1], [0.001, 0.999])
    xlim = (float(x_lo - pad), float(x_hi + pad))
    ylim = (float(y_lo - pad), float(y_hi + pad))
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    max_aspect = 1.28  # width/height; closer to square than raw data aspect
    if dx / dy > max_aspect:
        extra = 0.5 * (dx / max_aspect - dy)
        ylim = (ylim[0] - extra, ylim[1] + extra)

    # --- one-update sequence: before / trajectory / naive ---
    if args.stage_panels and not args.skip_mixture_panels:
        stage_lambdas = tuple(args.stage_lambdas) if args.stage_lambdas else lambdas

        def _sample_ckpt(ckpt: Path, seed_offset: int) -> np.ndarray | None:
            if not ckpt.exists():
                print(f"skip stage panel: missing {ckpt}")
                return None
            model, _ = load_checkpoint(ckpt, device)
            return sample_unconditional(
                model,
                args.n_gen,
                device=device,
                seed=20_000 + args.panel_seed + seed_offset,
                ode_step_size=args.ode_step_size,
            )

        # The task-0 and naive models do not depend on λ; only their reference does.
        x_gen_task0 = _sample_ckpt(
            runs / f"pretrain_task0_seed{args.panel_seed}" / "checkpoint.pt", 0
        )
        x_gen_naive = _sample_ckpt(_naive_dir(runs, args.panel_seed) / "checkpoint.pt", 1)

        if x_gen_task0 is not None:
            out = out_dir / "stage0_task0.png"
            plot_mixture_panel(
                out,
                x_gen=x_gen_task0,
                x_true=ds.subset(0).x,
                xlim=xlim,
                ylim=ylim,
                dpi=args.mixture_dpi,
                gen_label="Task-0 model",
                gen_color=STAGE_COLORS["task0"],
                stage_label="Before update",
            )
            print(f"Wrote {out}")

        for lam in stage_lambdas:
            tag = _lam_tag(lam)
            x_mix = load_mixture(data_dir, lam).x
            x_gen_traj = _sample_ckpt(
                _ft_dir(runs, "trajectory", lam, args.epochs, args.panel_seed)
                / "checkpoint.pt",
                2 + int(10 * lam),
            )
            panels = [
                ("stage1_trajectory", x_gen_traj, "Trajectory replay", "trajectory"),
                ("stage1_naive", x_gen_naive, "Naive finetuning", "naive"),
            ]
            for name, x_gen, gen_label, color_key in panels:
                if x_gen is None:
                    continue
                out = out_dir / f"{name}_{tag}.png"
                plot_mixture_panel(
                    out,
                    x_gen=x_gen,
                    x_true=x_mix,
                    xlim=xlim,
                    ylim=ylim,
                    dpi=args.mixture_dpi,
                    gen_label=gen_label,
                    gen_color=STAGE_COLORS[color_key],
                    stage_label=rf"After update, $\lambda={lam:g}$",
                )
                print(f"Wrote {out}")

    # --- mixture thumbnails (trajectory) ---
    if not args.skip_mixture_panels and not args.stage_panels:
        for lam in lambdas:
            d = _ft_dir(runs, "trajectory", lam, args.epochs, args.panel_seed)
            ckpt = d / "checkpoint.pt"
            if not ckpt.exists():
                print(f"skip panel λ={lam}: missing {ckpt}")
                continue
            model, _ = load_checkpoint(ckpt, device)
            x_gen = sample_unconditional(
                model,
                args.n_gen,
                device=device,
                seed=10_000 + args.panel_seed + int(10 * lam),
                ode_step_size=args.ode_step_size,
            )
            x_true = load_mixture(data_dir, lam).x
            out = out_dir / f"mixture_traj_{_lam_tag(lam)}.png"
            plot_mixture_panel(
                out,
                x_gen=x_gen,
                x_true=x_true,
                xlim=xlim,
                ylim=ylim,
                dpi=args.mixture_dpi,
            )
            print(f"Wrote {out}")

    # --- stripplot + loglik ---
    rows = collect_pi0_rows(
        runs,
        lambdas=lambdas,
        epochs=args.epochs,
        methods=list(args.methods),
    )
    missing_oracle = [r for r in rows if r["method"] == "oracle"]
    if "oracle" in args.methods and not missing_oracle:
        print("Note: no oracle metrics yet — stripplot will omit oracle until jobs finish.")
    plot_pi0_strip(out_dir / "pi0_stripplot.png", rows, lambdas=lambdas)
    plot_metric_box(
        out_dir / "energy_boxplot.png",
        rows,
        lambdas=lambdas,
        metric="energy_mixture",
        ylabel="Energy distance",
        legend_loc="upper right",
    )
    plot_metric_summary(
        out_dir / "loglik_boxplot.png",
        rows,
        lambdas=lambdas,
        metric="mean_loglik",
        ylabel=r"Mean target log-likelihood $\uparrow$",
    )

    csv_path = out_dir / "pi0_strip_data.csv"
    fieldnames = ["method", "lambda", "seed", "pi0_hat", "mean_loglik", "energy_mixture"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
