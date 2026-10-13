#!/usr/bin/env python3
"""Compact Stage B (c, p) data thumbnail (leaveout / mixture style).

Example:
  uv run python experiments/sim_conditional/stage_b_shift/plot_data.py
  uv run python experiments/sim_conditional/stage_b_shift/plot_data.py --n-contexts 6 --n-perturbations 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_conditional.stage_b_shift.common import (
    REPO_ROOT,
    load_stage_b_data,
    stage_b_data_root,
)

# Muted six-step red–cream–blue palette, adapted from the paper palette.
CTX_COLORS = [
    "#7b2d2f",  # burgundy
    "#a9454d",  # muted red
    "#c48277",  # dusty rose
    "#b8ac93",  # dimmed cream
    "#2f5364",  # muted navy
    "#7195a8",  # dusty blue (FT)
    "#91606a",
    "#657f8d",
    "#5a7180",
    "#987177",
]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-contexts", type=int, default=6)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-points-per-cell", type=int, default=120)
    p.add_argument(
        "--scatter-h",
        type=float,
        default=1.35,
        help="Scatter panel height in inches (thumbnail size).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or stage_b_data_root(
        REPO_ROOT,
        dim=args.dim,
        n_contexts=args.n_contexts,
        n_perturbations=args.n_perturbations,
    )
    arrays, meta = load_stage_b_data(data_dir)
    x = np.asarray(arrays["x"], dtype=np.float64)
    c = np.asarray(arrays["c"], dtype=np.int64)
    p = np.asarray(arrays["p"], dtype=np.int64)
    means = np.asarray(arrays["means"], dtype=np.float64)
    n_c = int(meta["n_contexts"])
    n_p = int(meta["n_perturbations"])
    ft_c = int(meta["ft_context"])
    ho_p = int(meta["heldout_pert"])
    ft_color = CTX_COLORS[ft_c % len(CTX_COLORS)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "legend.fontsize": 7.5,
        }
    )

    # Tight square frame around all cell means (+ samples).
    pad = 0.85
    pts = np.vstack([x[:, :2], means.reshape(-1, means.shape[-1])[:, :2]])
    x_lo, x_hi = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_lo, y_hi = float(pts[:, 1].min()), float(pts[:, 1].max())
    xlim = (x_lo - pad, x_hi + pad)
    ylim = (y_lo - pad, y_hi + pad)
    dx = max(xlim[1] - xlim[0], 1e-6)
    dy = max(ylim[1] - ylim[0], 1e-6)
    # Mild aspect clamp so the panel stays thumbnail-like.
    max_aspect = 1.15
    if dx / dy > max_aspect:
        extra = 0.5 * (dx / max_aspect - dy)
        ylim = (ylim[0] - extra, ylim[1] + extra)
        dy = ylim[1] - ylim[0]
    elif dy / dx > max_aspect:
        extra = 0.5 * (dy / max_aspect - dx)
        xlim = (xlim[0] - extra, xlim[1] + extra)
        dx = xlim[1] - xlim[0]

    scatter_h = float(args.scatter_h)
    scatter_w = scatter_h * (dx / dy)
    legend_in = 0.34
    side_pad_in = 0.035
    fig_w = scatter_w + 2 * side_pad_in
    fig_h = legend_in + scatter_h + 0.05
    fig = plt.figure(figsize=(fig_w, fig_h))
    left = side_pad_in / fig_w
    width = scatter_w / fig_w
    ax = fig.add_axes([left, 0.025 / fig_h, width, scatter_h / fig_h])

    rng = np.random.default_rng(0)
    for ci in range(n_c):
        for pj in range(n_p):
            m = (c == ci) & (p == pj)
            if not np.any(m):
                continue
            idx = np.flatnonzero(m)
            if idx.size > args.max_points_per_cell:
                idx = rng.choice(idx, size=args.max_points_per_cell, replace=False)
            is_heldout = ci == ft_c and pj == ho_p
            ax.scatter(
                x[idx, 0],
                x[idx, 1],
                s=5.2,
                alpha=0.16 if is_heldout else 0.38,
                c=ft_color if is_heldout else CTX_COLORS[ci % len(CTX_COLORS)],
                marker=MARKERS[pj % len(MARKERS)],
                edgecolors="none",
                linewidths=0,
                zorder=2,
                rasterized=True,
            )

    # Minimal mean glyphs keep the context/perturbation structure legible.
    for ci in range(n_c):
        for pj in range(n_p):
            is_heldout = ci == ft_c and pj == ho_p
            ax.scatter(
                [means[ci, pj, 0]],
                [means[ci, pj, 1]],
                s=10,
                alpha=0.55 if is_heldout else 1.0,
                c=ft_color if is_heldout else CTX_COLORS[ci % len(CTX_COLORS)],
                marker=MARKERS[pj % len(MARKERS)],
                edgecolors="white",
                linewidths=0.2,
                zorder=10,
            )

    heldout_mean = means[ft_c, ho_p, :2]
    ax.scatter(
        [heldout_mean[0]],
        [heldout_mean[1]],
        s=48,
        facecolors="none",
        edgecolors="0.2",
        marker=MARKERS[ho_p % len(MARKERS)],
        linewidths=0.8,
        zorder=11,
    )
    # Keep the callout outside the densest cloud.
    ax.annotate(
        "Held out",
        xy=heldout_mean,
        xytext=(-8, 10),
        textcoords="offset points",
        fontsize=6,
        color="0.25",
        ha="right",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": "0.35",
            "lw": 0.55,
            "shrinkA": 0,
            "shrinkB": 2,
        },
        zorder=12,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.5")

    c_handles = [
        mlines.Line2D(
            [],
            [],
            color=CTX_COLORS[i % len(CTX_COLORS)],
            marker="o",
            linestyle="None",
            markersize=4.0,
            markeredgecolor="none",
            label=rf"$c={i}$" + (r"$^\star$" if i == ft_c else ""),
        )
        for i in range(n_c)
    ]
    p_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.28",
            marker=MARKERS[j % len(MARKERS)],
            linestyle="None",
            # A rotated square has a larger visual footprint than the other
            # glyphs; reduce it slightly for optical alignment with its label.
            markersize=3.35 if j == 3 else 4.0,
            markeredgecolor="none",
            label=rf"$y={j}$",
        )
        for j in range(n_p)
    ]
    # Two compact rows above the frame (contexts, then perturbations).
    fig.legend(
        handles=c_handles,
        loc="upper center",
        ncol=min(n_c, 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.998),
        handlelength=0.65,
        columnspacing=0.24,
        borderpad=0.0,
        handletextpad=0.12,
        fontsize=5.5,
    )
    fig.legend(
        handles=p_handles,
        loc="upper center",
        ncol=n_p,
        frameon=False,
        bbox_to_anchor=(0.5, 0.895),
        handlelength=0.65,
        columnspacing=0.4,
        borderpad=0.0,
        handletextpad=0.12,
        fontsize=5.5,
    )

    out = data_dir / "data_overview.png"
    out_pdf = data_dir / "data_overview.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out} and {out_pdf}")


if __name__ == "__main__":
    main()
