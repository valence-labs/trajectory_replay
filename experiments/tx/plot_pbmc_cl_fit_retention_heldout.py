#!/usr/bin/env python3
"""Plot PBMC continual-learning fit/retention, colored by heldout performance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass(frozen=True)
class Point:
    family: str
    label: str
    fit: float
    retention: float
    heldout: float


POINTS = (
    Point("Naive freeze", "vf=0, pert=0, ctx=0", 0.233, 0.491, 0.235),
    Point("Naive freeze", "vf=0, pert=1, ctx=1", 0.232, 0.400, 0.207),
    Point("Naive full", "vf=1, pert=0, ctx=0", 0.290, 0.056, 0.151),
    Point("Naive full", "vf=1, pert=0, ctx=1", 0.290, 0.056, 0.151),
    Point("Naive full", "vf=1, pert=1, ctx=0", 0.289, 0.064, 0.154),
    Point("Naive full", "vf=1, pert=1, ctx=1", 0.289, 0.064, 0.154),
    Point("LoRA", "r=4", 0.273, 0.154, 0.214),
    Point("LoRA", "r=8", 0.279, 0.155, 0.217),
    Point("LoRA", "r=16", 0.287, 0.166, 0.225),
    Point("GFT / two-stage", "GFT sigmoid", 0.291, 0.061, 0.156),
    Point("GFT / two-stage", "GFT linear", 0.291, 0.060, 0.157),
    Point("GFT / two-stage", "two-stage 10%", 0.289, 0.059, 0.155),
    Point("GFT / two-stage", "two-stage 50%", 0.298, 0.091, 0.171),
    Point("Real replay", "lambda-grid", 0.498, 0.483, 0.350),
    Point("Real replay", "lambda-equal", 0.496, 0.500, 0.355),
    Point("Generative replay", "ODE lambda-grid", 0.475, 0.383, 0.320),
    Point("Generative replay", "ODE lambda-equal", 0.471, 0.403, 0.330),
    Point("Generative replay", "MeanFlow N=1 lambda-grid", 0.437, 0.306, 0.289),
    Point("Generative replay", "MeanFlow N=1 lambda-equal", 0.440, 0.340, 0.314),
    Point("Generative replay", "MeanFlow N=4 lambda-grid", 0.446, 0.316, 0.305),
    Point("Generative replay", "MeanFlow N=4 lambda-equal", 0.441, 0.345, 0.323),
    Point("Trajectory replay", "ODE lambda-grid", 0.496, 0.435, 0.332),
    Point("Trajectory replay", "ODE lambda-equal", 0.515, 0.450, 0.348),
    Point("Trajectory replay", "MeanFlow N=1 lambda-grid", 0.493, 0.430, 0.333),
    Point("Trajectory replay", "MeanFlow N=1 lambda-equal", 0.514, 0.449, 0.349),
    Point("Trajectory replay", "MeanFlow N=4 lambda-grid", 0.493, 0.431, 0.333),
    Point("Trajectory replay", "MeanFlow N=4 lambda-equal", 0.515, 0.451, 0.350),
)

MARKERS = {
    "Naive freeze": "s",
    "Naive full": "o",
    "LoRA": "D",
    "GFT / two-stage": "^",
    "Real replay": "*",
    "Generative replay": "P",
    "Trajectory replay": "X",
}


def main() -> None:
    output = (
        Path(__file__).parent
        / "tables"
        / "pbmc_cl_pca100_fit_retention_heldout.png"
    )
    norm = Normalize(
        vmin=min(point.heldout for point in POINTS),
        vmax=max(point.heldout for point in POINTS),
    )
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(10, 8))
    for family, marker in MARKERS.items():
        points = [point for point in POINTS if point.family == family]
        ax.scatter(
            [point.fit for point in points],
            [point.retention for point in points],
            c=[point.heldout for point in points],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=150 if family == "Real replay" else 105,
            linewidths=0.8,
            edgecolors="white",
            label=family,
            zorder=3,
        )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label(
        "Generalization (Pearson Δ, later-donor STATE test)", fontsize=14
    )

    ax.annotate(
        "Naive freeze\n(retention only)",
        xy=(0.233, 0.491),
        xytext=(0.245, 0.500),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "Real replay\n(fit + retention)",
        xy=(0.496, 0.500),
        xytext=(0.420, 0.480),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "Trajectory replay",
        xy=(0.515, 0.450),
        xytext=(0.440, 0.425),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "Naive full\n(fit only)",
        xy=(0.290, 0.060),
        xytext=(0.315, 0.030),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "LoRA",
        xy=(0.279, 0.155),
        xytext=(0.310, 0.185),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )

    ax.set_title(
        "Heldout is high only when both fit and retention are high",
        fontsize=17,
    )
    ax.set_xlabel("Fit (Pearson Δ, later-donor train)", fontsize=14)
    ax.set_ylabel("Retention (Pearson Δ, T0 train)", fontsize=14)
    ax.set_xlim(0.20, 0.535)
    ax.set_ylim(0.02, 0.525)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", frameon=False, fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Displayed final scores use K=500; λ-grid checkpoints were selected with K=200.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
