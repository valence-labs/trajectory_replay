#!/usr/bin/env python3
"""Plot final Replogle continual-learning fit and retention."""

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
    Point("Naive freeze", "vf=0, pert=0, ctx=0", 0.193, 0.629, 0.163),
    Point("Naive freeze", "vf=0, pert=1, ctx=1", 0.359, 0.595, 0.169),
    Point("Naive full", "vf=1, pert=0, ctx=0", 0.616, 0.216, 0.351),
    Point("Naive full", "vf=1, pert=0, ctx=1", 0.616, 0.216, 0.351),
    Point("Naive full", "vf=1, pert=1, ctx=0", 0.616, 0.214, 0.353),
    Point("Naive full", "vf=1, pert=1, ctx=1", 0.616, 0.214, 0.353),
    Point("LoRA", "r=4", 0.551, 0.259, 0.391),
    Point("LoRA", "r=8", 0.585, 0.260, 0.386),
    Point("LoRA", "r=16", 0.608, 0.264, 0.382),
    Point("GFT / two-stage", "GFT sigmoid", 0.616, 0.226, 0.361),
    Point("GFT / two-stage", "GFT linear", 0.615, 0.226, 0.363),
    Point("GFT / two-stage", "two-stage 10%", 0.618, 0.227, 0.361),
    Point("GFT / two-stage", "two-stage 50%", 0.617, 0.266, 0.377),
    Point("Real replay", "lambda-grid", 0.610, 0.632, 0.406),
    Point("Real replay", "lambda-equal", 0.613, 0.620, 0.401),
    Point("Generative replay", "ODE lambda-grid", 0.611, 0.594, 0.403),
    Point("Generative replay", "ODE lambda-equal", 0.613, 0.588, 0.400),
    Point("Generative replay", "MeanFlow N=1 lambda-grid", 0.611, 0.580, 0.403),
    Point("Generative replay", "MeanFlow N=1 lambda-equal", 0.613, 0.573, 0.400),
    Point("Generative replay", "MeanFlow N=4 lambda-grid", 0.609, 0.588, 0.407),
    Point("Generative replay", "MeanFlow N=4 lambda-equal", 0.613, 0.581, 0.400),
    Point("Trajectory replay", "ODE lambda-grid", 0.612, 0.619, 0.399),
    Point("Trajectory replay", "ODE lambda-equal", 0.610, 0.606, 0.398),
    Point("Trajectory replay", "MeanFlow N=1 lambda-grid", 0.613, 0.620, 0.404),
    Point("Trajectory replay", "MeanFlow N=1 lambda-equal", 0.608, 0.604, 0.392),
    Point("Trajectory replay", "MeanFlow N=4 lambda-grid", 0.613, 0.620, 0.404),
    Point("Trajectory replay", "MeanFlow N=4 lambda-equal", 0.608, 0.604, 0.392),
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
        / "replogle_cl_pca50_fit_retention_heldout.png"
    )
    norm = Normalize(vmin=0.34, vmax=0.41, clip=True)
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
        "Generalization (Pearson Δ, HepG2 heldout)", fontsize=14
    )

    ax.annotate(
        "Naive freeze\n(retention only)",
        xy=(0.193, 0.629),
        xytext=(0.225, 0.620),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "Real replay",
        xy=(0.610, 0.632),
        xytext=(0.505, 0.642),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "Trajectory replay",
        xy=(0.612, 0.619),
        xytext=(0.470, 0.603),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "LoRA",
        xy=(0.585, 0.260),
        xytext=(0.485, 0.300),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )
    ax.annotate(
        "Naive full\n(fit only)",
        xy=(0.616, 0.216),
        xytext=(0.525, 0.185),
        arrowprops={"arrowstyle": "-", "color": "0.65"},
        fontsize=12,
    )

    ax.set_title(
        "Heldout is high only when both fit and retention are high",
        fontsize=17,
    )
    ax.set_xlabel("Fit (Pearson Δ, HepG2 train)", fontsize=14)
    ax.set_ylabel("Retention (Pearson Δ, T0 train)", fontsize=14)
    ax.set_xlim(0.16, 0.645)
    ax.set_ylim(0.18, 0.655)
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
