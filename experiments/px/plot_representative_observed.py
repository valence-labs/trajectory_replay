#!/usr/bin/env python3
"""Render observed RxRx1 references for representative perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from experiments.px.dataset import METADATA_PATH, cp2rgb, encode
from experiments.px.plot_representative_generations import (
    EXPERIMENTS,
    PERTURBATIONS,
    REPO_ROOT,
    _decorate_axis,
    _mosaic,
    _style,
)


OUT_DIR = REPO_ROOT / "results_dir" / "px" / "images" / "representative"


def _rgb(path: str) -> np.ndarray:
    image = np.load(path)
    tensor = encode(torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0)
    rgb = cp2rgb(tensor).clamp(-1, 1)
    return (
        ((rgb + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )


def collect(metadata_path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    metadata = pl.read_csv(metadata_path)
    observed: dict[str, dict[str, np.ndarray]] = {}
    selected_paths: dict[str, dict[str, list[str]]] = {}
    for perturbation in PERTURBATIONS:
        observed[perturbation] = {}
        selected_paths[perturbation] = {}
        for batch in EXPERIMENTS:
            frame = metadata.filter(
                (pl.col("CPD_NAME") == perturbation)
                & (pl.col("BATCH") == batch)
                & (pl.col("SPLIT") == "test")
                & pl.col("is_eval_pert")
                & (pl.col("ANNOT") == "treated")
            ).sort("SAMPLE_KEY")
            paths = [str(path) for path in frame["path"].head(9).to_list()]
            if len(paths) != 9:
                raise ValueError(
                    f"{perturbation}/{batch}: expected nine test images, got {len(paths)}"
                )
            observed[perturbation][batch] = np.stack([_rgb(path) for path in paths])
            selected_paths[perturbation][batch] = paths
    provenance = {
        "metadata": str(metadata_path),
        "split": "test",
        "selection": "First nine SAMPLE_KEY rows within each perturbation and batch.",
        "rgb_projection": "experiments.px.dataset.cp2rgb; fixed clamp [-1,1]",
        "selected_paths": selected_paths,
    }
    return observed, provenance


def render_one(observed: dict[str, dict[str, np.ndarray]], perturbation: str) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    for column, batch in enumerate(EXPERIMENTS):
        _decorate_axis(
            axes[column],
            _mosaic(observed[perturbation][batch]),
            title=batch,
            ylabel="Observed" if column == 0 else None,
        )
    fig.suptitle(perturbation, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.35, w_pad=0.35)
    path = OUT_DIR / f"{perturbation.lower()}_observed.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_combined(observed: dict[str, dict[str, np.ndarray]]) -> None:
    _style()
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.4))
    for row, perturbation in enumerate(PERTURBATIONS):
        for column, batch in enumerate(EXPERIMENTS):
            _decorate_axis(
                axes[row, column],
                _mosaic(observed[perturbation][batch]),
                title=batch if row == 0 else None,
                ylabel=perturbation if column == 0 else None,
            )
    fig.tight_layout(pad=0.35, h_pad=0.45, w_pad=0.35)
    path = OUT_DIR / "rxrx1_observed_references.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observed, provenance = collect(Path(METADATA_PATH))
    for perturbation in PERTURBATIONS:
        render_one(observed, perturbation)
    render_combined(observed)
    (OUT_DIR / "observed_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()
