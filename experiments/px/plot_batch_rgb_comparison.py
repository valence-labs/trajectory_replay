#!/usr/bin/env python3
"""Render identical RxRx1 perturbations across biological experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from experiments.px.dataset import METADATA_PATH, cp2rgb, encode


def _to_rgb(path: str) -> np.ndarray:
    image = np.load(path)
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = encode(tensor).unsqueeze(0)
    rgb = cp2rgb(tensor).clamp(-1, 1).squeeze(0)
    return ((rgb + 1.0) * 0.5).permute(1, 2, 0).numpy()


def _grid(paths: list[str], *, gap: int = 2) -> np.ndarray:
    images = [_to_rgb(path) for path in paths]
    if len(images) != 4:
        raise ValueError(f"Expected four images, got {len(images)}")
    height, width, channels = images[0].shape
    canvas = np.ones(
        (2 * height + gap, 2 * width + gap, channels),
        dtype=np.float32,
    )
    canvas[:height, :width] = images[0]
    canvas[:height, width + gap :] = images[1]
    canvas[height + gap :, :width] = images[2]
    canvas[height + gap :, width + gap :] = images[3]
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(METADATA_PATH))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_dir/px/images/rxrx1_batch_rgb_comparison.png"),
    )
    parser.add_argument("--n-perturbations", type=int, default=4)
    args = parser.parse_args()

    metadata = pl.read_csv(args.metadata)
    eval_names = sorted(
        metadata.filter(
            (pl.col("ANNOT") == "treated") & pl.col("is_eval_pert")
        )["CPD_NAME"]
        .unique()
        .to_list()
    )
    selected_indices = np.linspace(
        0,
        len(eval_names) - 1,
        args.n_perturbations,
        dtype=int,
    )
    row_names = ["UNTREATED", *(eval_names[index] for index in selected_indices)]
    experiments = (
        metadata.select(["experiment_id", "BATCH"])
        .unique()
        .sort("experiment_id")
        .to_dicts()
    )

    selected: dict[str, dict[str, list[str]]] = {}
    fig, axes = plt.subplots(
        len(row_names),
        len(experiments),
        figsize=(6.3, 2.05 * len(row_names)),
        squeeze=False,
    )
    for row_id, name in enumerate(row_names):
        selected[name] = {}
        for col_id, experiment in enumerate(experiments):
            batch = str(experiment["BATCH"])
            annotation = "negative_control" if name == "UNTREATED" else "treated"
            subset = metadata.filter(
                (pl.col("BATCH") == batch)
                & (pl.col("SPLIT") == "test")
                & (pl.col("ANNOT") == annotation)
                & (pl.col("CPD_NAME") == name)
            ).sort("SAMPLE_KEY")
            paths = [str(path) for path in subset["path"].head(4).to_list()]
            if len(paths) != 4:
                raise RuntimeError(
                    f"{batch}/{name} has only {len(paths)} test images"
                )
            selected[name][batch] = paths
            ax = axes[row_id, col_id]
            ax.imshow(_grid(paths), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_id == 0:
                ax.set_title(batch, fontsize=12)
            if col_id == 0:
                ax.set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle(
        "RxRx1 test images: matched perturbations across experiments",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), pad=0.35, h_pad=0.55, w_pad=0.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        args.output.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    metadata_output = args.output.with_suffix(".json")
    metadata_output.write_text(
        json.dumps(
            {
                "source": str(args.metadata),
                "split": "test",
                "selection": (
                    "UNTREATED plus evenly spaced names in the sorted list of "
                    "100 benchmark perturbations; first four SAMPLE_KEY rows."
                ),
                "rgb_projection": "experiments.px.dataset.cp2rgb; fixed clamp [-1,1]",
                "selected_paths": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)
    print(metadata_output)


if __name__ == "__main__":
    main()
