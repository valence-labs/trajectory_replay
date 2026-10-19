#!/usr/bin/env python3
"""Render detailed and uncurated FMNIST appendix grids from saved samples."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from experiments.image_cl.plot_fmnist_appendix_examples import (
    REPO_ROOT,
    ROWS,
    SEEDS,
    _load_masip_grid,
    _load_tensor,
    _to_display,
)
from trajectory_replay.data.split_fmnist import build_split_fmnist_tasks
from trajectory_replay.eval.stream_kld import (
    classify_images,
    load_fmnist_kld_classifier,
)


TASK_COLORS = ("#8F0000", "#C1121F", "#F2C14E", "#68A0C0", "#003049")
OUTPUT_DIR = REPO_ROOT / "results_dir" / "image_cl" / "images" / "fmnist"
DATA_ROOT = REPO_ROOT / "data_dir" / "torchvision"


def _sources(seed: int, *, pool_tag: str = "") -> dict[str, Path]:
    root = REPO_ROOT / "results_dir" / "image_cl" / "fmnist"
    pool_root = REPO_ROOT / "results_dir" / "image_cl" / "images" / "fmnist" / "pools"
    tagged_suffix = f"_{pool_tag}" if pool_tag else ""
    generative_candidates = (
        pool_root / f"generative_seed{seed}_n512{tagged_suffix}.pt",
        pool_root / f"generative_seed{seed}_n512.pt",
        pool_root / f"generative_seed{seed}_n256_draw2.pt",
        pool_root / f"generative_seed{seed}_n256.pt",
    )
    trajectory_candidates = (
        pool_root / f"trajectory_seed{seed}_n512{tagged_suffix}.pt",
        pool_root / f"trajectory_seed{seed}_n512.pt",
        pool_root / f"trajectory_seed{seed}_n256_draw2.pt",
        pool_root / f"trajectory_seed{seed}_n256.pt",
    )
    generative_pool = next(
        (path for path in generative_candidates if path.is_file()),
        generative_candidates[-1],
    )
    trajectory_pool = next(
        (path for path in trajectory_candidates if path.is_file()),
        trajectory_candidates[-1],
    )
    return {
        "generative": (
            generative_pool
            if generative_pool.is_file()
            else root / f"generative_s{seed}" / "samples" / "task04_gen.pt"
        ),
        "trajectory": (
            trajectory_pool
            if trajectory_pool.is_file()
            else (
                root
                / f"trajectory_mf_n4_eq_tfix_s{seed}"
                / "samples"
                / "task04_gen.pt"
            )
        ),
        "masip": (
            REPO_ROOT
            / "results_dir"
            / "baselines"
            / "masip"
            / "split_fmnist"
            / "gr_diffusion_full_generation_distillation_steps_10_lambd_0.75_cnn"
            / str(seed)
            / "samples"
            / "0004.png"
        ),
    }


def _load_pools(
    seed: int, *, pool_tag: str = ""
) -> tuple[dict[str, torch.Tensor], dict[str, Path]]:
    sources = _sources(seed, pool_tag=pool_tag)
    pools = {
        "generative": _load_tensor(sources["generative"]),
        "trajectory": _load_tensor(sources["trajectory"]),
        "masip": _load_masip_grid(sources["masip"]),
    }
    return pools, sources


def _task_ids(predictions: torch.Tensor, task_classes: list[tuple[int, ...]]) -> list[int]:
    class_to_task = {
        int(class_id): task_id
        for task_id, classes in enumerate(task_classes)
        for class_id in classes
    }
    return [class_to_task[int(value)] for value in predictions.tolist()]


def _first_n_per_task(
    images: torch.Tensor,
    predictions: torch.Tensor,
    task_classes: list[tuple[int, ...]],
    *,
    n: int,
) -> tuple[list[list[torch.Tensor]], list[list[dict[str, int]]]]:
    selected: list[list[torch.Tensor]] = []
    metadata: list[list[dict[str, int]]] = []
    for task_id, classes in enumerate(task_classes):
        class_set = set(classes)
        indices = [
            index
            for index, prediction in enumerate(predictions.tolist())
            if int(prediction) in class_set
        ][:n]
        if len(indices) < n:
            raise RuntimeError(
                f"Need {n} distinct samples for task {task_id}; found {len(indices)}"
            )
        selected.append([images[index] for index in indices])
        metadata.append(
            [
                {
                    "sample_index": index,
                    "predicted_class": int(predictions[index]),
                }
                for index in indices
            ]
        )
    return selected, metadata


def _mini_grid(
    images: list[torch.Tensor],
    *,
    slots: int,
    ncol: int = 3,
) -> np.ndarray:
    arrays = [_to_display(image) for image in images]
    if not arrays:
        height = width = 32
        channels = 1
    else:
        height, width = arrays[0].shape[:2]
        channels = 1 if arrays[0].ndim == 2 else arrays[0].shape[2]
    nrow = (slots + ncol - 1) // ncol
    gap = 2
    shape = (
        (nrow * height + (nrow - 1) * gap, ncol * width + (ncol - 1) * gap)
        if channels == 1
        else (
            nrow * height + (nrow - 1) * gap,
            ncol * width + (ncol - 1) * gap,
            channels,
        )
    )
    canvas = np.zeros(shape, dtype=np.float32)
    for index in range(slots):
        row, col = divmod(index, ncol)
        y, x = row * (height + gap), col * (width + gap)
        if index < len(arrays):
            canvas[y : y + height, x : x + width] = arrays[index]
        else:
            for diagonal in range(min(height, width)):
                canvas[y + diagonal, x + diagonal] = 0.55
                canvas[y + diagonal, x + width - diagonal - 1] = 0.55
    return canvas


def _bordered_grid(
    images: torch.Tensor,
    task_ids: list[int],
    *,
    count: int = 64,
    nrow: int = 8,
    border: int = 2,
    gap: int = 2,
) -> np.ndarray:
    count = min(count, len(images))
    first = _to_display(images[0])
    height, width = first.shape[:2]
    tile_h, tile_w = height + 2 * border, width + 2 * border
    ncol = nrow
    rows = (count + ncol - 1) // ncol
    canvas = np.ones(
        (rows * tile_h + (rows - 1) * gap, ncol * tile_w + (ncol - 1) * gap, 3),
        dtype=np.float32,
    )
    for index in range(count):
        row, col = divmod(index, ncol)
        y = row * (tile_h + gap)
        x = col * (tile_w + gap)
        color = np.asarray(
            tuple(int(TASK_COLORS[task_ids[index]][i : i + 2], 16) for i in (1, 3, 5)),
            dtype=np.float32,
        ) / 255.0
        canvas[y : y + tile_h, x : x + tile_w] = color
        image = _to_display(images[index])
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        canvas[y + border : y + border + height, x + border : x + border + width] = image
    return canvas


def _balanced_real_pool(tasks: list, *, seed: int, count: int = 64) -> tuple[torch.Tensor, list[int]]:
    per_task = [count // len(tasks)] * len(tasks)
    for task_id in range(count % len(tasks)):
        per_task[task_id] += 1
    generator = torch.Generator().manual_seed(80_000 + seed)
    images = []
    task_ids = []
    for task, n_task in zip(tasks, per_task, strict=True):
        indices = torch.randperm(len(task.x_test), generator=generator)[:n_task]
        images.append(task.x_test[indices])
        task_ids.extend([task.task_id] * n_task)
    pool = torch.cat(images)
    order = torch.randperm(len(pool), generator=generator)
    return pool[order], [task_ids[int(index)] for index in order]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
        }
    )


def plot_detailed_seed(
    classifier: torch.nn.Module,
    *,
    seed: int,
    n_per_task: int = 6,
    include_masip: bool = True,
    pool_tag: str = "",
) -> Path:
    tasks = build_split_fmnist_tasks(
        seed=seed,
        data_root=DATA_ROOT,
        image_size=32,
        n_experiences=5,
        val_fraction=0.1,
        download=False,
    )
    task_classes = [task.classes for task in tasks]
    pools, sources = _load_pools(seed, pool_tag=pool_tag)
    selected: dict[str, list[list[torch.Tensor]]] = {}
    metadata: dict[str, object] = {}

    real_selected = []
    real_meta = []
    generator = torch.Generator().manual_seed(90_000 + seed)
    for task in tasks:
        indices = torch.randperm(len(task.x_test), generator=generator)[:n_per_task]
        real_selected.append([task.x_test[index] for index in indices])
        real_meta.append(
            [
                {"sample_index": int(index), "true_class": int(task.y_test[index])}
                for index in indices
            ]
        )
    selected["real"] = real_selected
    metadata["real"] = real_meta

    generated_methods = (
        ("masip", "generative", "trajectory")
        if include_masip
        else ("generative", "trajectory")
    )
    for method in generated_methods:
        predictions = classify_images(
            classifier,
            pools[method],
            device=torch.device("cpu"),
            batch_size=128,
        )
        selected[method], metadata[method] = _first_n_per_task(
            pools[method],
            predictions,
            task_classes,
            n=n_per_task,
        )

    _style()
    rows = ROWS if include_masip else tuple(row for row in ROWS if row[1] != "masip")
    fig, axes = plt.subplots(
        len(rows),
        5,
        figsize=(10.8, 7.0 if include_masip else 5.5),
    )
    for row, (row_label, method) in enumerate(rows):
        for task_id, ax in enumerate(axes[row]):
            grid = _mini_grid(
                selected[method][task_id],
                slots=n_per_task,
            )
            ax.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                classes = ",".join(str(value) for value in task_classes[task_id])
                ax.set_title(rf"$T_{{{task_id}}}$  ({classes})", fontsize=11, pad=5)
            if task_id == 0:
                ax.set_ylabel(
                    row_label,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=11,
                    labelpad=12,
                )
    fig.suptitle(
        rf"Fashion-MNIST after the final update (seed {seed})",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.995,
        bottom=0.02,
        top=0.90,
        wspace=0.06,
        hspace=0.16,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"per_task_n{n_per_task}"
        if include_masip
        else f"generative_vs_trajectory_n{n_per_task}"
    )
    if pool_tag:
        suffix += f"_{pool_tag}"
    output = OUTPUT_DIR / f"fmnist_seed{seed}_{suffix}.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "seed": seed,
                "n_slots_per_task": n_per_task,
                "include_masip": include_masip,
                "pool_tag": pool_tag,
                "selection_rule": (
                    f"Up to the first {n_per_task} classifier-assigned saved "
                    "samples per task; crossed slots indicate insufficient "
                    "samples in the fixed pool."
                ),
                "sources": {key: str(path) for key, path in sources.items()},
                "selected": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def plot_uncurated_seed(seed: int, classifier: torch.nn.Module) -> Path:
    tasks = build_split_fmnist_tasks(
        seed=seed,
        data_root=DATA_ROOT,
        image_size=32,
        n_experiences=5,
        val_fraction=0.1,
        download=False,
    )
    task_classes = [task.classes for task in tasks]
    pools, sources = _load_pools(seed)
    real_pool, real_task_ids = _balanced_real_pool(tasks, seed=seed)
    grids = {"real": _bordered_grid(real_pool, real_task_ids)}
    prediction_metadata = {}
    for method in ("masip", "generative", "trajectory"):
        pool = pools[method][:64]
        predictions = classify_images(
            classifier,
            pool,
            device=torch.device("cpu"),
            batch_size=128,
        )
        task_ids = _task_ids(predictions, task_classes)
        grids[method] = _bordered_grid(pool, task_ids)
        prediction_metadata[method] = {
            "predicted_classes": [int(value) for value in predictions],
            "assigned_tasks": task_ids,
            "task_counts": [task_ids.count(task_id) for task_id in range(5)],
        }

    _style()
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.0))
    for ax, (row_label, method) in zip(axes, ROWS, strict=True):
        ax.imshow(grids[method])
        ax.set_title(row_label, fontsize=11, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        Patch(facecolor=color, edgecolor="none", label=rf"$T_{{{task_id}}}$")
        for task_id, color in enumerate(TASK_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        rf"Uncurated final-stream samples (seed {seed})",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.15, top=0.84, wspace=0.06)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"fmnist_seed{seed}_uncurated_stream.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "seed": seed,
                "n_displayed_per_method": 64,
                "selection_rule": "First 64 saved samples; no filtering or visual selection.",
                "border_rule": "KLD-classifier-assigned task.",
                "sources": {key: str(path) for key, path in sources.items()},
                "predictions": prediction_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    classifier_dir = (
        REPO_ROOT
        / "baselines"
        / "diffusion_continual_learning"
        / "weights"
        / "cnn_fmnist"
    )
    classifier = load_fmnist_kld_classifier(
        classifier_dir,
        device=torch.device("cpu"),
    )
    outputs = [plot_detailed_seed(classifier, seed=42, n_per_task=6)]
    outputs.extend(plot_uncurated_seed(seed, classifier) for seed in SEEDS)
    for output in outputs:
        print(f"Wrote {output}, {output.with_suffix('.pdf')}, and metadata")


if __name__ == "__main__":
    main()
