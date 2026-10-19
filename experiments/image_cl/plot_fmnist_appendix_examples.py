#!/usr/bin/env python3
"""Render fixed, non-curated final-task FMNIST examples for the appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from trajectory_replay.data.split_fmnist import build_split_fmnist_tasks
from trajectory_replay.eval.stream_kld import (
    classify_images,
    load_fmnist_kld_classifier,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
ROWS = (
    ("Held-out real", "real"),
    ("Masip distillation", "masip"),
    ("Generative replay", "generative"),
    ("Trajectory replay", "trajectory"),
)


def _load_tensor(path: Path) -> torch.Tensor:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("x", "samples", "gens", "images"):
            if key in raw and torch.is_tensor(raw[key]):
                raw = raw[key]
                break
    if not torch.is_tensor(raw) or raw.ndim != 4:
        raise ValueError(f"{path}: expected an NCHW sample tensor")
    return raw.detach().float().cpu()


def _load_masip_grid(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.width % 10 or image.height % 10:
        raise ValueError(f"{path}: expected a 10x10 grid, got {image.size}")
    width, height = image.width // 10, image.height // 10
    tiles = []
    for row in range(10):
        for col in range(10):
            tile = image.crop(
                (col * width, row * height, (col + 1) * width, (row + 1) * height)
            )
            array = np.asarray(tile, dtype=np.float32) / 255.0
            tiles.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tiles)


def _first_per_task(
    images: torch.Tensor,
    predictions: torch.Tensor,
    task_classes: list[tuple[int, ...]],
) -> tuple[list[torch.Tensor], list[dict[str, int]]]:
    selected = []
    metadata = []
    for task_id, classes in enumerate(task_classes):
        class_set = set(classes)
        indices = [
            index
            for index, prediction in enumerate(predictions.tolist())
            if int(prediction) in class_set
        ]
        if not indices:
            raise RuntimeError(
                f"No generated sample assigned to task {task_id}, classes={classes}"
            )
        index = indices[0]
        selected.append(images[index])
        metadata.append(
            {
                "task_id": task_id,
                "sample_index": index,
                "predicted_class": int(predictions[index]),
            }
        )
    return selected, metadata


def _to_display(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu()
    if float(image.min()) < 0.0:
        image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)
    if image.shape[0] == 1:
        return image[0].numpy()
    return image.permute(1, 2, 0).numpy()


def plot_seed(seed: int, *, output_dir: Path, data_root: Path) -> Path:
    device = torch.device("cpu")
    classifier_dir = (
        REPO_ROOT
        / "baselines"
        / "diffusion_continual_learning"
        / "weights"
        / "cnn_fmnist"
    )
    classifier = load_fmnist_kld_classifier(classifier_dir, device=device)
    tasks = build_split_fmnist_tasks(
        seed=seed,
        data_root=data_root,
        image_size=32,
        n_experiences=5,
        val_fraction=0.1,
        download=False,
    )
    task_classes = [task.classes for task in tasks]

    real_images = []
    real_metadata = []
    real_rng = torch.Generator().manual_seed(90_000 + seed)
    for task in tasks:
        index = int(
            torch.randint(
                0,
                len(task.x_test),
                (1,),
                generator=real_rng,
            ).item()
        )
        real_images.append(task.x_test[index])
        real_metadata.append(
            {
                "task_id": task.task_id,
                "sample_index": index,
                "true_class": int(task.y_test[index]),
            }
        )

    fmnist_root = REPO_ROOT / "results_dir" / "image_cl" / "fmnist"
    source_paths = {
        "generative": (
            fmnist_root / f"generative_s{seed}" / "samples" / "task04_gen.pt"
        ),
        "trajectory": (
            fmnist_root
            / f"trajectory_mf_n4_eq_tfix_s{seed}"
            / "samples"
            / "task04_gen.pt"
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
    pools = {
        "generative": _load_tensor(source_paths["generative"]),
        "trajectory": _load_tensor(source_paths["trajectory"]),
        "masip": _load_masip_grid(source_paths["masip"]),
    }

    selected: dict[str, list[torch.Tensor]] = {"real": real_images}
    selection_metadata: dict[str, list[dict[str, int]]] = {"real": real_metadata}
    for method in ("masip", "generative", "trajectory"):
        predictions = classify_images(
            classifier,
            pools[method],
            device=device,
            batch_size=128,
        )
        selected[method], selection_metadata[method] = _first_per_task(
            pools[method],
            predictions,
            task_classes,
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
        }
    )
    fig, axes = plt.subplots(4, 5, figsize=(6.9, 5.1))
    for row, (row_label, method) in enumerate(ROWS):
        for task_id, ax in enumerate(axes[row]):
            image = _to_display(selected[method][task_id])
            if image.ndim == 2:
                ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(image, vmin=0.0, vmax=1.0)
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
        left=0.23,
        right=0.99,
        bottom=0.02,
        top=0.91,
        wspace=0.08,
        hspace=0.16,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fmnist_seed{seed}_per_task.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    metadata = {
        "seed": seed,
        "selection_rule": (
            "First saved generated sample whose fixed KLD-classifier prediction "
            "belongs to the task; no visual selection."
        ),
        "real_source": "held-out official Fashion-MNIST test split",
        "methods": {
            "masip": "Masip generative distillation, final task",
            "generative": "Generative replay ODE with adaptive lambda, final task",
            "trajectory": (
                "Trajectory replay MeanFlow N=4 with equal-task lambda, final task"
            ),
        },
        "sources": {key: str(path) for key, path in source_paths.items()},
        "task_classes": [list(classes) for classes in task_classes],
        "selected": selection_metadata,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "images" / "fmnist",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data_dir" / "torchvision",
    )
    args = parser.parse_args()
    output = plot_seed(
        int(args.seed),
        output_dir=args.output_dir,
        data_root=args.data_root,
    )
    print(f"Wrote {output}, {output.with_suffix('.pdf')}, and metadata")


if __name__ == "__main__":
    main()
