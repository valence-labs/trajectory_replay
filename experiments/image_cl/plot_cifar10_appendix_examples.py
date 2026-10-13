#!/usr/bin/env python3
"""Render nine distinct CIFAR-10 examples per task for all appendix seeds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.image_cl.plot_fmnist_appendix_grids import (
    REPO_ROOT,
    SEEDS,
    _first_n_per_task,
    _load_tensor,
    _mini_grid,
    _style,
)
from trajectory_replay.data.split_cifar10 import build_split_cifar10_tasks
from trajectory_replay.eval.stream_kld import (
    classify_images,
    load_fmnist_kld_classifier,
)


OUTPUT_DIR = REPO_ROOT / "results_dir" / "image_cl" / "images" / "cifar10"
DATA_ROOT = REPO_ROOT / "data_dir" / "torchvision"
ROWS = (
    ("Held-out real", "real"),
    ("Generative replay", "generative"),
    ("Trajectory replay", "trajectory"),
)


def plot_seed(
    seed: int,
    classifier: torch.nn.Module,
    *,
    n_per_task: int = 9,
    pool_tag: str = "",
) -> Path:
    tasks = build_split_cifar10_tasks(
        seed=seed,
        data_root=DATA_ROOT,
        image_size=32,
        n_experiences=5,
        val_fraction=0.1,
        download=False,
    )
    task_classes = [task.classes for task in tasks]
    pool_dir = OUTPUT_DIR / "pools"
    tagged_suffix = f"_{pool_tag}" if pool_tag else ""
    sources = {
        method: pool_dir / f"{method}_seed{seed}_n512{tagged_suffix}.pt"
        for method in ("generative", "trajectory")
    }
    pools = {method: _load_tensor(path) for method, path in sources.items()}

    selected: dict[str, list[list[torch.Tensor]]] = {}
    selection_metadata: dict[str, object] = {}
    real_selected = []
    real_metadata = []
    generator = torch.Generator().manual_seed(95_000 + seed)
    for task in tasks:
        indices = torch.randperm(len(task.x_test), generator=generator)[:n_per_task]
        real_selected.append([task.x_test[index] for index in indices])
        real_metadata.append(
            [
                {"sample_index": int(index), "true_class": int(task.y_test[index])}
                for index in indices
            ]
        )
    selected["real"] = real_selected
    selection_metadata["real"] = real_metadata

    for method in ("generative", "trajectory"):
        predictions = classify_images(
            classifier,
            pools[method],
            device=torch.device("cpu"),
            batch_size=128,
        )
        selected[method], selection_metadata[method] = _first_n_per_task(
            pools[method],
            predictions,
            task_classes,
            n=n_per_task,
        )

    _style()
    fig, axes = plt.subplots(3, 5, figsize=(10.8, 5.6))
    for row, (row_label, method) in enumerate(ROWS):
        for task_id, ax in enumerate(axes[row]):
            grid = _mini_grid(
                selected[method][task_id],
                slots=n_per_task,
                ncol=3,
            )
            ax.imshow(grid, vmin=0.0, vmax=1.0)
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
        rf"CIFAR-10 after the final update (seed {seed})",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.995,
        bottom=0.02,
        top=0.90,
        wspace=0.06,
        hspace=0.14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{pool_tag}" if pool_tag else ""
    output = (
        OUTPUT_DIR
        / f"cifar10_seed{seed}_generative_vs_trajectory_n9{suffix}.png"
    )
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "seed": seed,
                "n_per_task": n_per_task,
                "pool_tag": pool_tag,
                "selection_rule": (
                    f"First {n_per_task} classifier-assigned samples per task "
                    "from a fixed 512-sample pool; no visual selection."
                ),
                "methods": {
                    "generative": "Generative replay ODE with adaptive lambda",
                    "trajectory": "Trajectory replay ODE with adaptive lambda",
                },
                "sources": {key: str(path) for key, path in sources.items()},
                "selected": selection_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    classifier = load_fmnist_kld_classifier(
        REPO_ROOT
        / "baselines"
        / "diffusion_continual_learning"
        / "weights"
        / "cnn_cifar10",
        device=torch.device("cpu"),
    )
    for seed in SEEDS:
        output = plot_seed(seed, classifier)
        print(f"Wrote {output}, {output.with_suffix('.pdf')}, and metadata")


if __name__ == "__main__":
    main()
