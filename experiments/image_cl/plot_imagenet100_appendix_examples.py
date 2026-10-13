#!/usr/bin/env python3
"""Generate and render controlled ImageNet-100 appendix examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.models import ResNet50_Weights

from experiments.image_cl.image_cl_common import (
    load_checkpoint,
    resolve_latent_to_pixels,
)
from experiments.image_cl.plot_fmnist_appendix_grids import _mini_grid, _style
from trajectory_replay.data.split_imagenet import (
    DEFAULT_LATENT_DIR,
    build_split_imagenet100_tasks,
)
from trajectory_replay.eval.image_fid import generate_samples
from trajectory_replay.eval.stream_kld import load_imagenet2012_wnids


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "generative": "generative_g3_s{seed}",
    "trajectory": "trajectory_g3_s{seed}",
}
ROW_LABELS = (
    ("Held-out real", "real"),
    ("Generative replay", "generative"),
    ("Trajectory replay", "trajectory"),
)


def _category_names(synsets: list[str]) -> list[str]:
    wnids = load_imagenet2012_wnids()
    wnid_to_index = {wnid: index for index, wnid in enumerate(wnids)}
    names = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    return [str(names[wnid_to_index[synset]]) for synset in synsets]


def _selected_classes(task_classes: tuple[int, ...]) -> list[int]:
    """Deterministic first/middle/last categories from one task."""
    return [
        int(task_classes[0]),
        int(task_classes[len(task_classes) // 2]),
        int(task_classes[-1]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--ode-step-size", type=float, default=0.01)
    parser.add_argument("--n-per-class", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "images" / "imagenet100",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    tasks = build_split_imagenet100_tasks(
        seed=int(args.seed),
        latent_dir=args.latent_dir,
        n_experiences=3,
        val_fraction=0.1,
    )
    latent_blob = torch.load(
        args.latent_dir / "val.pt",
        map_location="cpu",
        weights_only=False,
    )
    synsets = list(latent_blob["synsets"])
    category_names = _category_names(synsets)
    classes_by_task = [_selected_classes(task.classes) for task in tasks]
    selected_classes = [class_id for task in classes_by_task for class_id in task]

    decode_args = argparse.Namespace(
        dataset="split_imagenet100",
        vae_id="stabilityai/sd-vae-ft-mse",
        fid_batch_size=int(args.batch_size) * 2,
    )
    to_pixels = resolve_latent_to_pixels(decode_args, device)
    assert to_pixels is not None

    real_latents = []
    real_labels = []
    real_indices: dict[str, list[int]] = {}
    generator = torch.Generator().manual_seed(90_000 + int(args.seed))
    for task in tasks:
        for class_id in classes_by_task[task.task_id]:
            indices = torch.nonzero(
                task.y_test == class_id,
                as_tuple=False,
            ).flatten()
            order = torch.randperm(len(indices), generator=generator)
            chosen = indices[order[: int(args.n_per_class)]]
            real_latents.append(task.x_test[chosen])
            real_labels.extend([class_id] * len(chosen))
            real_indices[str(class_id)] = [int(value) for value in chosen]
    real_images = to_pixels(torch.cat(real_latents))
    labels = torch.tensor(
        [
            class_id
            for class_id in selected_classes
            for _ in range(int(args.n_per_class))
        ],
        dtype=torch.long,
    )

    samples: dict[str, torch.Tensor] = {"real": real_images.cpu()}
    sources: dict[str, str] = {}
    sampling_seed = 80_000 + int(args.seed)
    for method, stem_template in METHODS.items():
        stem = stem_template.format(seed=int(args.seed))
        checkpoint = (
            REPO_ROOT
            / "results_dir"
            / "image_cl"
            / "imagenet100"
            / stem
            / "tasks"
            / "02"
            / "checkpoint.pt"
        )
        print(f"Loading {method}: {checkpoint}")
        model, _ = load_checkpoint(checkpoint, device)
        model.eval()
        samples[method] = generate_samples(
            model,
            len(labels),
            device=device,
            batch_size=int(args.batch_size),
            ode_method="euler",
            ode_step_size=float(args.ode_step_size),
            seed=sampling_seed,
            to_pixels=to_pixels,
            class_labels=labels,
            guidance_scale=float(args.cfg),
        )
        sources[method] = str(checkpoint)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cfg_tag = str(args.cfg).replace(".", "p")
    nfe = round(1.0 / float(args.ode_step_size))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = args.output_dir / (
        f"imagenet100_seed{args.seed}_cfg{cfg_tag}_euler{nfe}_samples.pt"
    )
    torch.save({"labels": labels, "samples": samples}, raw_output)

    _style()
    outputs = []
    for task_id, class_ids in enumerate(classes_by_task):
        fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.2))
        for row, (row_label, method) in enumerate(ROW_LABELS):
            for col, class_id in enumerate(class_ids):
                global_position = selected_classes.index(class_id)
                start = global_position * int(args.n_per_class)
                stop = start + int(args.n_per_class)
                grid = _mini_grid(
                    [image for image in samples[method][start:stop]],
                    slots=int(args.n_per_class),
                    ncol=3,
                )
                ax = axes[row, col]
                ax.imshow(grid, vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(category_names[class_id], fontsize=11, pad=5)
                if col == 0:
                    ax.set_ylabel(
                        row_label,
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=11,
                        labelpad=12,
                    )
        fig.suptitle(
            rf"ImageNet-100 $T_{{{task_id}}}$ after the final update "
            rf"(seed {args.seed}, CFG {args.cfg:g}, Euler {nfe})",
            fontsize=13,
            y=0.995,
        )
        fig.subplots_adjust(
            left=0.20,
            right=0.99,
            bottom=0.02,
            top=0.91,
            wspace=0.06,
            hspace=0.14,
        )
        output = args.output_dir / (
            f"imagenet100_seed{args.seed}_task{task_id}_"
            f"cfg{cfg_tag}_euler{nfe}.png"
        )
        fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        outputs.append(str(output))
        print(f"Wrote {output}")

    metadata = {
        "seed": int(args.seed),
        "cfg": float(args.cfg),
        "ode_method": "euler",
        "ode_step_size": float(args.ode_step_size),
        "nfe": nfe,
        "n_per_class": int(args.n_per_class),
        "sampling_seed": sampling_seed,
        "class_selection_rule": "first, middle, and last sorted class in each task",
        "classes_by_task": classes_by_task,
        "category_names": {
            str(class_id): category_names[class_id] for class_id in selected_classes
        },
        "real_indices": real_indices,
        "sources": sources,
        "raw_samples": str(raw_output),
        "figures": outputs,
    }
    metadata_path = args.output_dir / (
        f"imagenet100_seed{args.seed}_cfg{cfg_tag}_euler{nfe}.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
