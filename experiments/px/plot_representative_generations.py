#!/usr/bin/env python3
"""Render matched final-checkpoint PX generations across experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from experiments.px.dataset import (
    IMG_SIZE,
    METADATA_PATH,
    NUM_CHANNELS,
    NUM_EXPERIMENTS,
    NUM_PERTURBATIONS,
    cp2rgb,
    decode,
)
from experiments.px.model import ADMWrapper
from experiments.px.trainer import generate


REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATIONS = ("TRNT1", "NAA50", "EPPK1")
EXPERIMENTS = ("U2OS-01", "U2OS-02", "U2OS-03")
METHODS = {
    "Generative replay": (
        REPO_ROOT
        / "results_dir"
        / "px"
        / "cl_final"
        / "generative_ode_equal_s42"
        / "2026-09-18"
        / "13-06-29"
        / "DEBUG_23451482546800"
        / "tasks"
        / "02"
        / "checkpoint.ckpt"
    ),
    "Trajectory replay": (
        REPO_ROOT
        / "results_dir"
        / "px"
        / "cl_final"
        / "trajectory_ode_equal_s42"
        / "2026-09-18"
        / "13-06-28"
        / "DEBUG_23451482563184"
        / "tasks"
        / "02"
        / "checkpoint.ckpt"
    ),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def _strip_state_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module._orig_mod.", "module.")
    clean: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        clean[key] = value
    return clean


def _load_model(checkpoint_path: Path, device: torch.device) -> ADMWrapper:
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu",
    )
    state = _strip_state_prefix(checkpoint["ema"])
    model = ADMWrapper(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=1,
        use_cell_type_conditioning=False,
    )
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model = ADMWrapper(
            in_channels=NUM_CHANNELS,
            input_size=IMG_SIZE,
            y_dim=NUM_PERTURBATIONS,
            e_dim=NUM_EXPERIMENTS,
            c_dim=1,
            use_cell_type_conditioning=True,
        )
        model.load_state_dict(state)
    del checkpoint, state
    model = model.to(device)
    model.eval().requires_grad_(False)
    return model


def _condition_ids(metadata: pl.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    perturbation_ids: dict[str, int] = {}
    for name in PERTURBATIONS:
        values = (
            metadata.filter(
                (pl.col("CPD_NAME") == name)
                & pl.col("is_eval_pert")
                & (pl.col("ANNOT") == "treated")
            )["perturbation_id"]
            .unique()
            .to_list()
        )
        if len(values) != 1:
            raise ValueError(f"{name}: expected one perturbation id, got {values}")
        perturbation_ids[name] = int(values[0])

    experiment_ids: dict[str, int] = {}
    for batch in EXPERIMENTS:
        values = (
            metadata.filter(pl.col("BATCH") == batch)["experiment_id"]
            .unique()
            .to_list()
        )
        if len(values) != 1:
            raise ValueError(f"{batch}: expected one experiment id, got {values}")
        experiment_ids[batch] = int(values[0])
    return perturbation_ids, experiment_ids


def _noise(
    *,
    condition_seed: int,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(condition_seed)
    return torch.randn(
        (n_samples, NUM_CHANNELS, IMG_SIZE, IMG_SIZE),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


@torch.inference_mode()
def collect(
    *,
    metadata_path: Path,
    n_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = pl.read_csv(metadata_path)
    perturbation_ids, experiment_ids = _condition_ids(metadata)
    generated: dict[str, Any] = {}
    nfe: dict[str, Any] = {}

    for method_index, (method, checkpoint) in enumerate(METHODS.items()):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"Loading {method}: {checkpoint}", flush=True)
        model = _load_model(checkpoint, device)
        generated[method] = {}
        nfe[method] = {}
        for perturbation_index, perturbation in enumerate(PERTURBATIONS):
            generated[method][perturbation] = {}
            nfe[method][perturbation] = {}
            for batch_index, batch in enumerate(EXPERIMENTS):
                condition_seed = (
                    int(seed)
                    + 10_000 * perturbation_index
                    + 100 * batch_index
                )
                x0 = _noise(
                    condition_seed=condition_seed,
                    n_samples=n_samples,
                    device=device,
                )
                y = torch.full(
                    (n_samples,),
                    perturbation_ids[perturbation],
                    dtype=torch.long,
                    device=device,
                )
                e = torch.full(
                    (n_samples,),
                    experiment_ids[batch],
                    dtype=torch.long,
                    device=device,
                )
                c = torch.zeros(n_samples, dtype=torch.long, device=device)
                print(
                    f"  {perturbation}/{batch} seed={condition_seed}",
                    flush=True,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    samples, evaluations = generate(
                        model=model,
                        x0=x0,
                        y1=y,
                        e1=e,
                        c1=c,
                    )
                rgb = decode(cp2rgb(samples).clamp(-1, 1))
                generated[method][perturbation][batch] = (
                    rgb.permute(0, 2, 3, 1).cpu().numpy()
                )
                nfe[method][perturbation][batch] = int(evaluations)
        del model
        torch.cuda.empty_cache()
        print(f"Finished {method_index + 1}/{len(METHODS)} methods", flush=True)

    provenance = {
        "metadata": str(metadata_path),
        "perturbations": perturbation_ids,
        "experiments": experiment_ids,
        "methods": {name: str(path) for name, path in METHODS.items()},
        "checkpoint_task": 2,
        "seed": int(seed),
        "n_samples_per_condition": int(n_samples),
        "sampler": "dopri5",
        "same_noise_across_methods": True,
        "nfe": nfe,
    }
    return generated, provenance


def _mosaic(images: np.ndarray, *, gap: int = 2) -> np.ndarray:
    if images.shape[0] != 9:
        raise ValueError(f"Expected nine images, got {images.shape[0]}")
    height, width = images.shape[1:3]
    canvas = np.zeros(
        (3 * height + 2 * gap, 3 * width + 2 * gap, 3),
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, 3)
        top = row * (height + gap)
        left = column * (width + gap)
        canvas[top : top + height, left : left + width] = image
    return canvas


def _decorate_axis(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, pad=4)
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=90, labelpad=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")


def render_one(
    generated: dict[str, Any],
    perturbation: str,
    out: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.55))
    for row, method in enumerate(METHODS):
        for column, batch in enumerate(EXPERIMENTS):
            _decorate_axis(
                axes[row, column],
                _mosaic(generated[method][perturbation][batch]),
                title=batch if row == 0 else None,
                ylabel=method if column == 0 else None,
            )
    fig.suptitle(perturbation, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.35, h_pad=0.45, w_pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_combined(generated: dict[str, Any], out: Path) -> None:
    _style()
    methods = list(METHODS)
    fig, axes = plt.subplots(
        len(PERTURBATIONS) * len(methods),
        len(EXPERIMENTS),
        figsize=(7.0, 12.1),
    )
    for perturbation_index, perturbation in enumerate(PERTURBATIONS):
        for method_index, method in enumerate(methods):
            row = perturbation_index * len(methods) + method_index
            for column, batch in enumerate(EXPERIMENTS):
                label = f"{perturbation}\n{method}" if column == 0 else None
                _decorate_axis(
                    axes[row, column],
                    _mosaic(generated[method][perturbation][batch]),
                    title=batch if row == 0 else None,
                    ylabel=label,
                )
    fig.tight_layout(pad=0.35, h_pad=0.45, w_pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(METADATA_PATH))
    parser.add_argument("--n-samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "px" / "images" / "representative",
    )
    args = parser.parse_args()
    if int(args.n_samples) != 9:
        raise ValueError("The paper layout requires exactly nine samples per panel")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    generated, provenance = collect(
        metadata_path=args.metadata,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        device=device,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for perturbation in PERTURBATIONS:
        render_one(
            generated,
            perturbation,
            args.out_dir / f"{perturbation.lower()}_replay_generations.png",
        )
    render_combined(
        generated,
        args.out_dir / "rxrx1_representative_replay_generations.png",
    )
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
