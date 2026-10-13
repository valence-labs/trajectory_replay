#!/usr/bin/env python3
"""Render matched Euler and Dopri5 samples from a PX checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from experiments.px.dataset import (
    CellDataset,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--euler-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--legacy-cell-conditioning",
        action="store_true",
        help="Load checkpoints trained before constant cell conditioning was removed.",
    )
    return parser.parse_args()


@torch.no_grad()
def euler_sample(
    model: ADMWrapper,
    *,
    x0: torch.Tensor,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    x = x0.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full(
            (x.shape[0],),
            step * dt,
            device=x.device,
            dtype=x.dtype,
        )
        x = x + dt * model(x=x, t=t, y=y, e=e, c=c)
    return x


def rgb_grid(images: torch.Tensor) -> torch.Tensor:
    images = decode(cp2rgb(images).clamp(-1, 1)).float() / 255.0
    return make_grid(images.cpu(), nrow=8, padding=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = ADMWrapper(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=1,
        use_cell_type_conditioning=args.legacy_cell_conditioning,
    )
    checkpoint = torch.load(
        args.checkpoint,
        weights_only=True,
        map_location="cpu",
    )
    ema_state = checkpoint["ema"]
    prefixes = ("module._orig_mod.", "module.")
    model_state = {}
    for key, value in ema_state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        model_state[key] = value
    model.load_state_dict(model_state)
    del checkpoint, ema_state, model_state
    ema_model: ADMWrapper = model.to(device)
    ema_model.eval()

    metadata = pl.read_csv(METADATA_PATH)
    test_frame = metadata.filter(
        (pl.col("SPLIT") == "test")
        & pl.col("is_eval_pert")
        & (pl.col("ANNOT") == "treated")
    )
    loader = DataLoader(
        CellDataset(test_frame, train=False, load_control=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    batch = next(iter(loader))
    y = batch["perturbation_id"].long().to(device)
    e = batch["experiment_id"].long().to(device)
    c = batch["cell_type_id"].long().to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x0 = torch.randn(
        (y.shape[0], NUM_CHANNELS, IMG_SIZE, IMG_SIZE),
        device=device,
        generator=generator,
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        euler = euler_sample(
            ema_model,
            x0=x0,
            y=y,
            e=e,
            c=c,
            n_steps=args.euler_steps,
        )
        dopri, dopri_nfe = generate(
            model=ema_model,
            x0=x0.clone(),
            y1=y,
            e1=e,
            c1=c,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    euler_path = args.output_dir / f"euler{args.euler_steps}.png"
    dopri_path = args.output_dir / "dopri5.png"
    save_image(rgb_grid(euler), euler_path)
    save_image(rgb_grid(dopri), dopri_path)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
                "batch_size": int(y.shape[0]),
                "euler_nfe": args.euler_steps,
                "dopri5_nfe": dopri_nfe,
                "same_noise_and_conditions": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(euler_path)
    print(dopri_path)


if __name__ == "__main__":
    main()
