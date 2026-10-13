"""Composition-matched evaluation for the PX diffusion baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from experiments.px.cl_dataset import ExperimentTask
from experiments.px.dataset import CellDataset, cp2rgb, decode
from experiments.px.masip_trainer import ddim_sample
from experiments.px.utils.evaluation import FeatureExtractor, compute_fd, compute_kd


@torch.inference_mode()
def evaluate_diffusion_experiment(
    *,
    model: torch.nn.Module,
    task: ExperimentTask,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    num_steps: int,
    frame: pl.DataFrame | None = None,
    split_name: str = "test",
    detector: FeatureExtractor | None = None,
    return_features: bool = False,
) -> dict[str, Any]:
    dataset = CellDataset(
        task.test if frame is None else frame,
        train=False,
        load_control=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    detector = detector if detector is not None else FeatureExtractor(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inception_real: list[torch.Tensor] = []
    inception_fake: list[torch.Tensor] = []
    dino_real: list[torch.Tensor] = []
    dino_fake: list[torch.Tensor] = []
    model.eval()
    started = time.time()

    for batch in loader:
        real = batch["pert"].to(device, non_blocking=True)
        y = batch["perturbation_id"].long().to(device, non_blocking=True)
        e = batch["experiment_id"].long().to(device, non_blocking=True)
        c = batch["cell_type_id"].long().to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fake = ddim_sample(
                model,
                y=y,
                e=e,
                c=c,
                generator=generator,
                image_shape=tuple(real.shape[1:]),
                num_steps=num_steps,
            )
        fake_rgb = decode(cp2rgb(fake).clamp(-1, 1))
        real_rgb = decode(cp2rgb(real).clamp(-1, 1))
        fake_i, fake_d = detector(fake_rgb)
        real_i, real_d = detector(real_rgb)
        inception_fake.append(fake_i.cpu())
        inception_real.append(real_i.cpu())
        dino_fake.append(fake_d.cpu())
        dino_real.append(real_d.cpu())

    def array(values: list[torch.Tensor]) -> np.ndarray:
        return torch.cat(values).float().numpy()

    real_i = array(inception_real)
    fake_i = array(inception_fake)
    real_d = array(dino_real)
    fake_d = array(dino_fake)
    result: dict[str, Any] = {
        "task_id": task.task_id,
        "experiment_id": task.experiment_id,
        "batch_name": task.batch_name,
        "split": split_name,
        "n_real": len(dataset),
        "n_generated": int(fake_i.shape[0]),
        "eval_seed": seed,
        "sampler": "ddim",
        "sampling_steps": num_steps,
        "fid": float(compute_fd(real_i, fake_i)),
        "kid": float(compute_kd(real_i, fake_i)),
        "fdd_dinov2": float(compute_fd(real_d, fake_d)),
        "kdd_dinov2": float(compute_kd(real_d, fake_d)),
        "wall_seconds": time.time() - started,
    }
    if return_features:
        result["_features"] = {
            "inception_real": real_i,
            "inception_fake": fake_i,
            "dino_real": real_d,
            "dino_fake": fake_d,
        }
    return result
