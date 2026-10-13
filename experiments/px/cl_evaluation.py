"""Deterministic per-experiment PX FID/KID evaluation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from experiments.px.cl_dataset import ExperimentTask
from experiments.px.cl_trainer import euler_rollout
from experiments.px.dataset import CellDataset, cp2rgb, decode
from experiments.px.trainer import PXModel, generate
from experiments.px.utils.evaluation import FeatureExtractor, compute_fd, compute_kd


@torch.inference_mode()
def evaluate_experiment(
    *,
    model: PXModel,
    task: ExperimentTask,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    frame: pl.DataFrame | None = None,
    split_name: str = "test",
    detector: FeatureExtractor | None = None,
    sampler: str = "dopri5",
    euler_step_size: float = 0.1,
    evaluation_name: str | None = None,
    return_features: bool = False,
) -> dict[str, Any]:
    """Generate the official composition-matched test set for one experiment."""
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
    nfes: list[int] = []
    model.eval()
    started = time.time()

    for batch in loader:
        real = batch["pert"].to(device, non_blocking=True)
        y = batch["perturbation_id"].long().to(device, non_blocking=True)
        e = batch["experiment_id"].long().to(device, non_blocking=True)
        c = batch["cell_type_id"].long().to(device, non_blocking=True)
        x0 = torch.randn(
            real.shape,
            device=device,
            dtype=real.dtype,
            generator=generator,
        )
        if sampler == "dopri5":
            fake, nfe = generate(model=model, x0=x0, y1=y, e1=e, c1=c)
        elif sampler == "euler":
            fake = euler_rollout(
                model,
                y=y,
                e=e,
                c=c,
                generator=generator,
                image_shape=tuple(real.shape[1:]),
                step_size=euler_step_size,
            )
            nfe = int(round(1.0 / euler_step_size))
        else:
            raise ValueError(f"Unknown sampler={sampler!r}")
        nfes.append(int(nfe))
        fake_rgb = decode(cp2rgb(fake).clamp(-1, 1))
        real_rgb = decode(cp2rgb(real).clamp(-1, 1))
        fake_inception, fake_dino = detector(fake_rgb)
        real_inception, real_dino = detector(real_rgb)
        inception_fake.append(fake_inception.cpu())
        inception_real.append(real_inception.cpu())
        dino_fake.append(fake_dino.cpu())
        dino_real.append(real_dino.cpu())

    def array(values: list[torch.Tensor]) -> np.ndarray:
        return torch.cat(values).float().numpy()

    real_i = array(inception_real)
    fake_i = array(inception_fake)
    real_d = array(dino_real)
    fake_d = array(dino_fake)
    result: dict[str, Any] = {
        "task_id": task.task_id,
        "experiment_id": (
            task.experiment_id if evaluation_name is None else None
        ),
        "batch_name": (
            task.batch_name if evaluation_name is None else evaluation_name
        ),
        "split": split_name,
        "n_real": len(dataset),
        "n_generated": int(fake_i.shape[0]),
        "eval_seed": int(seed),
        "sampler": sampler,
        "euler_step_size": euler_step_size if sampler == "euler" else None,
        "fid": float(compute_fd(real_i, fake_i)),
        "kid": float(compute_kd(real_i, fake_i)),
        "fdd_dinov2": float(compute_fd(real_d, fake_d)),
        "kdd_dinov2": float(compute_kd(real_d, fake_d)),
        "mean_nfe_per_batch": float(np.mean(nfes)),
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


def aggregate_experiment_metrics(
    rows: list[dict[str, Any]],
    *,
    evaluation_name: str,
) -> dict[str, Any]:
    if not rows or any("_features" not in row for row in rows):
        raise ValueError("Rows with retained features are required")

    def joined(key: str) -> np.ndarray:
        return np.concatenate(
            [row["_features"][key] for row in rows],
            axis=0,
        )

    real_i = joined("inception_real")
    fake_i = joined("inception_fake")
    real_d = joined("dino_real")
    fake_d = joined("dino_fake")
    result = {
        "task_id": max(int(row["task_id"]) for row in rows),
        "experiment_id": None,
        "batch_name": evaluation_name,
        "split": rows[0]["split"],
        "n_real": int(real_i.shape[0]),
        "n_generated": int(fake_i.shape[0]),
        "eval_seed": [row["eval_seed"] for row in rows],
        "sampler": rows[0]["sampler"],
        "euler_step_size": rows[0].get("euler_step_size"),
        "sampling_steps": rows[0].get("sampling_steps"),
        "fid": float(compute_fd(real_i, fake_i)),
        "kid": float(compute_kd(real_i, fake_i)),
        "fdd_dinov2": float(compute_fd(real_d, fake_d)),
        "kdd_dinov2": float(compute_kd(real_d, fake_d)),
        "wall_seconds": float(sum(row["wall_seconds"] for row in rows)),
    }
    if all(row.get("mean_nfe_per_batch") is not None for row in rows):
        result["mean_nfe_per_batch"] = float(
            np.mean([row["mean_nfe_per_batch"] for row in rows])
        )
    return result
