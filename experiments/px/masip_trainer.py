"""Matched Masip-style diffusion continual-learning objectives for PX."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Literal

import polars as pl
import torch
from diffusers import DDIMScheduler

from experiments.px.cl_trainer import (
    _batch_to_device,
    _condition_arrays,
    _sample_old_conditions,
)
from experiments.px.trainer import TrainState
from experiments.px.utils.distributed import Distributed
from experiments.px.utils.infinite_dataloader import infinite_dataloader

MasipStrategy = Literal[
    "finetune",
    "cumulative",
    "generative_replay",
    "generative_distillation",
]


def make_ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=True,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )


def _stable_ddim_step(
    scheduler: DDIMScheduler,
    *,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Masip's deterministic DDIM step, including its zero-SNR epsilon."""
    if scheduler.num_inference_steps is None:
        raise ValueError("scheduler.set_timesteps() must be called first")
    timestep_int = int(timestep.item())
    step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    previous = timestep_int - step
    alpha = scheduler.alphas_cumprod[timestep_int].to(sample.device)
    alpha_previous = (
        scheduler.alphas_cumprod[previous].to(sample.device)
        if previous >= 0
        else scheduler.final_alpha_cumprod.to(sample.device)
    )
    beta = 1.0 - alpha
    predicted_x0 = (
        sample - beta.sqrt() * model_output.float()
    ) / (alpha + 1e-8).sqrt()
    predicted_x0 = predicted_x0.clamp(
        -scheduler.config.clip_sample_range,
        scheduler.config.clip_sample_range,
    )
    direction = (1.0 - alpha_previous).clamp(0.0, 1.0).sqrt()
    return alpha_previous.sqrt() * predicted_x0 + direction * model_output.float()


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    *,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    generator: torch.Generator,
    image_shape: tuple[int, int, int],
    num_steps: int,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    scheduler = make_ddim_scheduler()
    scheduler.set_timesteps(num_steps, device=y.device)
    x = (
        torch.randn(
            (y.shape[0], *image_shape),
            device=y.device,
            generator=generator,
        )
        if x0 is None
        else x0.clone()
    )
    model.eval()
    for timestep in scheduler.timesteps:
        t = timestep.expand(y.shape[0])
        prediction = model(x=x, t=t, y=y, e=e, c=c)
        x = _stable_ddim_step(
            scheduler,
            model_output=prediction,
            timestep=timestep,
            sample=x,
        )
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(
                f"Non-finite DDIM sample at timestep {int(timestep)}"
            )
    return x


def diffusion_loss(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    *,
    x1: torch.Tensor,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    generator: torch.Generator,
    teacher: torch.nn.Module | None = None,
) -> torch.Tensor:
    noise = torch.randn(
        x1.shape,
        device=x1.device,
        dtype=x1.dtype,
        generator=generator,
    )
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (x1.shape[0],),
        device=x1.device,
        generator=generator,
    ).long()
    noisy = scheduler.add_noise(x1, noise, timesteps)
    prediction = model(x=noisy, t=timesteps, y=y, e=e, c=c)
    if teacher is None:
        target = noise
    else:
        with torch.no_grad():
            target = teacher(x=noisy, t=timesteps, y=y, e=e, c=c)
    return torch.nn.functional.mse_loss(prediction, target)


def train_masip_task(
    *,
    state: TrainState,
    strategy: MasipStrategy,
    current_loader: Iterable,
    D: Distributed,
    num_steps: int,
    lr: float,
    seed: int,
    lambda_mix: float | None = None,
    replay_loss_scale: float = 0.75,
    teacher: torch.nn.Module | None = None,
    old_frames: Sequence[pl.DataFrame] = (),
    teacher_steps: int = 20,
    max_grad_norm: float = 1.0,
) -> dict[str, float | int | str | None]:
    replay = strategy in {"generative_replay", "generative_distillation"}
    if replay and (teacher is None or lambda_mix is None):
        raise ValueError(f"{strategy} requires a frozen teacher and lambda")

    state.opt = torch.optim.Adam(
        state.ddp.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    torch.manual_seed(seed + D.rank)
    torch.cuda.manual_seed(seed + D.rank)
    generator = torch.Generator(device=D.device)
    generator.manual_seed(seed + D.rank)
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed + D.rank)
    support = _condition_arrays(old_frames)
    scheduler = make_ddim_scheduler()
    iterator = infinite_dataloader(current_loader)
    loss_sum = 0.0
    new_sum = 0.0
    old_sum = 0.0
    started = time.time()

    for _ in range(num_steps):
        batch = next(iterator)
        x_new, y_new, e_new, c_new = _batch_to_device(batch, D.device)
        state.ddp.train()
        state.opt.zero_grad(set_to_none=True)
        if not replay:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_new = diffusion_loss(
                    state.ddp,
                    scheduler,
                    x1=x_new,
                    y=y_new,
                    e=e_new,
                    c=c_new,
                    generator=generator,
                )
                loss_old = torch.zeros((), device=D.device)
                loss = loss_new
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {strategy} loss")
            loss.backward()
        else:
            assert teacher is not None and lambda_mix is not None
            y_old, e_old, c_old = _sample_old_conditions(
                support,
                x_new.shape[0],
                generator=cpu_generator,
                device=D.device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x_old = ddim_sample(
                    teacher,
                    y=y_old,
                    e=e_old,
                    c=c_old,
                    generator=generator,
                    image_shape=tuple(x_new.shape[1:]),
                    num_steps=teacher_steps,
                )
            state.ddp.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_new = diffusion_loss(
                    state.ddp,
                    scheduler,
                    x1=x_new,
                    y=y_new,
                    e=e_new,
                    c=c_new,
                    generator=generator,
                )
            if not bool(torch.isfinite(loss_new)):
                raise FloatingPointError(f"Non-finite {strategy} new loss")
            ((1.0 - lambda_mix) * loss_new).backward()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_old = diffusion_loss(
                    state.ddp,
                    scheduler,
                    x1=x_old,
                    y=y_old,
                    e=e_old,
                    c=c_old,
                    generator=generator,
                    teacher=(
                        teacher
                        if strategy == "generative_distillation"
                        else None
                    ),
                )
            if not bool(torch.isfinite(loss_old)):
                raise FloatingPointError(f"Non-finite {strategy} old loss")
            (lambda_mix * replay_loss_scale * loss_old).backward()
            loss = (
                (1.0 - lambda_mix) * loss_new.detach()
                + lambda_mix * replay_loss_scale * loss_old.detach()
            )
        torch.nn.utils.clip_grad_norm_(
            state.ddp.parameters(),
            max_norm=max_grad_norm,
        )
        state.opt.step()
        state.global_step += 1
        state.ema.update(state.ddp.module, state.global_step)
        loss_sum += float(loss.detach())
        new_sum += float(loss_new.detach())
        old_sum += float(loss_old.detach())

    D.barrier()
    return {
        "strategy": strategy,
        "steps": num_steps,
        "global_step": state.global_step,
        "lambda": lambda_mix,
        "replay_loss_scale": replay_loss_scale if replay else None,
        "teacher_steps": teacher_steps if replay else None,
        "loss_mean": loss_sum / num_steps,
        "loss_new_mean": new_sum / num_steps,
        "loss_old_mean": old_sum / num_steps,
        "wall_seconds": time.time() - started,
    }
