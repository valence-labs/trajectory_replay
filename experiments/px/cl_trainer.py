"""Core experiment-incremental continual-learning losses for PX models."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Literal

import polars as pl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments.px.mean_flow import (
    identity_loss_from_anchor,
    identity_loss_from_endpoint,
)
from experiments.px.model import ADMMeanFlowWrapper
from experiments.px.trainer import TrainState
from experiments.px.utils.distributed import Distributed
from experiments.px.utils.infinite_dataloader import infinite_dataloader

log = logging.getLogger(__name__)

CLStrategy = Literal["naive", "cumulative", "real", "generative", "trajectory"]


def equal_task_lambda(task_id: int) -> float:
    if task_id < 1:
        raise ValueError("equal-task lambda is only defined after task 0")
    return float(task_id) / float(task_id + 1)


def frozen_teacher(model: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(model)
    teacher.eval().requires_grad_(False)
    return teacher


def _condition_arrays(frames: Sequence[pl.DataFrame]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    arrays: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frame in frames:
        arrays.append(
            (
                torch.tensor(frame["perturbation_id"].to_numpy(), dtype=torch.long),
                torch.tensor(frame["experiment_id"].to_numpy(), dtype=torch.long),
            )
        )
    return arrays


def _sample_old_conditions(
    support: list[tuple[torch.Tensor, torch.Tensor]],
    n: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not support or n < 1:
        raise ValueError("old condition support and n must be non-empty")
    task_choice = torch.randint(0, len(support), (n,), generator=generator)
    y = torch.empty(n, dtype=torch.long)
    e = torch.empty(n, dtype=torch.long)
    for task_id, (task_y, task_e) in enumerate(support):
        positions = torch.nonzero(task_choice == task_id, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        rows = torch.randint(0, task_y.numel(), (positions.numel(),), generator=generator)
        y[positions] = task_y[rows]
        e[positions] = task_e[rows]
    c = torch.zeros(n, dtype=torch.long)
    return y.to(device), e.to(device), c.to(device)


def _cfm_loss(
    model: DDP,
    *,
    x1: torch.Tensor,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    generator: torch.Generator,
    t_scalar: float | None = None,
) -> torch.Tensor:
    x0 = torch.randn(
        x1.shape,
        device=x1.device,
        dtype=x1.dtype,
        generator=generator,
    )
    if t_scalar is None:
        t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    else:
        t = torch.full(
            (x1.shape[0],),
            float(t_scalar),
            device=x1.device,
            dtype=x1.dtype,
        )
    t_view = t.reshape(-1, 1, 1, 1)
    xt = (1.0 - t_view) * x0 + t_view * x1
    target = x1 - x0
    prediction = model(x=xt, t=t, y=y, e=e, c=c)
    return torch.nn.functional.mse_loss(prediction, target)


@torch.no_grad()
def euler_rollout(
    model: torch.nn.Module,
    *,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    generator: torch.Generator,
    image_shape: tuple[int, int, int],
    step_size: float,
    t_end: float = 1.0,
) -> torch.Tensor:
    if step_size <= 0.0 or not (0.0 < t_end <= 1.0):
        raise ValueError("step_size must be positive and t_end must be in (0,1]")
    x = torch.randn(
        (y.shape[0], *image_shape),
        device=y.device,
        generator=generator,
    )
    n_steps = max(1, int(torch.ceil(torch.tensor(t_end / step_size)).item()))
    dt = float(t_end) / n_steps
    for step in range(n_steps):
        t = torch.full(
            (y.shape[0],),
            step * dt,
            device=y.device,
            dtype=x.dtype,
        )
        x = x + dt * model(x=x, t=t, y=y, e=e, c=c)
    return x


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["pert"].to(device, non_blocking=True)
    y = batch["perturbation_id"].long().to(device, non_blocking=True)
    e = batch["experiment_id"].long().to(device, non_blocking=True)
    c = batch["cell_type_id"].long().to(device, non_blocking=True)
    return x, y, e, c


def train_cl_task(
    *,
    state: TrainState,
    strategy: CLStrategy,
    current_loader: Iterable,
    D: Distributed,
    num_steps: int,
    lr: float,
    seed: int,
    lambda_mix: float | None,
    teacher: torch.nn.Module | None = None,
    old_frames: Sequence[pl.DataFrame] = (),
    old_loader: Iterable | None = None,
    replay_step_size: float = 0.05,
    max_grad_norm: float = 0.5,
    log_every: int = 250,
    mean_flow: ADMMeanFlowWrapper | None = None,
    replay_mean_flow: ADMMeanFlowWrapper | None = None,
    mean_flow_n_steps: int = 1,
    mean_flow_lr: float = 1e-4,
    mean_flow_max_grad_norm: float = 1.0,
    mean_flow_diag_fraction: float = 0.25,
    mean_flow_time_sampler: str = "logit_normal",
    mean_flow_time_mu: float = 2.0,
    mean_flow_time_sigma: float = 2.0,
    mean_flow_batch_size: int = 32,
) -> dict[str, float | int | str | None]:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if strategy in {"generative", "trajectory"} and teacher is None:
        raise ValueError(f"{strategy} requires a frozen teacher")
    if mean_flow is not None and strategy not in {"generative", "trajectory"}:
        raise ValueError("joint MeanFlow is only supported for synthetic replay")
    if strategy in {"real", "generative", "trajectory"}:
        if lambda_mix is None or not (0.0 < lambda_mix < 1.0):
            raise ValueError(f"{strategy} requires lambda_mix in (0,1)")

    state.opt = torch.optim.Adam(
        state.ddp.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    torch.manual_seed(seed + D.rank)
    torch.cuda.manual_seed(seed + D.rank)
    mean_flow_optimizer = (
        torch.optim.AdamW(
            mean_flow.parameters(),
            lr=mean_flow_lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        if mean_flow is not None
        else None
    )
    device_generator = torch.Generator(device=D.device)
    device_generator.manual_seed(seed + D.rank)
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed + D.rank)
    condition_support = _condition_arrays(old_frames)
    current_iter = infinite_dataloader(current_loader)
    old_iter = infinite_dataloader(old_loader) if old_loader is not None else None
    start = time.time()
    loss_sum = 0.0
    old_loss_sum = 0.0
    new_loss_sum = 0.0
    mean_flow_loss_sum = 0.0

    for local_step in range(1, num_steps + 1):
        state.ddp.train()
        current_batch = next(current_iter)
        x_new_all, y_new_all, e_new_all, c_new_all = _batch_to_device(
            current_batch, D.device
        )
        batch_size = x_new_all.shape[0]
        state.opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if strategy in {"naive", "cumulative"}:
                loss_new = _cfm_loss(
                    state.ddp,
                    x1=x_new_all,
                    y=y_new_all,
                    e=e_new_all,
                    c=c_new_all,
                    generator=device_generator,
                )
                loss_old = torch.zeros((), device=D.device)
                loss = loss_new
            elif strategy == "real":
                assert old_iter is not None and lambda_mix is not None
                old_batch = next(old_iter)
                x_old_all, y_old_all, e_old_all, c_old_all = _batch_to_device(
                    old_batch, D.device
                )
                n_old = min(max(1, int(batch_size * lambda_mix)), batch_size - 1)
                n_new = batch_size - n_old
                new_perm = torch.randperm(batch_size, generator=cpu_generator)
                old_perm = torch.randperm(x_old_all.shape[0], generator=cpu_generator)
                new_idx = new_perm[:n_new]
                old_idx = old_perm[:n_old]
                x_batch = torch.cat([x_new_all[new_idx], x_old_all[old_idx]])
                y_batch = torch.cat([y_new_all[new_idx], y_old_all[old_idx]])
                e_batch = torch.cat([e_new_all[new_idx], e_old_all[old_idx]])
                c_batch = torch.cat([c_new_all[new_idx], c_old_all[old_idx]])
                loss = _cfm_loss(
                    state.ddp,
                    x1=x_batch,
                    y=y_batch,
                    e=e_batch,
                    c=c_batch,
                    generator=device_generator,
                )
                loss_new = loss
                loss_old = loss.detach()
            elif strategy == "generative":
                assert teacher is not None and lambda_mix is not None
                n_old = min(max(1, int(batch_size * lambda_mix)), batch_size - 1)
                n_new = batch_size - n_old
                new_idx = torch.randperm(
                    batch_size, generator=cpu_generator
                )[:n_new]
                y_old, e_old, c_old = _sample_old_conditions(
                    condition_support,
                    n_old,
                    generator=cpu_generator,
                    device=D.device,
                )
                with torch.no_grad():
                    if replay_mean_flow is not None:
                        old_noise = torch.randn(
                            (n_old, *x_new_all.shape[1:]),
                            device=D.device,
                            generator=device_generator,
                        )
                        x_old = replay_mean_flow.sample_from_x0(
                            old_noise,
                            y=y_old,
                            e=e_old,
                            c=c_old,
                            n_steps=mean_flow_n_steps,
                        )
                    else:
                        x_old = euler_rollout(
                            teacher,
                            y=y_old,
                            e=e_old,
                            c=c_old,
                            generator=device_generator,
                            image_shape=tuple(x_new_all.shape[1:]),
                            step_size=replay_step_size,
                        )
                x_batch = torch.cat([x_new_all[new_idx], x_old])
                y_batch = torch.cat([y_new_all[new_idx], y_old])
                e_batch = torch.cat([e_new_all[new_idx], e_old])
                c_batch = torch.cat([c_new_all[new_idx], c_old])
                loss = _cfm_loss(
                    state.ddp,
                    x1=x_batch,
                    y=y_batch,
                    e=e_batch,
                    c=c_batch,
                    generator=device_generator,
                )
                loss_new = loss
                loss_old = loss.detach()
            else:
                assert teacher is not None and lambda_mix is not None
                n_old = batch_size // 2
                n_new = batch_size - n_old
                new_idx = torch.randperm(
                    batch_size, generator=cpu_generator
                )[:n_new]
                y_old, e_old, c_old = _sample_old_conditions(
                    condition_support,
                    n_old,
                    generator=cpu_generator,
                    device=D.device,
                )
                t_scalar = float(
                    torch.rand((), device=D.device, generator=device_generator).item()
                )
                t_scalar = max(t_scalar, 1e-4)
                loss_new = _cfm_loss(
                    state.ddp,
                    x1=x_new_all[new_idx],
                    y=y_new_all[new_idx],
                    e=e_new_all[new_idx],
                    c=c_new_all[new_idx],
                    generator=device_generator,
                    t_scalar=t_scalar,
                )
                with torch.no_grad():
                    if replay_mean_flow is not None:
                        old_noise = torch.randn(
                            (n_old, *x_new_all.shape[1:]),
                            device=D.device,
                            generator=device_generator,
                        )
                        x_t = replay_mean_flow.sample_from_x0(
                            old_noise,
                            y=y_old,
                            e=e_old,
                            c=c_old,
                            n_steps=mean_flow_n_steps,
                            t_end=t_scalar,
                        )
                    else:
                        x_t = euler_rollout(
                            teacher,
                            y=y_old,
                            e=e_old,
                            c=c_old,
                            generator=device_generator,
                            image_shape=tuple(x_new_all.shape[1:]),
                            step_size=replay_step_size,
                            t_end=t_scalar,
                        )
                    t_old = torch.full(
                        (n_old,),
                        t_scalar,
                        device=D.device,
                        dtype=x_t.dtype,
                    )
                    target_old = teacher(
                        x=x_t,
                        t=t_old,
                        y=y_old,
                        e=e_old,
                        c=c_old,
                    )
                prediction_old = state.ddp(
                    x=x_t,
                    t=t_old,
                    y=y_old,
                    e=e_old,
                    c=c_old,
                )
                loss_old = torch.nn.functional.mse_loss(
                    prediction_old, target_old
                )
                loss = (1.0 - lambda_mix) * loss_new + lambda_mix * loss_old

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite {strategy} loss at local step {local_step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            state.ddp.parameters(), max_norm=max_grad_norm
        )
        state.opt.step()

        mean_flow_loss = torch.zeros((), device=D.device)
        if mean_flow is not None:
            assert mean_flow_optimizer is not None and lambda_mix is not None
            mean_flow.train()
            mean_flow_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if strategy == "generative":
                    mf_count = min(mean_flow_batch_size, x_batch.shape[0])
                    mf_idx = torch.randperm(
                        x_batch.shape[0],
                        generator=cpu_generator,
                    )[:mf_count].to(D.device)
                    mean_flow_loss = identity_loss_from_endpoint(
                        mean_flow,
                        x1=x_batch[mf_idx].detach(),
                        y=y_batch[mf_idx],
                        e=e_batch[mf_idx],
                        c=c_batch[mf_idx],
                        generator=device_generator,
                        diag_fraction=mean_flow_diag_fraction,
                        time_sampler=mean_flow_time_sampler,
                        time_mu=mean_flow_time_mu,
                        time_sigma=mean_flow_time_sigma,
                    )
                else:
                    branch_limit = max(1, mean_flow_batch_size // 2)
                    mf_new_count = min(branch_limit, n_new)
                    mf_old_count = min(branch_limit, n_old)
                    mean_flow_new = identity_loss_from_endpoint(
                        mean_flow,
                        x1=x_new_all[new_idx[:mf_new_count]],
                        y=y_new_all[new_idx[:mf_new_count]],
                        e=e_new_all[new_idx[:mf_new_count]],
                        c=c_new_all[new_idx[:mf_new_count]],
                        generator=device_generator,
                        r_scalar=t_scalar,
                        diag_fraction=mean_flow_diag_fraction,
                        time_sampler=mean_flow_time_sampler,
                        time_mu=mean_flow_time_mu,
                        time_sigma=mean_flow_time_sigma,
                    )
                    r_old = torch.full(
                        (mf_old_count,),
                        t_scalar,
                        device=D.device,
                        dtype=x_t.dtype,
                    )
                    if mean_flow_time_sampler == "uniform":
                        u_old = torch.rand(
                            mf_old_count,
                            device=D.device,
                            generator=device_generator,
                        )
                    elif mean_flow_time_sampler == "logit_normal":
                        u_old = torch.randn(
                            mf_old_count,
                            device=D.device,
                            generator=device_generator,
                        )
                        u_old = torch.sigmoid(
                            u_old * float(mean_flow_time_sigma)
                            + float(mean_flow_time_mu)
                        )
                    else:
                        raise ValueError(
                            f"Unknown MeanFlow time sampler "
                            f"{mean_flow_time_sampler!r}"
                        )
                    t_old = r_old + (1.0 - r_old) * u_old
                    diagonal = (
                        torch.rand(
                            mf_old_count,
                            device=D.device,
                            generator=device_generator,
                        )
                        < mean_flow_diag_fraction
                    )
                    t_old = torch.where(diagonal, r_old, t_old)
                    mean_flow_old = identity_loss_from_anchor(
                        mean_flow,
                        z_r=x_t[:mf_old_count].detach(),
                        v_r=target_old[:mf_old_count].detach(),
                        r=r_old,
                        t=t_old,
                        y=y_old[:mf_old_count],
                        e=e_old[:mf_old_count],
                        c=c_old[:mf_old_count],
                    )
                    mean_flow_loss = (
                        (1.0 - lambda_mix) * mean_flow_new
                        + lambda_mix * mean_flow_old
                    )
            if not bool(torch.isfinite(mean_flow_loss)):
                raise FloatingPointError("Non-finite joint MeanFlow loss")
            mean_flow_loss.backward()
            if dist.is_initialized() and dist.get_world_size() > 1:
                for parameter in mean_flow.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(dist.get_world_size())
            torch.nn.utils.clip_grad_norm_(
                mean_flow.parameters(),
                max_norm=mean_flow_max_grad_norm,
            )
            mean_flow_optimizer.step()

        state.global_step += 1
        state.ema.update(state.ddp.module, state.global_step)

        loss_sum += float(loss.detach())
        old_loss_sum += float(loss_old.detach())
        new_loss_sum += float(loss_new.detach())
        mean_flow_loss_sum += float(mean_flow_loss.detach())
        if log_every > 0 and local_step % log_every == 0 and D.rank == 0:
            log.info(
                "%s local step %d/%d loss=%.6f old=%.6f new=%.6f",
                strategy,
                local_step,
                num_steps,
                loss_sum / local_step,
                old_loss_sum / local_step,
                new_loss_sum / local_step,
            )

    D.barrier()
    return {
        "strategy": strategy,
        "steps": num_steps,
        "global_step": state.global_step,
        "lambda": lambda_mix,
        "loss_mean": loss_sum / num_steps,
        "loss_old_mean": old_loss_sum / num_steps,
        "loss_new_mean": new_loss_sum / num_steps,
        "mean_flow_loss_mean": (
            mean_flow_loss_sum / num_steps if mean_flow is not None else None
        ),
        "wall_seconds": time.time() - start,
    }
