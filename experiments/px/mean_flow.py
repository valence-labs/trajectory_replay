"""ADM MeanFlow identity losses for PX replay."""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.autograd.functional import jvp

from experiments.px.model import ADMMeanFlowWrapper
from experiments.px.utils.distributed import Distributed
from experiments.px.utils.infinite_dataloader import infinite_dataloader


def sample_time_pairs(
    n: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    diag_fraction: float = 0.25,
    time_sampler: str = "logit_normal",
    time_mu: float = 2.0,
    time_sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if time_sampler == "uniform":
        values = torch.rand((n, 2), device=device, generator=generator)
    elif time_sampler == "logit_normal":
        values = torch.randn((n, 2), device=device, generator=generator)
        values = torch.sigmoid(values * float(time_sigma) + float(time_mu))
    else:
        raise ValueError(f"Unknown time_sampler={time_sampler!r}")
    values, _ = values.sort(dim=1)
    r, t = values[:, 0], values[:, 1]
    diagonal = torch.rand(n, device=device, generator=generator) < diag_fraction
    r = torch.where(diagonal, t, r)
    return r, t


def identity_loss_from_anchor(
    mean_flow: ADMMeanFlowWrapper,
    *,
    z_r: torch.Tensor,
    v_r: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    u = mean_flow.calc_u(x=z_r, r=r, t=t, y=y, e=e, c=c)

    def u_fn(
        z_value: torch.Tensor,
        r_value: torch.Tensor,
        t_value: torch.Tensor,
    ) -> torch.Tensor:
        return mean_flow.calc_u(
            x=z_value,
            r=r_value,
            t=t_value,
            y=y,
            e=e,
            c=c,
        )

    _, du_dr = jvp(
        u_fn,
        (z_r.detach(), r.detach(), t.detach()),
        (v_r.detach(), torch.ones_like(r), torch.zeros_like(t)),
        create_graph=False,
        strict=False,
    )
    target = (
        v_r + (t - r).reshape(-1, 1, 1, 1) * du_dr
    ).detach()
    return torch.nn.functional.mse_loss(u, target)


def identity_loss_from_endpoint(
    mean_flow: ADMMeanFlowWrapper,
    *,
    x1: torch.Tensor,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    generator: torch.Generator,
    diag_fraction: float = 0.25,
    r_scalar: float | None = None,
    time_sampler: str = "logit_normal",
    time_mu: float = 2.0,
    time_sigma: float = 2.0,
) -> torch.Tensor:
    x0 = torch.randn(
        x1.shape,
        device=x1.device,
        dtype=x1.dtype,
        generator=generator,
    )
    if r_scalar is None:
        r, t = sample_time_pairs(
            x1.shape[0],
            device=x1.device,
            generator=generator,
            diag_fraction=diag_fraction,
            time_sampler=time_sampler,
            time_mu=time_mu,
            time_sigma=time_sigma,
        )
    else:
        r = torch.full(
            (x1.shape[0],),
            float(r_scalar),
            device=x1.device,
            dtype=x1.dtype,
        )
        if time_sampler == "uniform":
            u = torch.rand(
                x1.shape[0],
                device=x1.device,
                generator=generator,
            )
        elif time_sampler == "logit_normal":
            u = torch.randn(
                x1.shape[0],
                device=x1.device,
                generator=generator,
            )
            u = torch.sigmoid(u * float(time_sigma) + float(time_mu))
        else:
            raise ValueError(f"Unknown time_sampler={time_sampler!r}")
        t = r + (1.0 - r) * u
        diagonal = (
            torch.rand(
                x1.shape[0],
                device=x1.device,
                generator=generator,
            )
            < diag_fraction
        )
        t = torch.where(diagonal, r, t)
    r_view = r.reshape(-1, 1, 1, 1)
    z_r = (1.0 - r_view) * x0 + r_view * x1
    return identity_loss_from_anchor(
        mean_flow,
        z_r=z_r,
        v_r=x1 - x0,
        r=r,
        t=t,
        y=y,
        e=e,
        c=c,
    )


def train_task0_mean_flow(
    *,
    mean_flow: ADMMeanFlowWrapper,
    loader: Iterable,
    D: Distributed,
    num_steps: int,
    lr: float,
    seed: int,
    max_grad_norm: float = 1.0,
    diag_fraction: float = 0.25,
    time_sampler: str = "logit_normal",
    time_mu: float = 2.0,
    time_sigma: float = 2.0,
) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(
        mean_flow.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    generator = torch.Generator(device=D.device)
    generator.manual_seed(seed + D.rank)
    iterator = infinite_dataloader(loader)
    loss_sum = 0.0
    started = time.time()
    mean_flow.train()
    for _ in range(num_steps):
        batch = next(iterator)
        x1 = batch["pert"].to(D.device, non_blocking=True)
        y = batch["perturbation_id"].long().to(D.device, non_blocking=True)
        e = batch["experiment_id"].long().to(D.device, non_blocking=True)
        c = batch["cell_type_id"].long().to(D.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = identity_loss_from_endpoint(
                mean_flow,
                x1=x1,
                y=y,
                e=e,
                c=c,
                generator=generator,
                diag_fraction=diag_fraction,
                time_sampler=time_sampler,
                time_mu=time_mu,
                time_sigma=time_sigma,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite task-0 MeanFlow loss")
        loss.backward()
        if dist.is_initialized() and dist.get_world_size() > 1:
            for parameter in mean_flow.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(dist.get_world_size())
        torch.nn.utils.clip_grad_norm_(
            mean_flow.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        loss_sum += float(loss.detach())
    D.barrier()
    return {
        "steps": num_steps,
        "loss_mean": loss_sum / num_steps,
        "wall_seconds": time.time() - started,
    }
