"""Joint instantaneous + MeanFlow continual-learning trainers.

Protocol (aligned with image CL):

- **Trajectory:** sample shared ``r``; roll ``replay_mean_flow`` (N jumps) or
  frozen CFM Euler to ``r``; evaluate old field; CFM field-match and MF identity
  share that state; MF also samples ``t >= r``. New branch = real data.
- **Generative:** sample old endpoints with ``replay_mean_flow`` (or ODE); both
  CFM and MF use **data-conditioned** losses on those samples and on real data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trajectory_replay.data.tx_dataset import unpack_tx_batch
from trajectory_replay.holdout.data import (
    SupportStratumDistribution,
    sample_support_conditions,
)
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.model.mean_flow import ConditionalMeanFlow
from trajectory_replay.training.mean_flow_distill import (
    DEFAULT_TIME_MU,
    DEFAULT_TIME_SAMPLER,
    DEFAULT_TIME_SIGMA,
    mean_flow_identity_loss_from_velocity,
    sample_meanflow_times,
)
from trajectory_replay.training.mixture_fm import (
    EpochEndCallback,
    _cfm_loss,
    _optimizer_step,
    _replay_counts,
    _sample_shared_time,
    _should_log,
    _trajectory_counts,
)
from trajectory_replay.training.traj_kappa import TrajKappaName, resolve_traj_kappa

StrategyName = Literal["generative", "trajectory"]


@torch.no_grad()
def _euler_rollout_to_times(
    model: ConditionalFlowMatching,
    x0: torch.Tensor,
    t_ends: torch.Tensor,
    *,
    step_size: float,
    c: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    z: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Vectorized Euler integrate from ``t=0`` to per-sample ``t_ends``."""
    if t_ends.shape != (x0.shape[0],):
        raise ValueError(f"t_ends must be shape (B,), got {tuple(t_ends.shape)}")
    if step_size <= 0.0:
        raise ValueError(f"step_size must be > 0, got {step_size}")

    dtype = x0.dtype
    device = x0.device
    t_ends = t_ends.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    t_max = float(t_ends.max().item())
    if t_max <= 0.0:
        return x0.clone()

    n_steps = max(1, int(math.ceil(t_max / float(step_size))))
    dt = t_max / n_steps
    x = x0.clone()
    t = torch.zeros(x0.shape[0], device=device, dtype=dtype)
    for _ in range(n_steps):
        remaining = (t_ends - t).clamp(min=0.0)
        active = remaining > 1e-8
        if not bool(active.any()):
            break
        dt_i = torch.minimum(torch.full_like(t, dt), remaining)
        v = model(x, t, c, p, z=z)
        dt_view = dt_i.view(-1, *([1] * (x.dim() - 1)))
        active_view = active.view(-1, *([1] * (x.dim() - 1)))
        x = torch.where(active_view, x + dt_view * v, x)
        t = t + dt_i
    return x


def _data_branch_zr_vr(
    x1: torch.Tensor,
    r: torch.Tensor,
    *,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form CFM path on endpoint data."""
    x1 = x1.to(device)
    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
    r_exp = r.view(-1, *([1] * (x1.dim() - 1)))
    z_r = (1.0 - r_exp) * x0 + r_exp * x1
    v_r = x1 - x0
    return z_r, v_r


# Back-compat alias.
_new_branch_zr_vr = _data_branch_zr_vr


def _sample_t_given_shared_r(
    n: int,
    r_scalar: float,
    *,
    device: torch.device,
    gen: torch.Generator,
    dtype: torch.dtype,
    diag_fraction: float,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 <= diag_fraction <= 1.0):
        raise ValueError(f"diag_fraction must be in [0, 1], got {diag_fraction}")
    r_scalar = float(min(max(r_scalar, 0.0), 1.0))
    r = torch.full((n,), r_scalar, device=device, dtype=dtype)
    if time_sampler == "uniform":
        u = torch.rand(n, device=device, generator=gen, dtype=dtype)
    elif time_sampler == "logit_normal":
        u = torch.randn(n, device=device, generator=gen, dtype=dtype)
        u = torch.sigmoid(u * float(time_sigma) + float(time_mu))
    else:
        raise ValueError(f"Unknown time_sampler={time_sampler!r}")
    t = r + (1.0 - r) * u
    equal = torch.rand(n, device=device, generator=gen) < float(diag_fraction)
    t = torch.where(equal, r, t)
    t = torch.maximum(t, r)
    return r, t


def _rollout_to_r(
    *,
    replay_mean_flow: ConditionalMeanFlow | None,
    model_ref: ConditionalFlowMatching,
    x0: torch.Tensor,
    r_scalar: float,
    mean_flow_n_steps: int,
    ode_step_size: float,
    c: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    z: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    r_scalar = float(min(max(r_scalar, 0.0), 1.0))
    if r_scalar <= 1e-8:
        return x0.clone()
    if replay_mean_flow is not None:
        return replay_mean_flow.sample_from_x0(
            x0, c, p, z=z, t_end=r_scalar, n_steps=int(mean_flow_n_steps)
        )
    r_vec = torch.full((x0.shape[0],), r_scalar, device=x0.device, dtype=x0.dtype)
    return _euler_rollout_to_times(
        model_ref, x0, r_vec, step_size=ode_step_size, c=c, p=p, z=z
    )


def mean_flow_two_branch_loss(
    mean_flow: ConditionalMeanFlow,
    model_ref: ConditionalFlowMatching,
    x_new: torch.Tensor,
    *,
    n_old: int,
    lambda_mix: float,
    device: torch.device,
    gen: torch.Generator,
    ode_step_size: float,
    diag_fraction: float = 0.25,
    adaptive_p: float | None = None,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    x_old: torch.Tensor | None = None,
    z_old: torch.Tensor | None = None,
    v_old: torch.Tensor | None = None,
    r_old: torch.Tensor | None = None,
    t_old: torch.Tensor | None = None,
    r_new: torch.Tensor | None = None,
    t_new: torch.Tensor | None = None,
    c_old: torch.Tensor | None = None,
    p_old: torch.Tensor | None = None,
    cond_z_old: dict[str, torch.Tensor] | None = None,
    c_new: torch.Tensor | None = None,
    p_new: torch.Tensor | None = None,
    cond_z_new: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """λ-weighted MeanFlow loss: old anchors + new data path."""
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if n_old < 1:
        raise ValueError(f"n_old must be >= 1, got {n_old}")
    n_new = int(x_new.shape[0])
    if n_new < 1:
        raise ValueError("x_new must be non-empty")

    time_kwargs = dict(
        diag_fraction=diag_fraction,
        time_sampler=time_sampler,
        time_mu=time_mu,
        time_sigma=time_sigma,
    )
    if r_new is None or t_new is None:
        r_new, t_new = sample_meanflow_times(n_new, device, gen, **time_kwargs)

    with torch.no_grad():
        if z_old is not None and v_old is not None and r_old is not None and t_old is not None:
            pass
        elif x_old is not None:
            if int(x_old.shape[0]) != n_old:
                raise ValueError(f"x_old batch {x_old.shape[0]} != n_old={n_old}")
            if r_old is None or t_old is None:
                r_old, t_old = sample_meanflow_times(n_old, device, gen, **time_kwargs)
            z_old, v_old = _data_branch_zr_vr(x_old, r_old, device=device, gen=gen)
        else:
            if r_old is None or t_old is None:
                r_old, t_old = sample_meanflow_times(n_old, device, gen, **time_kwargs)
            dtype = next(model_ref.parameters()).dtype
            x0 = model_ref.sample_noise(n_old, device=device, dtype=dtype, generator=gen)
            z_old = _euler_rollout_to_times(
                model_ref,
                x0,
                r_old,
                step_size=ode_step_size,
                c=c_old,
                p=p_old,
                z=cond_z_old,
            )
            v_old = model_ref(z_old, r_old, c_old, p_old, z=cond_z_old)
        z_new, v_new = _data_branch_zr_vr(x_new, r_new, device=device, gen=gen)

    loss_old, mse_old = mean_flow_identity_loss_from_velocity(
        mean_flow,
        z_old,
        v_old,
        r_old,
        t_old,
        c_old,
        p_old,
        z=cond_z_old,
        adaptive_p=adaptive_p,
    )
    loss_new, mse_new = mean_flow_identity_loss_from_velocity(
        mean_flow,
        z_new,
        v_new,
        r_new,
        t_new,
        c_new,
        p_new,
        z=cond_z_new,
        adaptive_p=adaptive_p,
    )
    loss = lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new
    return {
        "loss": loss,
        "loss_old": loss_old,
        "loss_new": loss_new,
        "mse_old": mse_old,
        "mse_new": mse_new,
    }


def train_unconditional_joint_mean_flow(
    model: ConditionalFlowMatching,
    mean_flow: ConditionalMeanFlow,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    strategy: StrategyName,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    ode_method: str = "euler",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    ode_step_size: float | None = 0.05,
    traj_kappa: TrajKappaName | str = "none",
    replay_mean_flow: ConditionalMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
    mf_lr: float | None = None,
    diag_fraction: float = 0.25,
    adaptive_p: float | None = None,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    max_grad_norm: float | None = None,
    mf_max_grad_norm: float | None = None,
    mf_use_adamw: bool = False,
    mf_betas: tuple[float, float] = (0.9, 0.999),
    mf_weight_decay: float = 0.0,
    params: Iterable[torch.nn.Parameter] | None = None,
    mf_params: Iterable[torch.nn.Parameter] | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Co-train instantaneous CFM + MeanFlow with shared old-branch simulation."""
    if strategy not in ("generative", "trajectory"):
        raise ValueError(f"strategy must be generative|trajectory, got {strategy!r}")
    if model.use_conditioning or model_ref.use_conditioning or mean_flow.use_conditioning:
        raise ValueError("joint MeanFlow CL currently supports unconditional models only")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if not (0.0 <= diag_fraction <= 1.0):
        raise ValueError(f"diag_fraction must be in [0, 1], got {diag_fraction}")
    if ode_step_size is None or float(ode_step_size) <= 0.0:
        raise ValueError("ode_step_size must be a positive float for joint MeanFlow CL")
    if int(mean_flow_n_steps) < 1:
        raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
    resolve_traj_kappa(traj_kappa, 0.5)

    ode_options = {"step_size": float(ode_step_size)}
    train_params = list(params if params is not None else model.parameters())
    mf_train_params = list(mf_params if mf_params is not None else mean_flow.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    mf_opt_cls = torch.optim.AdamW if mf_use_adamw else torch.optim.Adam
    opt_mf = mf_opt_cls(
        mf_train_params,
        lr=float(mf_lr if mf_lr is not None else lr),
        betas=mf_betas,
        weight_decay=mf_weight_decay,
    )

    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    mf_gen = torch.Generator(device=device)
    mf_gen.manual_seed(train_seed + 31_337)

    history: list[dict[str, float]] = []
    model.train()
    mean_flow.train()
    model_ref.eval()
    if replay_mean_flow is not None:
        replay_mean_flow.eval()

    time_kwargs = dict(
        diag_fraction=diag_fraction,
        time_sampler=time_sampler,
        time_mu=time_mu,
        time_sigma=time_sigma,
    )

    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_olds: list[float] = []
        loss_news: list[float] = []
        mf_losses: list[float] = []
        mf_olds: list[float] = []
        mf_news: list[float] = []
        mf_mse_olds: list[float] = []
        mf_mse_news: list[float] = []
        for (x1,) in loader:
            b = x1.shape[0]
            if strategy == "generative":
                n_new, n_old = _replay_counts(b, lambda_mix)
                perm = torch.randperm(b, generator=cpu_gen)
                x_new = x1[perm[:n_new]].to(device)
                with torch.no_grad():
                    if replay_mean_flow is not None:
                        x_old = replay_mean_flow.sample(
                            batch_size=n_old,
                            n_steps=mean_flow_n_steps,
                            generator=gen,
                        )
                    else:
                        x_old = model_ref.sample(
                            batch_size=n_old,
                            method=ode_method,
                            rtol=ode_rtol,
                            atol=ode_atol,
                            options=ode_options,
                            generator=gen,
                        )
                x_batch = torch.cat([x_new, x_old], dim=0)
                bb = x_batch.shape[0]
                x0 = torch.randn(
                    x_batch.shape, device=device, dtype=x_batch.dtype, generator=gen
                )
                t = torch.rand(bb, device=device, generator=gen)
                t_exp = t.view(-1, *([1] * (x_batch.dim() - 1)))
                xt = (1.0 - t_exp) * x0 + t_exp * x_batch
                v_star = x_batch - x0
                v_pred = model(xt, t, None, None)
                loss = F.mse_loss(v_pred, v_star)
                loss_new = F.mse_loss(v_pred[:n_new], v_star[:n_new])
                loss_old = F.mse_loss(v_pred[n_new:], v_star[n_new:])
                _optimizer_step(
                    opt, loss, params=train_params, max_grad_norm=max_grad_norm
                )
                mf = mean_flow_two_branch_loss(
                    mean_flow,
                    model_ref,
                    x_new,
                    n_old=n_old,
                    lambda_mix=lambda_mix,
                    device=device,
                    gen=mf_gen,
                    ode_step_size=float(ode_step_size),
                    x_old=x_old,
                    adaptive_p=adaptive_p,
                    **time_kwargs,
                )
            else:
                n_new, n_old = _trajectory_counts(b)
                perm = torch.randperm(b, generator=cpu_gen)
                x_new = x1[perm[:n_new]].to(device)
                r_scalar = float(_sample_shared_time(device=device, gen=gen))
                dtype = next(model.parameters()).dtype
                r_old, t_old = _sample_t_given_shared_r(
                    n_old,
                    r_scalar,
                    device=device,
                    gen=mf_gen,
                    dtype=dtype,
                    **time_kwargs,
                )
                r_new, t_new = _sample_t_given_shared_r(
                    n_new,
                    r_scalar,
                    device=device,
                    gen=mf_gen,
                    dtype=dtype,
                    **time_kwargs,
                )
                with torch.no_grad():
                    x0_old = model_ref.sample_noise(
                        n_old, device=device, dtype=dtype, generator=gen
                    )
                    z_r = _rollout_to_r(
                        replay_mean_flow=replay_mean_flow,
                        model_ref=model_ref,
                        x0=x0_old,
                        r_scalar=r_scalar,
                        mean_flow_n_steps=mean_flow_n_steps,
                        ode_step_size=float(ode_step_size),
                    )
                    v_r = model_ref(z_r, r_old, None, None)
                loss_old = F.mse_loss(model(z_r, r_old, None, None), v_r)
                loss_new = _cfm_loss(
                    model, x_new, device=device, gen=gen, t_scalar=r_scalar
                )
                kappa = resolve_traj_kappa(
                    traj_kappa, r_scalar, dtype=loss_old.dtype, device=loss_old.device
                )
                loss = kappa * (lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new)
                _optimizer_step(
                    opt, loss, params=train_params, max_grad_norm=max_grad_norm
                )
                mf = mean_flow_two_branch_loss(
                    mean_flow,
                    model_ref,
                    x_new,
                    n_old=n_old,
                    lambda_mix=lambda_mix,
                    device=device,
                    gen=mf_gen,
                    ode_step_size=float(ode_step_size),
                    z_old=z_r.detach(),
                    v_old=v_r.detach(),
                    r_old=r_old,
                    t_old=t_old,
                    r_new=r_new,
                    t_new=t_new,
                    adaptive_p=adaptive_p,
                    **time_kwargs,
                )

            _optimizer_step(
                opt_mf,
                mf["loss"],
                params=mf_train_params,
                max_grad_norm=mf_max_grad_norm,
            )

            losses.append(float(loss.item()))
            loss_olds.append(float(loss_old.item()))
            loss_news.append(float(loss_new.item()))
            mf_losses.append(float(mf["loss"].item()))
            mf_olds.append(float(mf["loss_old"].item()))
            mf_news.append(float(mf["loss_new"].item()))
            mf_mse_olds.append(float(mf["mse_old"].item()))
            mf_mse_news.append(float(mf["mse_new"].item()))

        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "mf_loss_mean": float(np.mean(mf_losses)),
            "mf_loss_old_mean": float(np.mean(mf_olds)),
            "mf_loss_new_mean": float(np.mean(mf_news)),
            "mf_mse_old_mean": float(np.mean(mf_mse_olds)),
            "mf_mse_new_mean": float(np.mean(mf_mse_news)),
            "lambda_mix": float(lambda_mix),
            "strategy": strategy,
            "mean_flow_n_steps": float(mean_flow_n_steps),
            "traj_kappa": str(traj_kappa) if strategy == "trajectory" else "none",
        }
        history.append(row)
        if _should_log(epoch, epochs):
            print(
                f"  epoch {epoch:5d}/{epochs}  "
                f"cfm {row['loss_mean']:.6f}  mf_mse {row['mf_mse_old_mean']:.6f}"
                f"/{row['mf_mse_new_mean']:.6f} (old/new)  "
                f"cfm old {row['loss_old_mean']:.6f} new {row['loss_new_mean']:.6f}  "
                f"N={mean_flow_n_steps}"
            )
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_conditional_joint_mean_flow(
    model: ConditionalFlowMatching,
    mean_flow: ConditionalMeanFlow,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    strategy: StrategyName,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    support_contexts: list[int],
    n_perturbations: int,
    replay_stratum_dist: SupportStratumDistribution,
    replay_condition_sampling: str = "proportional",
    replay_mean_flow: ConditionalMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
    ode_step_size: float = 0.05,
    traj_kappa: TrajKappaName | str = "none",
    mf_lr: float | None = None,
    diag_fraction: float = 0.25,
    adaptive_p: float | None = None,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    mf_max_grad_norm: float | None = None,
    mf_use_adamw: bool = False,
    mf_betas: tuple[float, float] = (0.9, 0.999),
    mf_weight_decay: float = 0.0,
    params: Iterable[torch.nn.Parameter] | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    epoch_offset: int = 0,
    log_total_epochs: int | None = None,
    finetune_stage: int | None = None,
    data_dim: int | None = None,
) -> list[dict[str, float]]:
    """Conditional Tx counterpart of :func:`train_unconditional_joint_mean_flow`."""
    del data_dim  # Kept for a shared Tx trainer dispatch interface.
    if strategy not in ("generative", "trajectory"):
        raise ValueError(f"strategy must be generative|trajectory, got {strategy!r}")
    if not model.use_conditioning or not model_ref.use_conditioning or not mean_flow.use_conditioning:
        raise ValueError("conditional joint MeanFlow requires conditional models")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if not support_contexts:
        raise ValueError("support_contexts must be non-empty")
    if int(mean_flow_n_steps) < 1 or float(ode_step_size) <= 0:
        raise ValueError("mean_flow_n_steps and ode_step_size must be positive")

    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    mf_params = list(mean_flow.parameters())
    mf_opt_cls = torch.optim.AdamW if mf_use_adamw else torch.optim.Adam
    opt_mf = mf_opt_cls(
        mf_params,
        lr=float(mf_lr if mf_lr is not None else lr),
        betas=mf_betas,
        weight_decay=mf_weight_decay,
    )
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    mf_gen = torch.Generator(device=device)
    mf_gen.manual_seed(train_seed + 31_337)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    cov_order = model.covariate_col_order
    time_kwargs = {
        "diag_fraction": diag_fraction,
        "time_sampler": time_sampler,
        "time_mu": time_mu,
        "time_sigma": time_sigma,
    }

    def old_conditions(n: int):
        sampled = sample_support_conditions(
            n,
            support_contexts=support_contexts,
            n_perturbations=n_perturbations,
            device=device,
            cpu_gen=cpu_gen,
            sampling=replay_condition_sampling,
            stratum_dist=replay_stratum_dist,
        )
        if len(sampled) == 3:
            return sampled
        c, p = sampled
        return c, p, None

    def subset_z(zb: dict[str, torch.Tensor] | None, idx: torch.Tensor):
        if not cov_order:
            return None
        assert zb is not None
        return {col: zb[col][idx].to(device=device, dtype=torch.long) for col in cov_order}

    def cfm_loss(
        x1: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None,
        *,
        t_scalar: float | None = None,
    ):
        n = int(x1.shape[0])
        x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
        t = (
            torch.rand(n, device=device, generator=gen)
            if t_scalar is None
            else torch.full((n,), t_scalar, device=device, dtype=x1.dtype)
        )
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        target = x1 - x0
        pred = model(xt, t, c, p, z=z)
        return F.mse_loss(pred, target), pred, target

    history: list[dict[str, float]] = []
    model.train()
    mean_flow.train()
    model_ref.eval()
    if replay_mean_flow is not None:
        replay_mean_flow.eval()
    total_epochs = log_total_epochs if log_total_epochs is not None else epoch_offset + epochs
    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        losses: list[float] = []
        old_losses: list[float] = []
        new_losses: list[float] = []
        mf_losses: list[float] = []
        for batch in loader:
            x1, cb, pb, zb = unpack_tx_batch(batch, cov_order)
            b = int(x1.shape[0])
            n_new, n_old = (
                _replay_counts(b, lambda_mix)
                if strategy == "generative"
                else _trajectory_counts(b)
            )
            perm = torch.randperm(b, generator=cpu_gen)
            idx = perm[:n_new]
            x_new = x1[idx].to(device)
            c_new = cb[idx].to(device=device, dtype=torch.long)
            p_new = pb[idx].to(device=device, dtype=torch.long)
            z_new = subset_z(zb, idx)
            c_old, p_old, z_old_cond = old_conditions(n_old)
            dtype = x_new.dtype

            if strategy == "generative":
                with torch.no_grad():
                    if replay_mean_flow is not None:
                        x_old = replay_mean_flow.sample(
                            batch_size=n_old,
                            c=c_old,
                            p=p_old,
                            z=z_old_cond,
                            n_steps=mean_flow_n_steps,
                            generator=gen,
                        )
                    else:
                        x_old = model_ref.sample(
                            batch_size=n_old,
                            c=c_old,
                            p=p_old,
                            z=z_old_cond,
                            method="euler",
                            options={"step_size": float(ode_step_size)},
                            generator=gen,
                        )
                z_batch = (
                    {
                        col: torch.cat([z_new[col], z_old_cond[col]], dim=0)
                        for col in cov_order
                    }
                    if cov_order
                    else None
                )
                loss, pred, target = cfm_loss(
                    torch.cat([x_new, x_old], dim=0),
                    torch.cat([c_new, c_old], dim=0),
                    torch.cat([p_new, p_old], dim=0),
                    z_batch,
                )
                loss_new = F.mse_loss(pred[:n_new], target[:n_new])
                loss_old = F.mse_loss(pred[n_new:], target[n_new:])
                mf = mean_flow_two_branch_loss(
                    mean_flow,
                    model_ref,
                    x_new,
                    n_old=n_old,
                    lambda_mix=lambda_mix,
                    device=device,
                    gen=mf_gen,
                    ode_step_size=ode_step_size,
                    x_old=x_old,
                    c_old=c_old,
                    p_old=p_old,
                    cond_z_old=z_old_cond,
                    c_new=c_new,
                    p_new=p_new,
                    cond_z_new=z_new,
                    adaptive_p=adaptive_p,
                    **time_kwargs,
                )
            else:
                r_scalar = float(_sample_shared_time(device=device, gen=gen))
                r_old, t_old = _sample_t_given_shared_r(
                    n_old, r_scalar, device=device, gen=mf_gen, dtype=dtype, **time_kwargs
                )
                r_new, t_new = _sample_t_given_shared_r(
                    n_new, r_scalar, device=device, gen=mf_gen, dtype=dtype, **time_kwargs
                )
                with torch.no_grad():
                    x0_old = model_ref.sample_noise(
                        n_old, device=device, dtype=dtype, generator=gen
                    )
                    z_r = _rollout_to_r(
                        replay_mean_flow=replay_mean_flow,
                        model_ref=model_ref,
                        x0=x0_old,
                        r_scalar=r_scalar,
                        mean_flow_n_steps=mean_flow_n_steps,
                        ode_step_size=ode_step_size,
                        c=c_old,
                        p=p_old,
                        z=z_old_cond,
                    )
                    v_r = model_ref(z_r, r_old, c_old, p_old, z=z_old_cond)
                loss_old = F.mse_loss(
                    model(z_r, r_old, c_old, p_old, z=z_old_cond), v_r
                )
                loss_new, _, _ = cfm_loss(
                    x_new, c_new, p_new, z_new, t_scalar=r_scalar
                )
                kappa = resolve_traj_kappa(
                    traj_kappa, r_scalar, dtype=loss_old.dtype, device=loss_old.device
                )
                loss = kappa * (lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new)
                mf = mean_flow_two_branch_loss(
                    mean_flow,
                    model_ref,
                    x_new,
                    n_old=n_old,
                    lambda_mix=lambda_mix,
                    device=device,
                    gen=mf_gen,
                    ode_step_size=ode_step_size,
                    z_old=z_r.detach(),
                    v_old=v_r.detach(),
                    r_old=r_old,
                    t_old=t_old,
                    r_new=r_new,
                    t_new=t_new,
                    c_old=c_old,
                    p_old=p_old,
                    cond_z_old=z_old_cond,
                    c_new=c_new,
                    p_new=p_new,
                    cond_z_new=z_new,
                    adaptive_p=adaptive_p,
                    **time_kwargs,
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if context_emb_train_rows is not None:
                ConditionalFlowMatching.mask_embedding_rows_grad(
                    model.context_emb.weight, context_emb_train_rows
                )
            if pert_emb_train_rows is not None:
                ConditionalFlowMatching.mask_embedding_rows_grad(
                    model.pert_emb.weight, pert_emb_train_rows
                )
            opt.step()
            _optimizer_step(
                opt_mf,
                mf["loss"],
                params=mf_params,
                max_grad_norm=mf_max_grad_norm,
            )
            losses.append(float(loss.item()))
            old_losses.append(float(loss_old.item()))
            new_losses.append(float(loss_new.item()))
            mf_losses.append(float(mf["loss"].item()))

        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(old_losses)),
            "loss_new_mean": float(np.mean(new_losses)),
            "mf_loss_mean": float(np.mean(mf_losses)),
            "lambda_mix": float(lambda_mix),
            "strategy": strategy,
            "mean_flow_n_steps": float(mean_flow_n_steps),
        }
        if finetune_stage is not None:
            row["finetune_stage"] = float(finetune_stage)
        history.append(row)
        if _should_log(local_epoch, epochs):
            stage_tag = f" stage {finetune_stage}" if finetune_stage is not None else ""
            print(
                f"  epoch {epoch:5d}/{total_epochs}{stage_tag}  cfm {row['loss_mean']:.6f}  "
                f"mf {row['mf_loss_mean']:.6f}  N={mean_flow_n_steps}"
            )
    return history
