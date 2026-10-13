"""Image-CL MeanFlow: spatial Diffusers-UNet student distilled from UNet CFM.

Replaces the earlier flatten→MLP adapter. The MeanFlow student keeps the same
Masip ``UNet2DModel`` capacity as the CFM teacher and uses dual-time conditioning
(``t`` as timestep, ``r`` as ``class_labels`` with ``class_embed_type='timestep'``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.functional import jvp as autograd_jvp
from torch.utils.data import DataLoader, TensorDataset

from trajectory_replay.model.diffusers_unet_fm import (
    DiffusersUNetFlowMatching,
    disable_fused_sdpa_if_needed,
)
from trajectory_replay.model.diffusers_unet_mean_flow import (
    DiffusersUNetMeanFlow,
    load_diffusers_unet_mean_flow_checkpoint,
    save_diffusers_unet_mean_flow_checkpoint,
)
from trajectory_replay.training.mean_flow_distill import (
    DEFAULT_TIME_MU,
    DEFAULT_TIME_SAMPLER,
    DEFAULT_TIME_SIGMA,
    MEAN_FLOW_CKPT_NAME,
    sample_meanflow_times,
)


def _broadcast_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (x.dim() - 1)))


def _linear_interpolant(s: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    s_exp = _broadcast_time(s, x0)
    return (1.0 - s_exp) * x0 + s_exp * x1


def _identity_loss_from_targets(
    mean_flow: DiffusersUNetMeanFlow,
    z_r: torch.Tensor,
    v_r: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    *,
    c: torch.Tensor | None = None,
    adaptive_p: float | None = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared JVP identity step given an already-known ``(z_r, v_r)`` anchor pair.

    ``u(z, r, t)`` is the average velocity conditioned on the state at ``r``, so
    that ``transport`` can jump forward: ``z_t = z_r + (t - r) u(z_r, r, t)``.
    Differentiating ``(t - r) u(z_r, r, t) = ∫_r^t v(z_s, s) ds`` in ``r`` gives

        ``u = v(z_r, r) + (t - r) * d/dr u(z_r, r, t)``

    where ``d/dr`` follows the trajectory (tangent ``(v_r, 1, 0)``). Note the
    ``+`` sign and the ``r``-derivative: differentiating in ``t`` instead would
    condition ``u`` on the state at ``t``, which is the data end under this
    time convention and is not what the sampler can supply.

    Uses a separate student forward for ``u`` (grad path) and
    ``torch.autograd.functional.jvp`` for the derivative only — Diffusers
    attention is incompatible with ``torch.func.jvp`` / flash-SDPA forward AD.

    ``v_r`` comes from a frozen CFM teacher
    (:func:`_spatial_mean_flow_identity_loss`).

    Returns ``(loss, mse, mse_per)``: the (optionally adaptively weighted) objective
    used for backprop, the unweighted batch-mean MSE for logging, and per-sample MSE.
    """
    disable_fused_sdpa_if_needed()
    labels = None
    if c is not None:
        labels = c.to(device=z_r.device, dtype=torch.long).view(-1)
        if labels.shape[0] != z_r.shape[0]:
            raise ValueError(
                f"c batch {labels.shape[0]} != z_r batch {z_r.shape[0]}"
            )
    # Gradient path (zhuyu-cs / paper Alg. 1 pattern).
    u = mean_flow.calc_u(z_r, r, t, labels)

    def u_fn(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        return mean_flow.calc_u(z_in, r_in, t_in, labels)

    # Target path: stopgrad on dudr via detach of u_target below.
    _, dudr = autograd_jvp(
        u_fn,
        (z_r.detach(), r.detach(), t.detach()),
        (v_r, torch.ones_like(r), torch.zeros_like(t)),
        create_graph=False,
        strict=False,
    )
    time_diff = _broadcast_time(t - r, v_r)
    u_target = (v_r + time_diff * dudr).detach()

    err = u - u_target
    mse_per = (err**2).reshape(err.shape[0], -1).mean(dim=-1)
    mse = mse_per.mean()
    if adaptive_p is not None and adaptive_p > 0:
        # Match zhuyu-cs: adaptive weights act on the per-sample squared L2
        # norm, not the per-pixel mean.  The fixed 1e-3 floor is calibrated to
        # that summed scale.  Keep MSE logging normalized per pixel.
        sq_norm_per = (err**2).reshape(err.shape[0], -1).sum(dim=-1)
        weights = 1.0 / (sq_norm_per.detach() + 1e-3).pow(adaptive_p)
        return (weights * sq_norm_per).mean(), mse.detach(), mse_per.detach()
    return mse, mse.detach(), mse_per.detach()


def _spatial_mean_flow_identity_loss(
    mean_flow: DiffusersUNetMeanFlow,
    teacher: DiffusersUNetFlowMatching | None,
    x0: torch.Tensor,
    x1: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    *,
    y: torch.Tensor | None = None,
    adaptive_p: float | None = 1.0,
    velocity: str = "ot",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MeanFlow identity loss on spatial states (FM: t=0 noise → t=1 data).

    ``velocity='ot'`` (default, zhuyu-style): ``v = x1 - x0`` on the interpolant.
    ``velocity='teacher'``: ``v_r`` from a frozen CFM at ``z_r``.
    """
    if velocity not in ("ot", "teacher"):
        raise ValueError(f"velocity must be 'ot' or 'teacher', got {velocity!r}")
    z_r = _linear_interpolant(r, x0, x1)
    labels = None if y is None else y.to(device=z_r.device, dtype=torch.long).view(-1)
    if velocity == "ot":
        v_r = x1 - x0
    else:
        if teacher is None:
            raise ValueError("velocity='teacher' requires a CFM teacher")
        with torch.no_grad():
            if labels is None:
                v_r = teacher(z_r, r)
            else:
                v_r = teacher(z_r, r, labels, guidance_scale=1.0)
    return _identity_loss_from_targets(
        mean_flow, z_r, v_r, r, t, c=labels, adaptive_p=adaptive_p
    )


def _unet_capacity_kwargs_from_teacher(teacher: DiffusersUNetFlowMatching) -> dict:
    """Copy Masip UNet capacity from a CFM teacher for MeanFlow distill."""
    ucfg = teacher.unet.config
    kw: dict = {
        "layers_per_block": int(ucfg.layers_per_block),
        "block_out_channels": list(ucfg.block_out_channels),
        "norm_num_groups": int(ucfg.norm_num_groups),
        "down_block_types": list(ucfg.down_block_types),
        "up_block_types": list(ucfg.up_block_types),
    }
    if getattr(ucfg, "attention_head_dim", None) is not None:
        kw["attention_head_dim"] = ucfg.attention_head_dim
    return kw


def _distill_batch_loss_spatial(
    mean_flow: DiffusersUNetMeanFlow,
    x1: torch.Tensor,
    *,
    device: torch.device,
    gen: torch.Generator,
    teacher: DiffusersUNetFlowMatching | None = None,
    y: torch.Tensor | None = None,
    diag_fraction: float = 0.75,
    adaptive_p: float | None = 1.0,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    velocity: str = "ot",
) -> dict[str, torch.Tensor]:
    """MeanFlow task-0 batch with paper/repo ``(r, t)`` sampling (logit-normal + diag)."""
    x1 = x1.to(device)
    b = x1.shape[0]
    labels = None if y is None else y.to(device=device, dtype=torch.long).view(-1)
    if labels is not None and labels.shape[0] != b:
        raise ValueError(f"y batch {labels.shape[0]} != x1 batch {b}")
    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
    r, t = sample_meanflow_times(
        b,
        device,
        gen,
        diag_fraction=diag_fraction,
        time_sampler=time_sampler,
        time_mu=time_mu,
        time_sigma=time_sigma,
    )
    loss, mse, mse_per = _spatial_mean_flow_identity_loss(
        mean_flow,
        teacher,
        x0,
        x1,
        r,
        t,
        y=labels,
        adaptive_p=adaptive_p,
        velocity=velocity,
    )
    diag_mask = r == t
    offdiag_mask = ~diag_mask
    mse_diag = mse_per[diag_mask].mean() if bool(diag_mask.any()) else mse.new_tensor(float("nan"))
    mse_offdiag = (
        mse_per[offdiag_mask].mean() if bool(offdiag_mask.any()) else mse.new_tensor(float("nan"))
    )
    return {
        "loss": loss,
        "mse": mse,
        "mse_diag": mse_diag,
        "mse_offdiag": mse_offdiag,
    }


def distill_mean_flow_from_image_teacher(
    teacher: DiffusersUNetFlowMatching,
    x_train: torch.Tensor,
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    train_seed: int,
    y_train: torch.Tensor | None = None,
    diag_fraction: float = 0.75,
    adaptive_p: float | None = 1.0,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    hidden_dim: int = 512,  # unused (API compat); UNet capacity matches teacher
    depth: int = 6,  # unused (API compat)
    eval_every: int = 0,
    parent_checkpoint: str = "",
    unet_kwargs: dict | None = None,
    use_adamw: bool = True,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 1.0,
) -> tuple[DiffusersUNetMeanFlow, DiffusersUNetMeanFlow, list[dict[str, float]]]:
    """Train task-0 spatial MeanFlow on data (OT velocity), matching CFM capacity.

    The CFM ``teacher`` is used only for UNet width / class-cond flags, not as
    a velocity field. Identity targets are ``v = x1 - x0`` (zhuyu-style).

    When class-conditional, pass matching ``y_train`` (no CFG during this step).
    Optimizer defaults match joint MeanFlow CL (AdamW, β₂=0.95, clip 1.0).
    """
    del hidden_dim, depth, eval_every  # UNet MeanFlow; no MLP / optional eval yet
    disable_fused_sdpa_if_needed()
    # Seed before constructing the MeanFlow student so its initial weights are
    # reproducible across train_seed values (must precede DiffusersUNetMeanFlow below).
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    teacher_cond = bool(getattr(teacher, "use_conditioning", False))
    if teacher_cond and y_train is None:
        raise ValueError("y_train is required when distilling a class-conditional teacher")
    if y_train is not None and y_train.shape[0] != x_train.shape[0]:
        raise ValueError(
            f"y_train length {y_train.shape[0]} != x_train length {x_train.shape[0]}"
        )

    capacity = unet_kwargs if unet_kwargs is not None else _unet_capacity_kwargs_from_teacher(teacher)
    num_classes = int(getattr(teacher, "num_classes", 0) or 0) if teacher_cond else 0
    guidance_scale = float(getattr(teacher, "guidance_scale", 1.0) or 1.0)
    mean_flow = DiffusersUNetMeanFlow(
        image_size=teacher.image_size,
        in_channels=teacher.in_channels,
        train_timesteps=teacher.train_timesteps,
        unet_kwargs=capacity,
        num_classes=num_classes,
        guidance_scale=guidance_scale,
    ).to(device)

    if y_train is None:
        dataset = TensorDataset(x_train.detach().cpu())
    else:
        dataset = TensorDataset(x_train.detach().cpu(), y_train.detach().cpu().long())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=False,
        num_workers=0,
    )
    from trajectory_replay.training.mixture_fm import _optimizer_step

    mf_params = list(mean_flow.parameters())
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    opt = opt_cls(mf_params, lr=lr, betas=betas, weight_decay=weight_decay)
    clip = (
        float(max_grad_norm)
        if max_grad_norm is not None and float(max_grad_norm) > 0.0
        else None
    )
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)

    seg_keys = ("mse_diag", "mse_offdiag")
    history: list[dict[str, float]] = []
    mean_flow.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        mses: list[float] = []
        seg: dict[str, list[float]] = {k: [] for k in seg_keys}
        for batch_tensors in loader:
            if len(batch_tensors) == 1:
                (x1,) = batch_tensors
                y1 = None
            else:
                x1, y1 = batch_tensors
            batch = _distill_batch_loss_spatial(
                mean_flow,
                x1,
                device=device,
                gen=gen,
                y=y1,
                diag_fraction=diag_fraction,
                adaptive_p=adaptive_p,
                time_sampler=time_sampler,
                time_mu=time_mu,
                time_sigma=time_sigma,
                velocity="ot",
            )
            _optimizer_step(
                opt, batch["loss"], params=mf_params, max_grad_norm=clip
            )
            losses.append(float(batch["loss"].item()))
            mses.append(float(batch["mse"].item()))
            for k in seg_keys:
                if k in batch and torch.isfinite(batch[k]):
                    seg[k].append(float(batch[k].item()))
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "mse_mean": float(np.mean(mses)) if mses else float("nan"),
        }
        for k in seg_keys:
            row[f"{k}_mean"] = float(np.mean(seg[k])) if seg[k] else float("nan")
        history.append(row)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            # The adaptive objective and per-pixel MSE have different scales;
            # MSE is the clearest convergence diagnostic.
            print(
                f"  meanflow distill {epoch:5d}/{epochs}  mse {row['mse_mean']:.6f}"
                f"  diag {row['mse_diag_mean']:.6f}"
                f"  offdiag {row['mse_offdiag_mean']:.6f}"
            )

    out_dir = Path(out_dir)
    save_diffusers_unet_mean_flow_checkpoint(
        out_dir / MEAN_FLOW_CKPT_NAME,
        mean_flow,
        parent_checkpoint=parent_checkpoint,
        extra={
            "distill_epochs": epochs,
            "diag_fraction": diag_fraction,
            "adaptive_p": adaptive_p,
            "time_sampler": time_sampler,
            "time_mu": time_mu,
            "time_sigma": time_sigma,
            "teacher_backbone": "diffusers_unet2d",
            "velocity": "ot",
        },
    )
    mean_flow.eval()
    # Replay object is the MeanFlow itself (spatial sample API).
    return mean_flow, mean_flow, history


@torch.no_grad()
def mean_flow_replay_fidelity(
    mean_flow: DiffusersUNetMeanFlow,
    teacher: DiffusersUNetFlowMatching,
    *,
    device: torch.device,
    n: int = 64,
    n_steps: int = 1,
    ode_method: str = "euler",
    ode_step_size: float = 0.1,
    seed: int = 0,
    class_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Compare the MeanFlow jump against the teacher ODE from shared noise.

    Guards the replay path: a mis-conditioned average-velocity field still fits
    the distill objective while producing pure-noise one-step samples, so the
    distill loss alone cannot detect it.

    For class-conditional models, pass ``class_ids`` (balanced labels are drawn
    from this set); sampling uses ``guidance_scale=1.0``.
    """
    dtype = next(mean_flow.parameters()).dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x0 = mean_flow.sample_noise(n, device=device, dtype=dtype, generator=gen)
    labels = None
    use_cond = bool(getattr(mean_flow, "use_conditioning", False)) or bool(
        getattr(teacher, "use_conditioning", False)
    )
    if use_cond:
        if not class_ids:
            raise ValueError("class_ids required for class-conditional MeanFlow fidelity")
        cpu_gen = torch.Generator()
        cpu_gen.manual_seed(seed)
        ids = torch.tensor(list(class_ids), dtype=torch.long)
        idx = torch.randint(0, ids.numel(), (n,), generator=cpu_gen)
        labels = ids[idx].to(device)
    x_teacher = teacher.sample_from_x0(
        x0,
        labels,
        method=ode_method,
        options={"step_size": ode_step_size},
        guidance_scale=1.0 if labels is not None else None,
    )
    x_mf = mean_flow.sample_from_x0(
        x0,
        labels,
        n_steps=n_steps,
        guidance_scale=1.0 if labels is not None else None,
    )
    denom = float((x_teacher**2).sum().item())
    rel_err = float(((x_mf - x_teacher) ** 2).sum().item()) / max(denom, 1e-12)
    return {
        "n": float(n),
        "n_steps": float(n_steps),
        "rel_err_vs_teacher": rel_err,
        "std_ratio": float(x_mf.std().item()) / max(float(x_teacher.std().item()), 1e-12),
    }


def load_spatial_mean_flow_replay(
    path: Path,
    device: torch.device,
    image_shape: tuple[int, int, int] | None = None,
) -> DiffusersUNetMeanFlow:
    del image_shape
    model, _cfg = load_diffusers_unet_mean_flow_checkpoint(path, device)
    model.eval()
    return model


def make_spatial_mean_flow_from_teacher(
    teacher: DiffusersUNetFlowMatching,
    *,
    device: torch.device | None = None,
) -> DiffusersUNetMeanFlow:
    """Fresh spatial MeanFlow with the same UNet capacity as ``teacher``."""
    disable_fused_sdpa_if_needed()
    teacher_cond = bool(getattr(teacher, "use_conditioning", False))
    capacity = _unet_capacity_kwargs_from_teacher(teacher)
    num_classes = int(getattr(teacher, "num_classes", 0) or 0) if teacher_cond else 0
    guidance_scale = float(getattr(teacher, "guidance_scale", 1.0) or 1.0)
    mean_flow = DiffusersUNetMeanFlow(
        image_size=teacher.image_size,
        in_channels=teacher.in_channels,
        train_timesteps=teacher.train_timesteps,
        unet_kwargs=capacity,
        num_classes=num_classes,
        guidance_scale=guidance_scale,
    )
    if device is not None:
        mean_flow.to(device)
    return mean_flow


@torch.no_grad()
def _euler_rollout_to_times_image(
    model: DiffusersUNetFlowMatching,
    x0: torch.Tensor,
    t_ends: torch.Tensor,
    *,
    step_size: float,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized Euler integrate from ``t=0`` to per-sample ``t_ends`` (images)."""
    if t_ends.shape != (x0.shape[0],):
        raise ValueError(f"t_ends must be shape (B,), got {tuple(t_ends.shape)}")
    if step_size <= 0.0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if bool(getattr(model, "use_conditioning", False)) and y is None:
        raise ValueError("class labels y are required for class-conditional rollout")

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
    labels = None if y is None else y.to(device=device, dtype=torch.long).view(-1)
    for _ in range(n_steps):
        remaining = (t_ends - t).clamp(min=0.0)
        active = remaining > 1e-8
        if not bool(active.any()):
            break
        dt_i = torch.minimum(torch.full_like(t, dt), remaining)
        if labels is None:
            v = model(x, t)
        else:
            v = model(x, t, labels, guidance_scale=1.0)
        dt_view = dt_i.view(-1, *([1] * (x.dim() - 1)))
        active_view = active.view(-1, *([1] * (x.dim() - 1)))
        x = torch.where(active_view, x + dt_view * v, x)
        t = t + dt_i
    return x


def _data_branch_zr_vr_image(
    x1: torch.Tensor,
    r: torch.Tensor,
    *,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form CFM interpolant anchors ``(z_r, v_r)`` on endpoint data."""
    x1 = x1.to(device)
    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
    r_exp = r.view(-1, *([1] * (x1.dim() - 1)))
    z_r = (1.0 - r_exp) * x0 + r_exp * x1
    v_r = x1 - x0
    return z_r, v_r


# Back-compat alias used by older tests / call sites.
_new_branch_zr_vr_image = _data_branch_zr_vr_image


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
    """Expand a shared ``r`` and sample ``t >= r`` (diag_fraction force ``t = r``)."""
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
    # Map u∈(0,1) into [r, 1]; diagonal mass sets t=r.
    t = r + (1.0 - r) * u
    equal = torch.rand(n, device=device, generator=gen) < float(diag_fraction)
    t = torch.where(equal, r, t)
    t = torch.maximum(t, r)
    return r, t


def _rollout_to_r_image(
    *,
    replay_mean_flow: DiffusersUNetMeanFlow | None,
    model_ref: DiffusersUNetFlowMatching,
    x0: torch.Tensor,
    r_scalar: float,
    y: torch.Tensor | None,
    mean_flow_n_steps: int,
    ode_step_size: float,
) -> torch.Tensor:
    """Roll noise to shared ``r`` with prev MeanFlow (N jumps) or CFM Euler fallback."""
    r_scalar = float(min(max(r_scalar, 0.0), 1.0))
    if r_scalar <= 1e-8:
        return x0.clone()
    if replay_mean_flow is not None:
        return replay_mean_flow.sample_from_x0(
            x0,
            c=y,
            t_end=r_scalar,
            n_steps=int(mean_flow_n_steps),
            guidance_scale=1.0 if y is not None else None,
        )
    r_vec = torch.full(
        (x0.shape[0],),
        r_scalar,
        device=x0.device,
        dtype=x0.dtype,
    )
    return _euler_rollout_to_times_image(
        model_ref, x0, r_vec, step_size=ode_step_size, y=y
    )


def _teacher_velocity_image(
    model_ref: DiffusersUNetFlowMatching,
    z: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor | None,
) -> torch.Tensor:
    if y is None:
        return model_ref(z, t)
    return model_ref(z, t, y, guidance_scale=1.0)


def spatial_mean_flow_two_branch_loss(
    mean_flow: DiffusersUNetMeanFlow,
    model_ref: DiffusersUNetFlowMatching,
    x_new: torch.Tensor,
    *,
    n_old: int,
    lambda_mix: float,
    device: torch.device,
    gen: torch.Generator,
    ode_step_size: float,
    y_new: torch.Tensor | None = None,
    y_old: torch.Tensor | None = None,
    diag_fraction: float = 0.75,
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
) -> dict[str, torch.Tensor]:
    """λ-weighted spatial MeanFlow loss: old anchors + new data path.

    Preferred call sites pass precomputed old ``(z_old, v_old, r_old, t_old)``
    (trajectory shared sim) or ``x_old`` (generative: treat MF samples as data).
    If neither is given, falls back to Euler of ``model_ref`` to sampled ``r``.
    """
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if n_old < 1:
        raise ValueError(f"n_old must be >= 1, got {n_old}")
    n_new = int(x_new.shape[0])
    if n_new < 1:
        raise ValueError("x_new must be non-empty")
    class_cond = bool(getattr(model_ref, "use_conditioning", False))
    if class_cond and y_new is None:
        raise ValueError("y_new is required for class-conditional two-branch loss")
    if class_cond and y_old is None and x_old is None and z_old is None:
        raise ValueError("y_old is required for class-conditional two-branch loss")

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
            if r_old.shape[0] != n_old or t_old.shape[0] != n_old:
                raise ValueError("precomputed old times must have length n_old")
        elif x_old is not None:
            if int(x_old.shape[0]) != n_old:
                raise ValueError(
                    f"x_old batch {x_old.shape[0]} != n_old={n_old}"
                )
            if r_old is None or t_old is None:
                r_old, t_old = sample_meanflow_times(n_old, device, gen, **time_kwargs)
            z_old, v_old = _data_branch_zr_vr_image(
                x_old, r_old, device=device, gen=gen
            )
        else:
            if r_old is None or t_old is None:
                r_old, t_old = sample_meanflow_times(n_old, device, gen, **time_kwargs)
            dtype = next(model_ref.parameters()).dtype
            x0 = model_ref.sample_noise(n_old, device=device, dtype=dtype, generator=gen)
            z_old = _euler_rollout_to_times_image(
                model_ref, x0, r_old, step_size=ode_step_size, y=y_old
            )
            v_old = _teacher_velocity_image(model_ref, z_old, r_old, y_old)
        z_new, v_new = _data_branch_zr_vr_image(x_new, r_new, device=device, gen=gen)

    loss_old, mse_old, _ = _identity_loss_from_targets(
        mean_flow, z_old, v_old, r_old, t_old, c=y_old, adaptive_p=adaptive_p
    )
    loss_new, mse_new, _ = _identity_loss_from_targets(
        mean_flow, z_new, v_new, r_new, t_new, c=y_new, adaptive_p=adaptive_p
    )
    loss = lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new
    return {
        "loss": loss,
        "loss_old": loss_old,
        "loss_new": loss_new,
        "mse_old": mse_old,
        "mse_new": mse_new,
    }


def train_image_joint_mean_flow(
    model: DiffusersUNetFlowMatching,
    mean_flow: DiffusersUNetMeanFlow,
    model_ref: DiffusersUNetFlowMatching,
    loader: DataLoader,
    *,
    strategy: str,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    ode_method: str = "euler",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    ode_step_size: float = 0.05,
    traj_kappa: str = "none",
    replay_mean_flow: DiffusersUNetMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
    old_class_ids: Sequence[int] | None = None,
    label_dropout: float = 0.1,
    mf_lr: float | None = None,
    diag_fraction: float = 0.75,
    adaptive_p: float | None = None,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    max_grad_norm: float | None = None,
    mf_max_grad_norm: float | None = 1.0,
    mf_use_adamw: bool = True,
    mf_betas: tuple[float, float] = (0.9, 0.95),
    mf_weight_decay: float = 0.0,
) -> list[dict[str, float]]:
    """Co-train image CFM + spatial MeanFlow with shared old-branch simulation.

    **Trajectory:** sample shared ``r``; roll ``MF^{k-1}`` (N jumps) to ``r``;
    evaluate frozen ``v^{k-1}``; CFM field-match and MF identity share that state;
    MF also samples ``t >= r``. New branch uses real data at the same ``r``.

    **Generative:** ``MF^{k-1}`` samples endpoints (N jumps to data); both CFM and
    MF use data-conditioned losses on those samples and on real task data.
    """
    from trajectory_replay.training.mixture_fm import (
        _cfm_loss,
        _dropout_labels,
        _optimizer_step,
        _replay_counts,
        _sample_old_labels,
        _sample_shared_time,
        _should_log,
        _trajectory_counts,
    )
    from trajectory_replay.training.traj_kappa import resolve_traj_kappa

    if strategy not in ("generative", "trajectory"):
        raise ValueError(f"strategy must be generative|trajectory, got {strategy!r}")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if not (0.0 <= diag_fraction <= 1.0):
        raise ValueError(f"diag_fraction must be in [0, 1], got {diag_fraction}")
    if int(mean_flow_n_steps) < 1:
        raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
    if float(ode_step_size) <= 0.0:
        raise ValueError("ode_step_size must be a positive float")
    class_cond = bool(getattr(model, "use_conditioning", False))
    if class_cond != bool(getattr(model_ref, "use_conditioning", False)):
        raise ValueError("model and model_ref conditioning flags must match")
    if class_cond != bool(getattr(mean_flow, "use_conditioning", False)):
        raise ValueError("mean_flow conditioning flag must match the CFM teacher")
    if class_cond and not old_class_ids:
        raise ValueError("old_class_ids required for class-conditional joint MeanFlow")
    resolve_traj_kappa(traj_kappa, 0.5)

    ode_options = {"step_size": float(ode_step_size)}
    train_params = list(model.parameters())
    mf_params = list(mean_flow.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
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
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    mf_gen = torch.Generator(device=device)
    mf_gen.manual_seed(train_seed + 31_337)

    null_id = int(getattr(model, "null_class_id", -1)) if class_cond else -1
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
        mf_mse_olds: list[float] = []
        mf_mse_news: list[float] = []
        for batch in loader:
            if class_cond:
                if len(batch) != 2:
                    raise ValueError("class-conditional loader must yield (x, y)")
                x1, y1 = batch
                y1 = y1.long()
            else:
                (x1,) = batch if not isinstance(batch, (list, tuple)) else batch
                y1 = None
            b = x1.shape[0]

            if strategy == "generative":
                n_new, n_old = _replay_counts(b, lambda_mix)
                perm = torch.randperm(b, generator=cpu_gen)
                x_new = x1[perm[:n_new]].to(device)
                y_new_raw = None if y1 is None else y1[perm[:n_new]].to(device)
                y_old = (
                    _sample_old_labels(old_class_ids, n_old, cpu_gen=cpu_gen).to(device)
                    if class_cond
                    else None
                )
                with torch.no_grad():
                    if replay_mean_flow is not None:
                        x_old = replay_mean_flow.sample(
                            c=y_old,
                            batch_size=n_old,
                            n_steps=mean_flow_n_steps,
                            generator=gen,
                            guidance_scale=1.0 if class_cond else None,
                        )
                    elif class_cond:
                        x_old = model_ref.sample(
                            c=y_old,
                            batch_size=n_old,
                            method=ode_method,
                            rtol=ode_rtol,
                            atol=ode_atol,
                            options=ode_options,
                            generator=gen,
                            guidance_scale=1.0,
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
                if class_cond:
                    assert y_new_raw is not None and y_old is not None
                    y_batch = _dropout_labels(
                        torch.cat([y_new_raw, y_old], dim=0),
                        null_id=null_id,
                        label_dropout=label_dropout,
                        cpu_gen=cpu_gen,
                    )
                else:
                    y_batch = None
                x_batch = torch.cat([x_new, x_old], dim=0)
                bb = x_batch.shape[0]
                x0 = torch.randn(
                    x_batch.shape, device=device, dtype=x_batch.dtype, generator=gen
                )
                t = torch.rand(bb, device=device, generator=gen)
                t_exp = t.view(-1, *([1] * (x_batch.dim() - 1)))
                xt = (1.0 - t_exp) * x0 + t_exp * x_batch
                v_star = x_batch - x0
                if class_cond:
                    v_pred = model(xt, t, y_batch, guidance_scale=1.0)
                else:
                    v_pred = model(xt, t)
                loss = F.mse_loss(v_pred, v_star)
                loss_new = F.mse_loss(v_pred[:n_new], v_star[:n_new])
                loss_old = F.mse_loss(v_pred[n_new:], v_star[n_new:])

                _optimizer_step(
                    opt, loss, params=train_params, max_grad_norm=max_grad_norm
                )

                y_new_mf = y_new_raw if class_cond else None
                mf = spatial_mean_flow_two_branch_loss(
                    mean_flow,
                    model_ref,
                    x_new,
                    n_old=n_old,
                    lambda_mix=lambda_mix,
                    device=device,
                    gen=mf_gen,
                    ode_step_size=float(ode_step_size),
                    y_new=y_new_mf,
                    y_old=y_old,
                    x_old=x_old,
                    adaptive_p=adaptive_p,
                    **time_kwargs,
                )
            else:
                # Trajectory: one shared r; MF^{k-1} roll + teacher v shared by CFM+MF.
                n_new, n_old = _trajectory_counts(b)
                perm = torch.randperm(b, generator=cpu_gen)
                x_new = x1[perm[:n_new]].to(device)
                y_new_raw = None if y1 is None else y1[perm[:n_new]].to(device)
                if class_cond:
                    assert y_new_raw is not None
                    y_new = _dropout_labels(
                        y_new_raw,
                        null_id=null_id,
                        label_dropout=label_dropout,
                        cpu_gen=cpu_gen,
                    )
                    y_old = _sample_old_labels(
                        old_class_ids, n_old, cpu_gen=cpu_gen
                    ).to(device)
                else:
                    y_new = None
                    y_old = None

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
                    z_r = _rollout_to_r_image(
                        replay_mean_flow=replay_mean_flow,
                        model_ref=model_ref,
                        x0=x0_old,
                        r_scalar=r_scalar,
                        y=y_old,
                        mean_flow_n_steps=mean_flow_n_steps,
                        ode_step_size=float(ode_step_size),
                    )
                    v_r = _teacher_velocity_image(model_ref, z_r, r_old, y_old)

                if class_cond:
                    v_pred_old = model(z_r, r_old, y_old, guidance_scale=1.0)
                else:
                    v_pred_old = model(z_r, r_old)
                loss_old = F.mse_loss(v_pred_old, v_r)
                loss_new = _cfm_loss(
                    model, x_new, device=device, gen=gen, t_scalar=r_scalar, y=y_new
                )
                kappa = resolve_traj_kappa(
                    traj_kappa, r_scalar, dtype=loss_old.dtype, device=loss_old.device
                )
                loss = kappa * (lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new)

                _optimizer_step(
                    opt, loss, params=train_params, max_grad_norm=max_grad_norm
                )

                y_new_mf = y_new_raw if class_cond else None
                mf = spatial_mean_flow_two_branch_loss(
                    mean_flow,
                    model_ref,
                    x_new,
                    n_old=n_old,
                    lambda_mix=lambda_mix,
                    device=device,
                    gen=mf_gen,
                    ode_step_size=float(ode_step_size),
                    y_new=y_new_mf,
                    y_old=y_old,
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
                opt_mf, mf["loss"], params=mf_params, max_grad_norm=mf_max_grad_norm
            )

            losses.append(float(loss.item()))
            loss_olds.append(float(loss_old.item()))
            loss_news.append(float(loss_new.item()))
            mf_losses.append(float(mf["loss"].item()))
            mf_mse_olds.append(float(mf["mse_old"].item()))
            mf_mse_news.append(float(mf["mse_new"].item()))

        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "mf_loss_mean": float(np.mean(mf_losses)),
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
                f"/{row['mf_mse_new_mean']:.6f} (old/new)  N={mean_flow_n_steps}"
            )
    return history
