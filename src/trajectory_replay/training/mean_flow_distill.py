"""Distill a MeanFlow average-velocity model from a frozen pretrained velocity field.

API mirrors :mod:`trajectory_replay.training.flow_map_distill` so the two
distillation backends stay interchangeable at the experiment layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.data import DataLoader

from trajectory_replay.model.backbones import DEFAULT_BACKBONE_TYPE
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.model.mean_flow import (
    ConditionalMeanFlow,
    build_conditional_mean_flow_from_config,
)
from trajectory_replay.training.flow_map_distill import (
    linear_interpolant,
)

MEAN_FLOW_CKPT_NAME = "mean_flow.pt"
MEAN_FLOW_METRICS_NAME = "mean_flow_metrics.json"

# MeanFlow paper / zhuyu-cs ImageNet defaults mirrored from their data→noise
# convention to this repo's noise→data convention (μ changes sign; σ does not).
DEFAULT_TIME_SAMPLER = "logit_normal"
DEFAULT_TIME_MU = 0.4
DEFAULT_TIME_SIGMA = 1.0


def sample_meanflow_times(
    n: int,
    device: torch.device,
    gen: torch.Generator,
    *,
    diag_fraction: float = 0.75,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    corner_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(r, t)`` as in zhuyu-cs/MeanFlow ``SILoss.sample_time_steps``.

    Draw two times, sort so ``r <= t``, then set a ``diag_fraction`` share to ``r = t``
    (paper/repo ``ratio_r_not_equal_t = 1 - diag_fraction``).

    Optional ``corner_fraction`` (project-local, not in the MeanFlow paper) then
    overwrites that share of the batch with noise-anchored windows ``r=0``.
    Half of those get the full one-step jump ``t=1``; the other half draw
    ``t ~ U(0,1]``. Use this when training for one-step (``N=1``) eval — under
    ``logit_normal`` the ``(0,1)`` corner is otherwise vanishingly rare. Leave
    at ``0`` for multi-step (``N>1``) training.
    """
    if not (0.0 <= diag_fraction <= 1.0):
        raise ValueError(f"diag_fraction must be in [0, 1], got {diag_fraction}")
    if not (0.0 <= corner_fraction <= 1.0):
        raise ValueError(f"corner_fraction must be in [0, 1], got {corner_fraction}")
    if time_sampler == "uniform":
        time_samples = torch.rand(n, 2, device=device, generator=gen)
    elif time_sampler == "logit_normal":
        time_samples = torch.randn(n, 2, device=device, generator=gen)
        time_samples = time_samples * float(time_sigma) + float(time_mu)
        time_samples = torch.sigmoid(time_samples)
    else:
        raise ValueError(f"Unknown time_sampler={time_sampler!r} (use 'logit_normal' or 'uniform')")
    sorted_samples, _ = torch.sort(time_samples, dim=1)
    r, t = sorted_samples[:, 0], sorted_samples[:, 1]
    equal_mask = torch.rand(n, device=device, generator=gen) < float(diag_fraction)
    r = torch.where(equal_mask, t, r)

    n_corner = int(round(n * float(corner_fraction)))
    if n_corner > 0:
        # Stable, generator-driven permutation so the overlay is reproducible.
        perm = torch.randperm(n, device=device, generator=gen)
        corner_idx = perm[:n_corner]
        n_full = n_corner // 2
        r = r.clone()
        t = t.clone()
        r[corner_idx] = 0.0
        t[corner_idx] = torch.rand(n_corner, device=device, generator=gen).clamp_min(1e-4)
        if n_full > 0:
            t[corner_idx[:n_full]] = 1.0
    return r, t

def _teacher_velocity(
    teacher: ConditionalFlowMatching,
    x: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor | None,
    p: torch.Tensor | None,
    z: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if teacher.use_conditioning:
        return teacher(x, t, c, p, z=z)
    return teacher(x, t)


def mean_flow_identity_loss_from_velocity(
    mean_flow: ConditionalMeanFlow,
    z_r: torch.Tensor,
    v_r: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    *,
    z: dict[str, torch.Tensor] | None = None,
    adaptive_p: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MeanFlow identity loss given a precomputed state/velocity at time ``r``.

    ``u(z, r, t)`` is the average velocity conditioned on the state at ``r``, to
    match ``transport``: ``z_t = z_r + (t - r) u(z_r, r, t)``. Differentiating
    ``(t - r) u(z_r, r, t) = ∫_r^t v(z_s, s) ds`` in ``r`` gives

        ``u_target = v_r + (t - r) * d/dr u(z_r, r, t)``

    via JVP along ``(v_r, 1, 0)``, then ``||u - stopgrad(u_target)||^2``. Under
    our FM time convention (t=0 noise → t=1 data) this r-form matches the
    sampler; the writeup's t-form is the dual under time reversal.

    Returns ``(loss, mse)``: ``loss`` is what to backprop (adaptively weighted if
    ``adaptive_p`` is set), ``mse`` is the raw unweighted error for logging.
    ``adaptive_p`` self-normalizes each sample by its own error, so a loss built
    from separately-weighted branches no longer represents a weighted mixture of
    squared errors — keep it off when mixing branches with a fixed ratio.
    """

    def u_fn(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        return mean_flow.calc_u(z_in, r_in, t_in, c, p, z=z)

    u = u_fn(z_r, r, t)
    primals = (z_r, r, t)
    tangents = (v_r, torch.ones_like(r), torch.zeros_like(t))
    _, dudr = jvp(u_fn, primals, tangents)
    time_diff = (t - r).view(-1, *([1] * (v_r.dim() - 1)))
    u_target = (v_r + time_diff * dudr).detach()

    err = u - u_target
    loss_per = (err**2).reshape(err.shape[0], -1).mean(dim=-1)
    mse = loss_per.detach().mean()
    if adaptive_p is not None and adaptive_p > 0:
        # Match the reference adaptive objective: its 1e-3 floor is calibrated
        # to the per-sample squared L2 norm, not a dimension-normalized MSE.
        sq_norm_per = (err**2).reshape(err.shape[0], -1).sum(dim=-1)
        weights = 1.0 / (sq_norm_per.detach() + 1e-3).pow(adaptive_p)
        return (weights * sq_norm_per).mean(), mse
    return loss_per.mean(), mse


def _mean_flow_identity_loss(
    mean_flow: ConditionalMeanFlow,
    teacher: ConditionalFlowMatching,
    x0: torch.Tensor,
    x1: torch.Tensor,
    c: torch.Tensor | None,
    p: torch.Tensor | None,
    r: torch.Tensor,
    t: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    adaptive_p: float | None = None,
) -> torch.Tensor:
    """MeanFlow loss with frozen teacher velocity on the linear CFM path."""
    z_r = linear_interpolant(r, x0, x1)
    with torch.no_grad():
        v_r = _teacher_velocity(teacher, z_r, r, c, p, z)
    loss, _mse = mean_flow_identity_loss_from_velocity(
        mean_flow,
        z_r,
        v_r,
        r,
        t,
        c,
        p,
        z=z,
        adaptive_p=adaptive_p,
    )
    return loss


def _distill_batch_loss(
    mean_flow: ConditionalMeanFlow,
    teacher: ConditionalFlowMatching,
    x1: torch.Tensor,
    c: torch.Tensor | None,
    p: torch.Tensor | None,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    diag_fraction: float = 0.75,
    adaptive_p: float | None = None,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    corner_fraction: float = 0.0,
) -> dict[str, torch.Tensor]:
    """MeanFlow distill batch with paper/repo ``(r, t)`` sampling (logit-normal + diag)."""
    x1 = x1.to(device)
    b = x1.shape[0]
    x0 = torch.randn(b, data_dim, device=device, generator=gen, dtype=x1.dtype)

    c_dev = p_dev = None
    if mean_flow.use_conditioning or teacher.use_conditioning:
        if c is None or p is None:
            raise ValueError("c and p required for conditional MeanFlow distill")
        c_dev = c.to(device=device, dtype=torch.long)
        p_dev = p.to(device=device, dtype=torch.long)
    z_dev = None
    if z is not None:
        z_dev = {col: z[col].to(device=device, dtype=torch.long) for col in z}

    r, t = sample_meanflow_times(
        b,
        device,
        gen,
        diag_fraction=diag_fraction,
        time_sampler=time_sampler,
        time_mu=time_mu,
        time_sigma=time_sigma,
        corner_fraction=corner_fraction,
    )
    loss = _mean_flow_identity_loss(
        mean_flow,
        teacher,
        x0,
        x1,
        c_dev,
        p_dev,
        r,
        t,
        z=z_dev,
        adaptive_p=adaptive_p,
    )
    return {
        "loss": loss,
        "loss_diag": loss.new_tensor(float("nan")),
        "loss_offdiag": loss.new_tensor(float("nan")),
    }


@torch.no_grad()
def eval_mean_flow_vs_teacher_ode(
    mean_flow: ConditionalMeanFlow,
    teacher: ConditionalFlowMatching,
    x1: torch.Tensor,
    c: torch.Tensor | None,
    p: torch.Tensor | None,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    eval_seed: int,
    n_eval: int = 256,
    ode_method: str = "euler",
    ode_step_size: float = 0.02,
    mean_flow_n_steps: int = 1,
) -> dict[str, float]:
    """MSE between MeanFlow rollout and teacher ODE endpoints."""
    n = min(int(n_eval), x1.shape[0])
    if n <= 0:
        return {"mean_flow_ode_mse": float("nan"), "n_eval": 0.0}

    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(eval_seed)
    idx = torch.randperm(x1.shape[0], generator=cpu_gen)[:n]

    c_sub = p_sub = None
    if mean_flow.use_conditioning or teacher.use_conditioning:
        if c is None or p is None:
            raise ValueError("c and p required for conditional eval")
        c_sub = c[idx].to(device=device, dtype=torch.long)
        p_sub = p[idx].to(device=device, dtype=torch.long)
    z_sub = None
    if z is not None:
        z_sub = {col: z[col][idx].to(device=device, dtype=torch.long) for col in z}

    gen = torch.Generator(device=device)
    gen.manual_seed(eval_seed)
    x0 = torch.randn(n, data_dim, device=device, generator=gen)
    x_mf = mean_flow.sample_from_x0(
        x0,
        c_sub,
        p_sub,
        z=z_sub,
        t_end=1.0,
        n_steps=mean_flow_n_steps,
    )
    options = {"step_size": float(ode_step_size)} if ode_step_size > 0 else None
    x_ode = teacher.sample_from_x0(
        x0,
        c_sub,
        p_sub,
        z=z_sub,
        method=ode_method,
        options=options,
    )
    mse = F.mse_loss(x_mf, x_ode).item()
    return {"mean_flow_ode_mse": float(mse), "n_eval": float(n)}


def train_mean_flow_distill(
    mean_flow: ConditionalMeanFlow,
    teacher: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    diag_fraction: float = 0.75,
    adaptive_p: float | None = None,
    time_sampler: str = DEFAULT_TIME_SAMPLER,
    time_mu: float = DEFAULT_TIME_MU,
    time_sigma: float = DEFAULT_TIME_SIGMA,
    corner_fraction: float = 0.0,
    params: Iterable[torch.nn.Parameter] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    unconditional: bool = False,
    use_adamw: bool = False,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    eval_x1: torch.Tensor | None = None,
    eval_c: torch.Tensor | None = None,
    eval_p: torch.Tensor | None = None,
    eval_z: dict[str, torch.Tensor] | None = None,
    eval_every: int = 0,
    eval_seed: int = 0,
    eval_n: int = 256,
    eval_ode_method: str = "euler",
    eval_ode_step_size: float = 0.02,
    eval_mean_flow_n_steps: int = 1,
) -> list[dict[str, float]]:
    """Train MeanFlow to match a frozen teacher velocity (identity + JVP).

    ``use_adamw``/``betas``/``weight_decay``/``max_grad_norm`` default to plain
    Adam with no clipping (unchanged behavior); set them to match the
    zhuyu-cs/MeanFlow reference recipe (``AdamW``, ``betas=(0.9, 0.95)``,
    ``max_grad_norm=1.0``) when desired.
    """
    from trajectory_replay.data.tx_dataset import unpack_tx_batch

    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    opt_params = list(params if params is not None else mean_flow.parameters())
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    opt = opt_cls(opt_params, lr=lr, betas=betas, weight_decay=weight_decay)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)

    history: list[dict[str, float]] = []
    mean_flow.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        diag_losses: list[float] = []
        offdiag_losses: list[float] = []
        for batch in loader:
            if unconditional:
                x1 = batch[0] if isinstance(batch, (tuple, list)) else batch
                cb = pb = zb = None
            else:
                x1, cb, pb, zb = unpack_tx_batch(batch, covariate_col_order)
            out = _distill_batch_loss(
                mean_flow,
                teacher,
                x1,
                cb,
                pb,
                z=zb,
                data_dim=data_dim,
                device=device,
                gen=gen,
                diag_fraction=diag_fraction,
                adaptive_p=adaptive_p,
                time_sampler=time_sampler,
                time_mu=time_mu,
                time_sigma=time_sigma,
                corner_fraction=corner_fraction,
            )
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if max_grad_norm is not None and float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(opt_params, float(max_grad_norm))
            opt.step()
            losses.append(out["loss"].item())
            if torch.isfinite(out["loss_diag"]):
                diag_losses.append(out["loss_diag"].item())
            if torch.isfinite(out["loss_offdiag"]):
                offdiag_losses.append(out["loss_offdiag"].item())

        row: dict[str, float] = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "loss_diag_mean": float(np.mean(diag_losses)) if diag_losses else float("nan"),
            "loss_offdiag_mean": float(np.mean(offdiag_losses)) if offdiag_losses else float("nan"),
        }
        if (
            eval_every > 0
            and eval_x1 is not None
            and (epoch == 1 or epoch % eval_every == 0 or epoch == epochs)
        ):
            eval_row = eval_mean_flow_vs_teacher_ode(
                mean_flow,
                teacher,
                eval_x1,
                eval_c,
                eval_p,
                z=eval_z,
                data_dim=data_dim,
                device=device,
                eval_seed=eval_seed + epoch,
                n_eval=eval_n,
                ode_method=eval_ode_method,
                ode_step_size=eval_ode_step_size,
                mean_flow_n_steps=eval_mean_flow_n_steps,
            )
            row.update(eval_row)
        history.append(row)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            msg = f"  epoch {epoch:5d}  loss {row['loss_mean']:.6f}"
            if not np.isnan(row["loss_diag_mean"]):
                msg += f"  diag {row['loss_diag_mean']:.6f}"
            if not np.isnan(row["loss_offdiag_mean"]):
                msg += f"  offdiag {row['loss_offdiag_mean']:.6f}"
            if "mean_flow_ode_mse" in row:
                msg += f"  mf_vs_ode_mse {row['mean_flow_ode_mse']:.6f}"
            print(msg)
    return history


def mean_flow_checkpoint_config(
    mean_flow: ConditionalMeanFlow,
    teacher_cfg: dict,
    *,
    parent_checkpoint: str,
    args: object | None = None,
    extra: dict | None = None,
) -> dict:
    cfg: dict[str, Any] = {
        "model_type": "conditional_mean_flow",
        "parent_checkpoint": parent_checkpoint,
        "parent_stage": teacher_cfg.get("stage", teacher_cfg.get("phase", "pretrain")),
        "data_dim": mean_flow.data_dim,
        "n_contexts": mean_flow.n_contexts,
        "n_perturbations": mean_flow.n_perturbations,
        "cond_emb_dim": mean_flow.cond_emb_dim,
        "time_emb_dim": mean_flow.time_emb_dim,
        "hidden_dim": mean_flow.hidden_dim,
        "depth": mean_flow.depth,
        "backbone_type": mean_flow.backbone_type,
        "use_conditioning": mean_flow.use_conditioning,
        "covariate_sizes": {
            col: mean_flow.covariate_embs[col].num_embeddings
            for col in mean_flow.covariate_col_order
        },
        "encoding": teacher_cfg.get("encoding"),
        "experiment": teacher_cfg.get("experiment", teacher_cfg.get("phase", "tx")),
    }
    if args is not None:
        for key in (
            "diag_fraction",
            "adaptive_p",
            "distill_epochs",
            "train_seed",
            "lr",
            "obsm_key",
            "split_col",
        ):
            if hasattr(args, key):
                cfg[key] = getattr(args, key)
    if extra:
        cfg.update(extra)
    return cfg


def save_mean_flow_checkpoint(
    path: Path,
    mean_flow: ConditionalMeanFlow,
    teacher_cfg: dict,
    *,
    parent_checkpoint: str,
    args: object | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "state_dict": mean_flow.state_dict(),
        "config": mean_flow_checkpoint_config(
            mean_flow,
            teacher_cfg,
            parent_checkpoint=parent_checkpoint,
            args=args,
            extra=extra,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_mean_flow_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ConditionalMeanFlow, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    mean_flow = build_conditional_mean_flow_from_config(cfg)
    mean_flow.load_state_dict(payload["state_dict"])
    mean_flow.to(device)
    return mean_flow, cfg


def make_mean_flow_from_teacher(
    teacher: ConditionalFlowMatching,
    *,
    cond_emb_dim: int | None = None,
    time_emb_dim: int | None = None,
    hidden_dim: int | None = None,
    depth: int | None = None,
    copy_embeddings: bool = True,
    freeze_condition_embeddings: bool = True,
) -> ConditionalMeanFlow:
    """Build a MeanFlow student with architecture kwargs defaulting to the teacher."""
    covariate_sizes = None
    if teacher.covariate_col_order:
        covariate_sizes = {
            col: teacher.covariate_embs[col].num_embeddings for col in teacher.covariate_col_order
        }
    mean_flow = ConditionalMeanFlow(
        data_dim=teacher.data_dim,
        n_contexts=teacher.n_contexts,
        n_perturbations=teacher.n_perturbations,
        cond_emb_dim=cond_emb_dim if cond_emb_dim is not None else teacher.cond_emb_dim,
        time_emb_dim=time_emb_dim if time_emb_dim is not None else teacher.time_emb_dim,
        hidden_dim=hidden_dim if hidden_dim is not None else teacher.hidden_dim,
        depth=depth if depth is not None else teacher.depth,
        backbone_type=getattr(teacher, "backbone_type", DEFAULT_BACKBONE_TYPE),
        covariate_sizes=covariate_sizes,
        use_conditioning=teacher.use_conditioning,
    )
    if copy_embeddings and teacher.use_conditioning:
        mean_flow.init_condition_embeddings_from(teacher)
    if freeze_condition_embeddings and mean_flow.use_conditioning:
        assert mean_flow.context_emb is not None and mean_flow.pert_emb is not None
        mean_flow.context_emb.weight.requires_grad_(False)
        mean_flow.pert_emb.weight.requires_grad_(False)
        for col in mean_flow.covariate_col_order:
            mean_flow.covariate_embs[col].weight.requires_grad_(False)
    return mean_flow


def resolve_mean_flow_checkpoint(pretrain_dir: Path) -> Path:
    """Return ``mean_flow.pt`` next to a pretrain ``checkpoint.pt``."""
    path = pretrain_dir / MEAN_FLOW_CKPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No {MEAN_FLOW_CKPT_NAME} in {pretrain_dir}")
    return path
