"""Distill a conditional flow map from a frozen pretrained velocity field."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trajectory_replay.model.flow_map import (
    ConditionalFlowMap,
    build_conditional_flow_map_from_config,
)
from trajectory_replay.model.flow_matching import ConditionalFlowMatching

FLOW_MAP_CKPT_NAME = "flow_map.pt"
FLOW_MAP_METRICS_NAME = "flow_map_metrics.json"


def linear_interpolant(
    s: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    """I_s = (1-s) x0 + s x1 with s shape (B,); broadcasts over flat or spatial x."""
    s_exp = s.view(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - s_exp) * x0 + s_exp * x1


def _split_diag_offdiag(batch_size: int, diag_fraction: float) -> tuple[int, int]:
    if not (0.0 <= diag_fraction <= 1.0):
        raise ValueError(f"diag_fraction must be in [0, 1], got {diag_fraction}")
    diag_bs = int(round(batch_size * diag_fraction))
    diag_bs = min(max(diag_bs, 0), batch_size)
    offdiag_bs = batch_size - diag_bs
    return diag_bs, offdiag_bs


def _sample_time_pairs(
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample s, t with 0 <= s < t <= 1."""
    s = torch.rand(n, device=device, generator=gen)
    gap = torch.rand(n, device=device, generator=gen) * (1.0 - s)
    gap = torch.clamp(gap, min=1e-6)
    t = torch.clamp(s + gap, max=1.0)
    return s, t


def _diagonal_distill_loss(
    flow_map: ConditionalFlowMap,
    teacher: ConditionalFlowMatching,
    x0: torch.Tensor,
    x1: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Match Phi(t,t,I_t) to teacher velocity at (I_t, t)."""
    It = linear_interpolant(t, x0, x1)
    phi_tt = flow_map.calc_b(It, t, c, p, z=z)
    with torch.no_grad():
        v_ref = teacher(It, t, c, p, z=z)
    return F.mse_loss(phi_tt, v_ref)


def _lsd_distill_loss(
    flow_map: ConditionalFlowMap,
    teacher: ConditionalFlowMatching,
    x0: torch.Tensor,
    x1: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    stopgrad_state: bool = True,
) -> torch.Tensor:
    """LSD: match d/dt X(s,t,I_s) to v_ref at the transported state."""
    Is = linear_interpolant(s, x0, x1)
    X_st, dt_X = flow_map.partial_t(Is, s, t, c, p, z=z)
    if stopgrad_state:
        b_teacher = teacher(X_st.detach(), t, c, p, z=z)
    else:
        b_teacher = teacher(X_st, t, c, p, z=z)
    return F.mse_loss(dt_X, b_teacher.detach())


def _distill_batch_loss(
    flow_map: ConditionalFlowMap,
    teacher: ConditionalFlowMatching,
    x1: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    loss_type: Literal["lsd"] = "lsd",
    diag_fraction: float = 0.25,
    stopgrad_state: bool = True,
) -> dict[str, torch.Tensor]:
    x1 = x1.to(device)
    c = c.to(device=device, dtype=torch.long)
    p = p.to(device=device, dtype=torch.long)
    z_dev = None
    if z is not None:
        z_dev = {col: z[col].to(device=device, dtype=torch.long) for col in z}
    b = x1.shape[0]
    x0 = torch.randn(b, data_dim, device=device, generator=gen)

    diag_bs, offdiag_bs = _split_diag_offdiag(b, diag_fraction)
    total_loss = torch.zeros((), device=device)
    diag_loss = torch.zeros((), device=device)
    offdiag_loss = torch.zeros((), device=device)

    if diag_bs > 0:
        t_diag = torch.rand(diag_bs, device=device, generator=gen)
        z_diag = None
        if z_dev is not None:
            z_diag = {col: z_dev[col][:diag_bs] for col in z_dev}
        diag_loss = _diagonal_distill_loss(
            flow_map,
            teacher,
            x0[:diag_bs],
            x1[:diag_bs],
            c[:diag_bs],
            p[:diag_bs],
            t_diag,
            z=z_diag,
        )
        total_loss = total_loss + diag_loss * diag_bs

    if offdiag_bs > 0:
        if loss_type != "lsd":
            raise ValueError(f"Unsupported loss_type {loss_type!r}")
        s, t = _sample_time_pairs(offdiag_bs, device, gen)
        z_off = None
        if z_dev is not None:
            z_off = {col: z_dev[col][diag_bs:] for col in z_dev}
        offdiag_loss = _lsd_distill_loss(
            flow_map,
            teacher,
            x0[diag_bs:],
            x1[diag_bs:],
            c[diag_bs:],
            p[diag_bs:],
            s,
            t,
            z=z_off,
            stopgrad_state=stopgrad_state,
        )
        total_loss = total_loss + offdiag_loss * offdiag_bs

    total_loss = total_loss / max(b, 1)
    return {
        "loss": total_loss,
        "loss_diag": diag_loss,
        "loss_offdiag": offdiag_loss,
    }


@torch.no_grad()
def eval_flow_map_vs_teacher_ode(
    flow_map: ConditionalFlowMap,
    teacher: ConditionalFlowMatching,
    x1: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    eval_seed: int,
    n_eval: int = 256,
    ode_method: str = "euler",
    ode_step_size: float = 0.02,
    map_n_steps: int = 1,
) -> dict[str, float]:
    """MSE between flow-map rollout and teacher ODE endpoints on a mini-batch."""
    n = min(int(n_eval), x1.shape[0])
    if n <= 0:
        return {"flow_map_ode_mse": float("nan"), "n_eval": 0.0}

    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(eval_seed)
    idx = torch.randperm(x1.shape[0], generator=cpu_gen)[:n]
    c_sub = c[idx].to(device=device, dtype=torch.long)
    p_sub = p[idx].to(device=device, dtype=torch.long)
    z_sub = None
    if z is not None:
        z_sub = {col: z[col][idx].to(device=device, dtype=torch.long) for col in z}

    gen = torch.Generator(device=device)
    gen.manual_seed(eval_seed)
    x0 = torch.randn(n, data_dim, device=device, generator=gen)
    x_map = flow_map.sample_from_x0(
        x0,
        c_sub,
        p_sub,
        z=z_sub,
        t_end=1.0,
        n_steps=map_n_steps,
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
    mse = F.mse_loss(x_map, x_ode).item()
    return {"flow_map_ode_mse": float(mse), "n_eval": float(n)}


def train_flow_map_distill(
    flow_map: ConditionalFlowMap,
    teacher: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    loss_type: Literal["lsd"] = "lsd",
    diag_fraction: float = 0.25,
    stopgrad_state: bool = True,
    params: Iterable[torch.nn.Parameter] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    eval_x1: torch.Tensor | None = None,
    eval_c: torch.Tensor | None = None,
    eval_p: torch.Tensor | None = None,
    eval_z: dict[str, torch.Tensor] | None = None,
    eval_every: int = 0,
    eval_seed: int = 0,
    eval_n: int = 256,
    eval_ode_method: str = "euler",
    eval_ode_step_size: float = 0.02,
    eval_map_n_steps: int = 1,
) -> list[dict[str, float]]:
    """Train flow map to approximate teacher rollouts (LSD + diagonal terms)."""
    from trajectory_replay.data.tx_dataset import unpack_tx_batch

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(params if params is not None else flow_map.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)

    history: list[dict[str, float]] = []
    flow_map.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        diag_losses: list[float] = []
        offdiag_losses: list[float] = []
        for batch in loader:
            x1, cb, pb, zb = unpack_tx_batch(batch, covariate_col_order)
            out = _distill_batch_loss(
                flow_map,
                teacher,
                x1,
                cb,
                pb,
                z=zb,
                data_dim=data_dim,
                device=device,
                gen=gen,
                loss_type=loss_type,
                diag_fraction=diag_fraction,
                stopgrad_state=stopgrad_state,
            )
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].item())
            if out["loss_diag"].numel() > 0:
                diag_losses.append(out["loss_diag"].item())
            if out["loss_offdiag"].numel() > 0:
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
            and eval_c is not None
            and eval_p is not None
            and (epoch == 1 or epoch % eval_every == 0 or epoch == epochs)
        ):
            eval_row = eval_flow_map_vs_teacher_ode(
                flow_map,
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
                map_n_steps=eval_map_n_steps,
            )
            row.update(eval_row)
        history.append(row)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            msg = f"  epoch {epoch:5d}  loss {row['loss_mean']:.6f}"
            if not np.isnan(row["loss_diag_mean"]):
                msg += f"  diag {row['loss_diag_mean']:.6f}"
            if not np.isnan(row["loss_offdiag_mean"]):
                msg += f"  offdiag {row['loss_offdiag_mean']:.6f}"
            if "flow_map_ode_mse" in row:
                msg += f"  map_vs_ode_mse {row['flow_map_ode_mse']:.6f}"
            print(msg)
    return history


def flow_map_checkpoint_config(
    flow_map: ConditionalFlowMap,
    teacher_cfg: dict,
    *,
    parent_checkpoint: str,
    args: object | None = None,
    extra: dict | None = None,
) -> dict:
    cfg: dict = {
        "model_type": "conditional_flow_map",
        "parent_checkpoint": parent_checkpoint,
        "parent_stage": teacher_cfg.get("stage", "pretrain"),
        "data_dim": flow_map.data_dim,
        "n_contexts": flow_map.n_contexts,
        "n_perturbations": flow_map.n_perturbations,
        "cond_emb_dim": flow_map.cond_emb_dim,
        "time_emb_dim": flow_map.time_emb_dim,
        "hidden_dim": flow_map.hidden_dim,
        "depth": flow_map.depth,
        "backbone_type": flow_map.backbone_type,
        "phi_rescale": flow_map.phi_rescale,
        "covariate_sizes": {
            col: flow_map.covariate_embs[col].num_embeddings for col in flow_map.covariate_col_order
        },
        "encoding": teacher_cfg.get("encoding"),
        "experiment": teacher_cfg.get("experiment", "tx"),
    }
    if args is not None:
        for key in (
            "loss_type",
            "diag_fraction",
            "stopgrad_state",
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


def save_flow_map_checkpoint(
    path: Path,
    flow_map: ConditionalFlowMap,
    teacher_cfg: dict,
    *,
    parent_checkpoint: str,
    args: object | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "state_dict": flow_map.state_dict(),
        "config": flow_map_checkpoint_config(
            flow_map,
            teacher_cfg,
            parent_checkpoint=parent_checkpoint,
            args=args,
            extra=extra,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_flow_map_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ConditionalFlowMap, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    flow_map = build_conditional_flow_map_from_config(cfg)
    flow_map.load_state_dict(payload["state_dict"])
    flow_map.to(device)
    return flow_map, cfg


def make_flow_map_from_teacher(
    teacher: ConditionalFlowMatching,
    *,
    cond_emb_dim: int | None = None,
    time_emb_dim: int | None = None,
    hidden_dim: int | None = None,
    depth: int | None = None,
    phi_rescale: float = 1.0,
    copy_embeddings: bool = True,
    freeze_condition_embeddings: bool = True,
) -> ConditionalFlowMap:
    """Build a flow map with architecture kwargs defaulting to the teacher."""
    covariate_sizes = None
    if teacher.covariate_col_order:
        covariate_sizes = {
            col: teacher.covariate_embs[col].num_embeddings for col in teacher.covariate_col_order
        }
    flow_map = ConditionalFlowMap(
        data_dim=teacher.data_dim,
        n_contexts=teacher.n_contexts,
        n_perturbations=teacher.n_perturbations,
        cond_emb_dim=cond_emb_dim if cond_emb_dim is not None else teacher.cond_emb_dim,
        time_emb_dim=time_emb_dim if time_emb_dim is not None else teacher.time_emb_dim,
        hidden_dim=hidden_dim if hidden_dim is not None else teacher.hidden_dim,
        depth=depth if depth is not None else teacher.depth,
        phi_rescale=phi_rescale,
        covariate_sizes=covariate_sizes,
    )
    if copy_embeddings:
        flow_map.init_condition_embeddings_from(teacher)
    if freeze_condition_embeddings:
        flow_map.context_emb.weight.requires_grad_(False)
        flow_map.pert_emb.weight.requires_grad_(False)
        for col in flow_map.covariate_col_order:
            flow_map.covariate_embs[col].weight.requires_grad_(False)
    return flow_map


def resolve_flow_map_checkpoint(pretrain_dir: Path) -> Path:
    """Return ``flow_map.pt`` next to a pretrain ``checkpoint.pt``."""
    path = pretrain_dir / FLOW_MAP_CKPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No {FLOW_MAP_CKPT_NAME} in {pretrain_dir}")
    return path
