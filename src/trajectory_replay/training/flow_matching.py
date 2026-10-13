"""Shared flow-matching training loops and ODE stratum evaluation utilities.

Used by ``experiments/sim/train_flow_matching.py``, ``experiments/sim/train_flow_matching_data_to_data.py``,
and ``experiments/sim/train_perturbation_forgetting.py``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trajectory_replay.eval.metrics import ode_stratum_metrics
from trajectory_replay.data.tx_dataset import unpack_tx_batch
from trajectory_replay.data.tx_strata import (
    control_indices_by_prefix,
    control_pool_key_for_stratum,
    iter_observed_strata,
    stratum_mask,
    stratum_row_to_dict,
)
from trajectory_replay.holdout.data import (
    OBSERVED_STRATUM_SAMPLINGS,
    SupportStratumDistribution,
    sample_support_cells,
    sample_support_conditions,
    stratum_seed,
)
from trajectory_replay.model.flow_map import ConditionalFlowMap
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.model.mean_flow import ConditionalMeanFlow
from trajectory_replay.training.gft_schedule import gft_beta_at_step
from trajectory_replay.training.traj_kappa import TrajKappaName, resolve_traj_kappa

# Distilled multi-jump samplers used as generative-replay backends (same sample API).
DistilledReplayModel = ConditionalFlowMap | ConditionalMeanFlow


def _resolve_distilled_replay(
    flow_map: ConditionalFlowMap | None,
    flow_map_n_steps: int,
    mean_flow: ConditionalMeanFlow | None,
    mean_flow_n_steps: int,
) -> tuple[DistilledReplayModel | None, int]:
    """Return at most one distilled rollout model and its jump count."""
    if flow_map is not None and mean_flow is not None:
        raise ValueError("pass at most one of flow_map and mean_flow")
    if mean_flow is not None:
        if mean_flow_n_steps < 1:
            raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
        return mean_flow, mean_flow_n_steps
    if flow_map is not None:
        if flow_map_n_steps < 1:
            raise ValueError(f"flow_map_n_steps must be >= 1, got {flow_map_n_steps}")
        return flow_map, flow_map_n_steps
    return None, 1


def control_indices_by_context(
    c: torch.Tensor,
    p: torch.Tensor,
    control_pert: int,
) -> dict[int, torch.Tensor]:
    """Map each context index to row indices of control (p == control_pert) cells."""
    c = c.long()
    p = p.long()
    pools: dict[int, list[int]] = {}
    for i in range(c.shape[0]):
        if int(p[i].item()) != control_pert:
            continue
        ci = int(c[i].item())
        pools.setdefault(ci, []).append(i)
    out: dict[int, torch.Tensor] = {}
    for ci, idxs in pools.items():
        if not idxs:
            raise ValueError(f"No control cells for context c={ci} (control_pert={control_pert})")
        out[ci] = torch.tensor(idxs, dtype=torch.long)
    return out


def subsample_control_pools(
    pools: dict[int, torch.Tensor],
    *,
    frac: float,
    seed: int,
    max_per_context: int | None = None,
    same_count_across_contexts: bool = True,
) -> tuple[dict[int, torch.Tensor], dict[str, float | int]]:
    """Subsample control row indices per context.

    When ``same_count_across_contexts`` is True (default), every context keeps the
    same number of control cells: ``max(1, round(frac * min_pool_size))`` (capped by
    ``max_per_context``). Sampling is deterministic per context (``seed + ci``).
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if not pools:
        raise ValueError("control pools are empty")

    sizes = {ci: int(idxs.numel()) for ci, idxs in pools.items()}
    min_size = min(sizes.values())
    if same_count_across_contexts:
        n_keep = max(1, int(round(frac * min_size)))
        if max_per_context is not None:
            n_keep = min(n_keep, max_per_context)
        n_keep = min(n_keep, min_size)
    else:
        n_keep = None

    out: dict[int, torch.Tensor] = {}
    kept_k = 0
    for ci, idxs in sorted(pools.items()):
        idx_list = idxs.cpu().tolist()
        if n_keep is not None:
            k = n_keep
        else:
            k = max(1, int(round(frac * len(idx_list))))
            if max_per_context is not None:
                k = min(k, max_per_context)
        k = min(k, len(idx_list))
        kept_k = k
        sub_rng = np.random.default_rng(seed + int(ci))
        chosen = sub_rng.choice(len(idx_list), size=k, replace=False)
        out[ci] = torch.tensor([idx_list[int(i)] for i in chosen], dtype=torch.long)

    meta: dict[str, float | int] = {
        "control_subsample_frac": float(frac),
        "control_subsample_seed": int(seed),
        "n_control_keep_per_context": int(kept_k),
        "n_control_full_min": int(min_size),
        "n_control_full_max": int(max(sizes.values())),
        "same_count_across_contexts": int(same_count_across_contexts),
    }
    return out, meta


def filter_dataset_control_subsample(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    control_pert: int,
    *,
    frac: float,
    seed: int,
    max_per_context: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Drop control cells not in the subsampled pool; keep all non-control rows.

    Same per-context retention rule as ``subsample_control_pools``. Perturbation
    cells are untouched, so both noise-to-data and data-to-data train on the
    reduced dataset.
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if frac >= 1.0:
        return x, c, p, {"control_subsampled": 0, "control_subsample_frac": 1.0}

    pools = control_indices_by_context(c, p, control_pert)
    sub_pools, meta = subsample_control_pools(
        pools,
        frac=frac,
        seed=seed,
        max_per_context=max_per_context,
        same_count_across_contexts=True,
    )
    is_ctrl = p.long() == int(control_pert)
    keep = ~is_ctrl
    for idxs in sub_pools.values():
        keep[idxs] = True

    n_before = int(x.shape[0])
    n_ctrl_before = int(is_ctrl.sum().item())
    n_ctrl_after = int((keep & is_ctrl).sum().item())
    meta["control_subsampled"] = 1
    meta["n_rows_before"] = n_before
    meta["n_rows_after"] = int(keep.sum().item())
    meta["n_control_before"] = n_ctrl_before
    meta["n_control_after"] = n_ctrl_after
    meta["n_control_dropped"] = n_ctrl_before - n_ctrl_after
    return x[keep], c[keep], p[keep], meta


def iter_observed_strata_pairs(
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    skip_contexts: frozenset[int] | None = None,
    skip_perturbations: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Unique (context, perturbation) pairs present in eval tensors, after filters."""
    if z is not None and covariate_col_order:
        return [(key[0], key[1]) for key in iter_observed_strata(
            c,
            p,
            z=z,
            covariate_col_order=covariate_col_order,
            skip_contexts=skip_contexts,
            skip_perturbations=skip_perturbations,
            skip_combos=skip_combos,
        )]
    c_np = c.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    uniq = np.unique(np.stack([c_np, p_np], axis=1), axis=0)
    pairs: list[tuple[int, int]] = []
    for row in uniq:
        ci, pj = int(row[0]), int(row[1])
        if skip_combos is not None and (ci, pj) in skip_combos:
            continue
        if skip_contexts is not None and ci in skip_contexts:
            continue
        if skip_perturbations is not None and pj in skip_perturbations:
            continue
        pairs.append((ci, pj))
    return pairs


def _z_batch_from_key(
    key: tuple[int, ...],
    n: int,
    device: torch.device,
    covariate_col_order: tuple[str, ...],
) -> dict[str, torch.Tensor] | None:
    if not covariate_col_order:
        return None
    return {
        col: torch.full((n,), key[2 + i], dtype=torch.long, device=device)
        for i, col in enumerate(covariate_col_order)
    }


def eval_ode_strata(
    model: nn.Module,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    encoding: object | None = None,
    n_contexts: int,
    n_perturbations: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    skip_perturbations: frozenset[int] | None = None,
    skip_contexts: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    model.eval()
    gen = torch.Generator(device=device)
    per_combo: list[dict[str, float | int]] = []
    mses: list[float] = []
    strata = iter_observed_strata(
        c,
        p,
        z=z,
        covariate_col_order=covariate_col_order,
        skip_contexts=skip_contexts,
        skip_perturbations=skip_perturbations,
        skip_combos=skip_combos,
    )
    for key in strata:
        ci, pj = key[0], key[1]
        m = stratum_mask(c, p, key, z=z, covariate_col_order=covariate_col_order)
        x1 = x[m].to(device)
        n = int(x1.shape[0])
        if n == 0:
            continue
        gen.manual_seed(stratum_seed(eval_seed, *key))
        cb = torch.full((n,), ci, dtype=torch.long, device=device)
        pb = torch.full((n,), pj, dtype=torch.long, device=device)
        zb = _z_batch_from_key(key, n, device, covariate_col_order)
        with torch.no_grad():
            if getattr(model, "use_conditioning", True):
                xh = model.sample(
                    cb,
                    pb,
                    zb,
                    method=ode_method,
                    rtol=ode_rtol,
                    atol=ode_atol,
                    options=ode_options,
                    generator=gen,
                )
            else:
                xh = model.sample(
                    batch_size=n,
                    method=ode_method,
                    rtol=ode_rtol,
                    atol=ode_atol,
                    options=ode_options,
                    generator=gen,
                )
        mse = F.mse_loss(xh, x1).item()
        metrics = ode_stratum_metrics(
            _state_as_matrix(x1).cpu().numpy(),
            _state_as_matrix(xh).cpu().numpy(),
            paired_mse=mse,
        )
        row: dict[str, float | int] = {"c": ci, "p": pj, "n": n, **metrics}
        if encoding is not None and covariate_col_order:
            for k, v in stratum_row_to_dict(
                key, encoding, covariate_col_order=covariate_col_order
            ).items():
                if k not in row:
                    row[k] = v  # type: ignore[assignment]
        per_combo.append(row)
        mses.append(mse)
    mean_mse = float(np.mean(mses)) if mses else float("nan")
    return per_combo, mean_mse


def eval_ode_strata_from_control(
    model: ConditionalFlowMatching,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    control_pert: int,
    n_contexts: int,
    n_perturbations: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    skip_perturbations: frozenset[int] | None = None,
    skip_contexts: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    """ODE from control x_0 (same c) to each (c, p); MSE vs true target stratum."""
    model.eval()
    control_pools = control_indices_by_context(c, p, control_pert)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(eval_seed)

    per_combo: list[dict[str, float | int]] = []
    mses: list[float] = []
    for ci, pj in iter_observed_strata_pairs(
        c,
        p,
        skip_contexts=skip_contexts,
        skip_perturbations=skip_perturbations,
        skip_combos=skip_combos,
    ):
        if ci not in control_pools:
            continue
        ctrl_idx = control_pools[ci]
        m = (c == ci) & (p == pj)
        x1 = x[m].to(device)
        n = int(x1.shape[0])
        if n == 0:
            continue
        pick = torch.randint(len(ctrl_idx), (n,), generator=cpu_gen)
        x0 = x[ctrl_idx[pick]].to(device)
        cb = torch.full((n,), ci, dtype=torch.long, device=device)
        pb = torch.full((n,), pj, dtype=torch.long, device=device)
        with torch.no_grad():
            xh = model.sample_from_x0(
                x0,
                cb,
                pb,
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
            )
        mse = F.mse_loss(xh, x1).item()
        metrics = ode_stratum_metrics(
            _state_as_matrix(x1).cpu().numpy(),
            _state_as_matrix(xh).cpu().numpy(),
            paired_mse=mse,
        )
        per_combo.append({"c": ci, "p": pj, "n": n, **metrics})
        mses.append(mse)
    mean_mse = float(np.mean(mses)) if mses else float("nan")
    return per_combo, mean_mse


@torch.no_grad()
def sample_combo_from_control(
    model: ConditionalFlowMatching,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    context: int,
    perturbation: int,
    control_pert: int,
    n: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    control_pools: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Integrate from ``n`` resampled control cells (same context) to target ``(context, perturbation)``."""
    if control_pools is None:
        control_pools = control_indices_by_context(c, p, control_pert)
    if context not in control_pools:
        raise ValueError(
            f"No control cells for context c={context} (control_pert={control_pert})"
        )
    ctrl_idx = control_pools[context]
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(eval_seed)
    pick = torch.randint(len(ctrl_idx), (n,), generator=cpu_gen)
    x0 = x[ctrl_idx[pick]].to(device)
    cb = torch.full((n,), context, dtype=torch.long, device=device)
    pb = torch.full((n,), perturbation, dtype=torch.long, device=device)
    return model.sample_from_x0(
        x0,
        cb,
        pb,
        method=ode_method,
        rtol=ode_rtol,
        atol=ode_atol,
        options=ode_options,
    )


@torch.no_grad()
def sample_combo_from_noise(
    model: ConditionalFlowMatching,
    *,
    context: int,
    perturbation: int,
    n: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Integrate from Gaussian noise to target ``(context, perturbation)``."""
    gen = torch.Generator(device=device)
    gen.manual_seed(eval_seed)
    cb = torch.full((n,), context, dtype=torch.long, device=device)
    pb = torch.full((n,), perturbation, dtype=torch.long, device=device)
    return model.sample(
        cb,
        pb,
        noise_scale=noise_scale,
        method=ode_method,
        rtol=ode_rtol,
        atol=ode_atol,
        options=ode_options,
        generator=gen,
    )


def eval_ode_all_strata(
    model: nn.Module,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    n_contexts: int,
    n_perturbations: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> tuple[list[dict[str, float | int]], float]:
    """One ODE solve per (c,p): same n as true samples, MSE(gen, true) pointwise."""
    return eval_ode_strata(
        model,
        x,
        c,
        p,
        n_contexts=n_contexts,
        n_perturbations=n_perturbations,
        device=device,
        eval_seed=eval_seed,
        ode_method=ode_method,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        ode_options=ode_options,
    )


def make_frozen_model_copy(model: ConditionalFlowMatching) -> ConditionalFlowMatching:
    """Deep-copy ``model`` with all parameters frozen (for GFT reference field v_theta_0)."""
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def resolve_two_stage_1_epochs(
    finetune_epochs: int,
    *,
    two_stage_1_epochs: int = 0,
    two_stage_1_epoch_fraction: float | None = None,
) -> int | None:
    """Return stage-1 epoch count, or ``None`` if two-stage finetuning is disabled.

    Stage 1 length can be set explicitly with ``two_stage_1_epochs`` or as a fraction of
    ``finetune_epochs`` via ``two_stage_1_epoch_fraction`` (mutually exclusive).
    """
    if two_stage_1_epochs > 0 and two_stage_1_epoch_fraction is not None:
        raise ValueError(
            "Specify only one of --two-stage-1-epochs or --two-stage-1-epoch-fraction"
        )
    if two_stage_1_epochs > 0:
        n = two_stage_1_epochs
    elif two_stage_1_epoch_fraction is not None:
        if not (0.0 < two_stage_1_epoch_fraction < 1.0):
            raise ValueError(
                "--two-stage-1-epoch-fraction must be in (0, 1), "
                f"got {two_stage_1_epoch_fraction}"
            )
        n = max(1, int(finetune_epochs * two_stage_1_epoch_fraction))
    else:
        return None
    if finetune_epochs < 2:
        raise ValueError(
            f"two-stage finetuning requires --finetune-epochs >= 2, got {finetune_epochs}"
        )
    if n >= finetune_epochs:
        raise ValueError(
            f"two-stage stage-1 length {n} must be < --finetune-epochs ({finetune_epochs})"
        )
    return n


def merge_training_histories(
    *parts: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Concatenate per-phase training histories (epochs must already be global)."""
    merged: list[dict[str, float]] = []
    for part in parts:
        merged.extend(part)
    return merged


def _should_log_training_epoch(local_epoch: int, phase_epochs: int) -> bool:
    return (
        local_epoch == 1
        or local_epoch % max(1, phase_epochs // 10) == 0
        or local_epoch == phase_epochs
    )


def _broadcast_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand ``t`` (B,) for elementwise ops with ``x`` of rank >= 2."""
    return t.view(-1, *([1] * (x.dim() - 1)))


def _state_as_matrix(x: torch.Tensor) -> torch.Tensor:
    """Flatten spatial batches to ``(B, D)`` for distributional metrics only."""
    if x.dim() == 2:
        return x
    return x.reshape(x.shape[0], -1)


def _sample_flow_noise(
    x1: torch.Tensor,
    *,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
) -> torch.Tensor:
    """Gaussian x_0 with the same layout as ``x1`` (flat vector or image batch)."""
    if x1.dim() == 2:
        return torch.randn(x1.shape[0], data_dim, device=device, generator=gen, dtype=x1.dtype)
    return torch.randn_like(x1)


def _apply_finetune_grad_masks(
    model: ConditionalFlowMatching,
    *,
    finetune_pert_index: int | None,
    finetune_context_index: int | None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
) -> None:
    if pert_emb_train_rows is not None:
        ConditionalFlowMatching.mask_embedding_rows_grad(
            model.pert_emb.weight, pert_emb_train_rows
        )
    elif finetune_pert_index is not None:
        ConditionalFlowMatching.mask_embedding_row_grad(
            model.pert_emb.weight, finetune_pert_index
        )
    if context_emb_train_rows is not None:
        ConditionalFlowMatching.mask_embedding_rows_grad(
            model.context_emb.weight, context_emb_train_rows
        )
    elif finetune_context_index is not None:
        ConditionalFlowMatching.mask_embedding_row_grad(
            model.context_emb.weight, finetune_context_index
        )


def _flow_matching_loss(
    model: ConditionalFlowMatching,
    x1: torch.Tensor,
    cb: torch.Tensor,
    pb: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    split_at: int | None = None,
    t_scalar: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, float, float]:
    x1 = x1.to(device)
    cb, pb = cb.to(device), pb.to(device)
    b = x1.shape[0]
    x0 = _sample_flow_noise(x1, data_dim=data_dim, device=device, gen=gen)
    if t_scalar is None:
        t = torch.rand(b, device=device, generator=gen)
    else:
        t = torch.full((b,), float(t_scalar), device=device, dtype=x1.dtype)
    t_exp = _broadcast_time(t, x1)
    xt = (1.0 - t_exp) * x0 + t_exp * x1
    v_star = x1 - x0
    v_pred = model(xt, t, cb, pb, z=z)
    if split_at is None:
        return F.mse_loss(v_pred, v_star)
    if split_at <= 0 or split_at >= b:
        raise ValueError(f"split_at must be in (0, {b}), got {split_at}")
    loss = F.mse_loss(v_pred, v_star)
    loss_ft = float(F.mse_loss(v_pred[:split_at], v_star[:split_at]).item())
    loss_ref = float(F.mse_loss(v_pred[split_at:], v_star[split_at:]).item())
    return loss, loss_ft, loss_ref


def _flow_matching_replay_step(
    model: ConditionalFlowMatching,
    opt: torch.optim.Optimizer,
    x1: torch.Tensor,
    cb: torch.Tensor,
    pb: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    split_at: int,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """One FM step on a mixed finetune+replay batch; split losses from a single forward."""
    loss_out = _flow_matching_loss(
        model,
        x1,
        cb,
        pb,
        z=z,
        data_dim=data_dim,
        device=device,
        gen=gen,
        split_at=split_at,
    )
    if not isinstance(loss_out, tuple):
        raise RuntimeError("split_at replay step expected tuple loss output")
    loss, loss_ft, loss_ref = loss_out
    _finetune_optimizer_step(
        model,
        opt,
        loss,
        finetune_pert_index=finetune_pert_index,
        finetune_context_index=finetune_context_index,
        context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
        max_grad_norm=max_grad_norm,
    )
    return {"loss": loss.item(), "loss_ft": loss_ft, "loss_ref": loss_ref}


def _finetune_optimizer_step(
    model: ConditionalFlowMatching,
    opt: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    max_grad_norm: float | None = None,
) -> bool:
    """Backward + optional grad clip + Adam step.

    Returns ``False`` (and skips the update) when ``loss`` is non-finite, so a single
    divergent replay/ODE draw cannot permanently NaN the weights.

    Grad clipping is off by default: clipping renormalizes step size and can confound
    λ-mixing comparisons. Pass ``max_grad_norm`` (e.g. 1.0) only as an explicit safety net.
    """
    if not torch.isfinite(loss).all():
        opt.zero_grad(set_to_none=True)
        return False
    opt.zero_grad(set_to_none=True)
    loss.backward()
    _apply_finetune_grad_masks(
        model,
        finetune_pert_index=finetune_pert_index,
        finetune_context_index=finetune_context_index,
        context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
    )
    if max_grad_norm is not None and max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
    opt.step()
    return True


def _flow_matching_step(
    model: ConditionalFlowMatching,
    opt: torch.optim.Optimizer,
    x1: torch.Tensor,
    cb: torch.Tensor,
    pb: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    max_grad_norm: float | None = None,
) -> float:
    loss = _flow_matching_loss(
        model, x1, cb, pb, z=z, data_dim=data_dim, device=device, gen=gen
    )
    _finetune_optimizer_step(
        model,
        opt,
        loss,
        finetune_pert_index=finetune_pert_index,
        finetune_context_index=finetune_context_index,
        context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
        max_grad_norm=max_grad_norm,
    )
    return loss.item()


def _ref_field_replay_loss(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    c_ref: torch.Tensor,
    p_ref: torch.Tensor,
    *,
    z_ref: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    replay_ode_method: str,
    replay_ode_rtol: float,
    replay_ode_atol: float,
    replay_ode_options: dict[str, float] | None,
    flow_map: ConditionalFlowMap | None = None,
    flow_map_n_steps: int = 1,
    mean_flow: ConditionalMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
    t_scalar: float | None = None,
) -> torch.Tensor:
    """Unweighted field MSE on ref rollout states (κ applied by the caller)."""
    n_ref = c_ref.shape[0]
    dtype = next(model_ref.parameters()).dtype
    x0_ref = model_ref.sample_noise(
        n_ref, device=device, dtype=dtype, generator=gen
    )
    # Sample t in (eps, 1] so ODE t_end matches the time label passed to v_θ.
    eps = 1e-4
    if t_scalar is None:
        t_scalar = float(
            (eps + (1.0 - eps) * torch.rand(1, device=device, generator=gen)).item()
        )
    else:
        t_scalar = float(t_scalar)
    t_ref = torch.full((n_ref,), t_scalar, device=device, dtype=dtype)
    distilled, distilled_n_steps = _resolve_distilled_replay(
        flow_map, flow_map_n_steps, mean_flow, mean_flow_n_steps
    )
    if distilled is not None:
        x_t_ref = distilled.sample_from_x0(
            x0_ref,
            c_ref,
            p_ref,
            z=z_ref,
            t_end=t_scalar,
            n_steps=distilled_n_steps,
        )
    else:
        x_t_ref = model_ref.sample_from_x0(
            x0_ref,
            c_ref,
            p_ref,
            z=z_ref,
            t_end=t_scalar,
            method=replay_ode_method,
            rtol=replay_ode_rtol,
            atol=replay_ode_atol,
            options=replay_ode_options,
        )
    v_pred = model(x_t_ref, t_ref, c_ref, p_ref, z=z_ref)
    with torch.no_grad():
        v_ref = model_ref(x_t_ref, t_ref, c_ref, p_ref, z=z_ref)
    return F.mse_loss(v_pred, v_ref)


def _ref_field_replay_loss_per_example_euler_grid(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    c_ref: torch.Tensor,
    p_ref: torch.Tensor,
    *,
    z_ref: dict[str, torch.Tensor] | None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    step_size: float,
) -> torch.Tensor:
    """Field MSE at continuous per-example times interpolated on an Euler grid."""
    if not (0.0 < step_size <= 1.0):
        raise ValueError(f"Euler step_size must be in (0, 1], got {step_size}")
    n_steps_float = 1.0 / step_size
    n_steps = int(round(n_steps_float))
    if not math.isclose(n_steps_float, n_steps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "per-example Euler-grid replay requires step_size to divide 1 exactly; "
            f"got {step_size}"
        )
    n_ref = c_ref.shape[0]
    dtype = next(model_ref.parameters()).dtype
    t_ref = torch.rand(n_ref, device=device, dtype=dtype, generator=gen)
    lower_steps = torch.clamp(
        torch.floor(t_ref / step_size).to(dtype=torch.long),
        max=n_steps - 1,
    )
    alpha = (t_ref - lower_steps.to(dtype=dtype) * step_size) / step_size
    x = model_ref.sample_noise(
        n_ref, device=device, dtype=dtype, generator=gen
    )
    x_selected = torch.empty_like(x)
    with torch.no_grad():
        for step in range(n_steps):
            t_now = torch.full(
                (n_ref,), step * step_size, device=device, dtype=dtype
            )
            x_next = x + step_size * model_ref(x, t_now, c_ref, p_ref, z=z_ref)
            take = lower_steps == step
            if torch.any(take):
                alpha_expanded = _broadcast_time(alpha[take], x[take])
                x_selected[take] = (
                    x[take] + alpha_expanded * (x_next[take] - x[take])
                )
            x = x_next
    v_pred = model(x_selected, t_ref, c_ref, p_ref, z=z_ref)
    with torch.no_grad():
        v_ref = model_ref(x_selected, t_ref, c_ref, p_ref, z=z_ref)
    return F.mse_loss(v_pred, v_ref)


def _z_dict_to_device(
    z: dict[str, torch.Tensor] | None,
    cols: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if not cols:
        return None
    if z is None:
        raise ValueError("covariate batch required when model has covariates")
    return {col: z[col].to(device=device, dtype=torch.long) for col in cols}


def _concat_z_dicts(
    z_a: dict[str, torch.Tensor] | None,
    z_b: dict[str, torch.Tensor] | None,
    cols: tuple[str, ...],
) -> dict[str, torch.Tensor] | None:
    if not cols:
        return None
    if z_a is None or z_b is None:
        raise ValueError("covariate batches required when model has covariates")
    device = z_b[cols[0]].device
    return {
        col: torch.cat(
            [z_a[col].to(device=device, dtype=torch.long), z_b[col]],
            dim=0,
        )
        for col in cols
    }


def _subset_z_dict(
    z: dict[str, torch.Tensor] | None,
    idx: torch.Tensor,
    cols: tuple[str, ...],
) -> dict[str, torch.Tensor] | None:
    if not cols:
        return None
    if z is None:
        raise ValueError("covariate batch required when model has covariates")
    return {col: z[col][idx] for col in cols}


def _sample_ref_replay_endpoints(
    model_ref: ConditionalFlowMatching,
    c_ref: torch.Tensor,
    p_ref: torch.Tensor,
    *,
    z_ref: dict[str, torch.Tensor] | None = None,
    device: torch.device,
    gen: torch.Generator,
    replay_ode_method: str,
    replay_ode_rtol: float,
    replay_ode_atol: float,
    replay_ode_options: dict[str, float] | None,
    flow_map: ConditionalFlowMap | None = None,
    flow_map_n_steps: int = 1,
    mean_flow: ConditionalMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
) -> torch.Tensor:
    distilled, distilled_n_steps = _resolve_distilled_replay(
        flow_map, flow_map_n_steps, mean_flow, mean_flow_n_steps
    )
    if distilled is not None:
        return distilled.sample(
            c_ref,
            p_ref,
            z=z_ref,
            t_end=1.0,
            n_steps=distilled_n_steps,
            generator=gen,
        )
    return model_ref.sample(
        c_ref,
        p_ref,
        z_ref,
        method=replay_ode_method,
        rtol=replay_ode_rtol,
        atol=replay_ode_atol,
        options=replay_ode_options,
        generator=gen,
    )


def _generative_replay_field_step(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    opt: torch.optim.Optimizer,
    x1_ft: torch.Tensor,
    cb_ft: torch.Tensor,
    pb_ft: torch.Tensor,
    c_ref: torch.Tensor,
    p_ref: torch.Tensor,
    *,
    lambda_mix: float,
    z_ft: dict[str, torch.Tensor] | None = None,
    z_ref: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    replay_ode_method: str,
    replay_ode_rtol: float,
    replay_ode_atol: float,
    replay_ode_options: dict[str, float] | None,
    flow_map: ConditionalFlowMap | None = None,
    flow_map_n_steps: int = 1,
    mean_flow: ConditionalMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
    traj_kappa: TrajKappaName | str = "none",
    trajectory_replay_loss_multiplier: float = 1.0,
    trajectory_time_sampling: Literal["batch", "per_example_euler_grid"] = "batch",
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """One trajectory-replay step with optional Thm. 2.4 κ(t) weighting.

    Samples a shared ``t`` and forms
    ``κ(t) · (λ L_traj(t) + (1-λ) L_CFM-new(t))``.
    Default ``traj_kappa='none'`` (κ≡1).
    """
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if trajectory_replay_loss_multiplier <= 0:
        raise ValueError(
            "trajectory replay loss multiplier must be > 0, "
            f"got {trajectory_replay_loss_multiplier}"
        )
    if trajectory_time_sampling == "batch":
        eps = 1e-4
        t_scalar = float(
            (eps + (1.0 - eps) * torch.rand(1, device=device, generator=gen)).item()
        )
        loss_ft = _flow_matching_loss(
            model, x1_ft, cb_ft, pb_ft, z=z_ft, data_dim=data_dim,
            device=device, gen=gen, t_scalar=t_scalar,
        )
        loss_ref = _ref_field_replay_loss(
            model, model_ref, c_ref, p_ref, z_ref=z_ref, data_dim=data_dim,
            device=device, gen=gen, replay_ode_method=replay_ode_method,
            replay_ode_rtol=replay_ode_rtol, replay_ode_atol=replay_ode_atol,
            replay_ode_options=replay_ode_options, flow_map=flow_map,
            flow_map_n_steps=flow_map_n_steps, mean_flow=mean_flow,
            mean_flow_n_steps=mean_flow_n_steps, t_scalar=t_scalar,
        )
        kappa = resolve_traj_kappa(
            traj_kappa, t_scalar, dtype=loss_ref.dtype, device=loss_ref.device
        )
    elif trajectory_time_sampling == "per_example_euler_grid":
        if replay_ode_method != "euler" or replay_ode_options is None:
            raise ValueError(
                "per_example_euler_grid requires Euler replay with a fixed step size"
            )
        if str(traj_kappa) != "none":
            raise ValueError("per_example_euler_grid currently requires traj_kappa='none'")
        loss_ft = _flow_matching_loss(
            model, x1_ft, cb_ft, pb_ft, z=z_ft, data_dim=data_dim,
            device=device, gen=gen,
        )
        loss_ref = _ref_field_replay_loss_per_example_euler_grid(
            model, model_ref, c_ref, p_ref, z_ref=z_ref, data_dim=data_dim,
            device=device, gen=gen, step_size=float(replay_ode_options["step_size"]),
        )
        kappa = torch.ones((), dtype=loss_ref.dtype, device=loss_ref.device)
    else:
        raise ValueError(
            f"Unknown trajectory_time_sampling {trajectory_time_sampling!r}"
        )
    loss = kappa * (
        lambda_mix * trajectory_replay_loss_multiplier * loss_ref
        + (1.0 - lambda_mix) * loss_ft
    )
    _finetune_optimizer_step(
        model,
        opt,
        loss,
        finetune_pert_index=finetune_pert_index,
        finetune_context_index=finetune_context_index,
        context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
        max_grad_norm=max_grad_norm,
    )
    return {
        "loss": loss.item(),
        "loss_ft": loss_ft.item(),
        "loss_ref": loss_ref.item(),
        "lambda_mix": float(lambda_mix),
        "trajectory_replay_loss_multiplier": float(
            trajectory_replay_loss_multiplier
        ),
        "traj_kappa_t": float(kappa.item()),
    }


def _gft_flow_matching_step(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    opt: torch.optim.Optimizer,
    x1: torch.Tensor,
    cb: torch.Tensor,
    pb: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    data_dim: int,
    device: torch.device,
    gen: torch.Generator,
    beta: float,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    x1 = x1.to(device)
    cb, pb = cb.to(device), pb.to(device)
    b = x1.shape[0]
    x0 = _sample_flow_noise(x1, data_dim=data_dim, device=device, gen=gen)
    t = torch.rand(b, device=device, generator=gen)
    t_exp = _broadcast_time(t, x1)
    xt = (1.0 - t_exp) * x0 + t_exp * x1
    v_star = x1 - x0
    v_pred = model(xt, t, cb, pb, z=z)
    with torch.no_grad():
        v_ref = model_ref(xt, t, cb, pb, z=z)
    loss_fm = F.mse_loss(v_pred, v_star)
    loss_reg = F.mse_loss(v_pred, v_ref)
    loss = loss_fm + beta * loss_reg
    if not torch.isfinite(loss).all():
        opt.zero_grad(set_to_none=True)
        return {
            "loss": float("nan"),
            "loss_fm": float("nan"),
            "loss_reg": float("nan"),
            "beta": float(beta),
        }
    opt.zero_grad(set_to_none=True)
    loss.backward()
    _apply_finetune_grad_masks(
        model,
        finetune_pert_index=finetune_pert_index,
        finetune_context_index=finetune_context_index,
        context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
    )
    opt.step()
    return {
        "loss": loss.item(),
        "loss_fm": loss_fm.item(),
        "loss_reg": loss_reg.item(),
        "beta": float(beta),
    }


def _data_to_data_flow_matching_step(
    model: ConditionalFlowMatching,
    opt: torch.optim.Optimizer,
    x0: torch.Tensor,
    x1: torch.Tensor,
    cb: torch.Tensor,
    pb: torch.Tensor,
    *,
    device: torch.device,
    gen: torch.Generator,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
) -> float:
    x0 = x0.to(device)
    x1 = x1.to(device)
    cb, pb = cb.to(device), pb.to(device)
    b = x1.shape[0]
    t = torch.rand(b, device=device, generator=gen)
    t_exp = _broadcast_time(t, x1)
    xt = (1.0 - t_exp) * x0 + t_exp * x1
    v_star = x1 - x0
    v_pred = model(xt, t, cb, pb)
    loss = F.mse_loss(v_pred, v_star)
    if not torch.isfinite(loss).all():
        opt.zero_grad(set_to_none=True)
        return float("nan")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    _apply_finetune_grad_masks(
        model,
        finetune_pert_index=finetune_pert_index,
        finetune_context_index=finetune_context_index,
    )
    opt.step()
    return loss.item()


def train_data_to_data_flow_matching(
    model: ConditionalFlowMatching,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    control_pert: int,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    train_seed: int,
    params: Iterable[torch.nn.Parameter] | None = None,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    exclude_identity_targets: bool = False,
    control_pools: dict[int, torch.Tensor] | None = None,
) -> list[dict[str, float]]:
    """Train with x_0 from control (p=control_pert, matched c) and x_1 from target cells.

    If ``exclude_identity_targets``, only cells with ``p != control_pert`` are used as x_1
    (no control→control / identity paths). Control cells remain available as x_0 sources.

    Pass ``control_pools`` to restrict which control rows are used as x_0 (e.g. subsampled).
    """
    if len(x) == 0:
        raise ValueError("training data is empty")
    if control_pools is None:
        control_pools = control_indices_by_context(c, p, control_pert)

    if exclude_identity_targets:
        target_idx = torch.nonzero(p != control_pert, as_tuple=False).squeeze(-1)
        if target_idx.numel() == 0:
            raise ValueError(
                f"No non-control target cells (control_pert={control_pert}) in training set"
            )
    else:
        target_idx = torch.arange(len(x), dtype=torch.long)

    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)

    n = int(target_idx.numel())
    drop_last = n >= batch_size
    steps_per_epoch = n // batch_size if drop_last else max(1, (n + batch_size - 1) // batch_size)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=cpu_gen)
        losses: list[float] = []
        for step in range(steps_per_epoch):
            start = step * batch_size
            idx = target_idx[perm[start : start + batch_size]]
            if idx.shape[0] < batch_size and drop_last:
                continue
            x1 = x[idx]
            cb = c[idx]
            pb = p[idx]
            x0 = torch.empty_like(x1)
            for ci_t in torch.unique(cb):
                ci = int(ci_t.item())
                mask = cb == ci
                pool = control_pools[ci]
                n_mask = int(mask.sum().item())
                picks = torch.randint(len(pool), (n_mask,), generator=cpu_gen)
                x0[mask] = x[pool[picks]]
            losses.append(
                _data_to_data_flow_matching_step(
                    model,
                    opt,
                    x0,
                    x1,
                    cb,
                    pb,
                    device=device,
                    gen=gen,
                    finetune_pert_index=finetune_pert_index,
                    finetune_context_index=finetune_context_index,
                )
            )
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": float(epoch), "loss_mean": mean_loss})
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  epoch {epoch:5d}  loss {mean_loss:.6f}")
    return history


def train_flow_matching(
    model: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    params: Iterable[torch.nn.Parameter] | None = None,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    epoch_offset: int = 0,
    log_total_epochs: int | None = None,
    finetune_stage: int | None = None,
) -> list[dict[str, float]]:
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    total_epochs = log_total_epochs if log_total_epochs is not None else epoch_offset + epochs

    history: list[dict[str, float]] = []
    model.train()
    cov_order = getattr(model, "covariate_col_order", ())
    for local_epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + local_epoch
        losses: list[float] = []
        for batch in loader:
            x1, cb, pb, zb = unpack_tx_batch(batch, cov_order)
            losses.append(
                _flow_matching_step(
                    model,
                    opt,
                    x1,
                    cb,
                    pb,
                    z=zb,
                    data_dim=data_dim,
                    device=device,
                    gen=gen,
                    finetune_pert_index=finetune_pert_index,
                    finetune_context_index=finetune_context_index,
                    context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
                )
            )
        mean_loss = float(np.nanmean(np.asarray(losses, dtype=np.float64)))
        row: dict[str, float] = {"epoch": float(global_epoch), "loss_mean": mean_loss}
        if finetune_stage is not None:
            row["finetune_stage"] = float(finetune_stage)
        history.append(row)
        if _should_log_training_epoch(local_epoch, epochs):
            stage_tag = f" stage {finetune_stage}" if finetune_stage is not None else ""
            print(
                f"  epoch {global_epoch:5d}/{total_epochs}{stage_tag}  loss {mean_loss:.6f}"
            )
    return history


def train_flow_matching_gft(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    beta_start: float,
    beta_end: float = 0.0,
    beta_schedule: str = "sigmoid",
    sigmoid_midpoint: float = 0.5,
    sigmoid_steepness: float = 10.0,
    params: Iterable[torch.nn.Parameter] | None = None,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    epoch_offset: int = 0,
    log_total_epochs: int | None = None,
    finetune_stage: int | None = None,
    global_step_offset: int = 0,
    gft_total_steps: int | None = None,
) -> list[dict[str, float]]:
    """Gradual fine-tuning (GFT): FM loss + beta(step) * ||v_theta - v_theta_0||^2."""
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)

    steps_per_epoch = len(loader)
    total_steps = gft_total_steps if gft_total_steps is not None else max(
        1, epochs * steps_per_epoch
    )
    global_step = global_step_offset
    total_epochs = log_total_epochs if log_total_epochs is not None else epoch_offset + epochs

    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    cov_order = getattr(model, "covariate_col_order", ())
    for local_epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + local_epoch
        losses: list[float] = []
        loss_fms: list[float] = []
        loss_regs: list[float] = []
        betas: list[float] = []
        for batch in loader:
            x1, cb, pb, zb = unpack_tx_batch(batch, cov_order)
            beta = gft_beta_at_step(
                global_step,
                total_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                schedule=beta_schedule,
                sigmoid_midpoint=sigmoid_midpoint,
                sigmoid_steepness=sigmoid_steepness,
            )
            step_out = _gft_flow_matching_step(
                model,
                model_ref,
                opt,
                x1,
                cb,
                pb,
                z=zb,
                data_dim=data_dim,
                device=device,
                gen=gen,
                beta=beta,
                finetune_pert_index=finetune_pert_index,
                finetune_context_index=finetune_context_index,
                context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
            )
            losses.append(step_out["loss"])
            loss_fms.append(step_out["loss_fm"])
            loss_regs.append(step_out["loss_reg"])
            betas.append(step_out["beta"])
            global_step += 1

        row = {
            "epoch": float(global_epoch),
            "loss_mean": float(np.nanmean(np.asarray(losses, dtype=np.float64))),
            "loss_fm_mean": float(np.nanmean(np.asarray(loss_fms, dtype=np.float64))),
            "loss_reg_mean": float(np.nanmean(np.asarray(loss_regs, dtype=np.float64))),
            "beta_mean": float(np.mean(betas)),
        }
        if finetune_stage is not None:
            row["finetune_stage"] = float(finetune_stage)
        history.append(row)
        if _should_log_training_epoch(local_epoch, epochs):
            stage_tag = f" stage {finetune_stage}" if finetune_stage is not None else ""
            print(
                f"  epoch {global_epoch:5d}/{total_epochs}{stage_tag}  "
                f"loss {row['loss_mean']:.6f}  "
                f"fm {row['loss_fm_mean']:.6f}  reg {row['loss_reg_mean']:.6f}  "
                f"beta {row['beta_mean']:.4f}"
            )
    return history


def _replay_batch_sizes(batch_size: int, replay_fraction: float) -> tuple[int, int]:
    """Return ``(n_finetune, n_replay)`` for generative (endpoint) replay.

    ``replay_fraction`` is the mixture weight ``λ`` realized as the fraction of the
    batch filled by synthetic old-task endpoints.
    """
    if not (0.0 < replay_fraction < 1.0):
        raise ValueError(f"replay_fraction must be in (0, 1), got {replay_fraction}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size == 1:
        return 1, 1
    n_replay = int(batch_size * replay_fraction)
    if n_replay < 1:
        n_replay = 1
    n_finetune = batch_size - n_replay
    if n_finetune < 1:
        raise ValueError(
            f"batch_size={batch_size} and replay_fraction={replay_fraction} "
            f"leave no finetune slots (n_finetune={n_finetune})"
        )
    return n_finetune, n_replay


def _trajectory_replay_batch_sizes(
    batch_size: int,
    replay_fraction: float = 0.5,
    *,
    sampling: Literal["equal", "lambda"] = "equal",
    replay_batch_multiplier: float = 1.0,
) -> tuple[int, int]:
    """Return new-task and old-trajectory Monte Carlo sample counts."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size == 1:
        return 1, 1
    if sampling == "equal":
        n_traj_base = batch_size // 2
        n_finetune = batch_size - n_traj_base
    elif sampling == "lambda":
        n_traj_base = min(max(int(round(batch_size * replay_fraction)), 1), batch_size - 1)
        n_finetune = batch_size - n_traj_base
    else:
        raise ValueError(
            f"trajectory batch sampling must be 'equal' or 'lambda', got {sampling!r}"
        )
    if replay_batch_multiplier <= 0:
        raise ValueError(
            f"trajectory replay batch multiplier must be > 0, got {replay_batch_multiplier}"
        )
    n_traj = max(1, int(round(n_traj_base * replay_batch_multiplier)))
    return n_finetune, n_traj


def train_flow_matching_generative_replay(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    support_contexts: list[int],
    n_perturbations: int,
    replay_fraction: float,
    replay_mode: Literal["endpoint", "field"] = "endpoint",
    replay_ode_method: str = "euler",
    replay_ode_rtol: float = 1e-5,
    replay_ode_atol: float = 1e-5,
    replay_ode_step_size: float | None = 0.05,
    replay_condition_sampling: str = "proportional",
    replay_stratum_dist: SupportStratumDistribution | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    epoch_offset: int = 0,
    log_total_epochs: int | None = None,
    finetune_stage: int | None = None,
    flow_map: ConditionalFlowMap | None = None,
    flow_map_n_steps: int = 1,
    mean_flow: ConditionalMeanFlow | None = None,
    mean_flow_n_steps: int = 1,
    traj_kappa: TrajKappaName | str = "none",
    trajectory_batch_sampling: Literal["equal", "lambda"] = "equal",
    trajectory_replay_batch_multiplier: float = 1.0,
    trajectory_replay_loss_multiplier: float = 1.0,
    trajectory_time_sampling: Literal["batch", "per_example_euler_grid"] = "batch",
    replay_p_remap_insert_at: int | None = None,
) -> list[dict[str, float]]:
    """Finetune on holdout cells plus replay on pretrain support ``(c, p)``.

    ``replay_mode="endpoint"`` (generative replay): roll ``model_ref`` (or a distilled
    ``flow_map`` / ``mean_flow``) to ``t=1``, re-noise, and apply CFM on a mixed batch.
    Here ``replay_fraction`` is ``λ``: the fraction of each batch that is synthetic
    old-task endpoints.

    ``replay_mode="field"`` (trajectory replay): split each batch 50/50 — half the
    slots for new-task CFM, half rolled to a random ``t`` under the frozen ref field
    for velocity matching. Here ``replay_fraction`` is ``λ`` in the weighted loss
    ``λ L_traj + (1-λ) L_CFM-new`` (not the batch mix ratio).
    ``traj_kappa`` optionally applies Thm. 2.4 weighting
    ``κ(t)·(λ L_traj + (1-λ) L_CFM-new)`` with a shared ``t``
    (default ``none`` = κ≡1; ``ot`` = ``t/(1-t)``).
    """
    if replay_mode not in ("endpoint", "field"):
        raise ValueError(f"replay_mode must be 'endpoint' or 'field', got {replay_mode!r}")
    if not (0.0 < replay_fraction < 1.0):
        raise ValueError(f"replay_fraction must be in (0, 1), got {replay_fraction}")
    if replay_mode == "field":
        resolve_traj_kappa(traj_kappa, 0.5)
        if trajectory_replay_loss_multiplier <= 0:
            raise ValueError(
                "trajectory replay loss multiplier must be > 0, "
                f"got {trajectory_replay_loss_multiplier}"
            )
    total_epochs = log_total_epochs if log_total_epochs is not None else epoch_offset + epochs
    if not support_contexts:
        raise ValueError("support_contexts must be non-empty for generative replay")
    if (
        replay_condition_sampling in OBSERVED_STRATUM_SAMPLINGS
        and replay_stratum_dist is None
    ):
        raise ValueError(
            f"replay_stratum_dist required for {replay_condition_sampling} sampling"
        )
    distilled, _ = _resolve_distilled_replay(
        flow_map, flow_map_n_steps, mean_flow, mean_flow_n_steps
    )

    replay_ode_options: dict[str, float] | None = None
    if replay_ode_step_size is not None:
        replay_ode_options = {"step_size": float(replay_ode_step_size)}

    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)

    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    if distilled is not None:
        distilled.eval()
    cov_order = getattr(model, "covariate_col_order", ())
    for local_epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + local_epoch
        losses: list[float] = []
        loss_fts: list[float] = []
        loss_refs: list[float] = []
        for batch in loader:
            x1, cb, pb, zb = unpack_tx_batch(batch, cov_order)
            b = x1.shape[0]
            if replay_mode == "field":
                n_ft, n_ref = _trajectory_replay_batch_sizes(
                    b,
                    replay_fraction,
                    sampling=trajectory_batch_sampling,
                    replay_batch_multiplier=trajectory_replay_batch_multiplier,
                )
            else:
                n_ft, n_ref = _replay_batch_sizes(b, replay_fraction)
            perm = torch.randperm(b, generator=cpu_gen)
            ft_idx = perm[:n_ft]
            x1_ft = x1[ft_idx].to(device)
            cb_ft = cb[ft_idx].to(device=device, dtype=torch.long)
            pb_ft = pb[ft_idx].to(device=device, dtype=torch.long)
            zb_ft = _z_dict_to_device(_subset_z_dict(zb, ft_idx, cov_order), cov_order, device)

            cond_out = sample_support_conditions(
                n_ref,
                support_contexts=support_contexts,
                n_perturbations=n_perturbations,
                device=device,
                cpu_gen=cpu_gen,
                sampling=replay_condition_sampling,
                stratum_dist=replay_stratum_dist,
            )
            z_ref = None
            if len(cond_out) == 3:
                c_ref, p_ref, z_ref = cond_out
            else:
                c_ref, p_ref = cond_out
            if replay_p_remap_insert_at is not None:
                p_ref = p_ref + (p_ref >= replay_p_remap_insert_at).long()
            if replay_mode == "endpoint":
                x1_ref = _sample_ref_replay_endpoints(
                    model_ref,
                    c_ref,
                    p_ref,
                    z_ref=z_ref,
                    device=device,
                    gen=gen,
                    replay_ode_method=replay_ode_method,
                    replay_ode_rtol=replay_ode_rtol,
                    replay_ode_atol=replay_ode_atol,
                    replay_ode_options=replay_ode_options,
                    flow_map=flow_map,
                    flow_map_n_steps=flow_map_n_steps,
                    mean_flow=mean_flow,
                    mean_flow_n_steps=mean_flow_n_steps,
                )
                x1_batch = torch.cat([x1_ft, x1_ref], dim=0)
                cb_batch = torch.cat([cb_ft, c_ref], dim=0)
                pb_batch = torch.cat([pb_ft, p_ref], dim=0)
                zb_batch = _concat_z_dicts(zb_ft, z_ref, cov_order)
                n_ft = x1_ft.shape[0]
                step_out = _flow_matching_replay_step(
                    model,
                    opt,
                    x1_batch,
                    cb_batch,
                    pb_batch,
                    z=zb_batch,
                    split_at=n_ft,
                    data_dim=data_dim,
                    device=device,
                    gen=gen,
                    finetune_pert_index=finetune_pert_index,
                    finetune_context_index=finetune_context_index,
                    context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
                )
                losses.append(step_out["loss"])
                loss_fts.append(step_out["loss_ft"])
                loss_refs.append(step_out["loss_ref"])
            else:
                step_out = _generative_replay_field_step(
                    model,
                    model_ref,
                    opt,
                    x1_ft,
                    cb_ft,
                    pb_ft,
                    c_ref,
                    p_ref,
                    lambda_mix=replay_fraction,
                    z_ft=zb_ft,
                    z_ref=z_ref,
                    data_dim=data_dim,
                    device=device,
                    gen=gen,
                    replay_ode_method=replay_ode_method,
                    replay_ode_rtol=replay_ode_rtol,
                    replay_ode_atol=replay_ode_atol,
                    replay_ode_options=replay_ode_options,
                    flow_map=flow_map,
                    flow_map_n_steps=flow_map_n_steps,
                    mean_flow=mean_flow,
                    mean_flow_n_steps=mean_flow_n_steps,
                    traj_kappa=traj_kappa,
                    trajectory_replay_loss_multiplier=trajectory_replay_loss_multiplier,
                    trajectory_time_sampling=trajectory_time_sampling,
                    finetune_pert_index=finetune_pert_index,
                    finetune_context_index=finetune_context_index,
                    context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
                )
                losses.append(step_out["loss"])
                loss_fts.append(step_out["loss_ft"])
                loss_refs.append(step_out["loss_ref"])

        row = {
            "epoch": float(global_epoch),
            "loss_mean": float(np.nanmean(np.asarray(losses, dtype=np.float64))),
            "loss_ft_mean": float(np.nanmean(np.asarray(loss_fts, dtype=np.float64))),
            "loss_ref_mean": float(np.nanmean(np.asarray(loss_refs, dtype=np.float64))),
        }
        if finetune_stage is not None:
            row["finetune_stage"] = float(finetune_stage)
        history.append(row)
        if _should_log_training_epoch(local_epoch, epochs):
            stage_tag = f" stage {finetune_stage}" if finetune_stage is not None else ""
            ref_tag = "field" if replay_mode == "field" else "ref"
            print(
                f"  epoch {global_epoch:5d}/{total_epochs}{stage_tag}  "
                f"loss {row['loss_mean']:.6f}  "
                f"ft {row['loss_ft_mean']:.6f}  {ref_tag} {row['loss_ref_mean']:.6f}"
            )
    return history


def train_flow_matching_real_replay(
    model: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    x_support: torch.Tensor,
    c_support: torch.Tensor,
    p_support: torch.Tensor,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    n_perturbations: int,
    replay_fraction: float,
    support_contexts: list[int],
    replay_condition_sampling: str = "proportional",
    replay_stratum_dist: SupportStratumDistribution | None = None,
    z_support: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    stratum_cells: dict[tuple[int, ...], torch.Tensor] | None = None,
    support_flat_idx: torch.Tensor | None = None,
    params: Iterable[torch.nn.Parameter] | None = None,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
    context_emb_train_rows: frozenset[int] | None = None,
    pert_emb_train_rows: frozenset[int] | None = None,
    epoch_offset: int = 0,
    log_total_epochs: int | None = None,
    finetune_stage: int | None = None,
) -> list[dict[str, float]]:
    """Finetune on holdout cells + FM on **real** pretrain support cells (not ref ODE).

    Same batch mixing and ``(c, p)`` sampling as :func:`train_flow_matching_generative_replay`,
    but replay endpoints are observed PCA vectors from the pretrain train pool.
    """
    total_epochs = log_total_epochs if log_total_epochs is not None else epoch_offset + epochs
    if stratum_cells is None or support_flat_idx is None:
        raise ValueError("stratum_cells and support_flat_idx are required")
    if not support_contexts:
        raise ValueError("support_contexts must be non-empty for real replay")
    if (
        replay_condition_sampling in OBSERVED_STRATUM_SAMPLINGS
        and replay_stratum_dist is None
    ):
        raise ValueError(
            f"replay_stratum_dist required for {replay_condition_sampling} sampling"
        )

    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)

    history: list[dict[str, float]] = []
    model.train()
    cov_order = getattr(model, "covariate_col_order", ()) or covariate_col_order
    for local_epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + local_epoch
        losses: list[float] = []
        loss_fts: list[float] = []
        loss_refs: list[float] = []
        for batch in loader:
            x1, cb, pb, zb = unpack_tx_batch(batch, cov_order)
            b = x1.shape[0]
            n_ft, n_ref = _replay_batch_sizes(b, replay_fraction)
            perm = torch.randperm(b, generator=cpu_gen)
            ft_idx = perm[:n_ft]
            x1_ft = x1[ft_idx].to(device)
            cb_ft = cb[ft_idx].to(device=device, dtype=torch.long)
            pb_ft = pb[ft_idx].to(device=device, dtype=torch.long)
            zb_ft = _z_dict_to_device(_subset_z_dict(zb, ft_idx, cov_order), cov_order, device)

            cell_out = sample_support_cells(
                n_ref,
                x_support,
                c_support,
                p_support,
                support_contexts=support_contexts,
                n_perturbations=n_perturbations,
                device=device,
                cpu_gen=cpu_gen,
                sampling=replay_condition_sampling,
                stratum_dist=replay_stratum_dist,
                stratum_cells=stratum_cells,
                support_flat_idx=support_flat_idx,
                z=z_support,
                covariate_col_order=cov_order,
            )
            z_ref = None
            if len(cell_out) == 4:
                x1_ref, c_ref, p_ref, z_ref = cell_out
            else:
                x1_ref, c_ref, p_ref = cell_out

            x1_batch = torch.cat([x1_ft, x1_ref], dim=0)
            cb_batch = torch.cat([cb_ft, c_ref], dim=0)
            pb_batch = torch.cat([pb_ft, p_ref], dim=0)
            zb_batch = _concat_z_dicts(zb_ft, z_ref, cov_order)
            n_ft = x1_ft.shape[0]
            step_out = _flow_matching_replay_step(
                model,
                opt,
                x1_batch,
                cb_batch,
                pb_batch,
                z=zb_batch,
                split_at=n_ft,
                data_dim=data_dim,
                device=device,
                gen=gen,
                finetune_pert_index=finetune_pert_index,
                finetune_context_index=finetune_context_index,
                context_emb_train_rows=context_emb_train_rows,
        pert_emb_train_rows=pert_emb_train_rows,
            )
            losses.append(step_out["loss"])
            loss_fts.append(step_out["loss_ft"])
            loss_refs.append(step_out["loss_ref"])

        row = {
            "epoch": float(global_epoch),
            "loss_mean": float(np.nanmean(np.asarray(losses, dtype=np.float64))),
            "loss_ft_mean": float(np.nanmean(np.asarray(loss_fts, dtype=np.float64))),
            "loss_ref_mean": float(np.nanmean(np.asarray(loss_refs, dtype=np.float64))),
        }
        if finetune_stage is not None:
            row["finetune_stage"] = float(finetune_stage)
        history.append(row)
        if _should_log_training_epoch(local_epoch, epochs):
            stage_tag = f" stage {finetune_stage}" if finetune_stage is not None else ""
            print(
                f"  epoch {global_epoch:5d}/{total_epochs}{stage_tag}  "
                f"loss {row['loss_mean']:.6f}  "
                f"ft {row['loss_ft_mean']:.6f}  replay {row['loss_ref_mean']:.6f}"
            )
    return history


def train_flow_matching_replay(
    model: ConditionalFlowMatching,
    x_ood: torch.Tensor,
    c_ood: torch.Tensor,
    p_ood: torch.Tensor,
    x_replay: torch.Tensor,
    c_replay: torch.Tensor,
    p_replay: torch.Tensor,
    *,
    data_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    replay_fraction: float,
    train_seed: int,
    params: Iterable[torch.nn.Parameter] | None = None,
    finetune_pert_index: int | None = None,
    finetune_context_index: int | None = None,
) -> list[dict[str, float]]:
    """Finetune with mixed batches: (1-r) OOD + r replay, every step."""
    if not (0.0 < replay_fraction < 1.0):
        raise ValueError(f"replay_fraction must be in (0, 1), got {replay_fraction}")
    if len(x_ood) == 0:
        raise ValueError("OOD finetune set is empty")
    if len(x_replay) == 0:
        raise ValueError("replay buffer is empty")

    n_replay = int(batch_size * replay_fraction)
    n_ood = batch_size - n_replay
    if n_ood < 1:
        raise ValueError(
            f"batch_size={batch_size} and replay_fraction={replay_fraction} "
            f"leave no OOD slots (n_ood={n_ood})"
        )

    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)

    drop_last = len(x_ood) >= n_ood
    if drop_last:
        steps_per_epoch = len(x_ood) // n_ood
    else:
        steps_per_epoch = max(1, (len(x_ood) + n_ood - 1) // n_ood)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for _ in range(steps_per_epoch):
            ood_idx = torch.randint(
                len(x_ood), (n_ood,), generator=cpu_gen, dtype=torch.long
            )
            rep_idx = torch.randint(
                len(x_replay), (n_replay,), generator=cpu_gen, dtype=torch.long
            )
            x1 = torch.cat([x_ood[ood_idx], x_replay[rep_idx]], dim=0)
            cb = torch.cat([c_ood[ood_idx], c_replay[rep_idx]], dim=0)
            pb = torch.cat([p_ood[ood_idx], p_replay[rep_idx]], dim=0)
            losses.append(
                _flow_matching_step(
                    model,
                    opt,
                    x1,
                    cb,
                    pb,
                    data_dim=data_dim,
                    device=device,
                    gen=gen,
                    finetune_pert_index=finetune_pert_index,
                    finetune_context_index=finetune_context_index,
                )
            )
        mean_loss = float(np.mean(losses))
        history.append({"epoch": float(epoch), "loss_mean": mean_loss})
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  epoch {epoch:5d}  loss {mean_loss:.6f}")
    return history
