"""Unconditional flow-matching trainers for mixture continual-learning sims."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.gft_schedule import gft_beta_at_step
from trajectory_replay.training.traj_kappa import TrajKappaName, resolve_traj_kappa

# Optional hook after each epoch: ``callback(epoch, history_row)``.
EpochEndCallback = Callable[[int, dict[str, Any]], None]


def _broadcast_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (x.dim() - 1)))


def _cfm_loss(
    model: ConditionalFlowMatching,
    x1: torch.Tensor,
    *,
    device: torch.device,
    gen: torch.Generator,
    t_scalar: float | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conditional FM loss. If ``t_scalar`` is set, use that time for the whole batch."""
    x1 = x1.to(device)
    b = x1.shape[0]
    # Image-safe: match full tensor shape (B,C,H,W) or (B,D).
    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
    if t_scalar is None:
        t = torch.rand(b, device=device, generator=gen)
    else:
        t = torch.full((b,), float(t_scalar), device=device, dtype=x1.dtype)
    t_exp = _broadcast_time(t, x1)
    xt = (1.0 - t_exp) * x0 + t_exp * x1
    v_star = x1 - x0
    labels = None if y is None else y.to(device=device, dtype=torch.long)
    v_pred = model(xt, t, labels, None)
    return F.mse_loss(v_pred, v_star)


def _sample_shared_time(
    *,
    device: torch.device,
    gen: torch.Generator,
    eps: float = 1e-4,
) -> float:
    """Sample one t ∈ (eps, 1] shared by both Thm. 2.4 integrand terms."""
    return float((eps + (1.0 - eps) * torch.rand(1, device=device, generator=gen)).item())


def _traj_field_loss(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    *,
    n: int,
    device: torch.device,
    gen: torch.Generator,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    mean_flow: Any | None = None,
    mean_flow_n_steps: int = 1,
    t_scalar: float | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unweighted field MSE at time ``t_scalar`` (sampled if omitted).

    When ``y`` is provided (class-conditional), teacher rollouts and both
    velocity evaluations are conditioned on those labels (no CFG).
    """
    dtype = next(model_ref.parameters()).dtype
    x0 = model_ref.sample_noise(n, device=device, dtype=dtype, generator=gen)
    if t_scalar is None:
        t_scalar = _sample_shared_time(device=device, gen=gen)
    t = torch.full((n,), float(t_scalar), device=device, dtype=dtype)
    labels = None if y is None else y.to(device=device, dtype=torch.long)
    if labels is not None and labels.shape != (n,):
        raise ValueError(f"y must be shape (n,), got {tuple(labels.shape)} for n={n}")

    def _rollout_from_x0(sampler: Any) -> torch.Tensor:
        kwargs: dict[str, Any] = {"t_end": float(t_scalar)}
        if mean_flow is not None:
            kwargs["n_steps"] = mean_flow_n_steps
        else:
            kwargs.update(
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
            )
        if labels is not None:
            kwargs["guidance_scale"] = 1.0
        return sampler(x0, labels, None, **kwargs)

    def _velocity(m: Any, x: torch.Tensor) -> torch.Tensor:
        if labels is None:
            return m(x, t, None, None)
        return m(x, t, labels, None, guidance_scale=1.0)

    with torch.no_grad():
        if mean_flow is not None:
            x_t = _rollout_from_x0(mean_flow.sample_from_x0)
        else:
            x_t = _rollout_from_x0(model_ref.sample_from_x0)
    v_pred = _velocity(model, x_t)
    with torch.no_grad():
        v_ref = _velocity(model_ref, x_t)
    return F.mse_loss(v_pred, v_ref)


def _dropout_labels(
    y: torch.Tensor,
    *,
    null_id: int,
    label_dropout: float,
    cpu_gen: torch.Generator,
) -> torch.Tensor:
    if label_dropout <= 0.0:
        return y
    drop = torch.rand(y.shape[0], generator=cpu_gen) < float(label_dropout)
    if not bool(drop.any()):
        return y
    out = y.clone()
    out[drop] = int(null_id)
    return out


def _sample_old_labels(
    old_class_ids: Sequence[int],
    n: int,
    *,
    cpu_gen: torch.Generator,
) -> torch.Tensor:
    from trajectory_replay.eval.image_fid import sample_labels_balanced

    return sample_labels_balanced(old_class_ids, int(n), generator=cpu_gen)


def _optimizer_step(
    opt: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    params: Iterable[torch.nn.Parameter],
    max_grad_norm: float | None = None,
) -> None:
    """Backward + optional global grad-norm clip + Adam step.

    Clipping is off by default (``max_grad_norm`` None/<=0) so λ-mixing
    comparisons stay comparable to prior unclipped runs. Pass 1.0 for large
    UNets that otherwise spike from occasional large gradients.
    """
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("non-finite training loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            list(params),
            max_norm=float(max_grad_norm),
            error_if_nonfinite=True,
        )
    opt.step()


def _log_epoch(epoch: int, total: int, row: dict[str, float]) -> None:
    extra = ""
    if "loss_old_mean" in row:
        extra = f"  old {row['loss_old_mean']:.6f}  new {row['loss_new_mean']:.6f}"
    print(f"  epoch {epoch:5d}/{total}  loss {row['loss_mean']:.6f}{extra}")


def _should_log(local_epoch: int, epochs: int) -> bool:
    return local_epoch == 1 or local_epoch % max(1, epochs // 10) == 0 or local_epoch == epochs


def _replay_counts(batch_size: int, lambda_mix: float) -> tuple[int, int]:
    """Return ``(n_new, n_old)`` for generative/real replay batch mixing."""
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2, got {batch_size}")
    n_old = int(batch_size * lambda_mix)
    n_old = min(max(n_old, 1), batch_size - 1)
    n_new = batch_size - n_old
    return n_new, n_old


def _trajectory_counts(batch_size: int) -> tuple[int, int]:
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2 for trajectory replay, got {batch_size}")
    n_old = batch_size // 2
    n_new = batch_size - n_old
    return n_new, n_old


def train_unconditional_fm(
    model: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Standard unconditional CFM on batches of ``x`` only."""
    if model.use_conditioning:
        raise ValueError("train_unconditional_fm expects use_conditioning=False")
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for (x1,) in loader:
            loss = _cfm_loss(model, x1, device=device, gen=gen)
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
        row = {"epoch": float(epoch), "loss_mean": float(np.mean(losses))}
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_class_conditional_fm(
    model: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    label_dropout: float = 0.1,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Class-conditional CFM on batches of ``(x, y)``.

    With probability ``label_dropout``, labels are replaced by the model's null
    class id so classifier-free guidance can be used at sample time.
    """
    if not model.use_conditioning:
        raise ValueError("train_class_conditional_fm expects use_conditioning=True")
    if not (0.0 <= label_dropout < 1.0):
        raise ValueError(f"label_dropout must be in [0, 1), got {label_dropout}")
    null_id = int(getattr(model, "null_class_id"))
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch in loader:
            if len(batch) != 2:
                raise ValueError(
                    "class-conditional loader must yield (x, y); "
                    f"got {len(batch)} tensors"
                )
            x1, y = batch
            y = y.long()
            if label_dropout > 0.0:
                drop = torch.rand(y.shape[0], generator=cpu_gen) < float(label_dropout)
                if bool(drop.any()):
                    y = y.clone()
                    y[drop] = null_id
            loss = _cfm_loss(model, x1, device=device, gen=gen, y=y)
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
        row = {"epoch": float(epoch), "loss_mean": float(np.mean(losses))}
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def _cfm_gft_losses(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    x1: torch.Tensor,
    *,
    device: torch.device,
    gen: torch.Generator,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(flow-matching loss, distillation loss)`` at a shared noisy point.

    Both velocities are read at the same ``(x_t, t)`` so the regularizer measures
    drift of the student away from the frozen previous-task field.
    """
    x1 = x1.to(device)
    b = x1.shape[0]
    x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=gen)
    t = torch.rand(b, device=device, generator=gen)
    t_exp = _broadcast_time(t, x1)
    xt = (1.0 - t_exp) * x0 + t_exp * x1
    v_star = x1 - x0
    labels = None if y is None else y.to(device=device, dtype=torch.long)
    if labels is None:
        v_pred = model(xt, t, None, None)
        with torch.no_grad():
            v_ref = model_ref(xt, t, None, None)
    else:
        v_pred = model(xt, t, labels, None)
        with torch.no_grad():
            v_ref = model_ref(xt, t, labels, None, guidance_scale=1.0)
    return F.mse_loss(v_pred, v_star), F.mse_loss(v_pred, v_ref)


def _train_fm_gft(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    class_conditional: bool,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
    sigmoid_midpoint: float,
    sigmoid_steepness: float,
    label_dropout: float,
    params: Iterable[torch.nn.Parameter] | None,
    max_grad_norm: float | None,
    on_epoch_end: EpochEndCallback | None,
) -> list[dict[str, float]]:
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    null_id = int(getattr(model, "null_class_id")) if class_conditional else -1

    total_steps = max(1, epochs * len(loader))
    global_step = 0
    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_fms: list[float] = []
        loss_regs: list[float] = []
        betas: list[float] = []
        for batch in loader:
            if class_conditional:
                if len(batch) != 2:
                    raise ValueError(
                        "class-conditional loader must yield (x, y); "
                        f"got {len(batch)} tensors"
                    )
                x1, y = batch
                y = _dropout_labels(
                    y.long(),
                    null_id=null_id,
                    label_dropout=label_dropout,
                    cpu_gen=cpu_gen,
                )
            else:
                (x1,) = batch
                y = None
            beta = gft_beta_at_step(
                global_step,
                total_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                schedule=beta_schedule,
                sigmoid_midpoint=sigmoid_midpoint,
                sigmoid_steepness=sigmoid_steepness,
            )
            loss_fm, loss_reg = _cfm_gft_losses(
                model, model_ref, x1, device=device, gen=gen, y=y
            )
            loss = loss_fm + beta * loss_reg
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
            loss_fms.append(float(loss_fm.item()))
            loss_regs.append(float(loss_reg.item()))
            betas.append(beta)
            global_step += 1
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_fm_mean": float(np.mean(loss_fms)),
            "loss_reg_mean": float(np.mean(loss_regs)),
            "beta_mean": float(np.mean(betas)),
        }
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_unconditional_fm_gft(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    beta_schedule: str = "sigmoid",
    sigmoid_midpoint: float = 0.5,
    sigmoid_steepness: float = 10.0,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Gradual fine-tuning: CFM loss plus ``beta(step)`` times the teacher-field MSE."""
    if model.use_conditioning or model_ref.use_conditioning:
        raise ValueError("train_unconditional_fm_gft expects use_conditioning=False")
    return _train_fm_gft(
        model,
        model_ref,
        loader,
        class_conditional=False,
        device=device,
        epochs=epochs,
        lr=lr,
        train_seed=train_seed,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        sigmoid_midpoint=sigmoid_midpoint,
        sigmoid_steepness=sigmoid_steepness,
        label_dropout=0.0,
        params=params,
        max_grad_norm=max_grad_norm,
        on_epoch_end=on_epoch_end,
    )


def train_class_conditional_fm_gft(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    beta_schedule: str = "sigmoid",
    sigmoid_midpoint: float = 0.5,
    sigmoid_steepness: float = 10.0,
    label_dropout: float = 0.1,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Class-conditional gradual fine-tuning.

    The frozen teacher is evaluated on the same labels the student sees, including
    the ones replaced by the null class, so the regularizer also covers the
    unconditional branch used by classifier-free guidance.
    """
    if not (model.use_conditioning and model_ref.use_conditioning):
        raise ValueError("train_class_conditional_fm_gft expects use_conditioning=True")
    if not (0.0 <= label_dropout < 1.0):
        raise ValueError(f"label_dropout must be in [0, 1), got {label_dropout}")
    return _train_fm_gft(
        model,
        model_ref,
        loader,
        class_conditional=True,
        device=device,
        epochs=epochs,
        lr=lr,
        train_seed=train_seed,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        sigmoid_midpoint=sigmoid_midpoint,
        sigmoid_steepness=sigmoid_steepness,
        label_dropout=label_dropout,
        params=params,
        max_grad_norm=max_grad_norm,
        on_epoch_end=on_epoch_end,
    )


def train_unconditional_generative_replay(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    ode_method: str = "euler",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    ode_step_size: float | None = 0.05,
    mean_flow: Any | None = None,
    mean_flow_n_steps: int = 1,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Generative replay: ``λ`` = fraction of batch from frozen-ref endpoints."""
    if model.use_conditioning or model_ref.use_conditioning:
        raise ValueError("unconditional generative replay requires use_conditioning=False")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if mean_flow_n_steps < 1:
        raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
    ode_options = {"step_size": float(ode_step_size)} if ode_step_size is not None else None
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    if mean_flow is not None:
        mean_flow.eval()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_olds: list[float] = []
        loss_news: list[float] = []
        for (x1,) in loader:
            b = x1.shape[0]
            n_new, n_old = _replay_counts(b, lambda_mix)
            perm = torch.randperm(b, generator=cpu_gen)
            x_new = x1[perm[:n_new]].to(device)
            with torch.no_grad():
                if mean_flow is not None:
                    x_old = mean_flow.sample(
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
            # Single CFM forward; report split diagnostics.
            x_batch = x_batch.to(device)
            bb = x_batch.shape[0]
            x0 = torch.randn(x_batch.shape, device=device, dtype=x_batch.dtype, generator=gen)
            t = torch.rand(bb, device=device, generator=gen)
            t_exp = _broadcast_time(t, x_batch)
            xt = (1.0 - t_exp) * x0 + t_exp * x_batch
            v_star = x_batch - x0
            v_pred = model(xt, t, None, None)
            loss = F.mse_loss(v_pred, v_star)
            loss_new = F.mse_loss(v_pred[:n_new], v_star[:n_new])
            loss_old = F.mse_loss(v_pred[n_new:], v_star[n_new:])
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
            loss_news.append(float(loss_new.item()))
            loss_olds.append(float(loss_old.item()))
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "lambda_mix": float(lambda_mix),
        }
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_unconditional_trajectory_replay(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    ode_method: str = "euler",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    ode_step_size: float | None = 0.05,
    mean_flow: Any | None = None,
    mean_flow_n_steps: int = 1,
    traj_kappa: TrajKappaName | str = "none",
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Trajectory replay with optional Thm. 2.4 time weighting.

    Each step samples a shared ``t``, evaluates both integrand terms at that ``t``,
    and forms
    ``κ(t) · (λ L_traj(t) + (1-λ) L_CFM-new(t))``.
    Default ``traj_kappa='none'`` (κ≡1, Prop. 2.2). Use ``traj_kappa='ot'`` for
    ``κ(t)=t/(1-t)`` on the linear OT path.
    """
    if model.use_conditioning or model_ref.use_conditioning:
        raise ValueError("unconditional trajectory replay requires use_conditioning=False")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if mean_flow_n_steps < 1:
        raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
    # Validate name early.
    resolve_traj_kappa(traj_kappa, 0.5)
    ode_options = {"step_size": float(ode_step_size)} if ode_step_size is not None else None
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    if mean_flow is not None:
        mean_flow.eval()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_olds: list[float] = []
        loss_news: list[float] = []
        for (x1,) in loader:
            b = x1.shape[0]
            n_new, n_old = _trajectory_counts(b)
            perm = torch.randperm(b, generator=cpu_gen)
            x_new = x1[perm[:n_new]].to(device)
            t_scalar = _sample_shared_time(device=device, gen=gen)
            loss_new = _cfm_loss(
                model, x_new, device=device, gen=gen, t_scalar=t_scalar
            )
            loss_old = _traj_field_loss(
                model,
                model_ref,
                n=n_old,
                device=device,
                gen=gen,
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_options=ode_options,
                mean_flow=mean_flow,
                mean_flow_n_steps=mean_flow_n_steps,
                t_scalar=t_scalar,
            )
            kappa = resolve_traj_kappa(
                traj_kappa, t_scalar, dtype=loss_old.dtype, device=loss_old.device
            )
            loss = kappa * (lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new)
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
            loss_olds.append(float(loss_old.item()))
            loss_news.append(float(loss_new.item()))
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "lambda_mix": float(lambda_mix),
            "traj_kappa": str(traj_kappa),
        }
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_class_conditional_generative_replay(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    old_class_ids: Sequence[int],
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    label_dropout: float = 0.1,
    ode_method: str = "euler",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    ode_step_size: float | None = 0.05,
    mean_flow: Any | None = None,
    mean_flow_n_steps: int = 1,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Class-conditional generative replay (λ = fraction of synthetic old endpoints).

    Old samples are drawn from the frozen teacher / MeanFlow conditioned on
    labels from ``old_class_ids`` (no CFG). Student CFM uses label dropout for CFG.
    """
    if not (model.use_conditioning and model_ref.use_conditioning):
        raise ValueError("class-conditional generative replay requires use_conditioning=True")
    if not old_class_ids:
        raise ValueError("old_class_ids must be non-empty for generative replay")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if mean_flow_n_steps < 1:
        raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
    if not (0.0 <= label_dropout < 1.0):
        raise ValueError(f"label_dropout must be in [0, 1), got {label_dropout}")
    null_id = int(getattr(model, "null_class_id"))
    ode_options = {"step_size": float(ode_step_size)} if ode_step_size is not None else None
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    if mean_flow is not None:
        mean_flow.eval()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_olds: list[float] = []
        loss_news: list[float] = []
        for batch in loader:
            if len(batch) != 2:
                raise ValueError(
                    "class-conditional loader must yield (x, y); "
                    f"got {len(batch)} tensors"
                )
            x1, y1 = batch
            y1 = y1.long()
            b = x1.shape[0]
            n_new, n_old = _replay_counts(b, lambda_mix)
            perm = torch.randperm(b, generator=cpu_gen)
            x_new = x1[perm[:n_new]].to(device)
            y_new = y1[perm[:n_new]].to(device)
            y_old = _sample_old_labels(old_class_ids, n_old, cpu_gen=cpu_gen).to(device)
            with torch.no_grad():
                if mean_flow is not None:
                    x_old = mean_flow.sample(
                        c=y_old,
                        batch_size=n_old,
                        n_steps=mean_flow_n_steps,
                        generator=gen,
                        guidance_scale=1.0,
                    )
                else:
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
            x_batch = torch.cat([x_new, x_old], dim=0)
            y_batch = torch.cat([y_new, y_old], dim=0)
            y_batch = _dropout_labels(
                y_batch, null_id=null_id, label_dropout=label_dropout, cpu_gen=cpu_gen
            )
            bb = x_batch.shape[0]
            x0 = torch.randn(x_batch.shape, device=device, dtype=x_batch.dtype, generator=gen)
            t = torch.rand(bb, device=device, generator=gen)
            t_exp = _broadcast_time(t, x_batch)
            xt = (1.0 - t_exp) * x0 + t_exp * x_batch
            v_star = x_batch - x0
            v_pred = model(xt, t, y_batch, None, guidance_scale=1.0)
            loss = F.mse_loss(v_pred, v_star)
            loss_new = F.mse_loss(v_pred[:n_new], v_star[:n_new])
            loss_old = F.mse_loss(v_pred[n_new:], v_star[n_new:])
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
            loss_news.append(float(loss_new.item()))
            loss_olds.append(float(loss_old.item()))
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "lambda_mix": float(lambda_mix),
        }
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_class_conditional_trajectory_replay(
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    old_class_ids: Sequence[int],
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    label_dropout: float = 0.1,
    ode_method: str = "euler",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    ode_step_size: float | None = 0.05,
    mean_flow: Any | None = None,
    mean_flow_n_steps: int = 1,
    traj_kappa: TrajKappaName | str = "none",
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Class-conditional trajectory replay with optional Thm. 2.4 time weighting."""
    if not (model.use_conditioning and model_ref.use_conditioning):
        raise ValueError("class-conditional trajectory replay requires use_conditioning=True")
    if not old_class_ids:
        raise ValueError("old_class_ids must be non-empty for trajectory replay")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if mean_flow_n_steps < 1:
        raise ValueError(f"mean_flow_n_steps must be >= 1, got {mean_flow_n_steps}")
    if not (0.0 <= label_dropout < 1.0):
        raise ValueError(f"label_dropout must be in [0, 1), got {label_dropout}")
    null_id = int(getattr(model, "null_class_id"))
    resolve_traj_kappa(traj_kappa, 0.5)
    ode_options = {"step_size": float(ode_step_size)} if ode_step_size is not None else None
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    history: list[dict[str, float]] = []
    model.train()
    model_ref.eval()
    if mean_flow is not None:
        mean_flow.eval()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_olds: list[float] = []
        loss_news: list[float] = []
        for batch in loader:
            if len(batch) != 2:
                raise ValueError(
                    "class-conditional loader must yield (x, y); "
                    f"got {len(batch)} tensors"
                )
            x1, y1 = batch
            y1 = y1.long()
            b = x1.shape[0]
            n_new, n_old = _trajectory_counts(b)
            perm = torch.randperm(b, generator=cpu_gen)
            x_new = x1[perm[:n_new]].to(device)
            y_new = y1[perm[:n_new]].to(device)
            y_new = _dropout_labels(
                y_new, null_id=null_id, label_dropout=label_dropout, cpu_gen=cpu_gen
            )
            y_old = _sample_old_labels(old_class_ids, n_old, cpu_gen=cpu_gen).to(device)
            t_scalar = _sample_shared_time(device=device, gen=gen)
            loss_new = _cfm_loss(
                model, x_new, device=device, gen=gen, t_scalar=t_scalar, y=y_new
            )
            loss_old = _traj_field_loss(
                model,
                model_ref,
                n=n_old,
                device=device,
                gen=gen,
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_options=ode_options,
                mean_flow=mean_flow,
                mean_flow_n_steps=mean_flow_n_steps,
                t_scalar=t_scalar,
                y=y_old,
            )
            kappa = resolve_traj_kappa(
                traj_kappa, t_scalar, dtype=loss_old.dtype, device=loss_old.device
            )
            loss = kappa * (lambda_mix * loss_old + (1.0 - lambda_mix) * loss_new)
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
            loss_olds.append(float(loss_old.item()))
            loss_news.append(float(loss_new.item()))
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "lambda_mix": float(lambda_mix),
            "traj_kappa": str(traj_kappa),
        }
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


def train_unconditional_real_replay(
    model: ConditionalFlowMatching,
    loader: DataLoader,
    x_support: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
    on_epoch_end: EpochEndCallback | None = None,
) -> list[dict[str, float]]:
    """Real replay: ``λ`` = fraction of batch drawn from observed old-task samples."""
    if model.use_conditioning:
        raise ValueError("unconditional real replay expects use_conditioning=False")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError(f"lambda_mix must be in (0, 1), got {lambda_mix}")
    if x_support.ndim != 2 or x_support.shape[0] < 1:
        raise ValueError("x_support must be (N, d) with N >= 1")
    train_params = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(train_params, lr=lr)
    torch.manual_seed(train_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(train_seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(train_seed)
    x_support = x_support.to(device)
    n_support = x_support.shape[0]
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        loss_olds: list[float] = []
        loss_news: list[float] = []
        for (x1,) in loader:
            b = x1.shape[0]
            n_new, n_old = _replay_counts(b, lambda_mix)
            perm = torch.randperm(b, generator=cpu_gen)
            x_new = x1[perm[:n_new]].to(device)
            idx = torch.randint(0, n_support, (n_old,), generator=cpu_gen)
            x_old = x_support[idx]
            x_batch = torch.cat([x_new, x_old], dim=0)
            bb = x_batch.shape[0]
            x0 = torch.randn(x_batch.shape, device=device, dtype=x_batch.dtype, generator=gen)
            t = torch.rand(bb, device=device, generator=gen)
            t_exp = _broadcast_time(t, x_batch)
            xt = (1.0 - t_exp) * x0 + t_exp * x_batch
            v_star = x_batch - x0
            v_pred = model(xt, t, None, None)
            loss = F.mse_loss(v_pred, v_star)
            loss_new = F.mse_loss(v_pred[:n_new], v_star[:n_new])
            loss_old = F.mse_loss(v_pred[n_new:], v_star[n_new:])
            _optimizer_step(opt, loss, params=train_params, max_grad_norm=max_grad_norm)
            losses.append(float(loss.item()))
            loss_news.append(float(loss_new.item()))
            loss_olds.append(float(loss_old.item()))
        row = {
            "epoch": float(epoch),
            "loss_mean": float(np.mean(losses)),
            "loss_old_mean": float(np.mean(loss_olds)),
            "loss_new_mean": float(np.mean(loss_news)),
            "lambda_mix": float(lambda_mix),
        }
        history.append(row)
        if _should_log(epoch, epochs):
            _log_epoch(epoch, epochs, row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)
    return history


StrategyName = Literal["naive", "generative", "trajectory", "real"]
