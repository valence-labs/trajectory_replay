"""Time weights κ(t) for trajectory replay (Thm. 2.4).

For conditional Gaussian paths with schedules α_t, σ_t,
``κ_t = α_t / (σ_t (α̇_t σ_t − α_t σ̇_t))`` (paper Eq. 17).

The linear OT / CFM path used in this repo is ``α_t = t``, ``σ_t = 1 − t``,
which simplifies to ``κ_t = t / (1 − t)``.

In training, κ multiplies the *full* integrand at a shared time ``t``:
``κ(t) · (λ L_traj(t) + (1−λ) L_CFM-new(t))``.

Training defaults to ``none`` (κ≡1); pass ``ot`` to enable the OT weight.
"""

from __future__ import annotations

from typing import Literal

import torch

TrajKappaName = Literal["none", "ot"]


def gaussian_path_kappa(
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    alpha_dot: torch.Tensor,
    sigma_dot: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """κ from Eq. (17) for general α_t, σ_t schedules."""
    denom = sigma * (alpha_dot * sigma - alpha * sigma_dot)
    return alpha / denom.clamp_min(eps)


def ot_kappa(
    t: torch.Tensor | float,
    *,
    eps: float = 1e-4,
    max_kappa: float | None = 1e4,
) -> torch.Tensor:
    """κ_t = t / (1 − t) for the OT path α_t = t, σ_t = 1 − t.

    Clamps ``t`` into ``(eps, 1 − eps)`` so κ stays finite near the endpoints.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(float(t))
    t = t.clamp(eps, 1.0 - eps)
    kappa = t / (1.0 - t)
    if max_kappa is not None:
        kappa = kappa.clamp_max(float(max_kappa))
    return kappa


def resolve_traj_kappa(
    name: TrajKappaName | str,
    t: torch.Tensor | float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a scalar (0-dim) weight for the traj MSE at time ``t``."""
    key = str(name).lower().strip()
    if key in ("none", "off", "uniform", "1"):
        w = torch.ones((), dtype=torch.float64)
    elif key in ("ot", "linear", "paper", "thm24"):
        w = ot_kappa(t)
    else:
        raise ValueError(
            f"Unknown traj_kappa={name!r}; expected one of 'none' (default), 'ot'."
        )
    if dtype is not None or device is not None:
        w = w.to(device=device, dtype=dtype or w.dtype)
    return w
