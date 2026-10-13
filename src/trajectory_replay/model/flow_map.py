"""Conditional flow map: residual displacement field Phi with X = x + (t-s) Phi.

Distilled from a pretrained velocity field v_ref for fast trajectory rollouts.
Parameterization follows Boffi et al. flow-maps (https://github.com/nmboffi/flow-maps).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trajectory_replay.model.backbones import (
    DEFAULT_BACKBONE_TYPE,
    build_flow_map_backbone,
    flow_map_backbone_in_dim,
    register_backbone,
    vector_field_module,
)
from trajectory_replay.model.flow_matching import (
    ConditionalFlowMatching,
    sinusoidal_time_embedding,
)


class ConditionalFlowMap(nn.Module):
    """Flow map X_psi(x, s, t | c, p[, z]) = x + (t - s) * Phi_psi(x, s, t, c, p[, z])."""

    def __init__(
        self,
        data_dim: int,
        n_contexts: int,
        n_perturbations: int,
        *,
        cond_emb_dim: int = 32,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        depth: int = 4,
        phi_rescale: float = 1.0,
        backbone_type: str = DEFAULT_BACKBONE_TYPE,
        backbone: nn.Module | None = None,
        covariate_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.n_contexts = n_contexts
        self.n_perturbations = n_perturbations
        self.cond_emb_dim = cond_emb_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.phi_rescale = float(phi_rescale)
        self.backbone_type = backbone_type
        self.covariate_col_order: tuple[str, ...] = tuple(covariate_sizes.keys()) if covariate_sizes else ()
        self.n_covariates = len(self.covariate_col_order)

        self.context_emb = nn.Embedding(n_contexts, cond_emb_dim)
        self.pert_emb = nn.Embedding(n_perturbations, cond_emb_dim)
        if covariate_sizes:
            self.covariate_embs = nn.ModuleDict(
                {
                    col: nn.Embedding(n_cat, cond_emb_dim)
                    for col, n_cat in covariate_sizes.items()
                }
            )
        else:
            self.covariate_embs = nn.ModuleDict()

        in_dim = flow_map_backbone_in_dim(
            data_dim=data_dim,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim,
            n_covariates=self.n_covariates,
        )
        if backbone is not None:
            register_backbone(self, backbone, backbone_type)
        else:
            register_backbone(
                self,
                build_flow_map_backbone(
                    backbone_type,
                    in_dim=in_dim,
                    out_dim=data_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                ),
                backbone_type,
            )

    def _prepare_z(
        self,
        z: dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if not self.covariate_col_order:
            return None
        if z is None:
            raise ValueError("covariate tensors required for models with covariates")
        out: dict[str, torch.Tensor] = {}
        for col in self.covariate_col_order:
            t = z[col].to(device=device, dtype=torch.long)
            if t.shape != (batch_size,):
                raise ValueError(f"z[{col!r}] must be shape ({batch_size},), got {tuple(t.shape)}")
            out[col] = t
        return out

    def _covariate_vector(self, z: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if not self.covariate_col_order:
            raise RuntimeError("_covariate_vector called without covariates")
        parts = [self.covariate_embs[col](z[col]) for col in self.covariate_col_order]  # type: ignore[index]
        return torch.cat(parts, dim=-1)

    def init_condition_embeddings_from(self, teacher: ConditionalFlowMatching) -> None:
        """Copy context / perturbation / covariate tables from a pretrained velocity field."""
        if (
            teacher.n_contexts != self.n_contexts
            or teacher.n_perturbations != self.n_perturbations
            or teacher.cond_emb_dim != self.cond_emb_dim
            or teacher.covariate_col_order != self.covariate_col_order
        ):
            raise ValueError("teacher embedding shapes must match flow map")
        with torch.no_grad():
            self.context_emb.weight.copy_(teacher.context_emb.weight)
            self.pert_emb.weight.copy_(teacher.pert_emb.weight)
            for col in self.covariate_col_order:
                self.covariate_embs[col].weight.copy_(teacher.covariate_embs[col].weight)

    def _embed(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must be (B, {self.data_dim}), got {tuple(x.shape)}")
        b = x.shape[0]
        if s.shape != (b,) or t.shape != (b,):
            raise ValueError("s and t must be shape (B,)")
        if c.shape != (b,) or p.shape != (b,):
            raise ValueError("c and p must be shape (B,)")
        z = self._prepare_z(z, b, x.device)
        se = sinusoidal_time_embedding(s, self.time_emb_dim)
        te = sinusoidal_time_embedding(t, self.time_emb_dim)
        ec = self.context_emb(c)
        ep = self.pert_emb(p)
        parts = [x, se, te, ec, ep]
        if self.covariate_col_order:
            parts.append(self._covariate_vector(z))
        return torch.cat(parts, dim=-1)

    def calc_phi(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Displacement rate Phi(s, t, x | c, p[, z])."""
        features = self._embed(x, s, t, c, p, z=z)
        return self.phi_rescale * vector_field_module(self)(features)

    def calc_b(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Instantaneous velocity at s = t: Phi(t, t, x | c, p[, z])."""
        return self.calc_phi(x, t, t, c, p, z=z)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Transport x from time s to t under condition (c, p[, z])."""
        dt = (t - s).unsqueeze(-1)
        return x + dt * self.calc_phi(x, s, t, c, p, z=z)

    def partial_t(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        z: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X(s,t,x), dX/dt) via JVP along t (for LSD distillation)."""
        x = x.detach()
        s = s.detach()
        c = c.detach()
        p = p.detach()
        z_det = None
        if z is not None:
            z_det = {col: z[col].detach() for col in self.covariate_col_order}
        t = t.detach().clone().requires_grad_(True)

        def forward_t(t_in: torch.Tensor) -> torch.Tensor:
            return self.forward(x, s, t_in, c, p, z=z_det)

        X, dt_X = torch.autograd.functional.jvp(
            forward_t,
            (t,),
            (torch.ones_like(t),),
            create_graph=True,
        )
        return X, dt_X

    @torch.no_grad()
    def rollout_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        *,
        z: dict[str, torch.Tensor] | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Transport ``x0`` from ``t_start`` to ``t_end`` in ``n_steps`` equal map jumps."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if not (0.0 <= t_start < t_end <= 1.0):
            raise ValueError(
                f"require 0 <= t_start < t_end <= 1, got t_start={t_start}, t_end={t_end}"
            )
        device = x0.device
        dtype = x0.dtype
        b = x0.shape[0]
        z = self._prepare_z(z, b, device)
        if n_steps == 1:
            s = torch.full((b,), float(t_start), device=device, dtype=dtype)
            t = torch.full((b,), float(t_end), device=device, dtype=dtype)
            return self.forward(x0, s, t, c, p, z=z)

        times = torch.linspace(
            float(t_start),
            float(t_end),
            n_steps + 1,
            device=device,
            dtype=dtype,
        )
        x = x0
        for i in range(n_steps):
            s = torch.full((b,), times[i].item(), device=device, dtype=dtype)
            t = torch.full((b,), times[i + 1].item(), device=device, dtype=dtype)
            x = self.forward(x, s, t, c, p, z=z)
        return x

    @torch.no_grad()
    def sample_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        *,
        z: dict[str, torch.Tensor] | None = None,
        t_end: float = 1.0,
        t_start: float = 0.0,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Transport ``x0`` from ``t_start`` to ``t_end`` (default one jump from 0)."""
        return self.rollout_from_x0(
            x0,
            c,
            p,
            z=z,
            t_start=t_start,
            t_end=t_end,
            n_steps=n_steps,
        )

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        p: torch.Tensor,
        *,
        z: dict[str, torch.Tensor] | None = None,
        noise_scale: float = 1.0,
        t_end: float = 1.0,
        t_start: float = 0.0,
        n_steps: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample x at ``t_end`` from x_0 ~ N(0, noise_scale^2 I) via ``n_steps`` map jumps."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        c = c.to(device=device, dtype=torch.long)
        p = p.to(device=device, dtype=torch.long)
        b = c.shape[0]
        z = self._prepare_z(z, b, device)
        x0 = noise_scale * torch.randn(
            b, self.data_dim, device=device, dtype=dtype, generator=generator
        )
        return self.sample_from_x0(
            x0,
            c,
            p,
            z=z,
            t_start=t_start,
            t_end=t_end,
            n_steps=n_steps,
        )


def build_conditional_flow_map_from_config(cfg: dict[str, Any]) -> ConditionalFlowMap:
    """Reconstruct a flow map from a saved checkpoint config dict."""
    backbone_type = str(cfg.get("backbone_type", DEFAULT_BACKBONE_TYPE))
    covariate_sizes = cfg.get("covariate_sizes")
    if covariate_sizes is None and "encoding" in cfg:
        enc = cfg["encoding"]
        order = tuple(enc.get("covariate_col_order") or ())
        covariate_sizes = {
            col: len(enc["covariate_categories"][col]) for col in order
        }
    return ConditionalFlowMap(
        data_dim=int(cfg["data_dim"]),
        n_contexts=int(cfg["n_contexts"]),
        n_perturbations=int(cfg["n_perturbations"]),
        cond_emb_dim=int(cfg["cond_emb_dim"]),
        time_emb_dim=int(cfg["time_emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        phi_rescale=float(cfg.get("phi_rescale", 1.0)),
        backbone_type=backbone_type,
        covariate_sizes=dict(covariate_sizes) if covariate_sizes else None,
    )
