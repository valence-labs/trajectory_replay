"""Conditional MeanFlow: average-velocity field u(x, r, t | c, p[, z]).

Distilled from a pretrained instantaneous velocity field for cheap multi-step
(or one-step) sampling. Architecture mirrors :class:`ConditionalFlowMap`
(dual time embeddings); the network predicts average velocity ``u`` such that

    x_t = x_r + (t - r) * u(x_r, r, t)

under our FM time convention (t=0 noise → t=1 data).

Reference: Geng et al., Mean Flows for One-step Generative Modeling;
PyTorch layout adapted from https://github.com/zhuyu-cs/MeanFlow.
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


class ConditionalMeanFlow(nn.Module):
    """MeanFlow average velocity ``u_psi(x, r, t | c, p[, z])``."""

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
        backbone_type: str = DEFAULT_BACKBONE_TYPE,
        backbone: nn.Module | None = None,
        covariate_sizes: dict[str, int] | None = None,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.n_contexts = n_contexts
        self.n_perturbations = n_perturbations
        self.cond_emb_dim = cond_emb_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.backbone_type = backbone_type
        self.use_conditioning = bool(use_conditioning)
        self.covariate_col_order: tuple[str, ...] = (
            tuple(covariate_sizes.keys()) if covariate_sizes else ()
        )
        self.n_covariates = len(self.covariate_col_order)

        if self.use_conditioning:
            self.context_emb: nn.Embedding | None = nn.Embedding(n_contexts, cond_emb_dim)
            self.pert_emb: nn.Embedding | None = nn.Embedding(n_perturbations, cond_emb_dim)
            if covariate_sizes:
                self.covariate_embs = nn.ModuleDict(
                    {col: nn.Embedding(n_cat, cond_emb_dim) for col, n_cat in covariate_sizes.items()}
                )
            else:
                self.covariate_embs = nn.ModuleDict()
        else:
            self.context_emb = None
            self.pert_emb = None
            self.covariate_embs = nn.ModuleDict()
            if self.covariate_col_order:
                raise ValueError("covariates require use_conditioning=True")

        in_dim = flow_map_backbone_in_dim(
            data_dim=data_dim,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim,
            n_covariates=self.n_covariates,
            use_conditioning=self.use_conditioning,
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
        if not self.use_conditioning:
            return
        if (
            teacher.n_contexts != self.n_contexts
            or teacher.n_perturbations != self.n_perturbations
            or teacher.cond_emb_dim != self.cond_emb_dim
            or teacher.covariate_col_order != self.covariate_col_order
        ):
            raise ValueError("teacher embedding shapes must match mean flow")
        assert self.context_emb is not None and self.pert_emb is not None
        with torch.no_grad():
            self.context_emb.weight.copy_(teacher.context_emb.weight)  # type: ignore[union-attr]
            self.pert_emb.weight.copy_(teacher.pert_emb.weight)  # type: ignore[union-attr]
            for col in self.covariate_col_order:
                self.covariate_embs[col].weight.copy_(teacher.covariate_embs[col].weight)

    def _embed(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None,
        p: torch.Tensor | None,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must be (B, {self.data_dim}), got {tuple(x.shape)}")
        b = x.shape[0]
        if r.shape != (b,) or t.shape != (b,):
            raise ValueError("r and t must be shape (B,)")
        re = sinusoidal_time_embedding(r, self.time_emb_dim)
        te = sinusoidal_time_embedding(t, self.time_emb_dim)
        parts = [x, re, te]
        if self.use_conditioning:
            if c is None or p is None:
                raise ValueError("c and p are required for conditional MeanFlow")
            if c.shape != (b,) or p.shape != (b,):
                raise ValueError("c and p must be shape (B,)")
            z = self._prepare_z(z, b, x.device)
            assert self.context_emb is not None and self.pert_emb is not None
            parts.extend([self.context_emb(c), self.pert_emb(p)])
            if self.covariate_col_order:
                parts.append(self._covariate_vector(z))
        return torch.cat(parts, dim=-1)

    def calc_u(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Average velocity u(x, r, t | c, p[, z])."""
        features = self._embed(x, r, t, c, p, z=z)
        return vector_field_module(self)(features)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`calc_u` (MeanFlow network output)."""
        return self.calc_u(x, r, t, c, p, z=z)

    def transport(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        z: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One MeanFlow jump: ``x_t = x_r + (t - r) * u(x_r, r, t)``."""
        dt = (t - r).unsqueeze(-1)
        return x + dt * self.calc_u(x, r, t, c, p, z=z)

    @torch.no_grad()
    def rollout_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        *,
        z: dict[str, torch.Tensor] | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Transport ``x0`` from ``t_start`` to ``t_end`` with ``n_steps`` MeanFlow jumps."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if not (0.0 <= t_start < t_end <= 1.0):
            raise ValueError(
                f"require 0 <= t_start < t_end <= 1, got t_start={t_start}, t_end={t_end}"
            )
        device = x0.device
        dtype = x0.dtype
        b = x0.shape[0]
        if self.use_conditioning:
            if c is None or p is None:
                raise ValueError("c and p required for conditional MeanFlow")
            c = c.to(device=device, dtype=torch.long)
            p = p.to(device=device, dtype=torch.long)
            z = self._prepare_z(z, b, device)
        else:
            c = None
            p = None
            z = None

        times = torch.linspace(
            float(t_start),
            float(t_end),
            n_steps + 1,
            device=device,
            dtype=dtype,
        )
        x = x0
        for i in range(n_steps):
            r = torch.full((b,), times[i].item(), device=device, dtype=dtype)
            t = torch.full((b,), times[i + 1].item(), device=device, dtype=dtype)
            x = self.transport(x, r, t, c, p, z=z)
        return x

    @torch.no_grad()
    def sample_from_x0(
        self,
        x0: torch.Tensor,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        *,
        z: dict[str, torch.Tensor] | None = None,
        t_end: float = 1.0,
        t_start: float = 0.0,
        n_steps: int = 1,
    ) -> torch.Tensor:
        return self.rollout_from_x0(
            x0, c, p, z=z, t_start=t_start, t_end=t_end, n_steps=n_steps
        )

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        *,
        z: dict[str, torch.Tensor] | None = None,
        batch_size: int | None = None,
        noise_scale: float = 1.0,
        t_end: float = 1.0,
        t_start: float = 0.0,
        n_steps: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from noise via ``n_steps`` MeanFlow jumps."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if self.use_conditioning:
            if c is None or p is None:
                raise ValueError("c and p required for conditional sample()")
            c = c.to(device=device, dtype=torch.long)
            p = p.to(device=device, dtype=torch.long)
            b = c.shape[0]
        else:
            if batch_size is not None:
                b = int(batch_size)
            elif c is not None:
                b = c.shape[0]
            else:
                raise ValueError("batch_size or c must be provided for unconditional sample()")
            c = None
            p = None
        x0 = noise_scale * torch.randn(
            b, self.data_dim, device=device, dtype=dtype, generator=generator
        )
        return self.sample_from_x0(
            x0, c, p, z=z, t_start=t_start, t_end=t_end, n_steps=n_steps
        )


def build_conditional_mean_flow_from_config(cfg: dict[str, Any]) -> ConditionalMeanFlow:
    """Reconstruct a MeanFlow model from a saved checkpoint config dict."""
    backbone_type = str(cfg.get("backbone_type", DEFAULT_BACKBONE_TYPE))
    covariate_sizes = cfg.get("covariate_sizes")
    if covariate_sizes is None and "encoding" in cfg:
        enc = cfg["encoding"]
        order = tuple(enc.get("covariate_col_order") or ())
        covariate_sizes = {
            col: len(enc["covariate_categories"][col]) for col in order
        }
    use_conditioning = bool(cfg.get("use_conditioning", True))
    return ConditionalMeanFlow(
        data_dim=int(cfg["data_dim"]),
        n_contexts=int(cfg["n_contexts"]),
        n_perturbations=int(cfg["n_perturbations"]),
        cond_emb_dim=int(cfg.get("cond_emb_dim", 32)),
        time_emb_dim=int(cfg["time_emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        backbone_type=backbone_type,
        covariate_sizes=dict(covariate_sizes) if covariate_sizes else None,
        use_conditioning=use_conditioning,
    )
