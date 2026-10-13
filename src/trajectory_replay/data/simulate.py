"""Gaussian single-cell-style simulation: x ~ N(μ(c, p), Σ).

Context c (categorical) sets a baseline mean; perturbation p shifts it. The map
(c, p) ↦ μ is pluggable so you can swap additive, rotated, or interaction
couplings without changing the sampler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np

Array = np.ndarray


def _as_cov(sigma: Union[float, Array], dim: int) -> Array:
    if np.ndim(sigma) == 0:
        return float(sigma) ** 2 * np.eye(dim, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (dim, dim):
        raise ValueError(f"sigma must be scalar or ({dim}, {dim}), got {sigma.shape}")
    return sigma


def random_orthogonal(rng: np.random.Generator, dim: int) -> Array:
    """Haar-ish orthogonal matrix via QR of a standard Gaussian matrix."""
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s


class MeanCoupling(ABC):
    """Maps discrete context c and perturbation p to a d-dimensional mean."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def n_contexts(self) -> int:
        ...

    @property
    @abstractmethod
    def n_perturbations(self) -> int:
        ...

    @abstractmethod
    def mean(self, c: int, p: int) -> Array:
        ...


@dataclass(frozen=True)
class AdditiveCoupling(MeanCoupling):
    """μ_{c,p} = μ_c + δ_p."""

    mu_context: Array
    delta_pert: Array

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu_context, dtype=np.float64)
        dlt = np.asarray(self.delta_pert, dtype=np.float64)
        object.__setattr__(self, "mu_context", mu)
        object.__setattr__(self, "delta_pert", dlt)
        if mu.ndim != 2 or dlt.ndim != 2:
            raise ValueError("mu_context and delta_pert must be 2D (n_c, d) and (n_p, d).")
        if mu.shape[1] != dlt.shape[1]:
            raise ValueError("mu_context and delta_pert must share the trailing dimension d.")

    @property
    def dim(self) -> int:
        return int(self.mu_context.shape[1])

    @property
    def n_contexts(self) -> int:
        return int(self.mu_context.shape[0])

    @property
    def n_perturbations(self) -> int:
        return int(self.delta_pert.shape[0])

    def mean(self, c: int, p: int) -> Array:
        return self.mu_context[c] + self.delta_pert[p]


@dataclass(frozen=True)
class RotatedAdditiveCoupling(MeanCoupling):
    """μ_{c,p} = μ_c + R_c @ δ_p with R_c orthogonal (perturbation direction depends on c)."""

    mu_context: Array
    delta_pert: Array
    rotation_per_context: Array

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu_context, dtype=np.float64)
        dlt = np.asarray(self.delta_pert, dtype=np.float64)
        r = np.asarray(self.rotation_per_context, dtype=np.float64)
        object.__setattr__(self, "mu_context", mu)
        object.__setattr__(self, "delta_pert", dlt)
        object.__setattr__(self, "rotation_per_context", r)
        d = mu.shape[1]
        if mu.shape[1] != dlt.shape[1]:
            raise ValueError("mu_context and delta_pert must share dimension d.")
        if r.shape != (mu.shape[0], d, d):
            raise ValueError("rotation_per_context must have shape (n_c, d, d).")

    @property
    def dim(self) -> int:
        return int(self.mu_context.shape[1])

    @property
    def n_contexts(self) -> int:
        return int(self.mu_context.shape[0])

    @property
    def n_perturbations(self) -> int:
        return int(self.delta_pert.shape[0])

    def mean(self, c: int, p: int) -> Array:
        return self.mu_context[c] + self.rotation_per_context[c] @ self.delta_pert[p]


@dataclass(frozen=True)
class HadamardInteractionCoupling(MeanCoupling):
    """μ_{c,p} = μ_c + δ_p + u_c ⊙ v_p (elementwise interaction)."""

    mu_context: Array
    delta_pert: Array
    u_context: Array
    v_perturb: Array

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu_context, dtype=np.float64)
        dlt = np.asarray(self.delta_pert, dtype=np.float64)
        u = np.asarray(self.u_context, dtype=np.float64)
        v = np.asarray(self.v_perturb, dtype=np.float64)
        object.__setattr__(self, "mu_context", mu)
        object.__setattr__(self, "delta_pert", dlt)
        object.__setattr__(self, "u_context", u)
        object.__setattr__(self, "v_perturb", v)
        if mu.shape[1] != dlt.shape[1]:
            raise ValueError("mu_context and delta_pert must share dimension d.")
        if u.shape != mu.shape or v.shape != dlt.shape:
            raise ValueError("u_context must match mu_context; v_perturb must match delta_pert.")

    @property
    def dim(self) -> int:
        return int(self.mu_context.shape[1])

    @property
    def n_contexts(self) -> int:
        return int(self.mu_context.shape[0])

    @property
    def n_perturbations(self) -> int:
        return int(self.delta_pert.shape[0])

    def mean(self, c: int, p: int) -> Array:
        return self.mu_context[c] + self.delta_pert[p] + self.u_context[c] * self.v_perturb[p]


@dataclass(frozen=True)
class GaussianSCDataset:
    """Samples x with integer indices c, p for context and perturbation."""

    x: Array
    c: Array
    p: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=np.float64))
        object.__setattr__(self, "c", np.asarray(self.c, dtype=np.int64))
        object.__setattr__(self, "p", np.asarray(self.p, dtype=np.int64))


def sample_gaussian_sc(
    coupling: MeanCoupling,
    *,
    samples_per_combo: int,
    rng: np.random.Generator,
    sigma: Union[float, Array] = 1.0,
) -> GaussianSCDataset:
    """Draw i.i.d. x ~ N(μ(c,p), Σ) for every (c, p), `samples_per_combo` times each."""
    if samples_per_combo < 1:
        raise ValueError("samples_per_combo must be >= 1.")
    d = coupling.dim
    cov = _as_cov(sigma, d)
    n_c, n_p = coupling.n_contexts, coupling.n_perturbations
    total = n_c * n_p * samples_per_combo
    x = np.empty((total, d), dtype=np.float64)
    c_idx = np.empty(total, dtype=np.int64)
    p_idx = np.empty(total, dtype=np.int64)
    row = 0
    for c in range(n_c):
        for p in range(n_p):
            mu = coupling.mean(c, p)
            slab = rng.multivariate_normal(mu, cov, size=samples_per_combo)
            x[row : row + samples_per_combo] = slab
            c_idx[row : row + samples_per_combo] = c
            p_idx[row : row + samples_per_combo] = p
            row += samples_per_combo
    return GaussianSCDataset(x=x, c=c_idx, p=p_idx)


def mean_grid(coupling: MeanCoupling) -> tuple[Array, Array, Array]:
    """Return (c_grid, p_grid, mu) with mu[k] = mean(c_grid[k], p_grid[k]) for all pairs."""
    n_c, n_p = coupling.n_contexts, coupling.n_perturbations
    c_grid = np.repeat(np.arange(n_c, dtype=np.int64), n_p)
    p_grid = np.tile(np.arange(n_p, dtype=np.int64), n_c)
    mu = np.stack([coupling.mean(int(c), int(p)) for c, p in zip(c_grid, p_grid)])
    return c_grid, p_grid, mu


def make_additive_coupling(
    rng: np.random.Generator,
    *,
    n_contexts: int,
    n_perturbations: int,
    dim: int,
    context_scale: float = 1.0,
    pert_scale: float = 0.5,
) -> AdditiveCoupling:
    mu = rng.normal(0.0, context_scale, size=(n_contexts, dim))
    delta = rng.normal(0.0, pert_scale, size=(n_perturbations, dim))
    return AdditiveCoupling(mu_context=mu, delta_pert=delta)


def make_rotated_additive_coupling(
    rng: np.random.Generator,
    *,
    n_contexts: int,
    n_perturbations: int,
    dim: int,
    context_scale: float = 1.0,
    pert_scale: float = 0.5,
) -> RotatedAdditiveCoupling:
    base = make_additive_coupling(
        rng,
        n_contexts=n_contexts,
        n_perturbations=n_perturbations,
        dim=dim,
        context_scale=context_scale,
        pert_scale=pert_scale,
    )
    rotations = np.stack([random_orthogonal(rng, dim) for _ in range(n_contexts)])
    return RotatedAdditiveCoupling(
        mu_context=base.mu_context,
        delta_pert=base.delta_pert,
        rotation_per_context=rotations,
    )


def make_hadamard_interaction_coupling(
    rng: np.random.Generator,
    *,
    n_contexts: int,
    n_perturbations: int,
    dim: int,
    context_scale: float = 1.0,
    pert_scale: float = 0.5,
    interaction_scale: float = 0.35,
) -> HadamardInteractionCoupling:
    mu = rng.normal(0.0, context_scale, size=(n_contexts, dim))
    delta = rng.normal(0.0, pert_scale, size=(n_perturbations, dim))
    u = rng.normal(0.0, interaction_scale, size=(n_contexts, dim))
    v = rng.normal(0.0, interaction_scale, size=(n_perturbations, dim))
    return HadamardInteractionCoupling(
        mu_context=mu, delta_pert=delta, u_context=u, v_perturb=v
    )
