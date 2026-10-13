"""Unconditional Gaussian mixture for continual-learning simulations.

Each component is a task. Samples are drawn i.i.d. from N(μ_k, σ²I); the
integer task label is stored only for train/finetune splits — models in the
mixture track are trained unconditionally.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class GaussianMixtureDataset:
    """Unconditional MoG samples with optional task ids (component indices)."""

    x: Array
    task: Array
    means: Array
    cov: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=np.float64))
        object.__setattr__(self, "task", np.asarray(self.task, dtype=np.int64))
        object.__setattr__(self, "means", np.asarray(self.means, dtype=np.float64))
        object.__setattr__(self, "cov", np.asarray(self.cov, dtype=np.float64))
        if self.x.ndim != 2:
            raise ValueError(f"x must be 2D (n, d), got {self.x.shape}")
        if self.task.shape != (self.x.shape[0],):
            raise ValueError("task must have shape (n,) matching x.")
        if self.means.ndim != 2 or self.means.shape[1] != self.x.shape[1]:
            raise ValueError("means must be (n_tasks, d).")
        d = self.x.shape[1]
        if self.cov.shape != (d, d):
            raise ValueError(f"cov must be ({d}, {d}), got {self.cov.shape}")

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_tasks(self) -> int:
        return int(self.means.shape[0])

    def mask_task(self, task_id: int) -> Array:
        return self.task == int(task_id)

    def subset(self, task_id: int) -> GaussianMixtureDataset:
        m = self.mask_task(task_id)
        return GaussianMixtureDataset(
            x=self.x[m],
            task=self.task[m],
            means=self.means,
            cov=self.cov,
        )


def make_two_mode_means(
    *,
    dim: int = 2,
    separation: float = 3.0,
    axis: int = 0,
) -> Array:
    """Two means along ``axis``, symmetric about the origin at ±separation/2."""
    if dim < 1:
        raise ValueError("dim must be >= 1.")
    if axis < 0 or axis >= dim:
        raise ValueError(f"axis must be in [0, {dim - 1}], got {axis}")
    means = np.zeros((2, dim), dtype=np.float64)
    means[0, axis] = -0.5 * float(separation)
    means[1, axis] = 0.5 * float(separation)
    return means


def make_polygon_means(
    n_modes: int,
    *,
    dim: int = 2,
    separation: float | None = None,
    radius: float | None = None,
) -> Array:
    """``n_modes`` means on a 2D regular polygon.

    Prefer ``radius`` (distance from the origin / noise prior to each vertex) when
    comparing across ``n_modes`` — that keeps transport length fixed. If only
    ``separation`` is given (or neither; default separation 3.0), the radius is
    derived so nearest-neighbor chord length equals ``separation`` (legacy).
    Passing both is an error.

    For ``dim > 2``, remaining coordinates are zero (signal lives in the first two axes).
    """
    if n_modes < 2:
        raise ValueError("n_modes must be >= 2.")
    if dim < 2:
        raise ValueError("dim must be >= 2 for polygon means.")
    if radius is not None and separation is not None:
        raise ValueError("Pass only one of radius= or separation=, not both.")
    if radius is not None:
        if float(radius) <= 0:
            raise ValueError("radius must be > 0.")
        r = float(radius)
    else:
        sep = 3.0 if separation is None else float(separation)
        if sep <= 0:
            raise ValueError("separation must be > 0.")
        # Chord length between adjacent vertices: 2 R sin(π/n) = separation.
        r = sep / (2.0 * np.sin(np.pi / float(n_modes)))
    angles = 2.0 * np.pi * np.arange(n_modes, dtype=np.float64) / float(n_modes)
    means = np.zeros((n_modes, dim), dtype=np.float64)
    means[:, 0] = r * np.cos(angles)
    means[:, 1] = r * np.sin(angles)
    return means


def sample_gaussian_mixture(
    means: Array,
    *,
    samples_per_task: int,
    rng: np.random.Generator,
    sigma: Union[float, Array] = 0.35,
) -> GaussianMixtureDataset:
    """Draw ``samples_per_task`` i.i.d. points from each N(μ_k, Σ)."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError(f"means must be 2D (n_tasks, d), got {means.shape}")
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be >= 1.")
    n_tasks, dim = means.shape
    cov = _as_cov(sigma, dim)
    total = n_tasks * samples_per_task
    x = np.empty((total, dim), dtype=np.float64)
    task = np.empty(total, dtype=np.int64)
    row = 0
    for k in range(n_tasks):
        slab = rng.multivariate_normal(means[k], cov, size=samples_per_task)
        x[row : row + samples_per_task] = slab
        task[row : row + samples_per_task] = k
        row += samples_per_task
    return GaussianMixtureDataset(x=x, task=task, means=means, cov=cov)


def sample_mixture_with_weights(
    means: Array,
    weights: Array,
    *,
    n_samples: int,
    rng: np.random.Generator,
    sigma: Union[float, Array] = 0.35,
) -> GaussianMixtureDataset:
    """Draw ``n_samples`` from Σ_k w_k N(μ_k, Σ) (for mixture-recovery checks)."""
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError(f"means must be 2D (n_tasks, d), got {means.shape}")
    if weights.shape != (means.shape[0],):
        raise ValueError("weights must have shape (n_tasks,).")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to 1.")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    n_tasks, dim = means.shape
    cov = _as_cov(sigma, dim)
    task = rng.choice(n_tasks, size=n_samples, p=weights)
    x = np.empty((n_samples, dim), dtype=np.float64)
    for k in range(n_tasks):
        idx = np.where(task == k)[0]
        if idx.size == 0:
            continue
        x[idx] = rng.multivariate_normal(means[k], cov, size=idx.size)
    return GaussianMixtureDataset(x=x, task=task, means=means, cov=cov)
