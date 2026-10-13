"""Distributional distances between sample clouds (true vs generated)."""

from __future__ import annotations

import numpy as np


def mean_l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between mean vectors of two samples, shape (n, d) and (m, d)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D arrays with the same number of columns.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    return float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)))


def energy_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int | None = 2000,
) -> float:
    """Energy distance between two i.i.d. samples, shape (n, d) and (m, d).

    Uses ``scipy.spatial.distance.cdist`` (pairwise ``(n, m)`` matrices) instead of
    broadcasting to ``(n, m, d)``, which OOMs once ambient dimension is moderate.
    Optionally subsamples each cloud (default 2000) for further safety.
    """
    from scipy.spatial.distance import cdist

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D arrays with the same number of columns.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    if max_samples is not None:
        rng = np.random.default_rng(0)
        if x.shape[0] > max_samples:
            x = x[rng.choice(x.shape[0], size=max_samples, replace=False)]
        if y.shape[0] > max_samples:
            y = y[rng.choice(y.shape[0], size=max_samples, replace=False)]
    d_xy = cdist(x, y, metric="euclidean")
    d_xx = cdist(x, x, metric="euclidean")
    d_yy = cdist(y, y, metric="euclidean")
    return float(2.0 * d_xy.mean() - d_xx.mean() - d_yy.mean())


def distribution_metrics(
    x_true: np.ndarray,
    x_gen: np.ndarray,
    *,
    dims: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    """MSE between means plus multivariate energy distance on ``dims``."""
    x_true = np.asarray(x_true, dtype=np.float64)
    x_gen = np.asarray(x_gen, dtype=np.float64)
    if dims is None:
        xt, xg = x_true, x_gen
    else:
        dim_list = [int(d) for d in dims]
        xt = x_true[:, dim_list]
        xg = x_gen[:, dim_list]
    mt = xt.mean(axis=0)
    mg = xg.mean(axis=0)
    return {
        "mse_mean_vectors": float(np.mean((mt - mg) ** 2)),
        "energy": energy_distance(xt, xg),
    }


def ode_stratum_metrics(
    x_true: np.ndarray,
    x_gen: np.ndarray,
    *,
    paired_mse: float,
    dims: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    """Per-stratum metrics for ODE eval (paired MSE + energy distance)."""
    return {"mse_to_true": float(paired_mse), **distribution_metrics(x_true, x_gen, dims=dims)}
