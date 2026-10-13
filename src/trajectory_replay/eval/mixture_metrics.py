"""Metrics for unconditional 2-mode mixture recovery."""

from __future__ import annotations

import numpy as np

from trajectory_replay.eval.metrics import energy_distance

Array = np.ndarray


def active_dims_from_means(means: Array, *, atol: float = 1e-8) -> list[int]:
    """Coordinates where mode means differ (signal subspace for this MoG)."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2 or means.shape[0] < 2:
        raise ValueError("means must have shape (k, d) with k >= 2")
    spread = means.max(axis=0) - means.min(axis=0)
    dims = np.where(spread > atol)[0].tolist()
    if not dims:
        # Degenerate: fall back to all coordinates.
        return list(range(means.shape[1]))
    return dims


def _slice_dims(x: Array, dims: list[int]) -> Array:
    return np.asarray(x, dtype=np.float64)[:, dims]


def hard_mode_assignment(x: Array, means: Array) -> Array:
    """Nearest-mean labels, shape ``(n,)``."""
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    # (n, k)
    d2 = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1).astype(np.int64)


def estimate_pi0(x_gen: Array, means: Array) -> float:
    """Hard-assignment estimate of mass on mode 0."""
    labels = hard_mode_assignment(x_gen, means)
    return float(np.mean(labels == 0))


def isotropic_gaussian_logpdf(x: Array, mean: Array, sigma: float) -> Array:
    """Per-point log-density of ``N(mean, sigma^2 I)``, shape ``(n,)``."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    d = x.shape[1]
    r2 = ((x - mean) ** 2).sum(axis=-1)
    return -0.5 * d * np.log(2.0 * np.pi) - d * np.log(sigma) - 0.5 * r2 / (sigma**2)


def mixture_mean_loglik(
    x: Array,
    *,
    means: Array,
    sigma: float,
    lambda_mix: float,
) -> float:
    """Mean log-likelihood under the true 2-Gaussian mixture ``p_λ``."""
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    lam = float(lambda_mix)
    if lam <= 0.0:
        return float(np.mean(isotropic_gaussian_logpdf(x, means[1], sigma)))
    if lam >= 1.0:
        return float(np.mean(isotropic_gaussian_logpdf(x, means[0], sigma)))
    log_p0 = np.log(lam) + isotropic_gaussian_logpdf(x, means[0], sigma)
    log_p1 = np.log(1.0 - lam) + isotropic_gaussian_logpdf(x, means[1], sigma)
    return float(np.mean(np.logaddexp(log_p0, log_p1)))


def per_mode_energy(
    x_gen: Array,
    x_true_by_mode: list[Array],
    means: Array,
    *,
    active_dims: list[int] | None = None,
) -> dict[str, float]:
    """Energy distance on hard-assigned gens vs true samples per mode.

    Reports ambient energy plus energy restricted to ``active_dims`` (coordinates
    where mode means differ). In high-d MoGs the noise axes dominate ambient
    energy; the active subspace tracks mode geometry / bridging.
    """
    labels = hard_mode_assignment(x_gen, means)
    means = np.asarray(means, dtype=np.float64)
    if active_dims is None:
        active_dims = active_dims_from_means(means)
    out: dict[str, float] = {
        "n_active_dims": float(len(active_dims)),
    }
    for k, x_true in enumerate(x_true_by_mode):
        sel = x_gen[labels == k]
        key = f"energy_mode{k}"
        if sel.shape[0] == 0 or x_true.shape[0] == 0:
            out[key] = float("nan")
            out[f"{key}_active"] = float("nan")
            out[f"n_assigned_mode{k}"] = float(sel.shape[0])
            continue
        out[key] = energy_distance(x_true, sel)
        out[f"{key}_active"] = energy_distance(
            _slice_dims(x_true, active_dims), _slice_dims(sel, active_dims)
        )
        out[f"n_assigned_mode{k}"] = float(sel.shape[0])
        out[f"mean_l2_mode{k}"] = float(
            np.linalg.norm(sel.mean(axis=0) - means[k])
        )
        out[f"mean_l2_mode{k}_active"] = float(
            np.linalg.norm(
                _slice_dims(sel, active_dims).mean(axis=0)
                - means[k, active_dims]
            )
        )
    return out


def mixture_recovery_metrics(
    x_gen: Array,
    *,
    means: Array,
    x_true_mode0: Array,
    x_true_mode1: Array,
    lambda_target: float | None = None,
    sigma: float | None = None,
    x_true_mixture: Array | None = None,
) -> dict[str, float]:
    """Mode mass + per-mode fidelity (+ optional mixture loglik / energy)."""
    means = np.asarray(means, dtype=np.float64)
    active_dims = active_dims_from_means(means)
    pi0 = estimate_pi0(x_gen, means)
    out: dict[str, float] = {
        "pi0_hat": pi0,
        "pi1_hat": 1.0 - pi0,
        "n_gen": float(x_gen.shape[0]),
    }
    if lambda_target is not None:
        out["lambda_target"] = float(lambda_target)
        out["pi0_abs_error"] = float(abs(pi0 - lambda_target))
        if sigma is not None:
            out["mean_loglik"] = mixture_mean_loglik(
                x_gen,
                means=means,
                sigma=float(sigma),
                lambda_mix=float(lambda_target),
            )
    if x_true_mixture is not None and np.asarray(x_true_mixture).shape[0] > 0:
        x_mix = np.asarray(x_true_mixture, dtype=np.float64)
        out["energy_mixture"] = energy_distance(x_mix, x_gen)
        out["energy_mixture_active"] = energy_distance(
            _slice_dims(x_mix, active_dims), _slice_dims(x_gen, active_dims)
        )
    out.update(
        per_mode_energy(
            x_gen, [x_true_mode0, x_true_mode1], means, active_dims=active_dims
        )
    )
    return out
