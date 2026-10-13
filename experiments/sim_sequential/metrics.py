"""Evaluation for the sequential Gaussian-ring experiment."""

from __future__ import annotations

import numpy as np

from trajectory_replay.data.mixture_mog import GaussianMixtureDataset
from trajectory_replay.eval.metrics import energy_distance
from trajectory_replay.eval.mixture_metrics import hard_mode_assignment


def categorical_kl(target: np.ndarray, estimate: np.ndarray) -> float:
    """Forward categorical KL, ``KL(target || estimate)``."""
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if target.shape != estimate.shape:
        raise ValueError("target and estimate must have the same shape")
    if np.any(target < 0) or np.any(estimate <= 0):
        raise ValueError("target must be non-negative and estimate strictly positive")
    if not np.isclose(target.sum(), 1.0) or not np.isclose(estimate.sum(), 1.0):
        raise ValueError("target and estimate must sum to one")
    active = target > 0
    return float(np.sum(target[active] * np.log(target[active] / estimate[active])))


def update_policy_weights(previous: np.ndarray, historical_weight: float) -> np.ndarray:
    """Unroll one replay update into task-level terminal weights."""
    previous = np.asarray(previous, dtype=np.float64)
    lam = float(historical_weight)
    if previous.ndim != 1 or previous.size < 1:
        raise ValueError("previous weights must be a non-empty vector")
    if not np.isclose(previous.sum(), 1.0):
        raise ValueError("previous weights must sum to one")
    if not 0.0 < lam < 1.0:
        raise ValueError("historical_weight must lie in (0, 1)")
    return np.concatenate([lam * previous, np.asarray([1.0 - lam])])


def balanced_reference(dataset: GaussianMixtureDataset, n_seen: int) -> np.ndarray:
    """Concatenate equal-size test sets for the first ``n_seen`` tasks."""
    return np.concatenate([dataset.subset(task_id).x for task_id in range(n_seen)], axis=0)


def evaluate_sequential_samples(
    x_gen: np.ndarray,
    test: GaussianMixtureDataset,
    *,
    n_seen: int,
    policy_weights: np.ndarray,
    mass_pseudocount: float = 0.5,
) -> dict:
    """Mass and geometry metrics after a sequential update."""
    if n_seen < 1 or n_seen > test.n_tasks:
        raise ValueError(f"n_seen must lie in [1, {test.n_tasks}]")
    if mass_pseudocount <= 0:
        raise ValueError("mass_pseudocount must be > 0")
    policy_weights = np.asarray(policy_weights, dtype=np.float64)
    if policy_weights.shape != (n_seen,):
        raise ValueError(f"policy_weights must have shape ({n_seen},)")

    means = test.means[:n_seen]
    labels = hard_mode_assignment(x_gen, means)
    counts = np.bincount(labels, minlength=n_seen).astype(np.float64)
    masses = (counts + mass_pseudocount) / (
        counts.sum() + mass_pseudocount * n_seen
    )
    uniform = np.full(n_seen, 1.0 / n_seen, dtype=np.float64)

    per_mode_energy: list[float] = []
    per_mode_count: list[int] = []
    per_task_distribution_energy: list[float] = []
    for task_id in range(n_seen):
        assigned = x_gen[labels == task_id]
        per_mode_count.append(int(assigned.shape[0]))
        per_task_distribution_energy.append(
            energy_distance(test.subset(task_id).x, x_gen)
        )
        if assigned.shape[0] == 0:
            per_mode_energy.append(float("nan"))
        else:
            per_mode_energy.append(energy_distance(test.subset(task_id).x, assigned))

    historical_energy = per_mode_energy[:-1]
    finite_historical = [x for x in historical_energy if np.isfinite(x)]
    overall_energy = energy_distance(balanced_reference(test, n_seen), x_gen)
    fit_distribution_energy = energy_distance(test.subset(n_seen - 1).x, x_gen)
    retention_distribution_energy = (
        energy_distance(balanced_reference(test, n_seen - 1), x_gen)
        if n_seen > 1
        else float("nan")
    )

    return {
        "n_seen": int(n_seen),
        "n_gen": int(x_gen.shape[0]),
        "mass_pseudocount": float(mass_pseudocount),
        "mass_counts": [int(x) for x in counts],
        "mass": masses.tolist(),
        "uniform_target_mass": uniform.tolist(),
        "policy_target_mass": policy_weights.tolist(),
        "mass_kl_uniform": categorical_kl(uniform, masses),
        "mass_kl_policy": categorical_kl(policy_weights, masses),
        "mass_l1_uniform": float(np.abs(uniform - masses).sum()),
        "energy_uniform_mixture": overall_energy,
        "per_mode_energy": per_mode_energy,
        "per_mode_count": per_mode_count,
        "per_task_distribution_energy": per_task_distribution_energy,
        "fit_mass": float(masses[-1]),
        "fit_mass_target_uniform": float(uniform[-1]),
        "fit_energy": float(per_mode_energy[-1]),
        "fit_distribution_energy": fit_distribution_energy,
        "retention_mass": float(masses[:-1].sum()) if n_seen > 1 else float("nan"),
        "retention_mass_target_uniform": (
            float(uniform[:-1].sum()) if n_seen > 1 else float("nan")
        ),
        "retention_energy_mean": (
            float(np.mean(finite_historical)) if finite_historical else float("nan")
        ),
        "retention_distribution_energy": retention_distribution_energy,
        "oldest_task_energy": float(per_mode_energy[0]),
    }
