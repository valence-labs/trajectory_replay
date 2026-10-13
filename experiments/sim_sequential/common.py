"""Data utilities for the sequential Gaussian-ring experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_replay.data.mixture_mog import GaussianMixtureDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_COLORS = ("#780000", "#C1121F", "#FDF0D5", "#669BBC", "#003049")


def sequential_results_root(repo_root: Path | None = None) -> Path:
    """Root for sequential-simulation artifacts."""
    return (repo_root or REPO_ROOT) / "results_dir" / "sim_sequential"


def sequential_data_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    return sequential_results_root(repo_root) / "data" / f"dim{dim}"


def make_clockwise_ring_means(
    *,
    n_tasks: int = 5,
    dim: int = 2,
    radius: float = 3.0,
) -> np.ndarray:
    """Regular-polygon means, starting at the top and proceeding clockwise."""
    if n_tasks < 2:
        raise ValueError("n_tasks must be >= 2")
    if dim < 2:
        raise ValueError("dim must be >= 2 for a ring")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    # k=0 is π/2 (top); subtracting angles gives clockwise traversal.
    angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(n_tasks) / float(n_tasks)
    means = np.zeros((n_tasks, dim), dtype=np.float64)
    means[:, 0] = radius * np.cos(angles)
    means[:, 1] = radius * np.sin(angles)
    return means


def _sample_split(
    means: np.ndarray,
    *,
    samples_per_task: int,
    sigma: float,
    rng: np.random.Generator,
) -> GaussianMixtureDataset:
    n_tasks, dim = means.shape
    cov = float(sigma) ** 2 * np.eye(dim, dtype=np.float64)
    x = np.concatenate(
        [
            rng.multivariate_normal(mean, cov, size=samples_per_task)
            for mean in means
        ],
        axis=0,
    )
    task = np.repeat(np.arange(n_tasks, dtype=np.int64), samples_per_task)
    return GaussianMixtureDataset(x=x, task=task, means=means, cov=cov)


def build_dataset_bundle(
    *,
    n_tasks: int = 5,
    dim: int = 2,
    radius: float = 3.0,
    sigma: float = 0.35,
    train_per_task: int = 10_000,
    val_per_task: int = 2_000,
    test_per_task: int = 5_000,
    data_seed: int = 0,
) -> tuple[dict[str, GaussianMixtureDataset], dict[str, Any]]:
    """Create independent train, validation, and test samples for every task."""
    counts = {
        "train": int(train_per_task),
        "val": int(val_per_task),
        "test": int(test_per_task),
    }
    if any(count < 1 for count in counts.values()):
        raise ValueError("all split counts must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    means = make_clockwise_ring_means(n_tasks=n_tasks, dim=dim, radius=radius)
    seed_sequence = np.random.SeedSequence(data_seed)
    split_seeds = seed_sequence.spawn(len(counts))
    splits = {
        split_name: _sample_split(
            means,
            samples_per_task=counts[split_name],
            sigma=sigma,
            rng=np.random.default_rng(split_seed),
        )
        for split_name, split_seed in zip(counts, split_seeds, strict=True)
    }
    nearest_neighbor_separation = 2.0 * radius * np.sin(np.pi / n_tasks)
    meta: dict[str, Any] = {
        "schema_version": 1,
        "n_tasks": int(n_tasks),
        "dim": int(dim),
        "radius": float(radius),
        "nearest_neighbor_separation": float(nearest_neighbor_separation),
        "sigma": float(sigma),
        "data_seed": int(data_seed),
        "task_order": list(range(n_tasks)),
        "ordering": "top_then_clockwise",
        "means": means.tolist(),
        "samples_per_task": counts,
    }
    return splits, meta


def save_dataset_bundle(
    out_dir: Path,
    splits: dict[str, GaussianMixtureDataset],
    meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, dataset in splits.items():
        np.savez_compressed(
            out_dir / f"{name}.npz",
            x=dataset.x,
            task=dataset.task,
            means=dataset.means,
            cov=dataset.cov,
        )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_dataset_bundle(
    data_dir: Path,
) -> tuple[dict[str, GaussianMixtureDataset], dict[str, Any]]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    splits: dict[str, GaussianMixtureDataset] = {}
    for name in ("train", "val", "test"):
        raw = np.load(data_dir / f"{name}.npz")
        splits[name] = GaussianMixtureDataset(
            x=raw["x"],
            task=raw["task"],
            means=raw["means"],
            cov=raw["cov"],
        )
    return splits, meta
