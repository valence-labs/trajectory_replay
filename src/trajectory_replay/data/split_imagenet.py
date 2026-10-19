"""Split ImageNet-100 latent protocol (Masip / Avalanche-style class orders).

Loads precomputed SD-VAE latents from::

    <latent_dir>/train.pt
    <latent_dir>/val.pt

as produced by ``scripts/encode_imagenet_vae_latents.py``, then builds
``n_experiences`` class-incremental tasks on the 100 CMC classes.

**Default protocol (image_cl):** ``n_experiences=3`` → nearly-equal chunks
**34 / 33 / 33** (100 is not divisible by 3). The older Masip-matched
**5 × 20** split remains available via ``n_experiences=5``.

Class order matches Avalanche ``NCScenario`` shuffling::

    torch.random.manual_seed(seed)
    order = arange(n_classes)[torch.randperm(n_classes)]

Chunked into contiguous blocks (nearly equal when ``n_classes`` is not
divisible by ``n_experiences``; remainder goes to the first tasks), with
classes sorted within each task (only the set matters). Official ImageNet
``val`` latents are used as ``x_test``; a ``val_fraction`` slice of the
train pool is held out as ``x_val`` (same pattern as Split-FMNIST /
Split-CIFAR10).

``TaskSplit.x_*`` are latent tensors ``(N, 4, 32, 32)`` in float32 (scaled
posterior means). Labels ``y_*`` are local ImageFolder indices ``0..99`` into
the cached ``synsets`` list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from trajectory_replay.data.split_fmnist import TaskSplit

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LATENT_DIR = _REPO_ROOT / "data_dir" / "imagenet100" / "latents_sdvae_ft_mse_256"

# Recorded Avalanche-compatible orders for Masip seeds (5 × 20 on 100 classes).
# Within-task lists are sorted; regenerated via ``_avalanche_nc_class_tasks``.
SPLIT_IMAGENET100_TASKS_E5: dict[int, list[list[int]]] = {
    42: [
        [10, 11, 13, 16, 19, 24, 42, 46, 52, 60, 62, 68, 70, 71, 78, 95, 96, 97, 98, 99],
        [1, 5, 9, 18, 32, 38, 40, 43, 50, 59, 67, 73, 75, 79, 82, 83, 86, 89, 93, 94],
        [3, 4, 6, 12, 20, 21, 22, 33, 35, 37, 44, 45, 48, 51, 58, 61, 63, 66, 74, 87],
        [7, 15, 17, 27, 29, 31, 34, 36, 41, 47, 56, 57, 65, 69, 72, 80, 81, 84, 85, 88],
        [0, 2, 8, 14, 23, 25, 26, 28, 30, 39, 49, 53, 54, 55, 64, 76, 77, 90, 91, 92],
    ],
    69: [
        [3, 5, 16, 25, 29, 34, 45, 49, 50, 58, 60, 69, 70, 75, 76, 91, 92, 94, 97, 99],
        [0, 2, 8, 10, 17, 22, 31, 32, 33, 35, 46, 52, 56, 62, 71, 79, 83, 84, 85, 93],
        [4, 13, 14, 15, 18, 19, 20, 28, 37, 40, 42, 43, 44, 47, 64, 65, 68, 72, 74, 98],
        [7, 9, 11, 12, 24, 27, 41, 48, 51, 53, 55, 57, 61, 66, 67, 77, 82, 89, 95, 96],
        [1, 6, 21, 23, 26, 30, 36, 38, 39, 54, 59, 63, 73, 78, 80, 81, 86, 87, 88, 90],
    ],
    1714: [
        [0, 6, 10, 17, 25, 31, 32, 39, 43, 45, 48, 50, 55, 62, 65, 71, 80, 83, 84, 87],
        [3, 13, 14, 15, 23, 26, 33, 40, 42, 49, 51, 54, 63, 73, 74, 81, 85, 95, 96, 99],
        [5, 7, 11, 12, 18, 20, 24, 29, 36, 44, 47, 53, 60, 67, 68, 76, 78, 82, 90, 92],
        [2, 4, 8, 16, 22, 28, 30, 34, 35, 46, 56, 57, 58, 66, 69, 75, 77, 86, 88, 93],
        [1, 9, 19, 21, 27, 37, 38, 41, 52, 59, 61, 64, 70, 72, 79, 89, 91, 94, 97, 98],
    ],
}

# Backward-compatible alias (Masip 5×20 table).
SPLIT_IMAGENET100_TASKS = SPLIT_IMAGENET100_TASKS_E5


def _nearly_equal_chunk_sizes(n_classes: int, n_experiences: int) -> list[int]:
    """Class counts per task; remainder distributed to the first tasks.

    Example: 100 classes / 3 tasks → ``[34, 33, 33]``.
    """
    base = n_classes // n_experiences
    rem = n_classes % n_experiences
    return [base + (1 if i < rem else 0) for i in range(n_experiences)]


def _avalanche_nc_class_tasks(
    seed: int,
    *,
    n_classes: int,
    n_experiences: int,
) -> list[list[int]]:
    """Avalanche ``NCScenario`` shuffle + nearly-equal chunking; classes sorted per task."""
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if n_experiences < 1:
        raise ValueError(f"n_experiences must be >= 1, got {n_experiences}")
    if n_experiences > n_classes:
        raise ValueError(
            f"n_experiences={n_experiences} exceeds n_classes={n_classes}"
        )
    order = list(range(n_classes))
    # Match Avalanche: mutate global RNG, then randperm. Restore afterwards.
    rng_state = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(int(seed))
        order = torch.as_tensor(order)[torch.randperm(len(order))].tolist()
    finally:
        torch.random.set_rng_state(rng_state)
    sizes = _nearly_equal_chunk_sizes(n_classes, n_experiences)
    out: list[list[int]] = []
    idx = 0
    for size in sizes:
        out.append(sorted(order[idx : idx + size]))
        idx += size
    return out


def class_order_for_seed(
    seed: int,
    *,
    n_classes: int = 100,
    n_experiences: int = 3,
) -> list[list[int]]:
    """Return per-task class ID lists for ``seed``.

    For the recorded Masip seeds with the 5×20 protocol, returns the frozen
    table (must match ``_avalanche_nc_class_tasks``). Other
    ``(seed, n_classes, n_experiences)`` triples are computed on the fly
    (including the default image_cl 3-task / 34-33-33 protocol).
    """
    if (
        n_classes == 100
        and n_experiences == 5
        and seed in SPLIT_IMAGENET100_TASKS_E5
    ):
        return [list(task) for task in SPLIT_IMAGENET100_TASKS_E5[seed]]
    return _avalanche_nc_class_tasks(
        seed, n_classes=n_classes, n_experiences=n_experiences
    )


def _split_indices(
    n: int,
    *,
    val_fraction: float,
    seed: int,
    task_id: int,
) -> tuple[list[int], list[int]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(task_id))
    perm = torch.randperm(n, generator=g).tolist()
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    return perm[n_val:], perm[:n_val]


def _load_latent_split(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing latent cache {path}. Run scripts/encode_imagenet_vae_latents.py first."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("z", "y", "synsets"):
        if key not in blob:
            raise KeyError(f"{path} missing required key {key!r}")
    z = blob["z"]
    y = blob["y"].long()
    if z.ndim != 4 or z.shape[1] != 4:
        raise ValueError(f"expected z shaped (N, 4, H, W), got {tuple(z.shape)}")
    if y.shape[0] != z.shape[0]:
        raise ValueError(f"z/y length mismatch: {z.shape[0]} vs {y.shape[0]}")
    return {
        "z": z.contiguous(),
        "y": y.contiguous(),
        "synsets": list(blob["synsets"]),
        "scaling_factor": float(blob.get("scaling_factor", 0.18215)),
        "vae_id": str(blob.get("vae_id", "")),
        "image_size": int(blob.get("image_size", 256)),
    }


def resolve_imagenet100_latent_dir(latent_dir: str | Path | None = None) -> Path:
    return Path(latent_dir) if latent_dir is not None else DEFAULT_LATENT_DIR


def build_split_imagenet100_tasks(
    *,
    seed: int,
    latent_dir: str | Path | None = None,
    n_experiences: int = 3,
    val_fraction: float = 0.1,
    n_classes: int = 100,
) -> list[TaskSplit]:
    """Build ImageNet-100 latent TaskSplits from cached ``train.pt`` / ``val.pt``.

    Default ``n_experiences=3`` (34/33/33 classes). Pass ``5`` for the Masip
    5×20 protocol.
    """
    root = resolve_imagenet100_latent_dir(latent_dir)
    train_blob = _load_latent_split(root / "train.pt")
    test_blob = _load_latent_split(root / "val.pt")  # official val → test

    if train_blob["synsets"] != test_blob["synsets"]:
        raise ValueError(
            f"train/val synset lists differ under {root}; re-encode both splits together."
        )
    synsets = train_blob["synsets"]
    if len(synsets) != n_classes:
        raise ValueError(
            f"expected {n_classes} synsets in cache, found {len(synsets)} under {root}"
        )

    task_classes = class_order_for_seed(
        seed, n_classes=n_classes, n_experiences=n_experiences
    )
    z_train = train_blob["z"]
    y_train = train_blob["y"]
    z_test = test_blob["z"]
    y_test = test_blob["y"]

    out: list[TaskSplit] = []
    for task_id, classes in enumerate(task_classes):
        class_set = {int(c) for c in classes}
        if len(class_set) != len(classes):
            raise ValueError(f"duplicate classes in task {task_id}: {classes}")
        class_tensor = torch.tensor(sorted(class_set), dtype=torch.long)
        train_pool_idx = torch.isin(y_train, class_tensor).nonzero(as_tuple=False).view(-1).tolist()
        test_idx = torch.isin(y_test, class_tensor).nonzero(as_tuple=False).view(-1).tolist()
        if len(train_pool_idx) < 2:
            raise RuntimeError(f"task {task_id}: need >=2 train samples, got {len(train_pool_idx)}")
        if not test_idx:
            raise RuntimeError(f"task {task_id}: empty test split for classes {sorted(class_set)}")

        train_rel, val_rel = _split_indices(
            len(train_pool_idx),
            val_fraction=val_fraction,
            seed=seed,
            task_id=task_id,
        )
        train_idx = [train_pool_idx[i] for i in train_rel]
        val_idx = [train_pool_idx[i] for i in val_rel]

        # float32 for FM training; keep CPU tensors (loaders move to device).
        out.append(
            TaskSplit(
                task_id=task_id,
                classes=tuple(sorted(class_set)),
                x_train=z_train[train_idx].to(dtype=torch.float32),
                y_train=y_train[train_idx].clone(),
                x_val=z_train[val_idx].to(dtype=torch.float32),
                y_val=y_train[val_idx].clone(),
                x_test=z_test[test_idx].to(dtype=torch.float32),
                y_test=y_test[test_idx].clone(),
            )
        )
    return out


__all__ = [
    "DEFAULT_LATENT_DIR",
    "SPLIT_IMAGENET100_TASKS",
    "SPLIT_IMAGENET100_TASKS_E5",
    "build_split_imagenet100_tasks",
    "class_order_for_seed",
    "resolve_imagenet100_latent_dir",
]
