"""Split CIFAR-10 protocol matching Masip / Avalanche class orders.

Produces per-task train / validation / test tensors in ``[-1, 1]`` at
``image_size`` x ``image_size``.  The class sets were verified from
``avalanche.benchmarks.SplitCIFAR10(n_experiences=5, seed=...)``; Avalanche
uses the same seeded PyTorch class permutation as Split-FMNIST.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torchvision import datasets, transforms

from trajectory_replay.data.split_fmnist import TaskSplit


# Canonicalized within each experience because only the class set matters.
AVALANCHE_SPLIT_CIFAR10_TASKS: dict[int, list[list[int]]] = {
    42: [[2, 6], [1, 8], [4, 5], [0, 9], [3, 7]],
    69: [[0, 2], [3, 5], [1, 4], [6, 9], [7, 8]],
    1714: [[3, 7], [0, 8], [1, 4], [2, 6], [5, 9]],
}


def class_order_for_seed(seed: int) -> list[list[int]]:
    if seed not in AVALANCHE_SPLIT_CIFAR10_TASKS:
        known = sorted(AVALANCHE_SPLIT_CIFAR10_TASKS)
        raise KeyError(
            f"No recorded Avalanche SplitCIFAR10 class order for seed={seed}. "
            f"Known seeds: {known}. Add the order after probing SplitCIFAR10."
        )
    return [list(task) for task in AVALANCHE_SPLIT_CIFAR10_TASKS[seed]]


def _cifar10_transform(image_size: int) -> transforms.Compose:
    # Train-only horizontal flips are applied by make_x_loader each epoch.
    steps: list[object] = [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(steps)


def _stack_subset(
    ds: datasets.CIFAR10,
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for i in indices:
        x, y = ds[i]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


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


def build_split_cifar10_tasks(
    *,
    seed: int,
    data_root: str | Path,
    image_size: int = 32,
    n_experiences: int = 5,
    val_fraction: float = 0.1,
    download: bool = True,
) -> list[TaskSplit]:
    """Build five two-class CIFAR-10 tasks with per-task validation splits."""
    if n_experiences != 5:
        raise ValueError("Only n_experiences=5 is supported (Masip CIFAR-10 protocol).")
    task_classes = class_order_for_seed(seed)

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=download,
        transform=_cifar10_transform(image_size),
    )
    # Validation uses Masip's deterministic eval transform, not train augmentation.
    val_ds = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=False,
        transform=_cifar10_transform(image_size),
    )
    test_ds = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=download,
        transform=_cifar10_transform(image_size),
    )

    train_targets = torch.as_tensor(train_ds.targets, dtype=torch.long)
    test_targets = torch.as_tensor(test_ds.targets, dtype=torch.long)
    out: list[TaskSplit] = []
    for task_id, classes in enumerate(task_classes):
        class_set = set(int(c) for c in classes)
        train_pool = [i for i, y in enumerate(train_targets.tolist()) if y in class_set]
        test_idx = [i for i, y in enumerate(test_targets.tolist()) if y in class_set]
        train_rel, val_rel = _split_indices(
            len(train_pool),
            val_fraction=val_fraction,
            seed=seed,
            task_id=task_id,
        )
        train_idx = [train_pool[i] for i in train_rel]
        val_idx = [train_pool[i] for i in val_rel]
        x_train, y_train = _stack_subset(train_ds, train_idx)
        x_val, y_val = _stack_subset(val_ds, val_idx)
        x_test, y_test = _stack_subset(test_ds, test_idx)
        out.append(
            TaskSplit(
                task_id=task_id,
                classes=tuple(sorted(class_set)),
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
            )
        )
    return out


__all__ = [
    "AVALANCHE_SPLIT_CIFAR10_TASKS",
    "build_split_cifar10_tasks",
    "class_order_for_seed",
]
