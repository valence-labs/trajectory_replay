"""Split Fashion-MNIST protocol matching Masip / Avalanche class orders.

Produces per-task train / val / test tensors in ``[-1, 1]`` at ``image_size``×``image_size``,
without depending on Avalanche at runtime. Class orders for seeds used in the Masip
FMNIST grid were verified against ``avalanche.benchmarks.SplitFMNIST``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

# Verified against Avalanche SplitFMNIST(n_experiences=5, seed=...).
AVALANCHE_SPLIT_FMNIST_TASKS: dict[int, list[list[int]]] = {
    42: [[2, 6], [8, 1], [4, 5], [0, 9], [3, 7]],
    69: [[0, 2], [3, 5], [1, 4], [9, 6], [8, 7]],
    1714: [[3, 7], [0, 8], [1, 4], [2, 6], [9, 5]],
}


@dataclass(frozen=True)
class TaskSplit:
    task_id: int
    classes: tuple[int, ...]
    x_train: torch.Tensor  # (N, C, H, W) in [-1, 1]
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def class_order_for_seed(seed: int) -> list[list[int]]:
    if seed not in AVALANCHE_SPLIT_FMNIST_TASKS:
        known = sorted(AVALANCHE_SPLIT_FMNIST_TASKS)
        raise KeyError(
            f"No recorded Avalanche SplitFMNIST class order for seed={seed}. "
            f"Known seeds: {known}. Add the order after probing SplitFMNIST."
        )
    return [list(t) for t in AVALANCHE_SPLIT_FMNIST_TASKS[seed]]


def _fmnist_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
        ]
    )


def _stack_subset(
    ds: datasets.FashionMNIST,
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
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def build_split_fmnist_tasks(
    *,
    seed: int,
    data_root: str | Path,
    image_size: int = 32,
    n_experiences: int = 5,
    val_fraction: float = 0.1,
    download: bool = True,
) -> list[TaskSplit]:
    """Build 5-task Split-FMNIST with a val carve from each task's train set."""
    if n_experiences != 5:
        raise ValueError("Only n_experiences=5 is supported (Masip FMNIST protocol).")
    tasks_classes = class_order_for_seed(seed)
    if len(tasks_classes) != n_experiences:
        raise RuntimeError("class order length mismatch")

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    tfm = _fmnist_transform(image_size)
    train_ds = datasets.FashionMNIST(root=str(root), train=True, download=download, transform=tfm)
    test_ds = datasets.FashionMNIST(root=str(root), train=False, download=download, transform=tfm)

    train_targets = torch.tensor(train_ds.targets, dtype=torch.long)
    test_targets = torch.tensor(test_ds.targets, dtype=torch.long)

    out: list[TaskSplit] = []
    for task_id, classes in enumerate(tasks_classes):
        class_set = set(int(c) for c in classes)
        train_pool = [i for i, y in enumerate(train_targets.tolist()) if y in class_set]
        test_idx = [i for i, y in enumerate(test_targets.tolist()) if y in class_set]
        tr_rel, va_rel = _split_indices(
            len(train_pool), val_fraction=val_fraction, seed=seed, task_id=task_id
        )
        tr_idx = [train_pool[i] for i in tr_rel]
        va_idx = [train_pool[i] for i in va_rel]
        x_tr, y_tr = _stack_subset(train_ds, tr_idx)
        x_va, y_va = _stack_subset(train_ds, va_idx)
        x_te, y_te = _stack_subset(test_ds, test_idx)
        out.append(
            TaskSplit(
                task_id=task_id,
                classes=tuple(sorted(class_set)),
                x_train=x_tr,
                y_train=y_tr,
                x_val=x_va,
                y_val=y_va,
                x_test=x_te,
                y_test=y_te,
            )
        )
    return out


def make_x_loader(
    x: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    random_horizontal_flip: bool = False,
) -> DataLoader:
    if random_horizontal_flip:
        ds: Dataset = _RandomHorizontalFlipTensorDataset(x)
    else:
        ds = TensorDataset(x)
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        drop_last=False,
        num_workers=0,
    )


def make_xy_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    random_horizontal_flip: bool = False,
) -> DataLoader:
    """DataLoader yielding ``(x, y)`` batches for class-conditional CFM."""
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if random_horizontal_flip:
        ds: Dataset = _RandomHorizontalFlipXYDataset(x, y)
    else:
        ds = TensorDataset(x, y.long())
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        drop_last=False,
        num_workers=0,
    )


class _RandomHorizontalFlipTensorDataset(Dataset):
    """Tensor dataset with Masip's stochastic CIFAR train augmentation."""

    def __init__(self, x: torch.Tensor) -> None:
        self.x = x

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        image = self.x[index]
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(-1,))
        return (image,)


class _RandomHorizontalFlipXYDataset(Dataset):
    """``(x, y)`` tensor dataset with stochastic horizontal flips on ``x`` only."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y.long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.x[index]
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(-1,))
        return image, self.y[index]
