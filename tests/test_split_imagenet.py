"""Tests for Split-ImageNet-100 latent TaskSplit builder."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trajectory_replay.data.split_imagenet import (
    DEFAULT_LATENT_DIR,
    SPLIT_IMAGENET100_TASKS,
    SPLIT_IMAGENET100_TASKS_E5,
    _avalanche_nc_class_tasks,
    _nearly_equal_chunk_sizes,
    build_split_imagenet100_tasks,
    class_order_for_seed,
)


def test_nearly_equal_chunk_sizes() -> None:
    assert _nearly_equal_chunk_sizes(100, 5) == [20, 20, 20, 20, 20]
    assert _nearly_equal_chunk_sizes(100, 3) == [34, 33, 33]
    assert _nearly_equal_chunk_sizes(10, 3) == [4, 3, 3]


def test_imagenet100_recorded_orders_match_avalanche_shuffle() -> None:
    for seed, tasks in SPLIT_IMAGENET100_TASKS_E5.items():
        assert class_order_for_seed(seed, n_experiences=5) == tasks
        assert _avalanche_nc_class_tasks(seed, n_classes=100, n_experiences=5) == tasks
        flat = [c for t in tasks for c in t]
        assert sorted(flat) == list(range(100))
        assert len(tasks) == 5
        assert all(len(t) == 20 for t in tasks)
    # Alias stays wired to the 5-task table.
    assert SPLIT_IMAGENET100_TASKS is SPLIT_IMAGENET100_TASKS_E5


def test_imagenet100_class_order_other_seed_computes() -> None:
    tasks = class_order_for_seed(123, n_classes=100, n_experiences=5)
    flat = [c for t in tasks for c in t]
    assert sorted(flat) == list(range(100))
    assert all(len(t) == 20 for t in tasks)


def test_imagenet100_default_is_three_tasks() -> None:
    tasks = class_order_for_seed(42)
    assert len(tasks) == 3
    assert [len(t) for t in tasks] == [34, 33, 33]
    flat = [c for t in tasks for c in t]
    assert sorted(flat) == list(range(100))
    # Same shuffle prefix as the 5-task protocol for seed 42 (Avalanche RNG).
    e5 = class_order_for_seed(42, n_experiences=5)
    assert set(tasks[0]).issuperset(set(e5[0]))


@pytest.mark.skipif(
    not (DEFAULT_LATENT_DIR / "train.pt").is_file(),
    reason="ImageNet-100 latent cache not present",
)
def test_build_split_imagenet100_tasks_shapes() -> None:
    tasks = build_split_imagenet100_tasks(seed=42, latent_dir=DEFAULT_LATENT_DIR)
    assert len(tasks) == 3
    for t, n_cls in zip(tasks, (34, 33, 33), strict=True):
        assert len(t.classes) == n_cls
        assert t.x_train.ndim == 4
        assert t.x_train.shape[1:] == (4, 32, 32)
        assert t.x_train.dtype == torch.float32
        assert t.x_train.shape[0] == t.y_train.shape[0]
        assert t.x_val.shape[0] == t.y_val.shape[0]
        assert t.x_test.shape[0] == t.y_test.shape[0]
        # ~1300 img/class × n_cls × 0.9 train; test = 50 × n_cls
        assert 30_000 <= t.x_train.shape[0] <= 45_000
        assert 3_000 <= t.x_val.shape[0] <= 6_000
        assert t.x_test.shape[0] == 50 * n_cls
        assert set(t.y_train.tolist()) | set(t.y_val.tolist()) == set(t.classes)
        assert set(t.y_test.tolist()) == set(t.classes)


@pytest.mark.skipif(
    not (DEFAULT_LATENT_DIR / "train.pt").is_file(),
    reason="ImageNet-100 latent cache not present",
)
def test_build_split_imagenet100_tasks_masip_five() -> None:
    tasks = build_split_imagenet100_tasks(
        seed=42, latent_dir=DEFAULT_LATENT_DIR, n_experiences=5
    )
    assert len(tasks) == 5
    for t in tasks:
        assert len(t.classes) == 20
        assert t.x_test.shape[0] == 1000
