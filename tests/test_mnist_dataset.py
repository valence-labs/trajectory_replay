"""Tests for MNIST processed split loaders."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from trajectory_replay.data.mnist_dataset import (
    _coerce_mnist_x,
    load_split,
)

IMAGE_SHAPE = (1, 28, 28)


def test_coerce_mnist_x_keeps_image_layout() -> None:
    flat = torch.randn(10, 784, dtype=torch.float64)
    out = _coerce_mnist_x(flat, image_shape=IMAGE_SHAPE)
    assert out.shape == (10, 1, 28, 28)
    assert out.dtype == torch.float32

    hw = torch.randn(10, 28, 28)
    out = _coerce_mnist_x(hw, image_shape=IMAGE_SHAPE)
    assert out.shape == (10, 1, 28, 28)

    chw = torch.randn(10, 1, 28, 28)
    out = _coerce_mnist_x(chw, image_shape=IMAGE_SHAPE)
    assert out.shape == (10, 1, 28, 28)


def test_load_split_preserves_image_tensor_on_disk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp)
        manifest = {
            "held_out_digit": 8,
            "data_dim": 784,
            "image_shape": [28, 28],
            "n_contexts": 1,
            "n_perturbations": 10,
        }
        (proc / "manifest.json").write_text(json.dumps(manifest))
        torch.save(
            {
                "x": torch.randn(32, 28, 28),
                "c": torch.zeros(32, dtype=torch.long),
                "p": torch.arange(32) % 9,
            },
            proc / "pretrain.pt",
        )
        split = load_split(proc, "pretrain")
        assert split.x.shape == (32, 1, 28, 28)
        assert split.x.dtype == torch.float32
        assert split.c.dtype == torch.long
        assert split.p.dtype == torch.long
