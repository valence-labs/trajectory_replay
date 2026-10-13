"""Tests for MNIST UNet backbone."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from trajectory_replay.model.flow_matching import (
    ConditionalFlowMatching,
    build_conditional_flow_matching_from_config,
)
from trajectory_replay.training.flow_matching import _flow_matching_loss


def test_unet_forward_flat_mnist_shape() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=9,
        cond_emb_dim=32,
        time_emb_dim=64,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
        unet_attention_resolutions="14",
    )
    x = torch.randn(4, 784)
    t = torch.rand(4)
    c = torch.zeros(4, dtype=torch.long)
    p = torch.tensor([0, 1, 2, 3])
    v = model(x, t, c, p)
    assert v.shape == (4, 784)


def test_unet_forward_image_batch() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=9,
        cond_emb_dim=32,
        time_emb_dim=64,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
        unet_attention_resolutions="14",
    )
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    c = torch.zeros(4, dtype=torch.long)
    p = torch.tensor([0, 1, 2, 3])
    v = model(x, t, c, p)
    assert v.shape == (4, 1, 28, 28)


def test_unet_sample_spatial_shape() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=9,
        cond_emb_dim=16,
        time_emb_dim=32,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
    )
    c = torch.zeros(3, dtype=torch.long)
    p = torch.tensor([0, 1, 2])
    gen = torch.Generator()
    gen.manual_seed(0)
    out = model.sample(
        c,
        p,
        method="euler",
        options={"step_size": 0.1},
        generator=gen,
    )
    assert out.shape == (3, 1, 28, 28)


def test_unet_flow_matching_loss_image_batch() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=9,
        cond_emb_dim=16,
        time_emb_dim=32,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
    )
    x1 = torch.randn(8, 1, 28, 28)
    c = torch.zeros(8, dtype=torch.long)
    p = torch.arange(8) % 9
    gen = torch.Generator()
    gen.manual_seed(0)
    loss = _flow_matching_loss(
        model,
        x1,
        c,
        p,
        data_dim=784,
        device=torch.device("cpu"),
        gen=gen,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_unet_checkpoint_roundtrip() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=9,
        cond_emb_dim=16,
        time_emb_dim=32,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
    )
    x = torch.randn(2, 784)
    t = torch.rand(2)
    c = torch.zeros(2, dtype=torch.long)
    p = torch.tensor([0, 1])
    ref = model(x, t, c, p)

    cfg = {
        "data_dim": 784,
        "n_contexts": 1,
        "n_perturbations": 9,
        "cond_emb_dim": 16,
        "time_emb_dim": 32,
        "hidden_dim": 256,
        "depth": 4,
        "backbone_type": "unet",
        "image_shape": [1, 28, 28],
        "unet_num_channels": 32,
        "unet_num_res_blocks": 1,
        "unet_attention_resolutions": "14",
        "unet_dropout": 0.0,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt)
        payload = torch.load(ckpt, weights_only=False)
        loaded = build_conditional_flow_matching_from_config(payload["config"])
        loaded.load_state_dict(payload["state_dict"])
        out = loaded(x, t, c, p)
    assert torch.allclose(ref, out)
