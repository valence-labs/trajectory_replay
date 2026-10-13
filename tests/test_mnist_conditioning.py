"""Tests for unconditional vs conditional flow matching models."""

from __future__ import annotations

import torch

from experiments.mnist.mnist_common import configure_mnist_finetune, pretrain_tensors
from trajectory_replay.data.mnist_dataset import MnistSplit
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import _flow_matching_loss


def test_unconditional_unet_ignores_conditions() -> None:
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
        use_conditioning=False,
    )
    assert model.context_emb is None
    assert model.pert_emb is None
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    v_a = model(x, t)
    v_b = model(x, t, c=torch.zeros(4, dtype=torch.long), p=torch.ones(4, dtype=torch.long))
    assert torch.allclose(v_a, v_b)
    assert v_a.shape == (4, 1, 28, 28)


def test_unconditional_sample_shape() -> None:
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
        use_conditioning=False,
    )
    out = model.sample(batch_size=3, method="euler", options={"step_size": 0.1})
    assert out.shape == (3, 1, 28, 28)


def test_unconditional_flow_matching_loss() -> None:
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
        use_conditioning=False,
    )
    x1 = torch.randn(8, 1, 28, 28)
    c = torch.zeros(8, dtype=torch.long)
    p = torch.arange(8) % 9
    gen = torch.Generator().manual_seed(0)
    loss = _flow_matching_loss(
        model,
        x1,
        c,
        p,
        data_dim=784,
        device=torch.device("cpu"),
        gen=gen,
    )
    assert torch.isfinite(loss)


def test_pretrain_tensors_keep_true_digit_labels() -> None:
    split = MnistSplit(
        x=torch.randn(20, 1, 28, 28),
        c=torch.zeros(20, dtype=torch.long),
        p=torch.tensor([0, 1, 8, 8, 9] * 4, dtype=torch.long),
    )
    _, _, p = pretrain_tensors(split, held_out=8, max_samples=None, seed=0)
    assert 8 not in p.tolist()
    assert 9 in p.tolist()


def test_configure_mnist_finetune_conditional_full_table() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=10,
        cond_emb_dim=8,
        time_emb_dim=16,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
        use_conditioning=True,
    )
    params, pert_mask, ctx_mask = configure_mnist_finetune(
        model,
        use_conditioning=True,
        held_out=8,
        train_vf=True,
        train_old_embeddings=False,
    )
    assert model.n_perturbations == 10
    assert pert_mask == 8
    assert ctx_mask is None
    assert len(params) > 0


def test_configure_mnist_finetune_unconditional_vf_only() -> None:
    model = ConditionalFlowMatching(
        data_dim=784,
        n_contexts=1,
        n_perturbations=9,
        cond_emb_dim=8,
        time_emb_dim=16,
        backbone_type="unet",
        image_shape=(1, 28, 28),
        unet_num_channels=32,
        unet_num_res_blocks=1,
        use_conditioning=False,
    )
    params, pert_mask, ctx_mask = configure_mnist_finetune(
        model,
        use_conditioning=False,
        held_out=8,
        train_vf=True,
        train_old_embeddings=False,
    )
    assert pert_mask is None and ctx_mask is None
    assert len(params) > 0
    assert model.n_perturbations == 9
