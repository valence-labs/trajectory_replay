"""Tests for image joint MeanFlow (shared-sim traj + generative data losses)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from trajectory_replay.model.diffusers_unet_fm import DiffusersUNetFlowMatching
from trajectory_replay.model.image_mean_flow import (
    _euler_rollout_to_times_image,
    _sample_t_given_shared_r,
    make_spatial_mean_flow_from_teacher,
    spatial_mean_flow_two_branch_loss,
    train_image_joint_mean_flow,
)


def _tiny_uncond() -> DiffusersUNetFlowMatching:
    return DiffusersUNetFlowMatching(
        image_size=8,
        in_channels=1,
        train_timesteps=100,
        unet_kwargs={
            "sample_size": 8,
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 1,
            "block_out_channels": (16, 32),
            "down_block_types": ("DownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "UpBlock2D"),
            "norm_num_groups": 8,
        },
    )


def test_image_euler_rollout_zero_is_identity() -> None:
    model = _tiny_uncond()
    x0 = torch.randn(2, 1, 8, 8)
    out = _euler_rollout_to_times_image(
        model, x0, torch.zeros(2), step_size=0.1
    )
    assert torch.allclose(out, x0)


def test_sample_t_given_shared_r_respects_diag() -> None:
    gen = torch.Generator().manual_seed(0)
    r, t = _sample_t_given_shared_r(
        64,
        0.3,
        device=torch.device("cpu"),
        gen=gen,
        dtype=torch.float32,
        diag_fraction=1.0,
    )
    assert torch.allclose(r, torch.full_like(r, 0.3))
    assert torch.allclose(t, r)


def test_spatial_two_branch_loss_generative_data_old() -> None:
    teacher = _tiny_uncond()
    mf = make_spatial_mean_flow_from_teacher(teacher)
    x_new = torch.randn(4, 1, 8, 8)
    x_old = torch.randn(3, 1, 8, 8)
    gen = torch.Generator().manual_seed(0)
    out = spatial_mean_flow_two_branch_loss(
        mf,
        teacher,
        x_new,
        n_old=3,
        lambda_mix=0.5,
        device=torch.device("cpu"),
        gen=gen,
        ode_step_size=0.25,
        diag_fraction=0.5,
        adaptive_p=None,
        x_old=x_old,
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_train_image_joint_mean_flow_generative_smoke() -> None:
    teacher = _tiny_uncond()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = _tiny_uncond()
    mf = make_spatial_mean_flow_from_teacher(teacher)
    replay = make_spatial_mean_flow_from_teacher(teacher)
    replay.eval()
    for p in replay.parameters():
        p.requires_grad_(False)
    x = torch.randn(8, 1, 8, 8)
    loader = DataLoader(TensorDataset(x), batch_size=4, shuffle=True)
    hist = train_image_joint_mean_flow(
        student,
        mf,
        teacher,
        loader,
        strategy="generative",
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        train_seed=0,
        lambda_mix=0.5,
        ode_step_size=0.25,
        diag_fraction=0.5,
        adaptive_p=None,
        replay_mean_flow=replay,
        mean_flow_n_steps=2,
        mf_use_adamw=True,
        mf_betas=(0.9, 0.95),
        mf_max_grad_norm=1.0,
    )
    assert len(hist) == 1
    assert hist[0]["mean_flow_n_steps"] == 2.0
    assert torch.isfinite(torch.tensor(hist[0]["loss_mean"]))


def test_train_image_joint_mean_flow_trajectory_smoke() -> None:
    teacher = _tiny_uncond()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = _tiny_uncond()
    mf = make_spatial_mean_flow_from_teacher(teacher)
    replay = make_spatial_mean_flow_from_teacher(teacher)
    replay.eval()
    for p in replay.parameters():
        p.requires_grad_(False)
    x = torch.randn(8, 1, 8, 8)
    loader = DataLoader(TensorDataset(x), batch_size=4, shuffle=True)
    hist = train_image_joint_mean_flow(
        student,
        mf,
        teacher,
        loader,
        strategy="trajectory",
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        train_seed=1,
        lambda_mix=0.4,
        ode_step_size=0.25,
        diag_fraction=0.5,
        adaptive_p=None,
        replay_mean_flow=replay,
        mean_flow_n_steps=4,
        mf_use_adamw=True,
        mf_betas=(0.9, 0.95),
        mf_max_grad_norm=1.0,
    )
    assert len(hist) == 1
    assert hist[0]["strategy"] == "trajectory"
    assert hist[0]["mean_flow_n_steps"] == 4.0
    assert torch.isfinite(torch.tensor(hist[0]["mf_mse_old_mean"]))
