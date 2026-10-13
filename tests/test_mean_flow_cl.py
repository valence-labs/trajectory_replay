"""Tests for joint instantaneous + MeanFlow CL (two-branch identity)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.mean_flow_cl import (
    _euler_rollout_to_times,
    mean_flow_two_branch_loss,
    train_unconditional_joint_mean_flow,
)
from trajectory_replay.training.mean_flow_distill import (
    make_mean_flow_from_teacher,
    sample_meanflow_times,
)


def _tiny_uncond(dim: int = 2) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=dim,
        n_contexts=1,
        n_perturbations=1,
        time_emb_dim=8,
        hidden_dim=32,
        depth=2,
        use_conditioning=False,
    )


def test_euler_rollout_to_zero_is_identity() -> None:
    model = _tiny_uncond()
    x0 = torch.randn(4, 2)
    t_ends = torch.zeros(4)
    out = _euler_rollout_to_times(model, x0, t_ends, step_size=0.1)
    assert torch.allclose(out, x0)


def test_euler_rollout_shape() -> None:
    model = _tiny_uncond()
    x0 = torch.randn(5, 2)
    t_ends = torch.tensor([0.1, 0.3, 0.5, 0.7, 1.0])
    out = _euler_rollout_to_times(model, x0, t_ends, step_size=0.05)
    assert out.shape == x0.shape
    assert torch.isfinite(out).all()


def test_two_branch_loss_finite_and_backward() -> None:
    teacher = _tiny_uncond()
    mf = make_mean_flow_from_teacher(teacher)
    x_new = torch.randn(8, 2)
    gen = torch.Generator().manual_seed(0)
    out = mean_flow_two_branch_loss(
        mf,
        teacher,
        x_new,
        n_old=6,
        lambda_mix=0.4,
        device=torch.device("cpu"),
        gen=gen,
        ode_step_size=0.1,
        diag_fraction=0.5,
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_two_branch_loss_is_lambda_weighted_mse() -> None:
    """Without adaptive weighting the total must be exactly the λ mixture.

    adaptive_p self-normalizes each sample by its own error, which silently
    destroys this identity, so the λ knob stops controlling the mixture.
    """
    teacher = _tiny_uncond()
    mf = make_mean_flow_from_teacher(teacher)
    x_new = torch.randn(8, 2)
    lam = 0.4
    out = mean_flow_two_branch_loss(
        mf,
        teacher,
        x_new,
        n_old=6,
        lambda_mix=lam,
        device=torch.device("cpu"),
        gen=torch.Generator().manual_seed(0),
        ode_step_size=0.1,
        diag_fraction=0.5,
    )
    expected = lam * out["loss_old"] + (1.0 - lam) * out["loss_new"]
    assert torch.allclose(out["loss"], expected)
    # With adaptive off, the reported MSE is the loss itself.
    assert torch.allclose(out["loss_old"], out["mse_old"])
    assert torch.allclose(out["loss_new"], out["mse_new"])


def test_cifar_time_schedule_favors_long_jumps() -> None:
    """zhuyu CIFAR one-step schedule should put more mass on large (t-r) than ImageNet."""
    def mean_gap(mu: float, sigma: float, diag: float, seed: int = 0) -> float:
        gen = torch.Generator().manual_seed(seed)
        r, t = sample_meanflow_times(
            8192,
            torch.device("cpu"),
            gen,
            diag_fraction=diag,
            time_mu=mu,
            time_sigma=sigma,
        )
        return float((t - r).mean().item())

    # ImageNet defaults vs CIFAR one-step defaults from zhuyu-cs/MeanFlow.
    gap_imagenet = mean_gap(-0.4, 1.0, 0.75)
    gap_cifar = mean_gap(-2.0, 2.0, 0.25)
    assert gap_cifar > gap_imagenet
    assert gap_cifar > 0.15


def test_joint_train_smoke_generative() -> None:
    ref = _tiny_uncond()
    model = _tiny_uncond()
    model.load_state_dict(ref.state_dict())
    mf = make_mean_flow_from_teacher(ref)
    x = torch.randn(64, 2)
    loader = DataLoader(TensorDataset(x), batch_size=16, shuffle=True, drop_last=True)
    history = train_unconditional_joint_mean_flow(
        model,
        mf,
        ref,
        loader,
        strategy="generative",
        device=torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        train_seed=0,
        lambda_mix=0.5,
        ode_step_size=0.2,
        diag_fraction=0.5,
    )
    assert len(history) == 2
    assert "mf_loss_mean" in history[-1]
    assert torch.isfinite(torch.tensor(history[-1]["mf_loss_mean"]))


def test_joint_train_smoke_trajectory() -> None:
    ref = _tiny_uncond()
    model = _tiny_uncond()
    model.load_state_dict(ref.state_dict())
    mf = make_mean_flow_from_teacher(ref)
    x = torch.randn(64, 2)
    loader = DataLoader(TensorDataset(x), batch_size=16, shuffle=True, drop_last=True)
    history = train_unconditional_joint_mean_flow(
        model,
        mf,
        ref,
        loader,
        strategy="trajectory",
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        train_seed=1,
        lambda_mix=0.3,
        ode_step_size=0.2,
    )
    assert len(history) == 1
    assert history[0]["strategy"] == "trajectory"


def test_joint_train_smoke_trajectory_with_replay_n_steps() -> None:
    ref = _tiny_uncond()
    model = _tiny_uncond()
    model.load_state_dict(ref.state_dict())
    mf = make_mean_flow_from_teacher(ref)
    replay = make_mean_flow_from_teacher(ref)
    replay.eval()
    for p in replay.parameters():
        p.requires_grad_(False)
    x = torch.randn(64, 2)
    loader = DataLoader(TensorDataset(x), batch_size=16, shuffle=True, drop_last=True)
    history = train_unconditional_joint_mean_flow(
        model,
        mf,
        ref,
        loader,
        strategy="trajectory",
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        train_seed=2,
        lambda_mix=0.3,
        ode_step_size=0.2,
        replay_mean_flow=replay,
        mean_flow_n_steps=4,
    )
    assert history[0]["mean_flow_n_steps"] == 4.0
    assert torch.isfinite(torch.tensor(history[0]["mf_loss_mean"]))
