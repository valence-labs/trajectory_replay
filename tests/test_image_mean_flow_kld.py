"""Tests for spatial Diffusers-UNet MeanFlow (image_cl)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from trajectory_replay.eval.stream_kld import histogram_kld_ard
from trajectory_replay.model.diffusers_unet_fm import DiffusersUNetFlowMatching
from trajectory_replay.model.diffusers_unet_mean_flow import (
    DiffusersUNetMeanFlow,
    load_diffusers_unet_mean_flow_checkpoint,
    save_diffusers_unet_mean_flow_checkpoint,
)
from trajectory_replay.model.image_mean_flow import (
    _spatial_mean_flow_identity_loss,
    distill_mean_flow_from_image_teacher,
    mean_flow_replay_fidelity,
)
from trajectory_replay.training.mixture_fm import (
    train_class_conditional_generative_replay,
    train_class_conditional_trajectory_replay,
    train_unconditional_generative_replay,
    train_unconditional_trajectory_replay,
)


def _tiny_unet_kwargs() -> dict:
    return {
        "block_out_channels": (32, 64),
        "down_block_types": ("DownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "UpBlock2D"),
        "norm_num_groups": 8,
        "layers_per_block": 1,
    }


def test_histogram_kld_identical() -> None:
    y = torch.arange(10).repeat(5)
    out = histogram_kld_ard(y, y, num_classes=10)
    assert out["stream_kld"] == 0.0
    assert out["stream_ard"] == 0.0


def test_unet_mean_flow_dual_time_and_shapes(tmp_path: Path) -> None:
    device = torch.device("cpu")
    mf = DiffusersUNetMeanFlow(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=_tiny_unet_kwargs()
    ).to(device)
    assert mf.unet.config.class_embed_type == "timestep"
    x = torch.randn(2, 1, 16, 16)
    r = torch.zeros(2)
    t = torch.ones(2) * 0.5
    u = mf.calc_u(x, r, t)
    assert u.shape == x.shape
    xt = mf.sample_from_x0(x, t_end=0.5, n_steps=1)
    assert xt.shape == x.shape
    samples = mf.sample(batch_size=3, n_steps=1)
    assert samples.shape == (3, 1, 16, 16)

    save_diffusers_unet_mean_flow_checkpoint(tmp_path / "mean_flow.pt", mf)
    loaded, cfg = load_diffusers_unet_mean_flow_checkpoint(tmp_path / "mean_flow.pt", device)
    assert cfg["model_type"] == "diffusers_unet_mean_flow"
    u2 = loaded.calc_u(x, r, t)
    assert torch.allclose(u, u2, atol=1e-5)


def test_unet_mean_flow_keeps_times_normalized() -> None:
    """MeanFlow must not inherit the CFM's DDPM-index time rescaling."""
    mf = DiffusersUNetMeanFlow(
        image_size=16,
        in_channels=1,
        train_timesteps=1000,
        unet_kwargs=_tiny_unet_kwargs(),
    )
    times = torch.tensor([0.0, 0.25, 1.0])
    assert torch.equal(mf._to_unet_timestep(times), times)

    # The CFM adapter still maps normalized time to Diffusers' index scale.
    cfm = DiffusersUNetFlowMatching(
        image_size=16,
        in_channels=1,
        train_timesteps=1000,
        unet_kwargs=_tiny_unet_kwargs(),
    )
    assert torch.equal(cfm._to_unet_timestep(times), times * 999.0)


def test_spatial_identity_loss_finite() -> None:
    device = torch.device("cpu")
    kw = _tiny_unet_kwargs()
    teacher = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    mf = DiffusersUNetMeanFlow(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    x0 = torch.randn(4, 1, 16, 16)
    x1 = torch.randn(4, 1, 16, 16)
    t = torch.rand(4)
    r = t.clone()  # diagonal: the (t - r) JVP term drops out of the target
    loss, mse, _mse_per = _spatial_mean_flow_identity_loss(
        mf, teacher, x0, x1, r, t, adaptive_p=1.0
    )
    assert torch.isfinite(loss) and torch.isfinite(mse)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mf.parameters())


def test_spatial_adaptive_loss_uses_squared_l2_norm() -> None:
    """The adaptive floor is calibrated to sum(error²), as in zhuyu-cs."""
    kw = _tiny_unet_kwargs()
    teacher = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    )
    mf = DiffusersUNetMeanFlow(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    )
    x0 = torch.randn(4, 1, 16, 16)
    x1 = torch.randn(4, 1, 16, 16)
    r = t = torch.rand(4)
    p = 0.75
    loss, _mse, mse_per = _spatial_mean_flow_identity_loss(
        mf, teacher, x0, x1, r, t, adaptive_p=p
    )

    sq_norm_per = mse_per * x0[0].numel()
    expected = (sq_norm_per / (sq_norm_per + 1e-3).pow(p)).mean()
    assert torch.allclose(loss.detach(), expected, rtol=1e-5, atol=1e-6)


def test_identity_loss_conditions_on_state_at_r() -> None:
    """The target must be built from ``z_r``, the state ``transport`` is given.

    Conditioning on ``z_t`` instead (the data end under this time convention)
    trains a field the sampler can never query, which silently turns one-step
    replay into noise.
    """
    device = torch.device("cpu")
    kw = _tiny_unet_kwargs()
    teacher = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    mf = DiffusersUNetMeanFlow(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    calls: list[torch.Tensor] = []
    original = mf.calc_u

    def spy(x: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *a, **kw):  # type: ignore[no-untyped-def]
        calls.append(x.detach().clone())
        return original(x, r, t, *a, **kw)

    mf.calc_u = spy  # type: ignore[method-assign]
    x0 = torch.randn(4, 1, 16, 16)
    x1 = torch.randn(4, 1, 16, 16)
    r = torch.zeros(4)
    t = torch.ones(4)
    _spatial_mean_flow_identity_loss(mf, teacher, x0, x1, r, t, adaptive_p=None)  # noqa: B018
    mf.calc_u = original  # type: ignore[method-assign]

    assert calls, "student was never evaluated"
    # r = 0 => z_r is exactly the noise sample, never the data endpoint x1.
    assert torch.allclose(calls[0], x0, atol=1e-6)
    assert not torch.allclose(calls[0], x1, atol=1e-3)


def test_one_step_jump_matches_teacher_after_distill(tmp_path: Path) -> None:
    """A short distill on a tiny UNet should already beat the untrained jump."""
    device = torch.device("cpu")
    kw = _tiny_unet_kwargs()
    teacher = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    teacher.eval()
    x = torch.randn(32, 1, 16, 16)
    mf, _replay, hist = distill_mean_flow_from_image_teacher(
        teacher,
        x,
        device=device,
        out_dir=tmp_path,
        epochs=3,
        batch_size=16,
        lr=1e-3,
        train_seed=0,
        unet_kwargs=kw,
    )
    assert hist[-1]["mse_mean"] < hist[0]["mse_mean"]
    fid = mean_flow_replay_fidelity(
        mf, teacher, device=device, n=8, n_steps=1, ode_step_size=0.25, seed=0
    )
    assert 0.0 < fid["rel_err_vs_teacher"] < 10.0
    assert fid["std_ratio"] > 0.0


def test_distill_one_epoch(tmp_path: Path) -> None:
    device = torch.device("cpu")
    kw = _tiny_unet_kwargs()
    teacher = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    x = torch.randn(16, 1, 16, 16)
    mf, replay, hist = distill_mean_flow_from_image_teacher(
        teacher,
        x,
        device=device,
        out_dir=tmp_path,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        train_seed=0,
        unet_kwargs=kw,
    )
    assert hist and (tmp_path / "mean_flow.pt").is_file()
    assert replay is mf
    assert mf.sample(batch_size=2, n_steps=1).shape == (2, 1, 16, 16)


def test_mixture_accepts_unet_mean_flow() -> None:
    device = torch.device("cpu")
    kw = _tiny_unet_kwargs()
    model = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    teacher = DiffusersUNetFlowMatching(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    teacher.eval()
    mf = DiffusersUNetMeanFlow(
        image_size=16, in_channels=1, train_timesteps=100, unet_kwargs=kw
    ).to(device)
    mf.eval()
    x = torch.randn(8, 1, 16, 16)
    loader = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)
    hist_g = train_unconditional_generative_replay(
        model,
        teacher,
        loader,
        device=device,
        epochs=1,
        lr=1e-3,
        train_seed=0,
        lambda_mix=0.5,
        mean_flow=mf,
        mean_flow_n_steps=1,
    )
    hist_t = train_unconditional_trajectory_replay(
        model,
        teacher,
        loader,
        device=device,
        epochs=1,
        lr=1e-3,
        train_seed=1,
        lambda_mix=0.5,
        mean_flow=mf,
        mean_flow_n_steps=1,
    )
    assert hist_g and hist_t


def test_class_conditional_mean_flow_distill_and_replay(tmp_path: Path) -> None:
    device = torch.device("cpu")
    kw = _tiny_unet_kwargs()
    teacher = DiffusersUNetFlowMatching(
        image_size=16,
        in_channels=1,
        train_timesteps=100,
        unet_kwargs=kw,
        num_classes=4,
        guidance_scale=1.5,
    ).to(device)
    teacher.eval()
    x = torch.randn(16, 1, 16, 16)
    y = torch.randint(0, 4, (16,))
    mf, replay, hist = distill_mean_flow_from_image_teacher(
        teacher,
        x,
        y_train=y,
        device=device,
        out_dir=tmp_path,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        train_seed=0,
    )
    assert hist and mf.use_conditioning and mf.num_classes == 4
    assert replay is mf
    labels = torch.tensor([0, 1, 2, 3])
    assert mf.sample(c=labels, n_steps=1, guidance_scale=1.0).shape == (4, 1, 16, 16)
    fid = mean_flow_replay_fidelity(
        mf,
        teacher,
        device=device,
        n=8,
        n_steps=1,
        ode_step_size=0.5,
        seed=0,
        class_ids=[0, 1, 2, 3],
    )
    assert "rel_err_vs_teacher" in fid

    model = DiffusersUNetFlowMatching(
        image_size=16,
        in_channels=1,
        train_timesteps=100,
        unet_kwargs=kw,
        num_classes=4,
        guidance_scale=1.5,
    ).to(device)
    loader = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=4, shuffle=False)
    hist_g = train_class_conditional_generative_replay(
        model,
        teacher,
        loader,
        old_class_ids=[0, 1],
        device=device,
        epochs=1,
        lr=1e-3,
        train_seed=0,
        lambda_mix=0.5,
        mean_flow=mf,
        mean_flow_n_steps=1,
    )
    hist_t = train_class_conditional_trajectory_replay(
        model,
        teacher,
        loader,
        old_class_ids=[0, 1],
        device=device,
        epochs=1,
        lr=1e-3,
        train_seed=1,
        lambda_mix=0.5,
        mean_flow=mf,
        mean_flow_n_steps=1,
        ode_step_size=0.5,
    )
    assert hist_g and hist_t


if __name__ == "__main__":
    import tempfile

    test_histogram_kld_identical()
    print("kld ok", flush=True)
    with tempfile.TemporaryDirectory() as d:
        test_unet_mean_flow_dual_time_and_shapes(Path(d))
    print("shapes ok", flush=True)
    test_spatial_identity_loss_finite()
    print("identity loss ok", flush=True)
    test_identity_loss_conditions_on_state_at_r()
    print("conditioning ok", flush=True)
    with tempfile.TemporaryDirectory() as d:
        test_distill_one_epoch(Path(d))
    print("distill ok", flush=True)
    with tempfile.TemporaryDirectory() as d:
        test_one_step_jump_matches_teacher_after_distill(Path(d))
    print("one-step fidelity ok", flush=True)
    test_mixture_accepts_unet_mean_flow()
    print("mixture ok", flush=True)
    with tempfile.TemporaryDirectory() as d:
        test_class_conditional_mean_flow_distill_and_replay(Path(d))
    print("class-cond mf ok", flush=True)
