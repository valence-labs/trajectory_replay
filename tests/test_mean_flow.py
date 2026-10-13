"""Tests for MeanFlow distillation (mirrors flow-map tests)."""

from __future__ import annotations

from pathlib import Path

import torch

from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.mean_flow_distill import (
    _mean_flow_identity_loss,
    load_mean_flow_checkpoint,
    make_mean_flow_from_teacher,
    save_mean_flow_checkpoint,
)


def _tiny_teacher(*, use_conditioning: bool = True) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=4,
        n_contexts=3,
        n_perturbations=2,
        cond_emb_dim=8,
        time_emb_dim=8,
        hidden_dim=32,
        depth=2,
        use_conditioning=use_conditioning,
    )


def test_mean_flow_transport_identity_at_r_equals_t() -> None:
    teacher = _tiny_teacher()
    mf = make_mean_flow_from_teacher(teacher)
    b = 5
    x = torch.randn(b, 4)
    t = torch.rand(b)
    c = torch.randint(0, 3, (b,))
    p = torch.randint(0, 2, (b,))
    out = mf.transport(x, t, t, c, p)
    assert torch.allclose(out, x, atol=1e-5)


def test_mean_flow_identity_loss_finite() -> None:
    teacher = _tiny_teacher()
    mf = make_mean_flow_from_teacher(teacher)
    b = 8
    x0 = torch.randn(b, 4)
    x1 = torch.randn(b, 4)
    c = torch.randint(0, 3, (b,))
    p = torch.randint(0, 2, (b,))
    r = torch.rand(b)
    t = torch.clamp(r + torch.rand(b) * (1 - r), max=1.0)
    loss = _mean_flow_identity_loss(mf, teacher, x0, x1, c, p, r, t)
    assert torch.isfinite(loss)
    loss.backward()


def test_mean_flow_unconditional_loss_finite() -> None:
    teacher = _tiny_teacher(use_conditioning=False)
    mf = make_mean_flow_from_teacher(teacher)
    assert not mf.use_conditioning
    b = 8
    x0 = torch.randn(b, 4)
    x1 = torch.randn(b, 4)
    r = torch.rand(b)
    t = torch.clamp(r + 0.1, max=1.0)
    loss = _mean_flow_identity_loss(mf, teacher, x0, x1, None, None, r, t)
    assert torch.isfinite(loss)


def test_mean_flow_one_step_sample_shape() -> None:
    teacher = _tiny_teacher(use_conditioning=False)
    mf = make_mean_flow_from_teacher(teacher)
    x0 = torch.randn(6, 4)
    out = mf.sample_from_x0(x0, n_steps=1)
    assert out.shape == x0.shape


def test_mean_flow_checkpoint_roundtrip(tmp_path: Path) -> None:
    teacher = _tiny_teacher()
    mf = make_mean_flow_from_teacher(teacher)
    ckpt = tmp_path / "mean_flow.pt"
    save_mean_flow_checkpoint(
        ckpt,
        mf,
        {"stage": "pretrain", "experiment": "test"},
        parent_checkpoint="checkpoint.pt",
    )
    loaded, cfg = load_mean_flow_checkpoint(ckpt, torch.device("cpu"))
    assert cfg["model_type"] == "conditional_mean_flow"
    b = 4
    x = torch.randn(b, 4)
    r = torch.zeros(b)
    t = torch.ones(b)
    c = torch.randint(0, 3, (b,))
    p = torch.randint(0, 2, (b,))
    assert torch.allclose(mf(x, r, t, c, p), loaded(x, r, t, c, p))


def test_mean_flow_replay_endpoint_shape() -> None:
    from trajectory_replay.training.flow_matching import _sample_ref_replay_endpoints

    teacher = _tiny_teacher()
    mf = make_mean_flow_from_teacher(teacher)
    mf.eval()
    device = torch.device("cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    c = torch.zeros(6, dtype=torch.long)
    p = torch.ones(6, dtype=torch.long)
    out = _sample_ref_replay_endpoints(
        teacher,
        c,
        p,
        device=device,
        gen=gen,
        replay_ode_method="euler",
        replay_ode_rtol=1e-5,
        replay_ode_atol=1e-5,
        replay_ode_options={"step_size": 0.25},
        mean_flow=mf,
        mean_flow_n_steps=2,
    )
    assert out.shape == (6, 4)


if __name__ == "__main__":
    test_mean_flow_transport_identity_at_r_equals_t()
    test_mean_flow_identity_loss_finite()
    test_mean_flow_unconditional_loss_finite()
    test_mean_flow_one_step_sample_shape()
    test_mean_flow_replay_endpoint_shape()
    print("mean_flow tests ok")
