"""Tests for conditional flow map distillation."""

from __future__ import annotations

from pathlib import Path

import torch

from trajectory_replay.model.flow_map import ConditionalFlowMap
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_map_distill import (
    _diagonal_distill_loss,
    _lsd_distill_loss,
    load_flow_map_checkpoint,
    make_flow_map_from_teacher,
    save_flow_map_checkpoint,
)


def _tiny_teacher() -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=4,
        n_contexts=2,
        n_perturbations=2,
        cond_emb_dim=8,
        time_emb_dim=8,
        hidden_dim=16,
        depth=3,
    )


def test_flow_map_identity_at_s_equals_t() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x = torch.randn(5, 4)
    t = torch.rand(5)
    c = torch.tensor([0, 1, 0, 1, 0])
    p = torch.tensor([0, 0, 1, 1, 0])
    out = flow_map(x, t, t, c, p)
    assert torch.allclose(out, x, atol=1e-6)


def test_flow_map_residual_forward() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x = torch.randn(3, 4)
    s = torch.zeros(3)
    t = torch.full((3,), 0.5)
    c = torch.zeros(3, dtype=torch.long)
    p = torch.zeros(3, dtype=torch.long)
    phi = flow_map.calc_phi(x, s, t, c, p)
    out = flow_map(x, s, t, c, p)
    assert torch.allclose(out, x + 0.5 * phi)


def test_partial_t_shape() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x = torch.randn(2, 4)
    s = torch.tensor([0.0, 0.1])
    t = torch.tensor([0.4, 0.8])
    c = torch.tensor([0, 1])
    p = torch.tensor([0, 1])
    X, dt_X = flow_map.partial_t(x, s, t, c, p)
    assert X.shape == (2, 4)
    assert dt_X.shape == (2, 4)


def test_distill_losses_finite() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x0 = torch.randn(4, 4)
    x1 = torch.randn(4, 4)
    c = torch.tensor([0, 1, 0, 1])
    p = torch.tensor([0, 0, 1, 1])
    t = torch.rand(4)
    s, t_off = torch.tensor([0.0, 0.1, 0.0, 0.2]), torch.tensor([0.5, 0.6, 0.8, 0.9])
    diag = _diagonal_distill_loss(flow_map, teacher, x0, x1, c, p, t)
    offdiag = _lsd_distill_loss(flow_map, teacher, x0, x1, c, p, s, t_off)
    assert torch.isfinite(diag)
    assert torch.isfinite(offdiag)


def _tiny_teacher_with_covariates() -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=4,
        n_contexts=2,
        n_perturbations=2,
        cond_emb_dim=8,
        time_emb_dim=8,
        hidden_dim=16,
        depth=3,
        covariate_sizes={"cell_type": 3},
    )


def test_flow_map_covariate_distill_losses_finite() -> None:
    teacher = _tiny_teacher_with_covariates()
    flow_map = make_flow_map_from_teacher(teacher)
    x0 = torch.randn(4, 4)
    x1 = torch.randn(4, 4)
    c = torch.tensor([0, 1, 0, 1])
    p = torch.tensor([0, 0, 1, 1])
    z = {"cell_type": torch.tensor([0, 1, 2, 1])}
    t = torch.rand(4)
    s, t_off = torch.tensor([0.0, 0.1, 0.0, 0.2]), torch.tensor([0.5, 0.6, 0.8, 0.9])
    diag = _diagonal_distill_loss(flow_map, teacher, x0, x1, c, p, t, z=z)
    offdiag = _lsd_distill_loss(flow_map, teacher, x0, x1, c, p, s, t_off, z=z)
    assert torch.isfinite(diag)
    assert torch.isfinite(offdiag)


def test_flow_map_covariate_checkpoint_roundtrip(tmp_path: Path) -> None:
    teacher = _tiny_teacher_with_covariates()
    flow_map = make_flow_map_from_teacher(teacher)
    ckpt = tmp_path / "flow_map.pt"
    save_flow_map_checkpoint(
        ckpt,
        flow_map,
        {"stage": "pretrain", "encoding": {"context_categories": ["A"], "perturbation_categories": ["PBS"]}},
        parent_checkpoint="checkpoint.pt",
    )
    loaded, cfg = load_flow_map_checkpoint(ckpt, torch.device("cpu"))
    assert cfg["covariate_sizes"] == {"cell_type": 3}
    x = torch.randn(2, 4)
    s = torch.zeros(2)
    t = torch.ones(2)
    c = torch.zeros(2, dtype=torch.long)
    p = torch.zeros(2, dtype=torch.long)
    z = {"cell_type": torch.tensor([1, 2])}
    assert torch.allclose(flow_map(x, s, t, c, p, z=z), loaded(x, s, t, c, p, z=z))


def test_flow_map_one_step_matches_forward() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x0 = torch.randn(3, 4)
    c = torch.tensor([0, 1, 0])
    p = torch.tensor([0, 0, 1])
    s = torch.zeros(3)
    t = torch.full((3,), 0.75)
    out = flow_map.sample_from_x0(x0, c, p, t_end=0.75, n_steps=1)
    assert torch.allclose(out, flow_map(x0, s, t, c, p))


def test_flow_map_multistep_equal_jumps() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x0 = torch.randn(2, 4)
    c = torch.tensor([0, 1])
    p = torch.tensor([0, 1])
    out = flow_map.rollout_from_x0(x0, c, p, t_start=0.0, t_end=1.0, n_steps=4)
    x = x0
    for i, (s_val, t_val) in enumerate([(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]):
        s = torch.full((2,), s_val)
        t = torch.full((2,), t_val)
        x = flow_map(x, s, t, c, p)
    assert torch.allclose(out, x)


def test_flow_map_multistep_partial_endpoint() -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    x0 = torch.randn(2, 4)
    c = torch.zeros(2, dtype=torch.long)
    p = torch.zeros(2, dtype=torch.long)
    one_step = flow_map.sample_from_x0(x0, c, p, t_end=0.6, n_steps=1)
    multi_step = flow_map.sample_from_x0(x0, c, p, t_end=0.6, n_steps=3)
    assert one_step.shape == multi_step.shape
    assert not torch.allclose(one_step, multi_step)


def test_flow_map_checkpoint_roundtrip(tmp_path: Path) -> None:
    teacher = _tiny_teacher()
    flow_map = make_flow_map_from_teacher(teacher)
    ckpt = tmp_path / "flow_map.pt"
    save_flow_map_checkpoint(
        ckpt,
        flow_map,
        {"stage": "pretrain", "encoding": {"context_categories": ["A"], "perturbation_categories": ["PBS"]}},
        parent_checkpoint="checkpoint.pt",
    )
    loaded, cfg = load_flow_map_checkpoint(ckpt, torch.device("cpu"))
    assert cfg["model_type"] == "conditional_flow_map"
    x = torch.randn(2, 4)
    s = torch.zeros(2)
    t = torch.ones(2)
    c = torch.zeros(2, dtype=torch.long)
    p = torch.zeros(2, dtype=torch.long)
    assert torch.allclose(flow_map(x, s, t, c, p), loaded(x, s, t, c, p))


if __name__ == "__main__":
    test_flow_map_identity_at_s_equals_t()
    test_flow_map_residual_forward()
    test_partial_t_shape()
    test_distill_losses_finite()
    print("flow_map tests ok")
