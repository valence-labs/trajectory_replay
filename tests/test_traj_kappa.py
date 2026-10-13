"""Unit tests for trajectory κ(t) weights (Thm. 2.4)."""

from __future__ import annotations

import torch

from trajectory_replay.training.traj_kappa import (
    gaussian_path_kappa,
    ot_kappa,
    resolve_traj_kappa,
)


def test_ot_kappa_matches_eq17_for_linear_path() -> None:
    t = torch.tensor([0.25, 0.5, 0.75])
    # α=t, σ=1-t ⇒ α̇=1, σ̇=-1 ⇒ κ = t / (1-t)
    alpha, sigma = t, 1.0 - t
    alpha_dot = torch.ones_like(t)
    sigma_dot = -torch.ones_like(t)
    expected = t / (1.0 - t)
    got = gaussian_path_kappa(alpha, sigma, alpha_dot, sigma_dot)
    assert torch.allclose(got, expected)
    assert torch.allclose(ot_kappa(t), expected)


def test_resolve_traj_kappa_none_is_one() -> None:
    w = resolve_traj_kappa("none", 0.8)
    assert float(w) == 1.0


def test_resolve_traj_kappa_ot_default_aliases() -> None:
    w = resolve_traj_kappa("ot", 0.5)
    assert abs(float(w) - 1.0) < 1e-6
    w2 = resolve_traj_kappa("paper", 0.5)
    assert abs(float(w2) - 1.0) < 1e-6
