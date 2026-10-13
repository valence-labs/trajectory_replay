"""Flow-matching replay step split-loss behavior."""

from __future__ import annotations

import torch

from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import (
    _flow_matching_loss,
    _flow_matching_replay_step,
    _generative_replay_field_step,
    _trajectory_replay_batch_sizes,
    make_frozen_model_copy,
)


def test_flow_matching_loss_split_from_single_forward() -> None:
    model = ConditionalFlowMatching(
        data_dim=8,
        n_contexts=2,
        n_perturbations=2,
        cond_emb_dim=16,
        hidden_dim=32,
        depth=2,
    )
    gen = torch.Generator(device=torch.device("cpu"))
    gen.manual_seed(0)
    x1 = torch.randn(10, 8)
    cb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pb = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    out = _flow_matching_loss(
        model,
        x1,
        cb,
        pb,
        data_dim=8,
        device=torch.device("cpu"),
        gen=gen,
        split_at=5,
    )
    assert isinstance(out, tuple)
    loss, loss_ft, loss_ref = out
    assert loss.ndim == 0
    assert loss_ft >= 0.0
    assert loss_ref >= 0.0


def test_flow_matching_replay_step_returns_split_losses() -> None:
    model = ConditionalFlowMatching(
        data_dim=8,
        n_contexts=2,
        n_perturbations=2,
        cond_emb_dim=16,
        hidden_dim=32,
        depth=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator(device=torch.device("cpu"))
    gen.manual_seed(1)
    x1 = torch.randn(8, 8)
    cb = torch.zeros(8, dtype=torch.long)
    pb = torch.zeros(8, dtype=torch.long)
    step = _flow_matching_replay_step(
        model,
        opt,
        x1,
        cb,
        pb,
        split_at=3,
        data_dim=8,
        device=torch.device("cpu"),
        gen=gen,
    )
    assert set(step) == {"loss", "loss_ft", "loss_ref"}
    assert step["loss"] >= 0.0


def test_trajectory_replay_batch_sizes_equal_split() -> None:
    assert _trajectory_replay_batch_sizes(8) == (4, 4)
    assert _trajectory_replay_batch_sizes(9) == (5, 4)
    assert _trajectory_replay_batch_sizes(1) == (1, 1)


def test_field_replay_step_weights_losses_by_lambda() -> None:
    model = ConditionalFlowMatching(
        data_dim=4,
        n_contexts=2,
        n_perturbations=2,
        cond_emb_dim=8,
        hidden_dim=32,
        depth=2,
    )
    model_ref = make_frozen_model_copy(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator(device=torch.device("cpu"))
    device = torch.device("cpu")

    x1_ft = torch.randn(4, 4)
    cb_ft = torch.zeros(4, dtype=torch.long)
    pb_ft = torch.zeros(4, dtype=torch.long)
    c_ref = torch.ones(4, dtype=torch.long)
    p_ref = torch.ones(4, dtype=torch.long)

    gen.manual_seed(0)
    step_a = _generative_replay_field_step(
        model,
        model_ref,
        opt,
        x1_ft,
        cb_ft,
        pb_ft,
        c_ref,
        p_ref,
        lambda_mix=0.25,
        data_dim=4,
        device=device,
        gen=gen,
        replay_ode_method="euler",
        replay_ode_rtol=1e-5,
        replay_ode_atol=1e-5,
        replay_ode_options={"step_size": 0.25},
    )
    expected = 0.25 * step_a["loss_ref"] + 0.75 * step_a["loss_ft"]
    assert abs(step_a["loss"] - expected) < 1e-5
    assert step_a["lambda_mix"] == 0.25
