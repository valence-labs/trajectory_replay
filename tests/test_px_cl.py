from __future__ import annotations

import polars as pl
import torch

from experiments.px.cl_dataset import _stratified_train_val
from experiments.px.cl_trainer import equal_task_lambda, euler_rollout
from experiments.px.mean_flow import identity_loss_from_endpoint
from experiments.px.masip_trainer import (
    ddim_sample,
    diffusion_loss,
    make_ddim_scheduler,
)
from experiments.px.model import ADMMeanFlowWrapper, ADMWrapper, get_model_cls


def test_px_stratified_train_val_is_disjoint() -> None:
    frame = pl.DataFrame(
        {
            "path": [f"{index}.npy" for index in range(20)],
            "perturbation_id": [0] * 10 + [1] * 10,
        }
    )
    train, val = _stratified_train_val(frame, val_fraction=0.2, seed=7)
    assert train.height == 16
    assert val.height == 4
    assert set(train["path"]).isdisjoint(set(val["path"]))
    assert train["perturbation_id"].n_unique() == 2
    assert val["perturbation_id"].n_unique() == 2


def test_px_equal_task_lambda() -> None:
    assert equal_task_lambda(1) == 0.5
    assert equal_task_lambda(2) == 2 / 3


def test_px_adm_forward_and_rollout() -> None:
    model = ADMWrapper(
        y_dim=5,
        e_dim=3,
        c_dim=1,
        input_size=16,
        in_channels=2,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 1),
        attention_resolutions=(2,),
        dropout=0.0,
    )
    x = torch.randn(2, 2, 16, 16)
    t = torch.tensor([0.2, 0.7])
    y = torch.tensor([1, 2])
    e = torch.tensor([0, 1])
    c = torch.zeros(2, dtype=torch.long)
    output = model(x=x, t=t, y=y, e=e, c=c)
    assert output.shape == x.shape

    generated = euler_rollout(
        model,
        y=y,
        e=e,
        c=c,
        generator=torch.Generator().manual_seed(3),
        image_shape=(2, 16, 16),
        step_size=0.5,
    )
    assert generated.shape == x.shape
    assert torch.isfinite(generated).all()


def test_px_adm_factory() -> None:
    model_cls = get_model_cls("ADM")
    model = model_cls(
        y_dim=5,
        e_dim=3,
        c_dim=1,
        input_size=16,
        in_channels=2,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 1),
        attention_resolutions=(2,),
        dropout=0.0,
    )
    assert isinstance(model, ADMWrapper)


def test_px_adm_mean_flow_identity_and_sampling() -> None:
    model = ADMMeanFlowWrapper(
        y_dim=5,
        e_dim=3,
        c_dim=1,
        input_size=16,
        in_channels=2,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 1),
        attention_resolutions=(2,),
        dropout=0.0,
    )
    x1 = torch.randn(2, 2, 16, 16)
    y = torch.tensor([1, 2])
    e = torch.tensor([0, 1])
    c = torch.zeros(2, dtype=torch.long)
    generator = torch.Generator().manual_seed(5)
    loss = identity_loss_from_endpoint(
        model,
        x1=x1,
        y=y,
        e=e,
        c=c,
        generator=generator,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    generated = model.sample_from_x0(
        torch.randn_like(x1),
        y=y,
        e=e,
        c=c,
        n_steps=1,
    )
    assert generated.shape == x1.shape
    assert torch.isfinite(generated).all()


def test_px_matched_masip_diffusion_loss_and_sampling() -> None:
    scheduler = make_ddim_scheduler()
    scheduler.set_timesteps(20)
    assert scheduler.config.rescale_betas_zero_snr
    assert scheduler.timesteps.tolist() == list(range(999, 0, -50))
    assert scheduler.alphas_cumprod[-1] == 0
    model = ADMWrapper(
        y_dim=5,
        e_dim=3,
        c_dim=1,
        input_size=16,
        in_channels=2,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 1),
        attention_resolutions=(2,),
        dropout=0.0,
    )
    x1 = torch.randn(2, 2, 16, 16)
    y = torch.tensor([1, 2])
    e = torch.tensor([0, 1])
    c = torch.zeros(2, dtype=torch.long)
    generator = torch.Generator().manual_seed(9)
    loss = diffusion_loss(
        model,
        scheduler,
        x1=x1,
        y=y,
        e=e,
        c=c,
        generator=generator,
    )
    assert torch.isfinite(loss)
    loss.backward()
    generated = ddim_sample(
        model,
        y=y,
        e=e,
        c=c,
        generator=torch.Generator().manual_seed(10),
        image_shape=(2, 16, 16),
        num_steps=2,
    )
    assert generated.shape == x1.shape
    assert torch.isfinite(generated).all()
