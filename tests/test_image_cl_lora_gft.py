"""Tests for the image-CL LoRA adapters and gradual fine-tuning trainer.

Runs as a plain script (``python tests/test_image_cl_lora_gft.py``) so it works
under ``sbatch/image_cl/run_lora_gft_tests.sbatch`` without a pytest dependency.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from experiments.image_cl.image_cl_common import (
    default_image_cl_run_name,
    train_one_task,
)
from trajectory_replay.model.diffusers_unet_fm import DiffusersUNetFlowMatching
from trajectory_replay.model.lora import (
    LoRAConv2d,
    has_lora,
    lora_parameters,
)
from trajectory_replay.training.gft_schedule import gft_beta_at_step

TINY_UNET = {
    "sample_size": 8,
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": 1,
    "block_out_channels": (32, 32),
    "norm_num_groups": 8,
    "down_block_types": ("DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D"),
}


def _tiny_model(num_classes: int = 0) -> DiffusersUNetFlowMatching:
    return DiffusersUNetFlowMatching(
        image_size=8,
        in_channels=1,
        unet_kwargs=TINY_UNET,
        num_classes=num_classes,
    )


def _assert_raises(exc: type[Exception], needle: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc as err:
        assert needle in str(err), f"expected {needle!r} in {err!r}"
        return
    raise AssertionError(f"expected {exc.__name__} mentioning {needle!r}")


def test_lora_conv_merge_matches_forward() -> None:
    for stride, padding in ((1, 1), (2, 1), (1, 0)):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 5, kernel_size=3, stride=stride, padding=padding)
        lor = LoRAConv2d(conv, rank=2, alpha=4.0)
        # lora_B starts at zero, so give it mass before checking the fold.
        with torch.no_grad():
            lor.lora_B.normal_()
        x = torch.randn(2, 3, 9, 9)
        lor.eval()
        adapted = lor(x)
        merged = lor.merge()
        assert isinstance(merged, nn.Conv2d)
        torch.testing.assert_close(adapted, merged(x), rtol=1e-5, atol=1e-5)


def test_lora_conv_rejects_grouped_convolutions() -> None:
    conv = nn.Conv2d(4, 4, kernel_size=3, groups=2)
    _assert_raises(ValueError, "groups=1", LoRAConv2d, conv, rank=2, alpha=4.0)


def test_unet_lora_wraps_convs_and_freezes_base() -> None:
    model = _tiny_model()
    n_conv_before = sum(1 for m in model.unet.modules() if isinstance(m, nn.Conv2d))
    model.apply_unet_lora(rank=2, alpha=4.0)
    assert has_lora(model.unet)
    adapters = lora_parameters(model.unet)
    assert adapters, "expected trainable adapter parameters"
    # Every adapted conv is now wrapped, leaving only the excluded output conv.
    remaining = [
        name
        for name, module in model.unet.named_modules()
        if isinstance(module, nn.Conv2d) and ".conv." not in f"{name}."
    ]
    assert remaining == ["conv_out"], remaining
    assert n_conv_before > 1
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in adapters}


def test_unet_lora_merge_is_identity_and_restores_state_dict() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    reference_keys = set(model.state_dict())
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0.25, 0.75])
    before = model(x, t)

    model.apply_unet_lora(rank=2, alpha=4.0)
    # Fresh adapters have lora_B = 0, so the wrapped model is unchanged.
    torch.testing.assert_close(before, model(x, t), rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        for parameter in lora_parameters(model.unet):
            parameter.normal_(std=0.05)
    adapted = model(x, t)
    model.merge_unet_lora()
    assert not has_lora(model.unet)
    assert set(model.state_dict()) == reference_keys
    torch.testing.assert_close(adapted, model(x, t), rtol=1e-5, atol=1e-5)
    assert all(p.requires_grad for p in model.unet.parameters())


def test_lora_training_leaves_output_conv_untouched() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    frozen_out = model.unet.conv_out.weight.detach().clone()
    frozen_in = model.unet.conv_in.weight.detach().clone()
    x = torch.randn(8, 1, 8, 8)
    history = train_one_task(
        "lora",
        model,
        None,
        x,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        lr=1e-2,
        train_seed=0,
        lambda_mix=None,
        traj_kappa="none",
        ode_method="euler",
        ode_step_size=0.5,
        ode_rtol=1e-5,
        ode_atol=1e-5,
        lora_rank=2,
        lora_alpha=4.0,
    )
    assert len(history) == 1
    assert not has_lora(model.unet)
    torch.testing.assert_close(model.unet.conv_out.weight, frozen_out)
    assert not torch.allclose(model.unet.conv_in.weight, frozen_in)


def test_gft_history_reports_schedule_and_matches_beta() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(8, 1, 8, 8)
    history = train_one_task(
        "gft",
        model,
        teacher,
        x,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
        lr=1e-3,
        train_seed=0,
        lambda_mix=None,
        traj_kappa="none",
        ode_method="euler",
        ode_step_size=0.5,
        ode_rtol=1e-5,
        ode_atol=1e-5,
        gft_beta_start=1.0,
        gft_beta_end=0.0,
        gft_beta_schedule="linear",
    )
    assert len(history) == 2
    for row in history:
        assert {"loss_fm_mean", "loss_reg_mean", "beta_mean"} <= set(row)
    # Linear cooling: the second epoch must sit strictly below the first.
    assert history[1]["beta_mean"] < history[0]["beta_mean"]


def test_gft_requires_a_teacher() -> None:
    model = _tiny_model()
    _assert_raises(
        ValueError,
        "requires a frozen model_ref",
        train_one_task,
        "gft",
        model,
        None,
        torch.randn(4, 1, 8, 8),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        lr=1e-3,
        train_seed=0,
        lambda_mix=None,
        traj_kappa="none",
        ode_method="euler",
        ode_step_size=0.5,
        ode_rtol=1e-5,
        ode_atol=1e-5,
    )


def test_gft_stays_closer_to_the_teacher_than_naive() -> None:
    """A large constant beta must pull the student toward the frozen field."""
    torch.manual_seed(0)
    base = _tiny_model()
    teacher = copy.deepcopy(base).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(16, 1, 8, 8)
    probe = torch.randn(4, 1, 8, 8)
    t = torch.full((4,), 0.5)
    with torch.no_grad():
        v_teacher = teacher(probe, t)

    drift = {}
    for name, kwargs in (
        ("naive", {}),
        (
            "gft",
            {
                "gft_beta_start": 50.0,
                "gft_beta_end": 50.0,
                "gft_beta_schedule": "constant",
            },
        ),
    ):
        student = copy.deepcopy(base)
        train_one_task(
            name,
            student,
            teacher if name == "gft" else None,
            x,
            device=torch.device("cpu"),
            epochs=2,
            batch_size=4,
            lr=1e-2,
            train_seed=0,
            lambda_mix=None,
            traj_kappa="none",
            ode_method="euler",
            ode_step_size=0.5,
            ode_rtol=1e-5,
            ode_atol=1e-5,
            **kwargs,
        )
        student.eval()
        with torch.no_grad():
            drift[name] = float((student(probe, t) - v_teacher).pow(2).mean())
    assert drift["gft"] < drift["naive"], drift


def test_class_conditional_gft_runs() -> None:
    from trajectory_replay.data.split_fmnist import make_xy_loader
    from trajectory_replay.training.mixture_fm import train_class_conditional_fm_gft

    torch.manual_seed(0)
    model = _tiny_model(num_classes=4)
    teacher = copy.deepcopy(model).eval()
    x = torch.randn(8, 1, 8, 8)
    y = torch.tensor([2, 2, 3, 3, 2, 3, 2, 3])
    loader = make_xy_loader(x, y, batch_size=4, shuffle=False, seed=0)
    history = train_class_conditional_fm_gft(
        model,
        teacher,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        train_seed=0,
        label_dropout=0.5,
    )
    assert len(history) == 1
    assert history[0]["loss_reg_mean"] >= 0.0


def test_gft_beta_schedule_endpoints() -> None:
    for schedule in ("linear", "sigmoid"):
        first = gft_beta_at_step(0, 10, beta_start=1.0, beta_end=0.0, schedule=schedule)
        last = gft_beta_at_step(9, 10, beta_start=1.0, beta_end=0.0, schedule=schedule)
        assert first > last
    constant = gft_beta_at_step(5, 10, beta_start=0.3, beta_end=0.0, schedule="constant")
    assert abs(constant - 0.3) < 1e-12


def test_run_names_carry_lora_rank_and_gft_schedule() -> None:
    lora = default_image_cl_run_name("split_fmnist", "lora", 42, lora_rank=16)
    assert lora == "image_cl/fmnist/lora_r16_s42"
    gft = default_image_cl_run_name("split_cifar10", "gft", 69, gft_beta_schedule="linear")
    assert gft == "image_cl/cifar10/gft_linear_s69"


if __name__ == "__main__":
    failures = 0
    for case_name, case in sorted(dict(globals()).items()):
        if not case_name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except Exception as err:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {case_name}: {type(err).__name__}: {err}", flush=True)
        else:
            print(f"PASS {case_name}", flush=True)
    raise SystemExit(1 if failures else 0)
