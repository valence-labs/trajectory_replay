"""Tests for Split-FMNIST class orders and Diffusers UNet CFM wrapper."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from experiments.image_cl.image_cl_common import (
    apply_image_cl_dataset_defaults,
    default_image_cl_run_name,
    image_cl_in_channels,
    image_cl_results_slug,
    image_cl_unet_kwargs,
    make_model,
    normalize_image_cl_run_name,
    resolve_cumulative_task_epochs,
    resolve_kld_clf_path,
)
from trajectory_replay.data.split_cifar10 import (
    AVALANCHE_SPLIT_CIFAR10_TASKS,
    class_order_for_seed as cifar10_class_order_for_seed,
)
from trajectory_replay.data.split_fmnist import AVALANCHE_SPLIT_FMNIST_TASKS, class_order_for_seed
from trajectory_replay.model.diffusers_unet_fm import (
    DEFAULT_DDIM_MEDIUM_3CH,
    DEFAULT_DDIM_MEDIUM_4CH,
)
from trajectory_replay.model.diffusers_unet_fm import DiffusersUNetFlowMatching


def test_avalanche_class_orders_recorded() -> None:
    assert class_order_for_seed(42)[0] == [2, 6]
    assert class_order_for_seed(69)[0] == [0, 2]
    assert class_order_for_seed(1714)[0] == [3, 7]
    for seed, tasks in AVALANCHE_SPLIT_FMNIST_TASKS.items():
        flat = [c for t in tasks for c in t]
        assert sorted(flat) == list(range(10)), seed
        assert len(tasks) == 5


def test_diffusers_unet_forward_shapes() -> None:
    m = DiffusersUNetFlowMatching(image_size=32)
    m.eval()
    x = torch.randn(2, 1, 32, 32)
    t = torch.tensor([0.1, 0.7])
    v = m(x, t)
    assert v.shape == x.shape


def test_avalanche_cifar10_class_orders_recorded() -> None:
    assert cifar10_class_order_for_seed(42)[0] == [2, 6]
    assert cifar10_class_order_for_seed(69)[0] == [0, 2]
    assert cifar10_class_order_for_seed(1714)[0] == [3, 7]
    for seed, tasks in AVALANCHE_SPLIT_CIFAR10_TASKS.items():
        flat = [class_id for task in tasks for class_id in task]
        assert sorted(flat) == list(range(10)), seed
        assert len(tasks) == 5


def test_cifar10_model_uses_masip_3ch_backbone() -> None:
    assert image_cl_in_channels("split_cifar10") == 3
    kwargs = image_cl_unet_kwargs("split_cifar10")
    assert kwargs == DEFAULT_DDIM_MEDIUM_3CH
    assert kwargs["layers_per_block"] == 2
    assert kwargs["block_out_channels"] == (128, 256, 256, 256)
    model = make_model(
        Namespace(dataset="split_cifar10", image_size=32, train_timesteps=1000)
    )
    assert model.image_shape == (3, 32, 32)
    assert model.unet.config.in_channels == 3
    assert tuple(model.unet.config.block_out_channels) == (128, 256, 256, 256)


def test_imagenet100_model_uses_4ch_backbone() -> None:
    assert image_cl_results_slug("split_imagenet100") == "imagenet100"
    assert image_cl_in_channels("split_imagenet100") == 4
    kwargs = image_cl_unet_kwargs("split_imagenet100")
    assert kwargs == DEFAULT_DDIM_MEDIUM_4CH
    assert kwargs["in_channels"] == 4
    assert kwargs["out_channels"] == 4
    model = make_model(
        Namespace(dataset="split_imagenet100", image_size=32, train_timesteps=1000)
    )
    assert model.image_shape == (4, 32, 32)
    assert model.unet.config.in_channels == 4
    x = torch.randn(2, 4, 32, 32)
    t = torch.tensor([0.1, 0.7])
    assert model(x, t).shape == x.shape


def test_image_cl_dataset_defaults_and_kld_path() -> None:
    fmnist = Namespace(
        dataset="split_fmnist",
        epochs=None,
        batch_size=None,
        ode_step_size=None,
        n_experiences=None,
        cumulative_schedule=None,
        no_kld=False,
        kld_clf_path=None,
    )
    apply_image_cl_dataset_defaults(fmnist)
    assert fmnist.epochs == 100
    assert fmnist.batch_size == 128
    assert fmnist.ode_step_size == 0.1
    assert fmnist.cumulative_schedule == "epochs"
    assert fmnist.mean_flow_time_mu == 2.0
    assert fmnist.mean_flow_time_sigma == 2.0
    assert fmnist.mean_flow_diag_fraction == 0.25
    assert resolve_kld_clf_path(fmnist, Path(".")).name == "cnn_fmnist"

    cifar = Namespace(
        dataset="split_cifar10",
        epochs=None,
        batch_size=None,
        ode_step_size=None,
        n_experiences=None,
        cumulative_schedule=None,
        no_kld=False,
        kld_clf_path=None,
    )
    apply_image_cl_dataset_defaults(cifar)
    assert cifar.epochs == 200
    assert cifar.batch_size == 64
    assert cifar.ode_step_size == 0.05
    assert cifar.cumulative_schedule == "epochs"
    assert cifar.mean_flow_time_mu == 2.0
    assert cifar.mean_flow_time_sigma == 2.0
    assert cifar.mean_flow_diag_fraction == 0.25
    assert resolve_kld_clf_path(cifar, Path(".")).name == "cnn_cifar10"

    imagenet = Namespace(
        dataset="split_imagenet100",
        epochs=None,
        batch_size=None,
        ode_step_size=None,
        n_experiences=None,
        cumulative_schedule=None,
        image_size=256,
        no_kld=False,
        kld_clf_path=None,
    )
    apply_image_cl_dataset_defaults(imagenet)
    assert imagenet.epochs == 667
    assert imagenet.batch_size == 256
    assert imagenet.ode_step_size == 0.02
    assert imagenet.n_experiences == 3
    assert imagenet.cumulative_schedule == "steps"
    assert imagenet.image_size == 32  # forced to latent grid
    assert imagenet.mean_flow_time_mu == 0.4
    assert imagenet.mean_flow_time_sigma == 1.0
    assert imagenet.mean_flow_diag_fraction == 0.75
    assert resolve_kld_clf_path(imagenet, Path(".")) is None

    # Explicit overrides are preserved.
    override = Namespace(
        dataset="split_cifar10",
        epochs=12,
        batch_size=8,
        ode_step_size=0.2,
        cumulative_schedule="steps",
    )
    apply_image_cl_dataset_defaults(override)
    assert (override.epochs, override.batch_size, override.ode_step_size) == (12, 8, 0.2)
    assert override.cumulative_schedule == "steps"


def test_resolve_cumulative_task_epochs_steps_vs_epochs() -> None:
    # Fixed-epoch schedule: steps grow with the loader.
    ep, steps = resolve_cumulative_task_epochs(
        base_epochs=100,
        steps_per_epoch=300,
        reference_steps_per_epoch=150,
        cumulative_schedule="epochs",
    )
    assert ep == 100 and steps == 30_000

    # Fixed-step schedule: shrink epochs so Adam steps ≈ task0 budget.
    ep, steps = resolve_cumulative_task_epochs(
        base_epochs=667,
        steps_per_epoch=450,  # ~3× task0
        reference_steps_per_epoch=150,
        cumulative_schedule="steps",
    )
    assert steps == 667 * 150
    assert ep == steps // 450
    assert ep * 450 <= steps
    # Task0 itself is unchanged.
    ep0, steps0 = resolve_cumulative_task_epochs(
        base_epochs=667,
        steps_per_epoch=150,
        reference_steps_per_epoch=150,
        cumulative_schedule="steps",
    )
    assert ep0 == 667 and steps0 == 667 * 150


def test_fid_rgb_passthrough_for_3ch() -> None:
    from trajectory_replay.eval.image_fid import _to_01_rgb

    gray = torch.full((2, 1, 8, 8), -1.0)
    rgb_gray = _to_01_rgb(gray)
    assert rgb_gray.shape == (2, 3, 8, 8)
    assert torch.allclose(rgb_gray, torch.zeros_like(rgb_gray))

    color = torch.stack(
        [
            torch.full((2, 8, 8), -1.0),
            torch.zeros(2, 8, 8),
            torch.ones(2, 8, 8),
        ],
        dim=1,
    )
    rgb = _to_01_rgb(color)
    assert rgb.shape == (2, 3, 8, 8)
    assert torch.allclose(rgb[:, 0], torch.zeros(2, 8, 8))
    assert torch.allclose(rgb[:, 1], torch.full((2, 8, 8), 0.5))
    assert torch.allclose(rgb[:, 2], torch.ones(2, 8, 8))


def test_fid_to_pixels_hook_decodes_before_compute() -> None:
    """FID path must decode latents when ``to_pixels`` is provided."""
    from trajectory_replay.eval.image_fid import compute_fid, fid_against_reals

    class _Toy(torch.nn.Module):
        def sample(self, **kwargs):
            b = int(kwargs["batch_size"])
            return torch.zeros(b, 4, 2, 2)

    def to_pixels(z: torch.Tensor) -> torch.Tensor:
        # Map any latent batch to constant gray RGB in [-1, 1].
        return torch.full((z.shape[0], 3, 8, 8), -1.0)

    real_latents = torch.zeros(4, 4, 2, 2)
    # Monkeypatch compute_fid to assert inputs are already RGB.
    import trajectory_replay.eval.image_fid as fid_mod

    seen: list[tuple[int, ...]] = []

    def _fake_compute_fid(real, fake, *, device):
        seen.append(tuple(real.shape))
        seen.append(tuple(fake.shape))
        assert real.shape[1] == 3 and fake.shape[1] == 3
        return 0.0

    orig = fid_mod.compute_fid
    fid_mod.compute_fid = _fake_compute_fid  # type: ignore[assignment]
    try:
        score = fid_against_reals(
            _Toy(),
            real_latents,
            device=torch.device("cpu"),
            n_gen=4,
            batch_size=4,
            to_pixels=to_pixels,
        )
    finally:
        fid_mod.compute_fid = orig  # type: ignore[assignment]
    assert score == 0.0
    assert seen == [(4, 3, 8, 8), (4, 3, 8, 8)]


def test_image_cl_run_name_dataset_slugs() -> None:
    assert image_cl_results_slug("split_fmnist") == "fmnist"
    assert image_cl_results_slug("split_cifar10") == "cifar10"
    assert default_image_cl_run_name("split_fmnist", "trajectory", 42) == (
        "image_cl/fmnist/trajectory_s42"
    )
    assert default_image_cl_run_name(
        "split_fmnist",
        "trajectory",
        42,
        replay_backend="mean_flow",
        lambda_mode="equal",
    ) == "image_cl/fmnist/trajectory_mf_n1_eq_s42"
    assert default_image_cl_run_name("split_cifar10", "naive", 69) == (
        "image_cl/cifar10/naive_s69"
    )
    assert default_image_cl_run_name(
        "split_imagenet100",
        "naive",
        42,
        n_experiences=3,
        unet_size="large",
        class_conditional=True,
    ) == "image_cl/imagenet100/naive_s42"
    assert default_image_cl_run_name(
        "split_imagenet100",
        "naive",
        42,
        n_experiences=5,
        unet_size="large",
        class_conditional=True,
    ) == "image_cl/imagenet100/naive_e5_s42"
    assert default_image_cl_run_name(
        "split_imagenet100",
        "cumulative",
        42,
        class_conditional=True,
        n_experiences=3,
        unet_size="large",
    ) == "image_cl/imagenet100/cumulative_s42"
    assert default_image_cl_run_name(
        "split_imagenet100",
        "cumulative",
        42,
        class_conditional=False,
        unet_size="large",
    ) == "image_cl/imagenet100/cumulative_uncond_s42"
    assert default_image_cl_run_name(
        "split_imagenet100",
        "naive",
        42,
        class_conditional=True,
        unet_size="medium",
    ) == "image_cl/imagenet100/naive_medium_s42"
    # Legacy flat names get the dataset slug inserted.
    assert normalize_image_cl_run_name("image_cl/trajectory_s42", "split_fmnist") == (
        "image_cl/fmnist/trajectory_s42"
    )
    # Already-slugged names are left alone (even if slug ≠ current dataset).
    assert normalize_image_cl_run_name("image_cl/fmnist/x", "split_fmnist") == (
        "image_cl/fmnist/x"
    )
    assert normalize_image_cl_run_name("image_cl/cifar10/x", "split_fmnist") == (
        "image_cl/cifar10/x"
    )
    # Custom paths outside image_cl/ are untouched.
    assert normalize_image_cl_run_name("custom/path", "split_fmnist") == "custom/path"


def test_image_cl_reeval_reconstructs_legacy_protocol() -> None:
    from experiments.image_cl.reeval_ode_fid import (
        _canonical_eval_seed,
        _infer_n_experiences,
    )

    old_imagenet = {
        "dataset": "split_imagenet100",
        "task_classes": [[0], [1], [2]],
    }
    assert _infer_n_experiences(old_imagenet) == 3
    assert _infer_n_experiences({"dataset": "split_imagenet100"}) == 3
    assert _infer_n_experiences({"dataset": "split_cifar10"}) == 5
    assert _infer_n_experiences(
        {"dataset": "split_imagenet100", "n_experiences": 5}
    ) == 5
    cfg = {"dataset": "split_imagenet100", "eval_seed": 7}
    assert _canonical_eval_seed(cfg, task_id=2, stream=True) == 1009
    assert _canonical_eval_seed(cfg, task_id=2, stream=False) == 2043


def test_class_conditional_unet_forward_and_sample() -> None:
    # Tiny backbone so this stays a CPU unit test.
    tiny = {
        "sample_size": 8,
        "in_channels": 4,
        "out_channels": 4,
        "layers_per_block": 1,
        "block_out_channels": (32, 32),
        "norm_num_groups": 8,
        "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
    }
    m = DiffusersUNetFlowMatching(
        image_size=8,
        in_channels=4,
        unet_kwargs=tiny,
        num_classes=10,
        guidance_scale=1.5,
    )
    assert m.use_conditioning
    assert m.null_class_id == 10
    assert int(m.unet.config.num_class_embeds) == 11
    m.eval()
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.1, 0.7])
    y = torch.tensor([3, 7])
    v = m(x, t, y)
    assert v.shape == x.shape
    # CFG path (eval + scale≠1) still returns correct shape.
    v_cfg = m(x, t, y, guidance_scale=2.0)
    assert v_cfg.shape == x.shape
    out = m.sample(c=y, method="euler", options={"step_size": 0.5})
    assert out.shape == (2, 4, 8, 8)


def test_make_xy_loader_shapes() -> None:
    from trajectory_replay.data.split_fmnist import make_xy_loader

    x = torch.randn(5, 4, 8, 8)
    y = torch.arange(5)
    loader = make_xy_loader(x, y, batch_size=2, shuffle=False, seed=0)
    bx, by = next(iter(loader))
    assert bx.shape == (2, 4, 8, 8)
    assert by.shape == (2,)
    assert by.dtype == torch.long


def test_make_model_class_conditional_imagenet() -> None:
    model = make_model(
        Namespace(
            dataset="split_imagenet100",
            image_size=32,
            train_timesteps=1000,
            class_conditional=True,
            num_classes=100,
            guidance_scale=1.5,
        )
    )
    assert model.use_conditioning
    assert model.num_classes == 100
    assert int(model.unet.config.num_class_embeds) == 101


def test_class_conditional_ode_replay_trainers() -> None:
    from trajectory_replay.data.split_fmnist import make_xy_loader
    from trajectory_replay.training.mixture_fm import (
        train_class_conditional_generative_replay,
        train_class_conditional_trajectory_replay,
    )

    tiny = {
        "sample_size": 8,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": (32, 32),
        "norm_num_groups": 8,
        "down_block_types": ("DownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "UpBlock2D"),
    }
    device = torch.device("cpu")
    model = DiffusersUNetFlowMatching(
        image_size=8, in_channels=1, unet_kwargs=tiny, num_classes=4, guidance_scale=1.5
    ).to(device)
    teacher = DiffusersUNetFlowMatching(
        image_size=8, in_channels=1, unet_kwargs=tiny, num_classes=4, guidance_scale=1.5
    ).to(device)
    teacher.eval()
    x = torch.randn(8, 1, 8, 8)
    y = torch.tensor([2, 2, 3, 3, 2, 3, 2, 3])
    loader = make_xy_loader(x, y, batch_size=4, shuffle=False, seed=0)
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
        ode_step_size=0.5,
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
        ode_step_size=0.5,
    )
    assert hist_g and hist_t
    assert "loss_old_mean" in hist_g[0] and "loss_old_mean" in hist_t[0]
