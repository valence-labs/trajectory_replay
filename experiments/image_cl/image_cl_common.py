"""Shared helpers for Split-FMNIST / CIFAR-10 / ImageNet-100 image CL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from trajectory_replay.data.split_cifar10 import (
    AVALANCHE_SPLIT_CIFAR10_TASKS,
    build_split_cifar10_tasks,
)
from trajectory_replay.data.split_fmnist import (
    AVALANCHE_SPLIT_FMNIST_TASKS,
    TaskSplit,
    build_split_fmnist_tasks,
    make_x_loader,
    make_xy_loader,
)
from trajectory_replay.data.split_imagenet import (
    DEFAULT_LATENT_DIR as DEFAULT_IMAGENET100_LATENT_DIR,
    SPLIT_IMAGENET100_TASKS,
    build_split_imagenet100_tasks,
)
from trajectory_replay.io.results import run_results_dir
from trajectory_replay.model.diffusers_unet_fm import (
    DEFAULT_DDIM_LARGE_4CH,
    DEFAULT_DDIM_MEDIUM,
    DEFAULT_DDIM_MEDIUM_3CH,
    DEFAULT_DDIM_MEDIUM_4CH,
    DiffusersUNetFlowMatching,
    build_diffusers_unet_fm_from_config,
    disable_fused_sdpa_if_needed,
    make_frozen_diffusers_copy,
)
from trajectory_replay.model.lora import lora_parameters
from trajectory_replay.training.gft_schedule import GFT_BETA_SCHEDULES
from trajectory_replay.training.mixture_fm import (
    train_class_conditional_fm,
    train_class_conditional_fm_gft,
    train_class_conditional_generative_replay,
    train_class_conditional_trajectory_replay,
    train_unconditional_fm,
    train_unconditional_fm_gft,
    train_unconditional_generative_replay,
    train_unconditional_trajectory_replay,
)


DEFAULT_LAMBDA_GRID = (0.1, 0.25, 0.5, 0.75, 0.9)

# Strategies that train on the incoming task only and never draw replay samples.
NO_REPLAY_STRATEGIES = ("naive", "cumulative", "joint", "lora", "gft")

# Dataset CLI name → results_dir/image_cl/<slug>/...
DATASET_RESULTS_SLUG: dict[str, str] = {
    "split_fmnist": "fmnist",
    "split_cifar10": "cifar10",
    "split_imagenet100": "imagenet100",
}

# Masip protocol defaults (FMNIST: 100 ep / batch 128 / N=10;
# CIFAR-10: 200 ep / batch 64 / N=20).
# ImageNet-100 latent FM: batch 256 (stabler Adam updates than the old 128
# FMNIST-inherited default; Lipman ImageNet-32 uses effective 1024).
# ImageNet uses 3 tasks (34/33/33 classes) by default — not Masip's 5×20;
# pass --n-experiences 5 for the Masip-matched ablation.
IMAGE_CL_DATASET_DEFAULTS: dict[str, dict[str, float | int | str]] = {
    "split_fmnist": {
        "epochs": 100,
        "batch_size": 128,
        "ode_step_size": 0.1,
        "n_experiences": 5,
        "cumulative_schedule": "epochs",
    },
    "split_cifar10": {
        "epochs": 200,
        "batch_size": 64,
        "ode_step_size": 0.05,
        "n_experiences": 5,
        "cumulative_schedule": "epochs",
    },
    "split_imagenet100": {
        # ~100k Adam steps/task @ batch 256 on a ~34-class experience
        # (pilot: 100k beat 200k on FID for medium/large, no EMA).
        "epochs": 667,
        "batch_size": 256,
        "ode_step_size": 0.02,  # Euler ~50 steps (pilots used 0.05 / ~20)
        "n_experiences": 3,
        # Growing |D| under fixed epochs blows up walltime; keep ~100k steps/task.
        "cumulative_schedule": "steps",
    },
}


def resolve_cumulative_task_epochs(
    *,
    base_epochs: int,
    steps_per_epoch: int,
    reference_steps_per_epoch: int,
    cumulative_schedule: str,
) -> tuple[int, int]:
    """Epochs (and target Adam steps) for one cumulative/joint task.

    ``epochs``: Masip/Avalanche-style — fixed epochs while the real-data union grows
    (CIFAR/FMNIST default).

    ``steps``: keep Adam steps ≈ ``base_epochs * reference_steps_per_epoch`` (usually
    task-0 sized) by shrinking epochs as ``steps_per_epoch`` grows (ImageNet default).
    """
    schedule = str(cumulative_schedule)
    if schedule not in ("epochs", "steps"):
        raise ValueError(
            f"cumulative_schedule must be 'epochs' or 'steps', got {schedule!r}"
        )
    spe = max(1, int(steps_per_epoch))
    if schedule == "epochs":
        target_steps = int(base_epochs) * spe
        return int(base_epochs), target_steps
    ref = max(1, int(reference_steps_per_epoch))
    target_steps = max(1, int(base_epochs) * ref)
    return max(1, target_steps // spe), target_steps


def image_cl_results_slug(dataset: str) -> str:
    """Map ``--dataset`` to the ``results_dir/image_cl/<slug>/`` folder name."""
    try:
        return DATASET_RESULTS_SLUG[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset={dataset!r}; known: {sorted(DATASET_RESULTS_SLUG)}"
        ) from exc


def default_image_cl_run_name(
    dataset: str,
    strategy: str,
    seed: int,
    *,
    replay_backend: str = "ode",
    lambda_mode: str = "grid",
    class_conditional: bool = False,
    unet_size: str = "medium",
    n_experiences: int | None = None,
    mean_flow_steps: int = 1,
    lora_rank: int = 8,
    gft_beta_schedule: str = "sigmoid",
) -> str:
    """Default ``--run-name``: CIFAR/FMNIST-style ``image_cl/<slug>/<strategy>[_tags]_s<seed>``.

    ImageNet locked recipe (class-cond, large, 3 experiences) uses clean names
    like ``naive_s42`` / ``cumulative_s42``. Tags are only added for deviations
    (``_uncond``, ``_medium``, ``_e5``, …) or λ/replay axes (``_mf_nN``, ``_eq``).
    """
    suffix = ""
    if dataset == "split_imagenet100":
        if not class_conditional:
            suffix += "_uncond"
        if str(unet_size) != "large":
            suffix += f"_{unet_size}"
        if n_experiences is not None and int(n_experiences) != 3:
            suffix += f"_e{int(n_experiences)}"
    else:
        if class_conditional:
            suffix += "_cc"
        if str(unet_size) == "large":
            suffix += "_large"
        if n_experiences is not None and int(n_experiences) != 5:
            suffix += f"_e{int(n_experiences)}"
    if replay_backend == "mean_flow" and strategy in ("generative", "trajectory"):
        suffix += f"_mf_n{int(mean_flow_steps)}"
    if strategy in ("cumulative", "joint"):
        # Joint upper bound — no λ / replay-backend suffix.
        return f"image_cl/{image_cl_results_slug(dataset)}/cumulative{suffix}_s{seed}"
    slug = image_cl_results_slug(dataset)
    if strategy == "lora":
        # Rank is part of the identity: the grid must not clobber itself.
        return f"image_cl/{slug}/lora_r{int(lora_rank)}{suffix}_s{seed}"
    if strategy == "gft":
        return f"image_cl/{slug}/gft_{gft_beta_schedule}{suffix}_s{seed}"
    if lambda_mode == "equal" and strategy in ("generative", "trajectory"):
        suffix += "_eq"
    return f"image_cl/{image_cl_results_slug(dataset)}/{strategy}{suffix}_s{seed}"


def normalize_image_cl_run_name(run_name: str, dataset: str) -> str:
    """Ensure ``image_cl/...`` run names sit under the dataset slug folder.

    - ``image_cl/fmnist/foo`` / ``image_cl/cifar10/foo`` → unchanged (already slugged).
    - ``image_cl/foo`` → ``image_cl/<slug>/foo`` (legacy flat names).
    - Anything else (custom paths under ``results_dir/``) → unchanged.
    """
    slug = image_cl_results_slug(dataset)
    prefix = "image_cl/"
    if not run_name.startswith(prefix):
        return run_name
    rest = run_name[len(prefix) :]
    known = set(DATASET_RESULTS_SLUG.values())
    first = rest.split("/", 1)[0]
    if first in known:
        return run_name
    return f"{prefix}{slug}/{rest}"


def equal_weight_lambda(task_id: int) -> float:
    """λ on the *old* term for equal weight over n=task_id+1 tasks: λ=(n-1)/n."""
    if task_id < 1:
        raise ValueError(f"equal_weight_lambda is for task_id>=1, got {task_id}")
    n = task_id + 1
    return float(task_id) / float(n)


def add_image_cl_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset",
        default="split_fmnist",
        choices=("split_fmnist", "split_cifar10", "split_imagenet100"),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Torchvision dataset root for FMNIST/CIFAR (default: <repo>/data_dir/torchvision).",
    )
    p.add_argument(
        "--latent-dir",
        type=Path,
        default=None,
        help=(
            "Cached SD-VAE latents for split_imagenet100 "
            f"(default: {DEFAULT_IMAGENET100_LATENT_DIR})."
        ),
    )
    p.add_argument(
        "--vae-id",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="HuggingFace VAE id used to decode latents for FID / sample grids "
        "(split_imagenet100 only).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Spatial size of the FM state. For split_imagenet100 this is the "
        "latent grid (32), not the pixel resolution (256).",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--n-experiences",
        type=int,
        default=None,
        help=(
            "Number of CL tasks. Default: 5 (FMNIST/CIFAR Masip), "
            "3 (ImageNet-100, 34/33/33 classes)."
        ),
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Stop after this many tasks (e.g. 1 for a single-task pilot). Default: all.",
    )


def add_image_cl_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-timesteps", type=int, default=1000, help="UNet time-embed scale.")
    p.add_argument(
        "--unet-size",
        choices=("medium", "large"),
        default="medium",
        help=(
            "UNet capacity. medium=Masip ddim_medium (~36M, default). "
            "large=LDM-style (128,256,512,512) latent UNet (~100M); "
            "intended for split_imagenet100 pilots."
        ),
    )
    p.add_argument(
        "--class-conditional",
        action="store_true",
        help=(
            "Train / sample with semantic class labels (Diffusers num_class_embeds). "
            "Supported for naive/cumulative; intended as an ImageNet-100 diagnostic."
        ),
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Label space size for --class-conditional (default: 100 ImageNet / 10 else).",
    )
    p.add_argument(
        "--label-dropout",
        type=float,
        default=0.1,
        help="CFG train dropout probability for class labels (null class).",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale at sample / FID / KLD time.",
    )


def add_image_cl_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--strategy",
        choices=(
            "naive",
            "cumulative",
            "joint",  # alias of cumulative (Masip naming)
            "lora",
            "gft",
            "generative",
            "trajectory",
        ),
        required=True,
        help="cumulative/joint: train CFM on all real data seen so far (upper target). "
        "lora/gft: memory-free finetuning controls that never replay.",
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument(
        "--lora-train-output-layer",
        action="store_true",
        help="Also adapt the UNet output convolution (frozen by default).",
    )
    p.add_argument("--gft-beta-start", type=float, default=1.0)
    p.add_argument("--gft-beta-end", type=float, default=0.0)
    p.add_argument(
        "--gft-beta-schedule",
        choices=GFT_BETA_SCHEDULES,
        default="sigmoid",
    )
    p.add_argument("--gft-sigmoid-midpoint", type=float, default=0.5)
    p.add_argument("--gft-sigmoid-steepness", type=float, default=10.0)
    p.add_argument(
        "--cumulative-schedule",
        choices=("epochs", "steps"),
        default=None,
        help=(
            "How to size each cumulative/joint task. "
            "'epochs': fixed --epochs while the real-data union grows (FMNIST/CIFAR). "
            "'steps': keep Adam steps ≈ epochs×steps/epoch(task0) by shrinking epochs "
            "as data accumulates (ImageNet default)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Train batch size. Default: 128 (FMNIST), 64 (CIFAR-10), 256 (ImageNet-100).",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help=(
            "Global grad-norm clip for CFM / replay trainers (0 disables). "
            "Use 1.0 for large UNets that otherwise spike."
        ),
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs per task. Default: 100 (FMNIST) or 200 (CIFAR-10).",
    )
    p.add_argument("--seed", type=int, default=42, help="CL stream seed (also Avalanche class order).")
    p.add_argument("--train-seed", type=int, default=None, help="Default: --seed.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_GRID),
        help="λ candidates when --lambda-mode=grid (generative/trajectory, task k>=1).",
    )
    p.add_argument(
        "--lambda-mode",
        choices=("grid", "equal"),
        default="grid",
        help="grid: select λ by val FID over --lambda-grid. "
        "equal: fixed λ=(n-1)/n at task k (n=k+1 tasks seen) — equal task weight, no sweep.",
    )
    p.add_argument(
        "--traj-kappa",
        choices=("none", "ot"),
        default="none",
        help="Trajectory time weight (Thm. 2.4). Ignored for naive/generative.",
    )
    p.add_argument(
        "--replay-backend",
        choices=("ode", "mean_flow"),
        default="ode",
        help=(
            "ode: Euler teacher replay. mean_flow: task-0 post-hoc distill + "
            "joint CFM/MeanFlow on later tasks; replay uses MF^{k-1} with "
            "--mean-flow-steps jumps (trajectory shares one rollout)."
        ),
    )
    p.add_argument(
        "--mean-flow-steps",
        type=int,
        default=1,
        help=(
            "MeanFlow jumps for replay when --replay-backend=mean_flow "
            "(N=1 one-step; use N=4 for multi-jump table rows)."
        ),
    )
    p.add_argument(
        "--distill-epochs",
        type=int,
        default=50,
        help="Task-0 MeanFlow epochs (k>=1 uses --epochs for the joint loop).",
    )
    p.add_argument(
        "--distill-lr",
        type=float,
        default=1e-4,
        help=(
            "Task-0 MeanFlow LR. Ignored whenever --mean-flow-lr is set (it always "
            "is via sbatch), so override MEAN_FLOW_LR instead."
        ),
    )
    p.add_argument(
        "--distill-batch-size",
        type=int,
        default=128,
        help="Task-0 MeanFlow distill batch size (default: 128; sbatch matches CFM --batch-size).",
    )
    p.add_argument(
        "--mean-flow-hidden-dim",
        type=int,
        default=512,
        help="Unused for image UNet MeanFlow (kept for CLI compat).",
    )
    p.add_argument(
        "--mean-flow-depth",
        type=int,
        default=6,
        help="Unused for image UNet MeanFlow (kept for CLI compat).",
    )
    p.add_argument(
        "--mean-flow-diag-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of MeanFlow (r,t) samples on the diagonal "
            "(zhuyu ratio_r_not_equal_t = 1 - this). Default: 0.25 "
            "(FMNIST/CIFAR), 0.75 (ImageNet)."
        ),
    )
    p.add_argument(
        "--mean-flow-adaptive-p",
        type=float,
        default=None,
        help=(
            "Adaptive MSE weighting power (zhuyu) for task-0 MeanFlow only; the "
            "joint CL loss always uses unweighted MSE so λ mixes comparable "
            "errors. Default: 0.75 (FMNIST/CIFAR), 1.0 (ImageNet). 0 disables."
        ),
    )
    p.add_argument(
        "--mean-flow-time-sampler",
        default="logit_normal",
        help="MeanFlow (r,t) sampler name.",
    )
    p.add_argument(
        "--mean-flow-time-mu",
        type=float,
        default=None,
        help=(
            "Logit-normal mu in this repo's noise-to-data convention. "
            "Default: +0.4 (ImageNet), +2.0 (FMNIST/CIFAR), mirroring the "
            "reference data-to-noise schedules."
        ),
    )
    p.add_argument(
        "--mean-flow-time-sigma",
        type=float,
        default=None,
        help="Logit-normal sigma. Default: 1.0 (ImageNet), 2.0 (FMNIST/CIFAR).",
    )
    p.add_argument(
        "--mean-flow-lr",
        type=float,
        default=1e-4,
        help="Joint MeanFlow LR (default: 1e-4; independent of CFM --lr).",
    )
    p.add_argument(
        "--mean-flow-max-grad-norm",
        type=float,
        default=1.0,
        help="Grad clip for the MeanFlow optimizer (0 disables).",
    )
    p.add_argument(
        "--mean-flow-use-adamw",
        type=int,
        default=1,
        choices=(0, 1),
        help="Use AdamW for MeanFlow (1) vs Adam (0).",
    )
    p.add_argument(
        "--mean-flow-adam-beta2",
        type=float,
        default=0.95,
        help="AdamW beta2 for MeanFlow (zhuyu recipe: 0.95).",
    )
    p.add_argument(
        "--mean-flow-weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay for MeanFlow.",
    )


def add_image_cl_ode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ode-method", default="euler")
    p.add_argument(
        "--ode-step-size",
        type=float,
        default=None,
        help=(
            "Euler step size for FID evaluation (and λ-selection val FID). "
            "Default: 0.1 (~10 steps, FMNIST), 0.05 (~20, CIFAR-10), 0.02 (~50, ImageNet-100)."
        ),
    )
    p.add_argument(
        "--replay-ode-step-size",
        type=float,
        default=None,
        help=(
            "Euler step size for ODE teacher replay during generative/trajectory training. "
            "Defaults to --ode-step-size when unset."
        ),
    )
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)


def apply_image_cl_dataset_defaults(args: argparse.Namespace) -> None:
    """Fill unset train/ODE knobs from the Masip dataset protocol."""
    try:
        defaults = IMAGE_CL_DATASET_DEFAULTS[args.dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported image-CL dataset: {args.dataset!r}") from exc
    if args.epochs is None:
        args.epochs = int(defaults["epochs"])
    if args.batch_size is None:
        args.batch_size = int(defaults["batch_size"])
    if args.ode_step_size is None:
        args.ode_step_size = float(defaults["ode_step_size"])
    if getattr(args, "replay_ode_step_size", None) is None:
        args.replay_ode_step_size = float(args.ode_step_size)
    if getattr(args, "n_experiences", None) is None:
        args.n_experiences = int(defaults["n_experiences"])
    if getattr(args, "cumulative_schedule", None) is None:
        args.cumulative_schedule = str(defaults["cumulative_schedule"])
    # Latent FM operates on the 32×32 VAE grid; ignore accidental pixel sizes.
    if args.dataset == "split_imagenet100":
        args.image_size = 32
    # MeanFlow uses noise→data time, opposite to zhuyu's data→noise path.
    # Mirror the reference logit-normal distributions by flipping μ:
    # CIFAR μ=-2 → +2; ImageNet μ=-0.4 → +0.4. σ and diagonal mass are unchanged.
    if getattr(args, "mean_flow_time_mu", None) is None:
        args.mean_flow_time_mu = 0.4 if args.dataset == "split_imagenet100" else 2.0
    if getattr(args, "mean_flow_time_sigma", None) is None:
        args.mean_flow_time_sigma = 1.0 if args.dataset == "split_imagenet100" else 2.0
    if getattr(args, "mean_flow_diag_fraction", None) is None:
        args.mean_flow_diag_fraction = (
            0.75 if args.dataset == "split_imagenet100" else 0.25
        )
    if getattr(args, "mean_flow_adaptive_p", None) is None:
        args.mean_flow_adaptive_p = (
            1.0 if args.dataset == "split_imagenet100" else 0.75
        )


def add_image_cl_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fid-val-samples",
        type=int,
        default=2048,
        help="Generated samples per task for λ selection (val FID).",
    )
    p.add_argument(
        "--fid-test-samples",
        type=int,
        default=10000,
        help="Generated samples for test FID / KLD (Masip uses 10k).",
    )
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument(
        "--kld-clf-path",
        type=Path,
        default=None,
        help="Dir with resnet.pth. Default selects Masip's cnn_fmnist or "
        "cnn_cifar10 weights based on --dataset. Unused for split_imagenet100 "
        "until a pretrained ImageNet KLD classifier is wired.",
    )
    p.add_argument(
        "--no-kld",
        action="store_true",
        help="Skip stream KLD evaluation.",
    )
    p.add_argument(
        "--per-class-fid-samples",
        type=int,
        default=0,
        help=(
            "If >0, also report mean per-class FID with this many gens/class "
            "(recommended for --class-conditional; 0 disables)."
        ),
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(args.device)


def resolve_data_root(args: argparse.Namespace, repo_root: Path) -> Path:
    return args.data_root if args.data_root is not None else repo_root / "data_dir" / "torchvision"


def resolve_latent_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "latent_dir", None) is not None:
        return Path(args.latent_dir)
    return Path(DEFAULT_IMAGENET100_LATENT_DIR)


def resolve_latent_to_pixels(
    args: argparse.Namespace,
    device: torch.device,
) -> Any | None:
    """Return a ``to_pixels`` callable for latent FM datasets, else ``None``.

    The callable maps scaled SD-VAE latents ``(N, 4, 32, 32)`` → RGB
    ``(N, 3, 256, 256)`` in ``[-1, 1]`` for FID / sample grids.
    """
    if args.dataset != "split_imagenet100":
        return None
    from trajectory_replay.model.sd_vae import FrozenSDVAE

    vae_id = str(getattr(args, "vae_id", "stabilityai/sd-vae-ft-mse"))
    print(f"Loading frozen VAE for latent→pixel decode: {vae_id}")
    vae = FrozenSDVAE.from_pretrained(vae_id, device=device)
    decode_bs = max(1, int(getattr(args, "fid_batch_size", 32)) // 2)

    def to_pixels(z: torch.Tensor) -> torch.Tensor:
        return vae.decode(z, batch_size=decode_bs)

    # Keep a handle so the VAE isn't GC'd while the closure lives.
    to_pixels.vae = vae  # type: ignore[attr-defined]
    return to_pixels


def build_image_cl_tasks(args: argparse.Namespace, repo_root: Path) -> list[TaskSplit]:
    """Dispatch the selected image-CL dataset using a common task API."""
    if args.dataset == "split_imagenet100":
        return build_split_imagenet100_tasks(
            seed=int(args.seed),
            latent_dir=resolve_latent_dir(args),
            n_experiences=int(args.n_experiences),
            val_fraction=float(args.val_fraction),
        )
    kwargs = {
        "seed": int(args.seed),
        "data_root": resolve_data_root(args, repo_root),
        "image_size": int(args.image_size),
        "n_experiences": int(args.n_experiences),
        "val_fraction": float(args.val_fraction),
    }
    if args.dataset == "split_fmnist":
        return build_split_fmnist_tasks(**kwargs)
    if args.dataset == "split_cifar10":
        return build_split_cifar10_tasks(**kwargs)
    raise ValueError(f"Unsupported image-CL dataset: {args.dataset!r}")


def image_cl_in_channels(dataset: str) -> int:
    if dataset == "split_fmnist":
        return 1
    if dataset == "split_cifar10":
        return 3
    if dataset == "split_imagenet100":
        return 4
    raise ValueError(f"Unsupported image-CL dataset: {dataset!r}")


def image_cl_unet_kwargs(dataset: str, unet_size: str = "medium") -> dict[str, Any]:
    """Return the backbone config for the selected dataset / size."""
    size = str(unet_size)
    if dataset == "split_fmnist":
        if size != "medium":
            raise ValueError(f"unet-size={size!r} is only supported for split_imagenet100")
        return dict(DEFAULT_DDIM_MEDIUM)
    if dataset == "split_cifar10":
        if size != "medium":
            raise ValueError(f"unet-size={size!r} is only supported for split_imagenet100")
        return dict(DEFAULT_DDIM_MEDIUM_3CH)
    if dataset == "split_imagenet100":
        if size == "medium":
            return dict(DEFAULT_DDIM_MEDIUM_4CH)
        if size == "large":
            return dict(DEFAULT_DDIM_LARGE_4CH)
        raise ValueError(f"Unknown unet-size={size!r}")
    raise ValueError(f"Unsupported image-CL dataset: {dataset!r}")


def image_cl_uses_hflip(dataset: str) -> bool:
    """CIFAR / ImageNet-100 use stochastic horizontal flips at train time."""
    return dataset in ("split_cifar10", "split_imagenet100")


def image_cl_num_classes(dataset: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    if dataset == "split_imagenet100":
        return 100
    if dataset in ("split_fmnist", "split_cifar10"):
        return 10
    raise ValueError(f"Unsupported image-CL dataset: {dataset!r}")


def make_model(args: argparse.Namespace) -> DiffusersUNetFlowMatching:
    disable_fused_sdpa_if_needed()
    in_channels = image_cl_in_channels(args.dataset)
    unet_size = str(getattr(args, "unet_size", "medium"))
    unet_kwargs = image_cl_unet_kwargs(args.dataset, unet_size=unet_size)
    class_conditional = bool(getattr(args, "class_conditional", False))
    num_classes = (
        image_cl_num_classes(args.dataset, getattr(args, "num_classes", None))
        if class_conditional
        else 0
    )
    guidance_scale = float(getattr(args, "guidance_scale", 1.0))
    return DiffusersUNetFlowMatching(
        image_size=args.image_size,
        in_channels=in_channels,
        train_timesteps=args.train_timesteps,
        unet_kwargs=unet_kwargs,
        num_classes=num_classes,
        guidance_scale=guidance_scale,
    )


def save_checkpoint(path: Path, model: DiffusersUNetFlowMatching, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[DiffusersUNetFlowMatching, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = build_diffusers_unet_fm_from_config(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, cfg


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def train_one_task(
    strategy: str,
    model: DiffusersUNetFlowMatching,
    model_ref: DiffusersUNetFlowMatching | None,
    x_train: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    train_seed: int,
    lambda_mix: float | None,
    traj_kappa: str,
    ode_method: str,
    ode_step_size: float,
    ode_rtol: float,
    ode_atol: float,
    mean_flow: Any | None = None,
    mean_flow_n_steps: int = 1,
    joint_mean_flow: Any | None = None,
    mf_lr: float | None = None,
    diag_fraction: float = 0.75,
    adaptive_p: float | None = None,
    time_sampler: str = "logit_normal",
    time_mu: float = 2.0,
    time_sigma: float = 2.0,
    mf_max_grad_norm: float | None = 1.0,
    mf_use_adamw: bool = True,
    mf_betas: tuple[float, float] = (0.9, 0.95),
    mf_weight_decay: float = 0.0,
    random_horizontal_flip: bool = False,
    y_train: torch.Tensor | None = None,
    old_class_ids: list[int] | tuple[int, ...] | None = None,
    label_dropout: float = 0.1,
    max_grad_norm: float | None = None,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_train_output_layer: bool = False,
    gft_beta_start: float = 1.0,
    gft_beta_end: float = 0.0,
    gft_beta_schedule: str = "sigmoid",
    gft_sigmoid_midpoint: float = 0.5,
    gft_sigmoid_steepness: float = 10.0,
) -> list[dict[str, float]]:
    class_cond = bool(getattr(model, "use_conditioning", False))
    clip = (
        float(max_grad_norm)
        if max_grad_norm is not None and float(max_grad_norm) > 0.0
        else None
    )
    if class_cond:
        if y_train is None:
            raise ValueError("y_train is required for class-conditional training")
        loader = make_xy_loader(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=train_seed,
            random_horizontal_flip=random_horizontal_flip,
        )
    else:
        loader = make_x_loader(
            x_train,
            batch_size=batch_size,
            shuffle=True,
            seed=train_seed,
            random_horizontal_flip=random_horizontal_flip,
        )

    if joint_mean_flow is not None:
        if model_ref is None:
            raise ValueError("joint_mean_flow requires a frozen model_ref teacher")
        if strategy not in ("generative", "trajectory"):
            raise ValueError(
                f"joint MeanFlow requires generative|trajectory, got {strategy!r}"
            )
        if lambda_mix is None:
            raise ValueError("joint MeanFlow requires lambda_mix")
        if class_cond and not old_class_ids:
            raise ValueError("old_class_ids required for class-conditional joint MeanFlow")
        from trajectory_replay.model.image_mean_flow import train_image_joint_mean_flow

        return train_image_joint_mean_flow(
            model,
            joint_mean_flow,
            model_ref,
            loader,
            strategy=strategy,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            lambda_mix=float(lambda_mix),
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_step_size=ode_step_size,
            traj_kappa=traj_kappa,
            replay_mean_flow=mean_flow,
            mean_flow_n_steps=mean_flow_n_steps,
            old_class_ids=list(old_class_ids) if old_class_ids is not None else None,
            label_dropout=float(label_dropout),
            mf_lr=mf_lr,
            diag_fraction=diag_fraction,
            adaptive_p=adaptive_p,
            time_sampler=time_sampler,
            time_mu=time_mu,
            time_sigma=time_sigma,
            max_grad_norm=clip,
            mf_max_grad_norm=mf_max_grad_norm,
            mf_use_adamw=mf_use_adamw,
            mf_betas=mf_betas,
            mf_weight_decay=mf_weight_decay,
        )

    if strategy == "lora":
        # Freeze the backbone, train only the adapters, then fold them back in so
        # the checkpoint written after this task is an ordinary UNet.
        model.apply_unet_lora(
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            train_output_layer=bool(lora_train_output_layer),
        )
        adapter_params = lora_parameters(model.unet)
        if not adapter_params:
            raise ValueError("LoRA produced no trainable parameters")
        try:
            if class_cond:
                history = train_class_conditional_fm(
                    model,
                    loader,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    train_seed=train_seed,
                    label_dropout=float(label_dropout),
                    params=adapter_params,
                    max_grad_norm=clip,
                )
            else:
                history = train_unconditional_fm(
                    model,
                    loader,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    train_seed=train_seed,
                    params=adapter_params,
                    max_grad_norm=clip,
                )
        finally:
            model.merge_unet_lora()
        return history

    if strategy == "gft":
        if model_ref is None:
            raise ValueError("gradual fine-tuning requires a frozen model_ref teacher")
        gft_kwargs: dict[str, Any] = {
            "device": device,
            "epochs": epochs,
            "lr": lr,
            "train_seed": train_seed,
            "beta_start": float(gft_beta_start),
            "beta_end": float(gft_beta_end),
            "beta_schedule": str(gft_beta_schedule),
            "sigmoid_midpoint": float(gft_sigmoid_midpoint),
            "sigmoid_steepness": float(gft_sigmoid_steepness),
            "max_grad_norm": clip,
        }
        if class_cond:
            return train_class_conditional_fm_gft(
                model,
                model_ref,
                loader,
                label_dropout=float(label_dropout),
                **gft_kwargs,
            )
        return train_unconditional_fm_gft(model, model_ref, loader, **gft_kwargs)

    if class_cond:
        if strategy in ("naive", "cumulative", "joint") or model_ref is None:
            return train_class_conditional_fm(
                model,
                loader,
                device=device,
                epochs=epochs,
                lr=lr,
                train_seed=train_seed,
                label_dropout=float(label_dropout),
                max_grad_norm=clip,
            )
        assert lambda_mix is not None
        if not old_class_ids:
            raise ValueError(
                "old_class_ids is required for class-conditional generative/trajectory replay"
            )
        if strategy == "generative":
            return train_class_conditional_generative_replay(
                model,
                model_ref,
                loader,
                old_class_ids=list(old_class_ids),
                device=device,
                epochs=epochs,
                lr=lr,
                train_seed=train_seed,
                lambda_mix=float(lambda_mix),
                label_dropout=float(label_dropout),
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_step_size=ode_step_size,
                mean_flow=mean_flow,
                mean_flow_n_steps=mean_flow_n_steps,
                max_grad_norm=clip,
            )
        if strategy == "trajectory":
            return train_class_conditional_trajectory_replay(
                model,
                model_ref,
                loader,
                old_class_ids=list(old_class_ids),
                device=device,
                epochs=epochs,
                lr=lr,
                train_seed=train_seed,
                lambda_mix=float(lambda_mix),
                label_dropout=float(label_dropout),
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_step_size=ode_step_size,
                traj_kappa=traj_kappa,
                mean_flow=mean_flow,
                mean_flow_n_steps=mean_flow_n_steps,
                max_grad_norm=clip,
            )
        raise ValueError(
            f"class-conditional training does not support strategy={strategy!r}"
        )

    if strategy in ("naive", "cumulative", "joint") or model_ref is None:
        return train_unconditional_fm(
            model,
            loader,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            max_grad_norm=clip,
        )
    assert lambda_mix is not None
    if strategy == "generative":
        return train_unconditional_generative_replay(
            model,
            model_ref,
            loader,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            lambda_mix=float(lambda_mix),
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_step_size=ode_step_size,
            mean_flow=mean_flow,
            mean_flow_n_steps=mean_flow_n_steps,
            max_grad_norm=clip,
        )
    if strategy == "trajectory":
        return train_unconditional_trajectory_replay(
            model,
            model_ref,
            loader,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            lambda_mix=float(lambda_mix),
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_step_size=ode_step_size,
            traj_kappa=traj_kappa,
            mean_flow=mean_flow,
            mean_flow_n_steps=mean_flow_n_steps,
            max_grad_norm=clip,
        )
    raise ValueError(f"Unknown strategy {strategy}")


def resolve_kld_clf_path(args: argparse.Namespace, repo_root: Path) -> Path | None:
    """Return Masip CNN weights dir for FMNIST/CIFAR; ``None`` for ImageNet-100.

    ImageNet-100 uses an off-the-shelf torchvision ResNet50 (see
    ``resolve_image_cl_kld_classifier``) rather than a local checkpoint path.
    """
    if bool(getattr(args, "no_kld", False)):
        return None
    if args.dataset == "split_imagenet100":
        return None
    if args.kld_clf_path is not None:
        return Path(args.kld_clf_path)
    classifier_dir = {
        "split_fmnist": "cnn_fmnist",
        "split_cifar10": "cnn_cifar10",
    }.get(args.dataset, "__missing__")
    if classifier_dir == "__missing__":
        raise ValueError(f"Unsupported image-CL dataset: {args.dataset!r}")
    return (
        repo_root
        / "baselines"
        / "diffusion_continual_learning"
        / "weights"
        / classifier_dir
    )


def image_cl_kld_num_classes(dataset: str) -> int:
    if dataset == "split_imagenet100":
        return 100
    if dataset in ("split_fmnist", "split_cifar10"):
        return 10
    raise ValueError(f"Unsupported image-CL dataset: {dataset!r}")


def resolve_image_cl_kld_classifier(
    args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
) -> Any | None:
    """Load the stream-KLD probe for the selected dataset, or ``None`` if disabled."""
    if bool(getattr(args, "no_kld", False)):
        return None
    if args.dataset == "split_imagenet100":
        from trajectory_replay.eval.stream_kld import (
            load_imagenet100_kld_classifier,
            load_imagenet100_synsets_from_latent_dir,
        )

        synsets = load_imagenet100_synsets_from_latent_dir(resolve_latent_dir(args))
        return load_imagenet100_kld_classifier(synsets, device=device)

    kld_path = resolve_kld_clf_path(args, repo_root)
    if kld_path is None:
        return None
    from trajectory_replay.eval.stream_kld import load_fmnist_kld_classifier

    print(f"Loading KLD classifier from {kld_path}")
    return load_fmnist_kld_classifier(kld_path, device=device)


def clone_train_state(
    src: DiffusersUNetFlowMatching,
    device: torch.device,
) -> DiffusersUNetFlowMatching:
    """Fresh trainable copy of ``src`` weights (for λ candidate branches)."""
    import copy

    dst = copy.deepcopy(src)
    dst.to(device)
    dst.train()
    for p in dst.parameters():
        p.requires_grad_(True)
    return dst


__all__ = [
    "AVALANCHE_SPLIT_CIFAR10_TASKS",
    "AVALANCHE_SPLIT_FMNIST_TASKS",
    "DEFAULT_IMAGENET100_LATENT_DIR",
    "DEFAULT_LAMBDA_GRID",
    "SPLIT_IMAGENET100_TASKS",
    "TaskSplit",
    "DiffusersUNetFlowMatching",
    "add_image_cl_data_args",
    "add_image_cl_model_args",
    "add_image_cl_train_args",
    "add_image_cl_ode_args",
    "add_image_cl_eval_args",
    "apply_image_cl_dataset_defaults",
    "build_image_cl_tasks",
    "build_split_cifar10_tasks",
    "build_split_fmnist_tasks",
    "build_split_imagenet100_tasks",
    "image_cl_in_channels",
    "image_cl_num_classes",
    "image_cl_unet_kwargs",
    "image_cl_uses_hflip",
    "clone_train_state",
    "dump_json",
    "equal_weight_lambda",
    "load_checkpoint",
    "make_frozen_diffusers_copy",
    "make_model",
    "resolve_data_root",
    "resolve_device",
    "image_cl_kld_num_classes",
    "resolve_image_cl_kld_classifier",
    "resolve_kld_clf_path",
    "resolve_latent_dir",
    "resolve_latent_to_pixels",
    "run_results_dir",
    "save_checkpoint",
    "train_one_task",
]
