"""Shared helpers for MNIST flow-matching experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trajectory_replay.data.mnist_dataset import (
    MnistManifest,
    MnistSplit,
    default_processed_dir,
    load_manifest,
    load_split,
)
from trajectory_replay.holdout.data import (
    build_support_stratum_distribution,
    mask_replay_pool,
    subsample_per_stratum,
)
from trajectory_replay.model.backbones import vector_field_module
from trajectory_replay.model.flow_matching import (
    ConditionalFlowMatching,
    build_conditional_flow_matching_from_config,
)
from trajectory_replay.io.results import run_results_dir


def add_mnist_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory with pretrain.pt / finetune.pt / manifest.json.",
    )
    p.add_argument(
        "--held-out-digit",
        type=int,
        default=None,
        help="Digit held out for finetune (default: manifest held_out_digit).",
    )


def add_mnist_conditioning_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--unconditional",
        action="store_true",
        help="Train/eval without context or perturbation embeddings (state + time only).",
    )


def model_uses_conditioning(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "unconditional", False))


def resolve_held_out_digit(args: argparse.Namespace, manifest: MnistManifest) -> int:
    return manifest.held_out_digit if args.held_out_digit is None else args.held_out_digit


def add_mnist_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backbone-type",
        choices=("mlp", "unet"),
        default="unet",
        help="Vector field backbone (default: unet for MNIST).",
    )
    p.add_argument("--cond-emb-dim", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256, help="MLP width (ignored for UNet).")
    p.add_argument("--depth", type=int, default=4, help="MLP depth (ignored for UNet).")
    p.add_argument("--unet-num-channels", type=int, default=64)
    p.add_argument("--unet-num-res-blocks", type=int, default=2)
    p.add_argument("--unet-attention-resolutions", default="14")
    p.add_argument("--unet-dropout", type=float, default=0.0)


def add_mnist_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=None, help="Cap pretrain set size.")


def add_mnist_ode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument(
        "--eval-max-samples-per-stratum",
        type=int,
        default=64,
        help="Max images simulated per (context, digit) stratum in post-train ODE eval.",
    )


def resolve_processed_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    return args.processed_dir if args.processed_dir is not None else default_processed_dir(repo_root)


def pretrain_tensors(
    split: MnistSplit,
    *,
    held_out: int,
    max_samples: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, c, p = mask_replay_pool(
        split.x,
        split.c,
        split.p,
        holdout_mode="perturbation",
        held_out_pert=held_out,
        held_out_context=None,
    )
    if max_samples is not None and len(x) > max_samples:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(x), generator=rng)[:max_samples]
        x, c, p = x[idx], c[idx], p[idx]
    return x, c, p


def make_mnist_model(manifest: MnistManifest, args: argparse.Namespace) -> ConditionalFlowMatching:
    if model_uses_conditioning(args):
        n_pert_model = manifest.n_perturbations
    else:
        n_pert_model = manifest.n_perturbations - 1
    return ConditionalFlowMatching(
        data_dim=manifest.data_dim,
        n_contexts=manifest.n_contexts,
        n_perturbations=n_pert_model,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        backbone_type=args.backbone_type,
        image_shape=manifest.image_shape,
        unet_num_channels=args.unet_num_channels,
        unet_num_res_blocks=args.unet_num_res_blocks,
        unet_attention_resolutions=args.unet_attention_resolutions,
        unet_dropout=args.unet_dropout,
        use_conditioning=model_uses_conditioning(args),
    )


def checkpoint_config(model: ConditionalFlowMatching, manifest: MnistManifest, args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "experiment": "mnist",
        "backbone_type": model.backbone_type,
        "data_dim": model.data_dim,
        "image_shape": list(model.image_shape) if model.image_shape is not None else None,
        "n_contexts": model.n_contexts,
        "n_perturbations": model.n_perturbations,
        "cond_emb_dim": model.cond_emb_dim,
        "time_emb_dim": model.time_emb_dim,
        "hidden_dim": model.hidden_dim,
        "depth": model.depth,
        "held_out_digit": manifest.held_out_digit,
        "unet_num_channels": model.unet_num_channels,
        "unet_num_res_blocks": model.unet_num_res_blocks,
        "unet_attention_resolutions": model.unet_attention_resolutions,
        "unet_dropout": model.unet_dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_seed": args.train_seed,
        "use_conditioning": model.use_conditioning,
    }
    return cfg


def load_mnist_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ConditionalFlowMatching, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = build_conditional_flow_matching_from_config(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, cfg


def resolve_pretrain_checkpoint(repo_root: Path, pretrain_run: str | None, pretrain_checkpoint: Path | None) -> Path:
    if pretrain_checkpoint is not None:
        return pretrain_checkpoint
    if pretrain_run is None:
        raise ValueError("Provide --pretrain-run or --pretrain-checkpoint")
    path = run_results_dir(repo_root, pretrain_run) / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {path}")
    return path


def subsample_eval_tensors(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    max_per_stratum: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cap post-train ODE eval pool size per (c, p) stratum."""
    if max_per_stratum is None or max_per_stratum <= 0:
        return x, c, p
    return subsample_per_stratum(x, c, p, max_per_stratum=max_per_stratum, seed=seed)


def pretrain_support_tensors(
    processed_dir: Path,
    *,
    held_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pretrain support pool with true (c, p) indices (held-out digit excluded)."""
    split = load_split(processed_dir, "pretrain")
    return mask_replay_pool(
        split.x,
        split.c,
        split.p,
        holdout_mode="perturbation",
        held_out_pert=held_out,
        held_out_context=None,
    )


def finetune_tensors(
    processed_dir: Path,
    *,
    held_out: int,
    use_conditioning: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split = load_split(processed_dir, "finetune")
    x, c, p = split.x, split.c, split.p
    if (p != held_out).any():
        raise ValueError(f"finetune split should contain only digit {held_out}")
    if use_conditioning:
        p = torch.full_like(p, held_out)
    return x, c, p


def configure_mnist_finetune(
    model: ConditionalFlowMatching,
    *,
    use_conditioning: bool,
    held_out: int,
    train_vf: bool,
    train_old_embeddings: bool,
) -> tuple[list[nn.Parameter], int | None, int | None]:
    if use_conditioning:
        return model.configure_finetune(
            holdout_mode="perturbation",
            train_vector_field=train_vf,
            train_old_pert_embeddings=train_old_embeddings,
            held_out_pert=held_out,
        )

    for param in model.parameters():
        param.requires_grad = False
    trainable: list[nn.Parameter] = []
    if train_vf:
        vf = vector_field_module(model)
        for param in vf.parameters():
            param.requires_grad = True
            trainable.append(param)
    return trainable, None, None


def assert_pretrain_conditioning_compatible(
    pre_cfg: dict[str, Any],
    use_conditioning: bool,
) -> None:
    from trajectory_replay.model.flow_matching import resolve_use_conditioning_from_config

    pre_use = resolve_use_conditioning_from_config(pre_cfg)
    if pre_use != use_conditioning:
        raise ValueError(
            f"Finetune use_conditioning={use_conditioning} but pretrain checkpoint "
            f"has use_conditioning={pre_use}. Use a matching pretrain run."
        )


def assert_pretrain_embedding_table_compatible(
    pre_cfg: dict[str, Any],
    manifest: MnistManifest,
    *,
    use_conditioning: bool,
) -> None:
    """Reject legacy n-1 conditional checkpoints (insert-at-finetune protocol)."""
    if not use_conditioning:
        return
    expected = manifest.n_perturbations
    actual = int(pre_cfg.get("n_perturbations", -1))
    if actual != expected:
        raise ValueError(
            f"Pretrain checkpoint has n_perturbations={actual}, but MNIST conditional "
            f"requires the full table n_perturbations={expected}. Re-run pretrain with "
            "the current scripts (legacy expand-at-finetune checkpoints are incompatible)."
        )


def build_replay_stratum_dist(
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    support_contexts: list[int],
) -> Any:
    return build_support_stratum_distribution(c, p, support_contexts)


def save_digit_grid(
    path: Path,
    images: torch.Tensor,
    *,
    nrow: int = 8,
    title: str = "",
) -> None:
    """Plot flat or image tensors as a grayscale grid."""
    x = images.detach().cpu()
    if x.dim() == 2:
        side = int(x.shape[1] ** 0.5)
        x = x.view(-1, 1, side, side)
    n = min(x.shape[0], nrow * nrow)
    x = x[:n]
    cols = nrow
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(x[i, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_seen_digit_sample_grids(
    model: nn.Module,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    out_dir: Path,
    device: torch.device,
    use_conditioning: bool,
    plot_n_per_digit: int,
    max_digits: int = 3,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> list[int]:
    """Save real/gen grids for the first ``max_digits`` labels (pretrain-style)."""
    plotted: list[int] = []
    plot_labels = sorted({int(v) for v in p.unique().tolist()})[:max_digits]
    for label in plot_labels:
        idx = (p == label).nonzero(as_tuple=True)[0][:plot_n_per_digit]
        if len(idx) == 0:
            continue
        real = x[idx]
        if use_conditioning:
            gen = model.sample(
                c[idx].to(device),
                p[idx].to(device),
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
            ).cpu()
            tag = f"p{label}"
        else:
            gen = model.sample(
                batch_size=len(idx),
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
            ).cpu()
            tag = f"digit{label}"
        save_digit_grid(out_dir / f"real_{tag}.png", real, title=f"Real {tag}")
        save_digit_grid(out_dir / f"gen_{tag}.png", gen, title=f"Generated {tag}")
        plotted.append(label)
    return plotted


def save_real_gen_comparison(
    path: Path,
    real: torch.Tensor | None,
    gen: torch.Tensor,
    *,
    nrow: int = 8,
    title: str = "",
) -> None:
    """Two-row grid: real on top (optional), generated on bottom."""
    gen_x = gen.detach().cpu()
    if gen_x.dim() == 2:
        side = int(gen_x.shape[1] ** 0.5)
        gen_x = gen_x.view(-1, 1, side, side)
    n = min(gen_x.shape[0], nrow)
    gen_x = gen_x[:n]

    rows = 2 if real is not None else 1
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.2, rows * 1.2))
    if rows == 1:
        axes = np.atleast_2d(axes)
    else:
        axes = np.atleast_2d(axes)

    if real is not None:
        real_x = real.detach().cpu()
        if real_x.dim() == 2:
            side = int(real_x.shape[1] ** 0.5)
            real_x = real_x.view(-1, 1, side, side)
        real_x = real_x[:n]
        for j in range(n):
            axes[0, j].axis("off")
            axes[0, j].imshow(real_x[j, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            if j == 0:
                axes[0, j].set_ylabel("real", rotation=0, labelpad=20, va="center")

    row = 1 if real is not None else 0
    for j in range(n):
        axes[row, j].axis("off")
        axes[row, j].imshow(gen_x[j, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        if j == 0:
            axes[row, j].set_ylabel("gen", rotation=0, labelpad=20, va="center")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def resolve_mnist_run_checkpoint(
    repo_root: Path,
    *,
    run_name: str | None,
    checkpoint: Path | None,
) -> Path:
    if checkpoint is not None:
        path = checkpoint
    elif run_name is not None:
        path = run_results_dir(repo_root, run_name) / "checkpoint.pt"
    else:
        raise ValueError("Provide --run-name or --checkpoint")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def pick_indices_for_digit(
    p: torch.Tensor,
    digit: int,
    n: int,
    *,
    seed: int,
) -> torch.Tensor:
    idx = (p == digit).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return idx
    if len(idx) <= n:
        return idx[:n]
    rng = torch.Generator().manual_seed(seed + digit * 1_009)
    pick = torch.randperm(len(idx), generator=rng)[:n]
    return idx[pick]


def make_pretrain_loader(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x, c, p),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(x) >= batch_size,
    )


def eval_plot_model(model: ConditionalFlowMatching, held_out: int) -> ConditionalFlowMatching:
    """Return the model for plotting (full-table checkpoints use true perturbation indices)."""
    del held_out
    return model
