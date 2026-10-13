#!/usr/bin/env python3
"""Plot real vs generated MNIST samples per digit (conditional or unconditional)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from trajectory_replay.io.results import run_results_dir
from trajectory_replay.model.flow_matching import resolve_use_conditioning_from_config
from trajectory_replay.utils.cli import resolve_device

from experiments.mnist.mnist_common import (
    add_mnist_conditioning_args,
    add_mnist_data_args,
    add_mnist_ode_args,
    finetune_tensors,
    load_manifest,
    load_mnist_checkpoint,
    pick_indices_for_digit,
    pretrain_support_tensors,
    resolve_held_out_digit,
    resolve_mnist_run_checkpoint,
    resolve_processed_dir,
    save_digit_grid,
    save_real_gen_comparison,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_mnist_data_args(p)
    add_mnist_ode_args(p)
    add_mnist_conditioning_args(p)
    p.add_argument("--run-name", default=None, help="Run under results_dir/ with checkpoint.pt.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--data-split",
        choices=("pretrain", "finetune", "both"),
        default="both",
        help="Which processed split supplies real images (default: both).",
    )
    p.add_argument(
        "--digits",
        default=None,
        help="Comma-separated digits to plot (default: all perturbations if conditional, "
        "else digits present in the chosen data split).",
    )
    p.add_argument("--n-per-digit", type=int, default=8)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/eval_plots).",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_digits_arg(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_eval_data(
    processed_dir: Path,
    *,
    held_out: int,
    data_split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if data_split in ("pretrain", "both"):
        chunks.append(pretrain_support_tensors(processed_dir, held_out=held_out))
    if data_split in ("finetune", "both"):
        chunks.append(finetune_tensors(processed_dir, held_out=held_out, use_conditioning=True))
    if not chunks:
        raise ValueError(f"unknown data_split {data_split!r}")
    x = torch.cat([c[0] for c in chunks], dim=0)
    c = torch.cat([c[1] for c in chunks], dim=0)
    p = torch.cat([c[2] for c in chunks], dim=0)
    return x, c, p


def resolve_digits_to_plot(
    *,
    use_conditioning: bool,
    data_digits: list[int],
    explicit: list[int] | None,
    n_perturbations: int,
) -> list[int]:
    if explicit is not None:
        return explicit
    if use_conditioning:
        return list(range(n_perturbations))
    return sorted(set(data_digits))


@torch.no_grad()
def sample_generated(
    model: torch.nn.Module,
    *,
    digit: int,
    n: int,
    device: torch.device,
    use_conditioning: bool,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(eval_seed + digit * 10_007)
    if use_conditioning:
        cb = torch.zeros(n, dtype=torch.long, device=device)
        pb = torch.full((n,), digit, dtype=torch.long, device=device)
        return model.sample(
            cb,
            pb,
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            options=ode_options,
            generator=gen,
        ).cpu()
    return model.sample(
        batch_size=n,
        method=ode_method,
        rtol=ode_rtol,
        atol=ode_atol,
        options=ode_options,
        generator=gen,
    ).cpu()


def save_overview_grid(
    path: Path,
    *,
    digits: list[int],
    reals: dict[int, torch.Tensor | None],
    gens: dict[int, torch.Tensor],
    use_conditioning: bool,
    title: str,
) -> None:
    n_cols = len(digits)
    n_rows = 2 if any(reals[d] is not None for d in digits) else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.6))
    if n_cols == 1:
        axes = np.atleast_2d(axes.T).T
    elif n_rows == 1:
        axes = np.atleast_2d(axes)

    for j, digit in enumerate(digits):
        gen = gens[digit]
        if gen.dim() == 2:
            side = int(gen.shape[1] ** 0.5)
            gen = gen.view(-1, 1, side, side)
        img = gen[0, 0].numpy()

        if n_rows == 2:
            real = reals.get(digit)
            ax_real = axes[0, j]
            ax_real.axis("off")
            if real is not None:
                r = real[0, 0].numpy() if real.dim() == 4 else real[0].numpy()
                ax_real.imshow(r, cmap="gray", vmin=0.0, vmax=1.0)
            ax_real.set_title(f"digit {digit}")
            row = 1
        else:
            row = 0

        ax_gen = axes[row, j]
        ax_gen.axis("off")
        ax_gen.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        if n_rows == 1:
            label = f"p={digit}" if use_conditioning else f"digit {digit}"
            ax_gen.set_title(label)

    mode = "conditional" if use_conditioning else "unconditional"
    fig.suptitle(f"{title} ({mode})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def resolve_eval_conditioning(args: argparse.Namespace, cfg: dict) -> bool:
    """Use checkpoint config; error only if ``--unconditional`` conflicts."""
    use_conditioning = resolve_use_conditioning_from_config(cfg)
    if args.unconditional and use_conditioning:
        raise ValueError("Checkpoint is conditional but --unconditional was passed.")
    if args.unconditional:
        return False
    return use_conditioning


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    device = resolve_device(args.device)
    processed_dir = resolve_processed_dir(args, repo_root)
    manifest = load_manifest(processed_dir)
    held_out = resolve_held_out_digit(args, manifest)

    ckpt_path = resolve_mnist_run_checkpoint(
        repo_root, run_name=args.run_name, checkpoint=args.checkpoint
    )
    model, cfg = load_mnist_checkpoint(ckpt_path, device)
    use_conditioning = resolve_eval_conditioning(args, cfg)
    model.eval()

    x, c, p = load_eval_data(processed_dir, held_out=held_out, data_split=args.data_split)
    data_digits = sorted({int(v) for v in p.unique().tolist()})
    digits = resolve_digits_to_plot(
        use_conditioning=use_conditioning,
        data_digits=data_digits,
        explicit=parse_digits_arg(args.digits),
        n_perturbations=model.n_perturbations if use_conditioning else manifest.n_perturbations,
    )

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.run_name is not None:
        out_dir = run_results_dir(repo_root, args.run_name) / "eval_plots"
    else:
        out_dir = ckpt_path.parent / "eval_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ode_options = {"step_size": float(args.ode_step_size)} if args.ode_step_size is not None else None
    sample_kw = dict(
        device=device,
        use_conditioning=use_conditioning,
        eval_seed=args.eval_seed,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options,
    )

    cond_label = "conditional" if use_conditioning else "unconditional"
    print(
        f"Plot MNIST eval: {cond_label}, digits={digits}, n_per_digit={args.n_per_digit}, "
        f"data_split={args.data_split}, checkpoint={ckpt_path.name}"
    )

    reals: dict[int, torch.Tensor | None] = {}
    gens: dict[int, torch.Tensor] = {}
    for digit in digits:
        idx = pick_indices_for_digit(p, digit, args.n_per_digit, seed=args.eval_seed)
        real = x[idx] if len(idx) > 0 else None
        gen = sample_generated(model, digit=digit, n=args.n_per_digit, **sample_kw)

        reals[digit] = real
        gens[digit] = gen

        tag = f"digit_{digit}"
        if real is not None:
            save_digit_grid(out_dir / f"real_{tag}.png", real, nrow=args.n_per_digit, title=f"Real {tag}")
        save_digit_grid(
            out_dir / f"gen_{tag}.png",
            gen,
            nrow=args.n_per_digit,
            title=f"Generated {tag}" + ("" if use_conditioning else " (unconditional)"),
        )
        save_real_gen_comparison(
            out_dir / f"compare_{tag}.png",
            real,
            gen,
            nrow=args.n_per_digit,
            title=f"{tag}: real vs generated",
        )

    save_overview_grid(
        out_dir / "overview_all_digits.png",
        digits=digits,
        reals=reals,
        gens=gens,
        use_conditioning=use_conditioning,
        title="MNIST eval overview",
    )

    meta = {
        "checkpoint": str(ckpt_path),
        "use_conditioning": use_conditioning,
        "data_split": args.data_split,
        "digits": digits,
        "n_per_digit": args.n_per_digit,
        "eval_seed": args.eval_seed,
        "held_out_digit": held_out,
        "stage": cfg.get("stage"),
        "run_name": args.run_name,
    }
    with (out_dir / "eval_plots.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved eval plots to {out_dir}")


if __name__ == "__main__":
    main()
