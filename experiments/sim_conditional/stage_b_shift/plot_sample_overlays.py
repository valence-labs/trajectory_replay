#!/usr/bin/env python3
"""Stage B overlays: real (taupe) vs generated, plus heldout-test panels.

Regenerates from existing FT checkpoints (no retraining).

Outputs under ``…/dim{D}/figures/``:
  - heldout_test_overlays.png — strip of real vs gen on (c*, p*) test
  - overlay_{run}.png — full FT/pretrain/heldout overview for selected runs

Example:
  uv run python experiments/sim_conditional/stage_b_shift/plot_sample_overlays.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.default"] = "it"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
)

from experiments.sim_conditional.stage_b_shift.common import (
    POOL_FINETUNE,
    POOL_HELDOUT,
    POOL_PRETRAIN,
    REPO_ROOT,
    SPLIT_TEST,
    SPLIT_VAL,
    load_checkpoint,
    load_stage_b_data,
    mask_rows,
    sample_cp,
    save_overlay_scatter,
    stage_b_data_root,
    stage_b_runs_root,
)
from trajectory_replay.utils.cli import resolve_device

TRUE_COLOR = "#b7a99a"
GEN_COLOR = "#2a9d8f"

# (strategy, lambda|None, panel title)
DEFAULT_HELDOUT_PANELS = [
    ("naive", None, "Naive FT"),
    ("generative", 0.1, r"Gen $\lambda{=}0.1$"),
    ("generative", 0.5, r"Gen $\lambda{=}0.5$"),
    ("generative", 0.75, r"Gen $\lambda{=}0.75$"),
    ("trajectory", 0.1, r"Traj $\lambda{=}0.1$"),
    ("trajectory", 0.75, r"Traj $\lambda{=}0.75$"),
]

DEFAULT_OVERVIEW_RUNS = [
    ("naive", None),
    ("generative", 0.1),
    ("generative", 0.75),
    ("trajectory", 0.1),
    ("trajectory", 0.75),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-contexts", type=int, default=6)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--n-gen", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _run_name(strategy: str, lam: float | None, epochs: int, seed: int) -> str:
    if strategy == "naive":
        return f"ft_naive_e{epochs}_seed{seed}"
    return f"ft_{strategy}_lam{lam}_e{epochs}_seed{seed}"


def _ckpt(runs: Path, strategy: str, lam: float | None, epochs: int, seed: int) -> Path:
    return runs / "finetune" / _run_name(strategy, lam, epochs, seed) / "checkpoint.pt"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runs = stage_b_runs_root(
        REPO_ROOT,
        dim=args.dim,
        n_contexts=args.n_contexts,
        n_perturbations=args.n_perturbations,
    )
    data_dir = stage_b_data_root(
        REPO_ROOT,
        dim=args.dim,
        n_contexts=args.n_contexts,
        n_perturbations=args.n_perturbations,
    )
    arrays, meta = load_stage_b_data(data_dir)
    heldout_cell = tuple(map(int, meta["heldout_combo"]))
    ft_c, ho_p = heldout_cell
    pretrain_cells = [tuple(map(int, cp)) for cp in meta["pretrain_cells"]]
    finetune_cells = [tuple(map(int, cp)) for cp in meta["finetune_cells"]]

    m_ho = mask_rows(arrays, pool=POOL_HELDOUT, split=SPLIT_TEST)
    x_ho = np.asarray(arrays["x"][m_ho], dtype=np.float64)

    out_dir = args.out_dir or (runs / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared axis limits from heldout real + a bit of padding.
    pad = 0.6
    x_lo, x_hi = np.quantile(x_ho[:, 0], [0.002, 0.998])
    y_lo, y_hi = np.quantile(x_ho[:, 1], [0.002, 0.998])
    xlim = (float(x_lo - pad), float(x_hi + pad))
    ylim = (float(y_lo - pad), float(y_hi + pad))

    # --- Heldout test strip ---
    n = len(DEFAULT_HELDOUT_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.35), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    legend_done = False
    for ax, (strat, lam, title) in zip(axes, DEFAULT_HELDOUT_PANELS):
        ckpt = _ckpt(runs, strat, lam, args.epochs, args.seed)
        ax.scatter(
            x_ho[:, 0],
            x_ho[:, 1],
            s=8,
            alpha=0.40,
            c=TRUE_COLOR,
            marker="o",
            edgecolors="none",
            label="Real" if not legend_done else None,
            zorder=2,
        )
        if not ckpt.exists():
            ax.set_title(f"{title}\n(missing)", fontsize=9)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            continue
        model, _ = load_checkpoint(ckpt, device)
        x_gen = sample_cp(
            model,
            ft_c,
            ho_p,
            args.n_gen,
            device=device,
            seed=10_000 + args.seed + int(1000 * (lam or 0)),
            ode_step_size=args.ode_step_size,
        )
        ax.scatter(
            x_gen[:, 0],
            x_gen[:, 1],
            s=9,
            alpha=0.65,
            c=GEN_COLOR,
            marker="o",
            edgecolors="none",
            label="Generated" if not legend_done else None,
            zorder=3,
        )
        legend_done = True
        ax.set_title(title, fontsize=9)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    handles = [
        mlines.Line2D(
            [],
            [],
            color=TRUE_COLOR,
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real (heldout test)",
        ),
        mlines.Line2D(
            [],
            [],
            color=GEN_COLOR,
            marker="o",
            linestyle="None",
            markersize=6,
            label="Generated",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        rf"Stage B heldout test $(c={ft_c},y={ho_p})$  |  seed {args.seed}",
        fontsize=11,
        y=1.14,
    )
    fig.tight_layout()
    held_path = out_dir / f"heldout_test_overlays_seed{args.seed}.png"
    fig.savefig(held_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {held_path}")

    # --- Full overview overlays for selected runs (same cells: real vs gen) ---
    demo = list(finetune_cells) + [heldout_cell] + pretrain_cells[:2]
    # Restrict real cloud to the demo cells so overlay is apples-to-apples.
    true_masks = []
    for ci, pj in demo:
        if (ci, pj) == heldout_cell:
            true_masks.append(
                mask_rows(
                    arrays,
                    pool=POOL_HELDOUT,
                    split=SPLIT_TEST,
                    contexts=[ci],
                    perturbations=[pj],
                )
            )
        elif ci == int(meta["ft_context"]):
            true_masks.append(
                mask_rows(
                    arrays,
                    pool=POOL_FINETUNE,
                    split=SPLIT_VAL,
                    contexts=[ci],
                    perturbations=[pj],
                )
            )
        else:
            true_masks.append(
                mask_rows(
                    arrays,
                    pool=POOL_PRETRAIN,
                    split=SPLIT_VAL,
                    contexts=[ci],
                    perturbations=[pj],
                )
            )
    m_demo = np.zeros(arrays["x"].shape[0], dtype=bool)
    for m in true_masks:
        m_demo |= m
    x_true = arrays["x"][m_demo]
    c_true = arrays["c"][m_demo]
    p_true = arrays["p"][m_demo]

    for strat, lam in DEFAULT_OVERVIEW_RUNS:
        ckpt = _ckpt(runs, strat, lam, args.epochs, args.seed)
        if not ckpt.exists():
            print(f"skip overview {strat} λ={lam}: missing {ckpt}")
            continue
        model, _ = load_checkpoint(ckpt, device)
        gens = {
            (ci, pj): sample_cp(
                model,
                ci,
                pj,
                300,
                device=device,
                seed=20_000 + args.seed + ci * 17 + pj,
                ode_step_size=args.ode_step_size,
            )
            for ci, pj in demo
        }
        if strat == "naive":
            title = "Stage B FT naive"
            tag = "naive"
        else:
            title = rf"Stage B FT {strat} $\lambda={lam}$"
            tag = f"{strat}_lam{lam}"
        out = out_dir / f"overlay_{tag}_seed{args.seed}.png"
        save_overlay_scatter(
            out,
            x_true=x_true,
            c_true=c_true,
            p_true=p_true,
            gens=gens,
            title=title,
            figsize=(3.8, 3.5),
        )
        # Per-run heldout-only zoom
        out_ho = out_dir / f"heldout_{tag}_seed{args.seed}.png"
        save_overlay_scatter(
            out_ho,
            x_true=x_ho,
            x_gen=gens[heldout_cell],
            title=rf"{title}  |  heldout test",
            figsize=(3.0, 2.8),
            max_true=1000,
            max_gen=args.n_gen,
        )
        print(f"Wrote {out}")
        print(f"Wrote {out_ho}")


if __name__ == "__main__":
    main()
