"""Shared paths / loaders / model helpers for Stage B (c, p) shift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.sim_conditional.common import REPO_ROOT, conditional_results_root
from trajectory_replay.eval.metrics import energy_distance, mean_l2_distance
from trajectory_replay.model.flow_matching import ConditionalFlowMatching

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
POOL_PRETRAIN, POOL_FINETUNE, POOL_HELDOUT = 0, 1, 2


def stage_b_config_tag(*, n_contexts: int, n_perturbations: int) -> str:
    """Subfolder name under dim{{D}}, e.g. ``c4_p4`` / ``c10_p4``."""
    return f"c{int(n_contexts)}_p{int(n_perturbations)}"


def stage_b_data_root(
    repo_root: Path | None = None,
    *,
    dim: int = 2,
    n_contexts: int = 6,
    n_perturbations: int = 4,
) -> Path:
    tag = stage_b_config_tag(n_contexts=n_contexts, n_perturbations=n_perturbations)
    return (
        conditional_results_root(repo_root)
        / "stage_b_shift"
        / "data"
        / f"dim{dim}"
        / tag
    )


def stage_b_runs_root(
    repo_root: Path | None = None,
    *,
    dim: int = 2,
    n_contexts: int = 6,
    n_perturbations: int = 4,
) -> Path:
    tag = stage_b_config_tag(n_contexts=n_contexts, n_perturbations=n_perturbations)
    return conditional_results_root(repo_root) / "stage_b_shift" / f"dim{dim}" / tag


def resolve_stage_b_roots(
    repo_root: Path | None = None,
    *,
    dim: int,
    data_dir: Path | None = None,
    n_contexts: int | None = None,
    n_perturbations: int | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve data/runs roots from explicit (c,p) or by reading ``meta.json``.

    If ``data_dir`` is given, ``n_contexts`` / ``n_perturbations`` are taken from meta
    (and must match CLI overrides when provided).
    """
    root = repo_root or REPO_ROOT
    if data_dir is not None:
        arrays, meta = load_stage_b_data(data_dir)
        del arrays
        n_c = int(meta["n_contexts"])
        n_p = int(meta["n_perturbations"])
        if n_contexts is not None and int(n_contexts) != n_c:
            raise SystemExit(f"data n_contexts={n_c} != --n-contexts {n_contexts}")
        if n_perturbations is not None and int(n_perturbations) != n_p:
            raise SystemExit(
                f"data n_perturbations={n_p} != --n-perturbations {n_perturbations}"
            )
        runs = stage_b_runs_root(root, dim=dim, n_contexts=n_c, n_perturbations=n_p)
        return data_dir, runs, meta

    n_c = 6 if n_contexts is None else int(n_contexts)
    n_p = 4 if n_perturbations is None else int(n_perturbations)
    data = stage_b_data_root(root, dim=dim, n_contexts=n_c, n_perturbations=n_p)
    runs = stage_b_runs_root(root, dim=dim, n_contexts=n_c, n_perturbations=n_p)
    if not (data / "meta.json").exists():
        raise SystemExit(f"missing Stage B data at {data} (build with --n-contexts {n_c})")
    _, meta = load_stage_b_data(data)
    return data, runs, meta


def load_stage_b_data(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = np.load(data_dir / "cp.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    arrays = {k: raw[k] for k in raw.files}
    return arrays, meta


def mask_rows(
    arrays: dict[str, np.ndarray],
    *,
    pool: int | None = None,
    split: int | None = None,
    contexts: list[int] | None = None,
    perturbations: list[int] | None = None,
) -> np.ndarray:
    m = np.ones(arrays["x"].shape[0], dtype=bool)
    if pool is not None:
        m &= arrays["pool"] == int(pool)
    if split is not None:
        m &= arrays["split"] == int(split)
    if contexts is not None:
        m &= np.isin(arrays["c"], np.asarray(contexts, dtype=np.int64))
    if perturbations is not None:
        m &= np.isin(arrays["p"], np.asarray(perturbations, dtype=np.int64))
    return m


def make_cp_loader(
    x: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ct = torch.from_numpy(np.asarray(c, dtype=np.int64))
    pt = torch.from_numpy(np.asarray(p, dtype=np.int64))
    ds = TensorDataset(xt, ct, pt)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=gen
    )


def build_stage_b_model(
    dim: int,
    n_contexts: int,
    n_perturbations: int,
    *,
    cond_emb_dim: int = 32,
    time_emb_dim: int = 64,
    hidden_dim: int = 256,
    depth: int = 4,
) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=dim,
        n_contexts=n_contexts,
        n_perturbations=n_perturbations,
        cond_emb_dim=cond_emb_dim,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        use_conditioning=True,
    )


def save_checkpoint(
    path: Path,
    model: ConditionalFlowMatching,
    *,
    config: dict[str, Any],
    history: list[dict[str, float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict(), "config": config}
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[ConditionalFlowMatching, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    cfg = dict(payload["config"])
    model = build_stage_b_model(
        int(cfg["data_dim"]),
        int(cfg["n_contexts"]),
        int(cfg["n_perturbations"]),
        cond_emb_dim=int(cfg.get("cond_emb_dim", 32)),
        time_emb_dim=int(cfg.get("time_emb_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        depth=int(cfg.get("depth", 4)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def sample_cp(
    model: ConditionalFlowMatching,
    context: int,
    perturbation: int,
    n: int,
    *,
    device: torch.device,
    seed: int,
    ode_method: str = "euler",
    ode_step_size: float = 0.05,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    options = {"step_size": float(ode_step_size)}
    chunks: list[np.ndarray] = []
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        cb = torch.full((b,), int(context), dtype=torch.long, device=device)
        pb = torch.full((b,), int(perturbation), dtype=torch.long, device=device)
        x = model.sample(cb, pb, method=ode_method, options=options, generator=gen)
        chunks.append(x.detach().cpu().numpy())
        done += b
    return np.concatenate(chunks, axis=0)


def evaluate_cells(
    model: ConditionalFlowMatching,
    arrays: dict[str, np.ndarray],
    cells: list[tuple[int, int]],
    *,
    pool: int | None,
    split: int,
    device: torch.device,
    seed: int,
    n_gen: int = 500,
) -> dict[str, Any]:
    """Energy / mean-L2 of generated vs true cloud for each (c, p) cell."""
    per: dict[str, dict[str, float]] = {}
    energies: list[float] = []
    mean_l2s: list[float] = []
    for i, (ci, pj) in enumerate(cells):
        m = mask_rows(
            arrays,
            pool=pool,
            split=split,
            contexts=[ci],
            perturbations=[pj],
        )
        if not np.any(m):
            # Heldout lives only in heldout pool / test; allow pool=None fallback.
            m = (arrays["c"] == ci) & (arrays["p"] == pj) & (arrays["split"] == split)
        x_true = arrays["x"][m]
        if x_true.shape[0] == 0:
            continue
        x_gen = sample_cp(
            model,
            ci,
            pj,
            min(n_gen, max(200, x_true.shape[0])),
            device=device,
            seed=seed + 17 * i + ci * 31 + pj,
        )
        e = float(energy_distance(x_true, x_gen))
        ml = float(mean_l2_distance(x_true, x_gen))
        per[f"{ci},{pj}"] = {"energy": e, "mean_l2": ml, "n_true": int(x_true.shape[0])}
        energies.append(e)
        mean_l2s.append(ml)
    return {
        "per_cell": per,
        "energy_mean": float(np.mean(energies)) if energies else float("nan"),
        "mean_l2_mean": float(np.mean(mean_l2s)) if mean_l2s else float("nan"),
        "n_cells": len(energies),
    }


def evaluate_fit_forget(
    model: ConditionalFlowMatching,
    arrays: dict[str, np.ndarray],
    *,
    finetune_cells: list[tuple[int, int]],
    pretrain_cells: list[tuple[int, int]],
    heldout_cell: tuple[int, int],
    split: int,
    device: torch.device,
    seed: int,
    n_gen: int = 500,
    pretrained_forget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit on FT cells, forget on pretrain cells, transfer on heldout (c*, p*)."""
    fit = evaluate_cells(
        model,
        arrays,
        finetune_cells,
        pool=POOL_FINETUNE,
        split=split,
        device=device,
        seed=seed,
        n_gen=n_gen,
    )
    forget = evaluate_cells(
        model,
        arrays,
        pretrain_cells,
        pool=POOL_PRETRAIN,
        split=split,
        device=device,
        seed=seed + 101,
        n_gen=n_gen,
    )
    heldout = evaluate_cells(
        model,
        arrays,
        [heldout_cell],
        pool=POOL_HELDOUT,
        split=SPLIT_TEST if split == SPLIT_TEST else split,
        device=device,
        seed=seed + 211,
        n_gen=n_gen,
    )
    ho_key = f"{heldout_cell[0]},{heldout_cell[1]}"
    ho_per = heldout["per_cell"].get(ho_key, {})
    out: dict[str, Any] = {
        "fit_energy_mean": fit["energy_mean"],
        "fit_mean_l2_mean": fit["mean_l2_mean"],
        "forget_energy_mean": forget["energy_mean"],
        "forget_mean_l2_mean": forget["mean_l2_mean"],
        "heldout_energy": float(ho_per.get("energy", float("nan"))),
        "heldout_mean_l2": float(ho_per.get("mean_l2", float("nan"))),
        "fit": fit,
        "forget": forget,
        "heldout": heldout,
    }
    if pretrained_forget is not None:
        out["delta_forget_energy_mean"] = float(
            out["forget_energy_mean"] - float(pretrained_forget["energy_mean"])
        )
        out["delta_forget_mean_l2_mean"] = float(
            out["forget_mean_l2_mean"] - float(pretrained_forget["mean_l2_mean"])
        )
    return out


def save_heldout_shift_scatter(
    path: Path,
    *,
    x_true: np.ndarray,
    x_gen_heldout: np.ndarray,
    title: str,
    max_true: int = 2500,
    max_gen: int = 600,
    true_color: str = "0.72",
    gen_color: str = "#c1121f",
    figsize: tuple[float, float] = (3.8, 3.5),
) -> None:
    """All true cells in gray; only generated heldout in red (shift readout)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    x_true = np.asarray(x_true, dtype=np.float64)
    x_gen = np.asarray(x_gen_heldout, dtype=np.float64)
    if x_true.shape[0] > max_true:
        x_true = x_true[rng.choice(x_true.shape[0], size=max_true, replace=False)]
    if x_gen.shape[0] > max_gen:
        x_gen = x_gen[rng.choice(x_gen.shape[0], size=max_gen, replace=False)]
    ax.scatter(
        x_true[:, 0],
        x_true[:, 1],
        s=7,
        alpha=0.28,
        c=true_color,
        marker="o",
        edgecolors="none",
        label="Real (all cells)",
        zorder=2,
    )
    ax.scatter(
        x_gen[:, 0],
        x_gen[:, 1],
        s=14,
        alpha=0.75,
        c=gen_color,
        marker="o",
        edgecolors="none",
        label="Generated heldout",
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.92, fontsize=8, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_stylized_heldout_overlay(
    path: Path,
    *,
    x_true: np.ndarray,
    x_gen: np.ndarray,
    gen_label: str = "Generated",
    gen_color: str = "#1f6aa5",
    real_color: str = "#8a8178",
    scatter_h: float = 1.15,
    legend_fontsize: float = 8.5,
    max_true: int = 2000,
    max_gen: int = 1500,
    pad: float = 0.55,
) -> None:
    """Small square Real-vs-gen panel (Stage A leaveout / mixture thumbnail style).

    No axis ticks/labels; legend above the frame; equal aspect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "legend.fontsize": legend_fontsize,
        }
    )
    rng = np.random.default_rng(0)
    x_true = np.asarray(x_true, dtype=np.float64)
    x_gen = np.asarray(x_gen, dtype=np.float64)
    if x_true.shape[0] > max_true:
        x_true = x_true[rng.choice(x_true.shape[0], size=max_true, replace=False)]
    if x_gen.shape[0] > max_gen:
        x_gen = x_gen[rng.choice(x_gen.shape[0], size=max_gen, replace=False)]

    pts = np.vstack([x_true, x_gen])
    x_lo, x_hi = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_lo, y_hi = float(pts[:, 1].min()), float(pts[:, 1].max())
    xlim = (x_lo - pad, x_hi + pad)
    ylim = (y_lo - pad, y_hi + pad)
    dx = max(xlim[1] - xlim[0], 1e-6)
    dy = max(ylim[1] - ylim[0], 1e-6)
    # Mild aspect clamp so very elongated clouds still read as a compact panel.
    max_aspect = 1.28
    if dx / dy > max_aspect:
        extra = 0.5 * (dx / max_aspect - dy)
        ylim = (ylim[0] - extra, ylim[1] + extra)
        dy = ylim[1] - ylim[0]
    elif dy / dx > max_aspect:
        extra = 0.5 * (dy / max_aspect - dx)
        xlim = (xlim[0] - extra, xlim[1] + extra)
        dx = xlim[1] - xlim[0]

    scatter_w = scatter_h * (dx / dy)
    legend_in = 0.26
    side_pad_in = 0.06
    fig_w = scatter_w + 2 * side_pad_in
    fig_h = legend_in + scatter_h + 0.06
    fig = plt.figure(figsize=(fig_w, fig_h))
    left = side_pad_in / fig_w
    width = scatter_w / fig_w
    ax = fig.add_axes([left, 0.05 / fig_h, width, scatter_h / fig_h])

    ax.scatter(
        x_true[:, 0],
        x_true[:, 1],
        s=5.0,
        c=real_color,
        alpha=0.48,
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )
    ax.scatter(
        x_gen[:, 0],
        x_gen[:, 1],
        s=6.5,
        c=gen_color,
        alpha=0.82,
        edgecolors="none",
        zorder=3,
        rasterized=True,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.4")

    marker_size = 7.5 if scatter_h < 1.6 else 6.5
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=marker_size,
                markerfacecolor=real_color,
                markeredgecolor="none",
                label="Real",
            ),
            plt.Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=marker_size,
                markerfacecolor=gen_color,
                markeredgecolor="none",
                label=gen_label,
            ),
        ],
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.0,
        columnspacing=1.0,
        borderpad=0.0,
        handletextpad=0.35,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_overlay_scatter(
    path: Path,
    *,
    x_true: np.ndarray,
    c_true: np.ndarray | None = None,
    p_true: np.ndarray | None = None,
    gens: dict[tuple[int, int], np.ndarray] | None = None,
    x_gen: np.ndarray | None = None,
    title: str,
    max_true: int = 400,
    max_gen: int = 400,
    true_color: str = "#b7a99a",
    figsize: tuple[float, float] = (3.8, 3.5),
) -> None:
    """Scatter with grey/taupe real vs colored generated (+ Real/Generated legend).

    Prefer ``gens`` (per-(c,p) colored). If only ``x_gen`` is given, use a single
    gen color (teal).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx_colors = ["#457b9d", "#c1121f", "#8a8178", "#2a9d8f"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)

    x_true = np.asarray(x_true, dtype=np.float64)
    if x_true.shape[0] > max_true:
        idx = rng.choice(x_true.shape[0], size=max_true, replace=False)
        x_true_sc = x_true[idx]
        c_sc = None if c_true is None else np.asarray(c_true)[idx]
        p_sc = None if p_true is None else np.asarray(p_true)[idx]
    else:
        x_true_sc, c_sc, p_sc = x_true, c_true, p_true

    if c_sc is None or p_sc is None:
        ax.scatter(
            x_true_sc[:, 0],
            x_true_sc[:, 1],
            s=10,
            alpha=0.45,
            c=true_color,
            marker="o",
            edgecolors="none",
            label="Real",
            zorder=2,
        )
    else:
        c_sc = np.asarray(c_sc)
        p_sc = np.asarray(p_sc)
        first = True
        for ci in np.unique(c_sc):
            for pj in np.unique(p_sc):
                m = (c_sc == ci) & (p_sc == pj)
                if not np.any(m):
                    continue
                ax.scatter(
                    x_true_sc[m, 0],
                    x_true_sc[m, 1],
                    s=10,
                    alpha=0.40,
                    c=true_color,
                    marker=markers[int(pj) % len(markers)],
                    edgecolors="none",
                    label="Real" if first else None,
                    zorder=2,
                )
                first = False

    gen_labeled = False
    if gens:
        for (ci, pj), xg in gens.items():
            xg = np.asarray(xg, dtype=np.float64)
            if xg.shape[0] > max_gen:
                xg = xg[rng.choice(xg.shape[0], size=max_gen, replace=False)]
            ax.scatter(
                xg[:, 0],
                xg[:, 1],
                s=12,
                alpha=0.70,
                c=ctx_colors[int(ci) % len(ctx_colors)],
                marker=markers[int(pj) % len(markers)],
                edgecolors="none",
                label="Generated" if not gen_labeled else None,
                zorder=3,
            )
            gen_labeled = True
    elif x_gen is not None:
        xg = np.asarray(x_gen, dtype=np.float64)
        if xg.shape[0] > max_gen:
            xg = xg[rng.choice(xg.shape[0], size=max_gen, replace=False)]
        ax.scatter(
            xg[:, 0],
            xg[:, 1],
            s=12,
            alpha=0.70,
            c="#2a9d8f",
            marker="o",
            edgecolors="none",
            label="Generated",
            zorder=3,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.92, fontsize=8, markerscale=1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_pretrain_scatter(
    path: Path,
    *,
    x_true: np.ndarray,
    c_true: np.ndarray,
    p_true: np.ndarray,
    gens: dict[tuple[int, int], np.ndarray],
    title: str,
) -> None:
    """Backward-compatible wrapper around :func:`save_overlay_scatter`."""
    save_overlay_scatter(
        path,
        x_true=x_true,
        c_true=c_true,
        p_true=p_true,
        gens=gens,
        title=title,
    )


__all__ = [
    "REPO_ROOT",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_TEST",
    "POOL_PRETRAIN",
    "POOL_FINETUNE",
    "POOL_HELDOUT",
    "stage_b_config_tag",
    "stage_b_data_root",
    "stage_b_runs_root",
    "resolve_stage_b_roots",
    "load_stage_b_data",
    "mask_rows",
    "make_cp_loader",
    "build_stage_b_model",
    "save_checkpoint",
    "load_checkpoint",
    "sample_cp",
    "evaluate_cells",
    "evaluate_fit_forget",
    "save_overlay_scatter",
    "save_stylized_heldout_overlay",
    "save_heldout_shift_scatter",
    "save_pretrain_scatter",
]
