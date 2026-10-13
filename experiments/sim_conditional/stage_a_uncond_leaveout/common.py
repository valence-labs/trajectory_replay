"""Shared helpers for Stage A *unconditional* leave-out MoG.

Sibling of ``stage_a_leaveout`` (conditional). Same data geometry; model has
``use_conditioning=False`` so fit/forget must compete in a joint density.
"""

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

from experiments.sim_conditional.common import (
    REPO_ROOT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    conditional_results_root,
    mask_rows,
)
from trajectory_replay.eval.metrics import energy_distance, mean_l2_distance
from trajectory_replay.model.flow_matching import ConditionalFlowMatching

__all__ = [
    "REPO_ROOT",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_TEST",
    "stage_a_uncond_data_root",
    "stage_a_uncond_runs_root",
    "load_stage_a_data",
    "mask_rows",
    "make_x_loader",
    "build_unconditional_model",
    "save_checkpoint",
    "load_checkpoint",
    "sample_unconditional",
    "assign_nearest_mean",
    "evaluate_modes",
    "evaluate_fit_forget",
    "save_scatter",
]


def stage_a_uncond_data_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    return (
        conditional_results_root(repo_root)
        / "stage_a_uncond_leaveout"
        / "data"
        / f"dim{dim}"
    )


def stage_a_uncond_runs_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    return conditional_results_root(repo_root) / "stage_a_uncond_leaveout" / f"dim{dim}"


def load_stage_a_data(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = np.load(data_dir / "classes.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    arrays = {k: raw[k] for k in raw.files}
    return arrays, meta


def make_x_loader(
    x: np.ndarray, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ds = TensorDataset(t)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=gen
    )


def build_unconditional_model(
    dim: int,
    *,
    time_emb_dim: int = 64,
    hidden_dim: int = 256,
    depth: int = 4,
) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=dim,
        n_contexts=1,
        n_perturbations=1,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        use_conditioning=False,
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
    model = build_unconditional_model(
        int(cfg["data_dim"]),
        time_emb_dim=int(cfg.get("time_emb_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        depth=int(cfg.get("depth", 4)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def sample_unconditional(
    model: ConditionalFlowMatching,
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
        x = model.sample(
            batch_size=b,
            method=ode_method,
            options=options,
            generator=gen,
        )
        chunks.append(x.detach().cpu().numpy())
        done += b
    return np.concatenate(chunks, axis=0)


def assign_nearest_mean(x: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Nearest-mean labels in ``{0..K-1}``."""
    d2 = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1).astype(np.int64)


def evaluate_modes(
    model: ConditionalFlowMatching,
    arrays: dict[str, np.ndarray],
    classes: list[int],
    *,
    split: int,
    device: torch.device,
    seed: int,
    n_gen: int = 2000,
    ode_method: str = "euler",
    ode_step_size: float = 0.05,
) -> dict[str, Any]:
    """Sample joint model, assign to nearest mean, score vs true class clouds."""
    means = np.asarray(arrays["means"], dtype=np.float64)
    x_gen = sample_unconditional(
        model,
        n_gen,
        device=device,
        seed=seed,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
    )
    assign = assign_nearest_mean(x_gen, means)
    per: dict[str, dict[str, float]] = {}
    energies: list[float] = []
    mean_l2s: list[float] = []
    for k in classes:
        m = mask_rows(arrays, classes=[k], split=split)
        x_true = arrays["x"][m]
        x_k = x_gen[assign == int(k)]
        n_k = int(x_k.shape[0])
        mass = float(n_k) / float(max(n_gen, 1))
        if n_k < 8:
            e = float("nan")
            ml = float("nan")
        else:
            e = float(energy_distance(x_true, x_k))
            ml = float(mean_l2_distance(x_true, x_k))
        per[str(k)] = {
            "energy": e,
            "mean_l2": ml,
            "n_true": int(x_true.shape[0]),
            "n_gen_assigned": n_k,
            "mass": mass,
        }
        if np.isfinite(e):
            energies.append(e)
            mean_l2s.append(ml)
    # Full-cloud energy vs union of requested classes (secondary).
    m_all = mask_rows(arrays, classes=classes, split=split)
    e_mix = float(energy_distance(arrays["x"][m_all], x_gen))
    return {
        "per_class": per,
        "energy_mean": float(np.mean(energies)) if energies else float("nan"),
        "mean_l2_mean": float(np.mean(mean_l2s)) if mean_l2s else float("nan"),
        "energy_mixture_vs_classes": e_mix,
        "n_gen": int(n_gen),
        "mass_by_class": {str(k): per[str(k)]["mass"] for k in classes},
    }


def evaluate_fit_forget(
    model: ConditionalFlowMatching,
    arrays: dict[str, np.ndarray],
    *,
    heldout_class: int,
    pretrain_classes: list[int],
    split: int,
    device: torch.device,
    seed: int,
    n_gen: int = 2000,
    pretrained_forget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fit = evaluate_modes(
        model,
        arrays,
        [heldout_class],
        split=split,
        device=device,
        seed=seed,
        n_gen=n_gen,
    )
    forget = evaluate_modes(
        model,
        arrays,
        pretrain_classes,
        split=split,
        device=device,
        seed=seed + 101,
        n_gen=n_gen,
    )
    # Also one joint sample for heldout mass under full K means.
    means = np.asarray(arrays["means"], dtype=np.float64)
    x_gen = sample_unconditional(
        model, n_gen, device=device, seed=seed + 7, ode_method="euler", ode_step_size=0.05
    )
    assign = assign_nearest_mean(x_gen, means)
    heldout_mass = float((assign == int(heldout_class)).mean())

    out: dict[str, Any] = {
        "fit_energy": fit["per_class"][str(heldout_class)]["energy"],
        "fit_mean_l2": fit["per_class"][str(heldout_class)]["mean_l2"],
        "fit_mass": fit["per_class"][str(heldout_class)]["mass"],
        "heldout_mass_joint": heldout_mass,
        "forget_energy_mean": forget["energy_mean"],
        "forget_mean_l2_mean": forget["mean_l2_mean"],
        "fit": fit,
        "forget": forget,
    }
    if pretrained_forget is not None:
        out["delta_forget_energy_mean"] = float(
            out["forget_energy_mean"] - float(pretrained_forget["energy_mean"])
        )
        out["delta_forget_mean_l2_mean"] = float(
            out["forget_mean_l2_mean"] - float(pretrained_forget["mean_l2_mean"])
        )
    return out


def save_scatter(
    path: Path,
    *,
    x_true: np.ndarray,
    x_gen: np.ndarray,
    means: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(
        x_true[:, 0], x_true[:, 1], s=4, alpha=0.15, c="0.7", edgecolors="none", label="true"
    )
    ax.scatter(
        x_gen[:, 0], x_gen[:, 1], s=6, alpha=0.35, c="C0", edgecolors="none", label="gen"
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(means.shape[0], 1)))
    for k in range(means.shape[0]):
        ax.scatter(
            [means[k, 0]],
            [means[k, 1]],
            marker="x",
            s=90,
            color=colors[k % len(colors)],
            linewidths=2,
            zorder=5,
            label=f"μ{k}" if k < 4 else None,
        )
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
