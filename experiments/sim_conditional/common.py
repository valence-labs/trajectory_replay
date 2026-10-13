"""Shared helpers for Stage A conditional leave-out MoG."""

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

from trajectory_replay.eval.metrics import energy_distance, mean_l2_distance
from trajectory_replay.model.flow_matching import ConditionalFlowMatching

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2


def conditional_results_root(repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    return root / "results_dir" / "sim_conditional"


def stage_a_data_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    return conditional_results_root(repo_root) / "stage_a_leaveout" / "data" / f"dim{dim}"


def stage_a_runs_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    return conditional_results_root(repo_root) / "stage_a_leaveout" / f"dim{dim}"


def load_stage_a_data(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = np.load(data_dir / "classes.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    arrays = {k: raw[k] for k in raw.files}
    return arrays, meta


def mask_rows(
    arrays: dict[str, np.ndarray],
    *,
    classes: list[int] | None = None,
    split: int | None = None,
) -> np.ndarray:
    y = arrays["y"]
    s = arrays["split"]
    m = np.ones(y.shape[0], dtype=bool)
    if classes is not None:
        m &= np.isin(y, np.asarray(classes, dtype=np.int64))
    if split is not None:
        m &= s == int(split)
    return m


def make_cp_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Class label -> context id; dummy perturbation p≡0."""
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ct = torch.from_numpy(np.asarray(y, dtype=np.int64))
    pt = torch.zeros_like(ct)
    ds = TensorDataset(xt, ct, pt)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=gen
    )


def build_conditional_model(
    dim: int,
    n_classes: int,
    *,
    cond_emb_dim: int = 32,
    time_emb_dim: int = 64,
    hidden_dim: int = 256,
    depth: int = 4,
) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(
        data_dim=dim,
        n_contexts=n_classes,
        n_perturbations=1,
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
    model = build_conditional_model(
        int(cfg["data_dim"]),
        int(cfg["n_classes"]),
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
def sample_class(
    model: ConditionalFlowMatching,
    class_id: int,
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
        cb = torch.full((b,), int(class_id), dtype=torch.long, device=device)
        pb = torch.zeros((b,), dtype=torch.long, device=device)
        x = model.sample(
            cb,
            pb,
            method=ode_method,
            options=options,
            generator=gen,
        )
        chunks.append(x.detach().cpu().numpy())
        done += b
    return np.concatenate(chunks, axis=0)


def evaluate_classes(
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
    """Per-class energy / mean-L2 vs true split cloud; plus mean aggregates."""
    per: dict[str, dict[str, float]] = {}
    energies: list[float] = []
    mean_l2s: list[float] = []
    for k in classes:
        m = mask_rows(arrays, classes=[k], split=split)
        x_true = arrays["x"][m]
        n = min(int(n_gen), int(x_true.shape[0]))
        x_gen = sample_class(
            model,
            k,
            n,
            device=device,
            seed=seed + 17 * int(k),
            ode_method=ode_method,
            ode_step_size=ode_step_size,
        )
        e = float(energy_distance(x_true, x_gen))
        ml = float(mean_l2_distance(x_true, x_gen))
        per[str(k)] = {"energy": e, "mean_l2": ml, "n_true": int(x_true.shape[0]), "n_gen": n}
        energies.append(e)
        mean_l2s.append(ml)
    return {
        "per_class": per,
        "energy_mean": float(np.mean(energies)) if energies else float("nan"),
        "mean_l2_mean": float(np.mean(mean_l2s)) if mean_l2s else float("nan"),
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
    fit = evaluate_classes(
        model,
        arrays,
        [heldout_class],
        split=split,
        device=device,
        seed=seed,
        n_gen=n_gen,
    )
    forget = evaluate_classes(
        model,
        arrays,
        pretrain_classes,
        split=split,
        device=device,
        seed=seed + 101,
        n_gen=n_gen,
    )
    out: dict[str, Any] = {
        "fit_energy": fit["per_class"][str(heldout_class)]["energy"],
        "fit_mean_l2": fit["per_class"][str(heldout_class)]["mean_l2"],
        "forget_energy_mean": forget["energy_mean"],
        "forget_mean_l2_mean": forget["mean_l2_mean"],
        "fit": fit,
        "forget": forget,
    }
    if pretrained_forget is not None:
        # Δforget = current - pretrained (positive => worse retention)
        out["delta_forget_energy_mean"] = float(
            out["forget_energy_mean"] - float(pretrained_forget["energy_mean"])
        )
        out["delta_forget_mean_l2_mean"] = float(
            out["forget_mean_l2_mean"] - float(pretrained_forget["mean_l2_mean"])
        )
    return out


def save_class_scatter(
    path: Path,
    arrays: dict[str, np.ndarray],
    gens: dict[int, np.ndarray],
    *,
    means: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, max(means.shape[0], 1)))
    for k in range(means.shape[0]):
        m = mask_rows(arrays, classes=[k], split=SPLIT_VAL)
        ax.scatter(
            arrays["x"][m, 0],
            arrays["x"][m, 1],
            s=4,
            alpha=0.15,
            color="0.7",
            label="true val" if k == 0 else None,
        )
    for k, xg in gens.items():
        ax.scatter(
            xg[:, 0],
            xg[:, 1],
            s=6,
            alpha=0.35,
            color=colors[k % len(colors)],
            label=f"gen c={k}",
        )
        ax.scatter(
            [means[k, 0]],
            [means[k, 1]],
            marker="x",
            s=80,
            color=colors[k % len(colors)],
            linewidths=2,
        )
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
