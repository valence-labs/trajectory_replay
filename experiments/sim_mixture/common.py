"""Shared helpers for unconditional mixture continual-learning experiments."""

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

from trajectory_replay.data.mixture_mog import (
    GaussianMixtureDataset,
    make_two_mode_means,
    sample_gaussian_mixture,
    sample_mixture_with_weights,
)
from trajectory_replay.eval.mixture_metrics import mixture_recovery_metrics
from trajectory_replay.io.results import dump_metrics, run_results_dir, safe_run_name
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.model.mean_flow import ConditionalMeanFlow
from trajectory_replay.training.mixture_fm import (
    train_unconditional_fm,
    train_unconditional_generative_replay,
    train_unconditional_real_replay,
    train_unconditional_trajectory_replay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAMBDAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def mixture_data_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    root = repo_root or REPO_ROOT
    return root / "results_dir" / "sim_mixture" / "data" / f"dim{dim}"


def mixture_runs_root(repo_root: Path | None = None, *, dim: int = 2) -> Path:
    root = repo_root or REPO_ROOT
    return root / "results_dir" / "sim_mixture" / f"dim{dim}"


def build_two_task_data(
    *,
    dim: int = 2,
    separation: float = 3.0,
    sigma: float = 0.35,
    samples_per_task: int = 10_000,
    data_seed: int = 0,
) -> GaussianMixtureDataset:
    rng = np.random.default_rng(data_seed)
    means = make_two_mode_means(dim=dim, separation=separation, axis=0)
    return sample_gaussian_mixture(
        means, samples_per_task=samples_per_task, rng=rng, sigma=sigma
    )


def save_dataset_bundle(
    out_dir: Path,
    ds: GaussianMixtureDataset,
    *,
    sigma: float,
    separation: float,
    data_seed: int,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    mixture_n_samples: int = 10_000,
) -> dict[str, Any]:
    """Save task splits + reference mixtures for each λ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dim": int(ds.dim),
        "sigma": float(sigma),
        "separation": float(separation),
        "data_seed": int(data_seed),
        "n_per_task": int((ds.task == 0).sum()),
        "means": ds.means.tolist(),
        "lambdas": [float(x) for x in lambdas],
        "mixture_n_samples": int(mixture_n_samples),
    }
    np.savez_compressed(
        out_dir / "tasks.npz",
        x=ds.x,
        task=ds.task,
        means=ds.means,
        cov=ds.cov,
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rng = np.random.default_rng(data_seed + 1)
    for lam in lambdas:
        mix = sample_mixture_with_weights(
            ds.means,
            weights=np.array([lam, 1.0 - lam], dtype=np.float64),
            n_samples=mixture_n_samples,
            rng=rng,
            sigma=sigma,
        )
        tag = _lam_tag(lam)
        np.savez_compressed(
            out_dir / f"mixture_{tag}.npz",
            x=mix.x,
            task=mix.task,
            means=mix.means,
            cov=mix.cov,
            lambda_mix=np.asarray([lam], dtype=np.float64),
        )
    return meta


def load_tasks(data_dir: Path) -> tuple[GaussianMixtureDataset, dict[str, Any]]:
    raw = np.load(data_dir / "tasks.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    ds = GaussianMixtureDataset(
        x=raw["x"], task=raw["task"], means=raw["means"], cov=raw["cov"]
    )
    return ds, meta


def load_mixture(data_dir: Path, lambda_mix: float) -> GaussianMixtureDataset:
    raw = np.load(data_dir / f"mixture_{_lam_tag(lambda_mix)}.npz")
    return GaussianMixtureDataset(
        x=raw["x"], task=raw["task"], means=raw["means"], cov=raw["cov"]
    )


def _lam_tag(lam: float) -> str:
    return f"lam{lam:.1f}".replace(".", "p")


def make_x_loader(x: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ds = TensorDataset(t)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=gen)


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
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": config,
    }
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ConditionalFlowMatching, dict[str, Any]]:
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
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
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
            rtol=ode_rtol,
            atol=ode_atol,
            options=options,
            generator=gen,
        )
        chunks.append(x.detach().cpu().numpy())
        done += b
    return np.concatenate(chunks, axis=0)


def evaluate_model(
    model: ConditionalFlowMatching,
    ds: GaussianMixtureDataset,
    *,
    device: torch.device,
    n_gen: int,
    seed: int,
    lambda_target: float | None = None,
    sigma: float | None = None,
    x_true_mixture: np.ndarray | None = None,
    ode_method: str = "euler",
    ode_step_size: float = 0.05,
) -> tuple[dict[str, float], np.ndarray]:
    x_gen = sample_unconditional(
        model,
        n_gen,
        device=device,
        seed=seed,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
    )
    metrics = mixture_recovery_metrics(
        x_gen,
        means=ds.means,
        x_true_mode0=ds.subset(0).x,
        x_true_mode1=ds.subset(1).x,
        lambda_target=lambda_target,
        sigma=sigma,
        x_true_mixture=x_true_mixture,
    )
    return metrics, x_gen


@torch.no_grad()
def sample_mean_flow(
    model: ConditionalMeanFlow,
    n: int,
    *,
    device: torch.device,
    seed: int,
    n_steps: int = 1,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    chunks: list[np.ndarray] = []
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        x = model.sample(batch_size=b, n_steps=n_steps, generator=gen)
        chunks.append(x.detach().cpu().numpy())
        done += b
    return np.concatenate(chunks, axis=0)


def evaluate_mean_flow(
    model: ConditionalMeanFlow,
    ds: GaussianMixtureDataset,
    *,
    device: torch.device,
    n_gen: int,
    seed: int,
    n_steps: int = 1,
    lambda_target: float | None = None,
    sigma: float | None = None,
    x_true_mixture: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    x_gen = sample_mean_flow(
        model, n_gen, device=device, seed=seed, n_steps=n_steps
    )
    metrics = mixture_recovery_metrics(
        x_gen,
        means=ds.means,
        x_true_mode0=ds.subset(0).x,
        x_true_mode1=ds.subset(1).x,
        lambda_target=lambda_target,
        sigma=sigma,
        x_true_mixture=x_true_mixture,
    )
    return metrics, x_gen


def save_scatter(
    path: Path,
    *,
    x_gen: np.ndarray,
    means: np.ndarray,
    title: str,
    x_true: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if x_true is not None:
        ax.scatter(x_true[:, 0], x_true[:, 1], s=6, alpha=0.15, c="0.7", edgecolors="none", label="true")
    ax.scatter(x_gen[:, 0], x_gen[:, 1], s=8, alpha=0.35, c="C0", edgecolors="none", label="gen")
    ax.scatter(means[:, 0], means[:, 1], s=120, marker="X", c=["C1", "C3"], edgecolors="k", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_finetune_strategy(
    strategy: str,
    model: ConditionalFlowMatching,
    model_ref: ConditionalFlowMatching,
    loader: DataLoader,
    *,
    x_support: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    train_seed: int,
    lambda_mix: float,
    ode_method: str,
    ode_step_size: float | None,
    mean_flow=None,
    mean_flow_n_steps: int = 1,
    traj_kappa: str = "none",
    max_grad_norm: float | None = None,
    on_epoch_end=None,
) -> list[dict[str, float]]:
    if strategy == "naive":
        return train_unconditional_fm(
            model,
            loader,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            max_grad_norm=max_grad_norm,
            on_epoch_end=on_epoch_end,
        )
    if strategy == "generative":
        return train_unconditional_generative_replay(
            model,
            model_ref,
            loader,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            lambda_mix=lambda_mix,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            mean_flow=mean_flow,
            mean_flow_n_steps=mean_flow_n_steps,
            max_grad_norm=max_grad_norm,
            on_epoch_end=on_epoch_end,
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
            lambda_mix=lambda_mix,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            mean_flow=mean_flow,
            mean_flow_n_steps=mean_flow_n_steps,
            traj_kappa=traj_kappa,
            max_grad_norm=max_grad_norm,
            on_epoch_end=on_epoch_end,
        )
    if strategy == "real":
        return train_unconditional_real_replay(
            model,
            loader,
            x_support,
            device=device,
            epochs=epochs,
            lr=lr,
            train_seed=train_seed,
            lambda_mix=lambda_mix,
            max_grad_norm=max_grad_norm,
            on_epoch_end=on_epoch_end,
        )
    raise ValueError(f"Unknown strategy {strategy!r}")


def clone_trainable(model_ref: ConditionalFlowMatching, device: torch.device) -> ConditionalFlowMatching:
    """Trainable deep copy of a pretrained checkpoint."""
    import copy

    model = copy.deepcopy(model_ref)
    for p in model.parameters():
        p.requires_grad = True
    model.to(device)
    model.train()
    return model
