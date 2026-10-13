#!/usr/bin/env python3
"""Plot fixed-seed final generations for the sequential ring experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.sim_sequential.common import (
    DEFAULT_TASK_COLORS,
    REPO_ROOT,
    load_dataset_bundle,
    sequential_data_root,
    sequential_results_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--points", type=int, default=2500)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--seed-selection",
        choices=("fixed", "best"),
        default="fixed",
        help="Use one prespecified seed or the lowest-energy seed per method.",
    )
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument(
        "--generative-policy", choices=("equal", "adaptive"), default="adaptive"
    )
    parser.add_argument(
        "--trajectory-policy", choices=("equal", "adaptive"), default="adaptive"
    )
    return parser.parse_args()


def _policy_key(config: dict) -> tuple[str, float | None]:
    return config["lambda_mode"], config.get("lambda_constant")


def _policy_label(key: tuple[str, float | None]) -> str:
    mode, constant = key
    if mode == "constant":
        return rf"constant $\lambda={float(constant):g}$"
    if mode == "equal":
        return "equal-task"
    if mode == "adaptive":
        return "validation-selected"
    return "none"


def _load_complete_runs(runs_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        config_path = run_dir / "config.json"
        stream_path = run_dir / "stream_metrics.json"
        samples_path = run_dir / "tasks" / "04" / "samples.npz"
        if not (config_path.is_file() and stream_path.is_file() and samples_path.is_file()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        stream = json.loads(stream_path.read_text(encoding="utf-8"))
        if config.get("smoke") or len(stream) != 5:
            continue
        energy = float(stream[-1]["metrics"]["energy_uniform_mixture"])
        x = np.load(samples_path)["x"]
        if not math.isfinite(energy) or not bool(np.isfinite(x).all()):
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "config": config,
                "stream": stream,
                "samples": x,
                "energy": energy,
            }
        )
    return runs


def _select_fixed_run(
    runs: list[dict],
    strategy: str,
    *,
    model_seed: int,
    policy: str,
) -> tuple[dict, tuple[str, float | None]]:
    for run in runs:
        config = run["config"]
        if (
            config["strategy"] == strategy
            and int(config["seed"]) == model_seed
            and config["lambda_mode"] == policy
        ):
            return run, _policy_key(config)
    raise ValueError(
        f"No complete finite {strategy}/{policy} run for seed {model_seed}"
    )


def _select_best_run(
    runs: list[dict], strategy: str, *, policy: str
) -> tuple[dict, tuple[str, float | None]]:
    eligible = [
        run
        for run in runs
        if run["config"]["strategy"] == strategy
        and run["config"]["lambda_mode"] == policy
    ]
    if not eligible:
        raise ValueError(f"No complete finite {strategy}/{policy} run")
    selected = min(eligible, key=lambda run: float(run["energy"]))
    return selected, _policy_key(selected["config"])


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    return x[rng.choice(x.shape[0], size=n, replace=False)]


def _draw_ring(ax: plt.Axes, means: np.ndarray) -> None:
    closed = np.vstack([means[:, :2], means[0, :2]])
    ax.plot(closed[:, 0], closed[:, 1], color="0.78", linewidth=0.7, zorder=0)
    ax.scatter(
        means[:, 0],
        means[:, 1],
        marker="+",
        s=30,
        linewidths=0.9,
        color="0.25",
        zorder=4,
    )


def _draw_cloud(
    ax: plt.Axes,
    x: np.ndarray,
    means: np.ndarray,
    *,
    points: int,
    rng: np.random.Generator,
) -> None:
    x = _subsample(x, points, rng)
    labels = np.argmin(
        np.sum((x[:, None, :2] - means[None, :, :2]) ** 2, axis=2),
        axis=1,
    )
    for task_id, color in enumerate(DEFAULT_TASK_COLORS):
        cloud = x[labels == task_id]
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=5,
            color=color,
            alpha=0.48,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    _draw_ring(ax, means)


def plot_final_generations(
    runs: list[dict],
    test,
    *,
    out: Path,
    points: int,
    model_seed: int,
    seed_selection: str,
    subsample_seed: int,
    generative_policy: str,
    trajectory_policy: str,
) -> dict:
    if seed_selection == "best":
        naive, naive_key = _select_best_run(runs, "naive", policy="none")
        generative, generative_key = _select_best_run(
            runs, "generative", policy=generative_policy
        )
        trajectory, trajectory_key = _select_best_run(
            runs, "trajectory", policy=trajectory_policy
        )
    else:
        naive, naive_key = _select_fixed_run(
            runs, "naive", model_seed=model_seed, policy="none"
        )
        generative, generative_key = _select_fixed_run(
            runs,
            "generative",
            model_seed=model_seed,
            policy=generative_policy,
        )
        trajectory, trajectory_key = _select_fixed_run(
            runs,
            "trajectory",
            model_seed=model_seed,
            policy=trajectory_policy,
        )
    selections = [
        ("Naive", naive, naive_key),
        ("Generative replay", generative, generative_key),
        ("Trajectory replay", trajectory, trajectory_key),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.75), sharex=True, sharey=True)
    rng = np.random.default_rng(subsample_seed)
    n_per_task = max(1, points // test.n_tasks)
    target = np.concatenate(
        [
            _subsample(test.subset(task_id).x, n_per_task, rng)
            for task_id in range(test.n_tasks)
        ],
        axis=0,
    )
    _draw_cloud(axes[0], target, test.means, points=points, rng=rng)
    axes[0].set_title("Target")

    metadata: dict[str, dict] = {}
    for ax, (title, run, policy_key) in zip(
        axes[1:], selections, strict=True
    ):
        _draw_cloud(ax, run["samples"], test.means, points=points, rng=rng)
        final = run["stream"][-1]["metrics"]
        ax.set_title(title)
        ax.text(
            0.5,
            0.015,
            rf"$D_{{KL}}={final['mass_kl_uniform']:.3f}$, "
            rf"$E={final['energy_uniform_mixture']:.3f}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        metadata[run["config"]["strategy"]] = {
            "policy": _policy_label(policy_key),
            "model_seed": int(run["config"]["seed"]),
            "seed_selection": seed_selection,
            "run": run["run_dir"].name,
            "energy": float(run["energy"]),
        }

    extent = float(np.max(np.abs(test.x[:, :2]))) + 0.25
    for ax in axes:
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=5.5,
            label=rf"$T_{{{task_id}}}$",
        )
        for task_id, color in enumerate(DEFAULT_TASK_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=test.n_tasks,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.25,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), pad=0.18, w_pad=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    root = sequential_results_root(REPO_ROOT)
    runs_dir = args.runs_dir or root / "runs"
    data_dir = args.data_dir or sequential_data_root(REPO_ROOT, dim=2)
    splits, _ = load_dataset_bundle(data_dir)
    test = splits["test"]
    out = args.out or root / "figures" / "final_generations.png"
    runs = _load_complete_runs(runs_dir)
    metadata = plot_final_generations(
        runs,
        test,
        out=out,
        points=args.points,
        model_seed=args.model_seed,
        seed_selection=args.seed_selection,
        subsample_seed=args.subsample_seed,
        generative_policy=args.generative_policy,
        trajectory_policy=args.trajectory_policy,
    )
    print(f"Wrote {out}, {out.with_suffix('.pdf')}, and selection metadata")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
