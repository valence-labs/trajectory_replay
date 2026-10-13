"""Save run metrics and true-vs-generated 2D visualizations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal

TxMetricsSuite = Literal["vcb", "celleval"]

_METRICS_DIR_NAMES: dict[TxMetricsSuite, str] = {
    "vcb": "metrics_vcb",
    "celleval": "metrics_celleval",
}
_COMPARISONS_DIR_NAMES: dict[TxMetricsSuite, str] = {
    "vcb": "comparisons_vcb",
    "celleval": "comparisons_celleval",
}

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

from trajectory_replay.eval.metrics import distribution_metrics
from trajectory_replay.model.flow_matching import ConditionalFlowMatching


def safe_run_name(name: str) -> str:
    """Sanitize a run name; allow nested subdirs under ``results_dir/`` (e.g. ``sweep_ft/run_a``)."""
    name = name.strip().replace("\\", "/")
    parts = [p for p in name.split("/") if p and p not in (".", "..")]
    return "/".join(parts) if parts else "unnamed_run"


DEFAULT_TX_RESULTS_BUCKET = "ho_d2_indistrib_small"


def tx_results_bucket(bucket: str | None = None) -> str:
    """Active tx results bucket (override with env ``TX_RESULTS_BUCKET``)."""
    if bucket is not None:
        return bucket
    return os.environ.get("TX_RESULTS_BUCKET", DEFAULT_TX_RESULTS_BUCKET)


def tx_results_root(repo_root: Path, bucket: str | None = None) -> Path:
    """``<repo_root>/results_dir/<bucket>/`` for transcriptomics finetuning runs."""
    return repo_root / "results_dir" / tx_results_bucket(bucket)


def tx_run_dir(repo_root: Path, run_name: str, bucket: str | None = None) -> Path:
    """Run output under the active tx results bucket."""
    return tx_results_root(repo_root, bucket) / safe_run_name(run_name)


def tx_metrics_dir(
    repo_root: Path,
    bucket: str | None = None,
    *,
    suite: TxMetricsSuite = "vcb",
) -> Path:
    """Aggregated metrics tables under ``metrics_vcb/`` or ``metrics_celleval/``."""
    return tx_results_root(repo_root, bucket) / _METRICS_DIR_NAMES[suite]


def tx_comparisons_dir(
    repo_root: Path,
    name: str = "pbmc_split1",
    bucket: str | None = None,
    *,
    suite: TxMetricsSuite = "vcb",
) -> Path:
    """Method comparison plots under ``comparisons_vcb/`` or ``comparisons_celleval/``."""
    return tx_results_root(repo_root, bucket) / _COMPARISONS_DIR_NAMES[suite] / name


def resolve_tx_path(repo_root: Path, rel: str, bucket: str | None = None) -> Path:
    """Resolve manifest/checkpoint paths (absolute, legacy ``results_dir/...``, or bucket-relative)."""
    rel = rel.strip().replace("\\", "/")
    if not rel:
        raise ValueError("empty path")
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    if rel.startswith("results_dir/"):
        return (repo_root / rel).resolve()
    return (tx_results_root(repo_root, bucket) / rel).resolve()


def tx_path_rel(repo_root: Path, path: Path, bucket: str | None = None) -> str:
    """Path relative to the active bucket (for storing in ``metrics.json``)."""
    path = path.resolve()
    root = tx_results_root(repo_root, bucket).resolve()
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        pass
    try:
        return str(path.relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def tx_path_display(repo_root: Path, path: Path, bucket: str | None = None) -> str:
    """Human-readable path under ``results_dir/<bucket>/...``."""
    path = path.resolve()
    try:
        rel = path.relative_to(tx_results_root(repo_root, bucket).resolve())
        return f"results_dir/{tx_results_bucket(bucket)}/{rel}".replace("\\", "/")
    except ValueError:
        return str(path)


def run_results_dir(repo_root: Path, run_name: str) -> Path:
    """Sim / legacy runs under ``<repo_root>/results_dir/<run-name>/`` (no tx bucket)."""
    return repo_root / "results_dir" / safe_run_name(run_name)


def namespace_to_jsonable(ns: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v) if v is not None else None
        elif isinstance(v, torch.device):
            out[k] = str(v)
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def dump_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _panel_title(ci: int, pj: int, stats: dict[str, float]) -> str:
    return (
        f"c={ci}, p={pj}\n"
        f"mse(μ)={stats['mse_mean_vectors']:.3f}  "
        f"E={stats['energy']:.3f}"
    )


def _per_condition_stats(
    x_true: np.ndarray,
    x_gen: np.ndarray,
    *,
    proj_axes: tuple[int, int],
) -> dict[str, float]:
    st = distribution_metrics(x_true, x_gen, dims=list(proj_axes))
    return {"n": float(x_true.shape[0]), **st}


def _render_one_panel(
    ax: Any,
    ci: int,
    pj: int,
    x: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    model: ConditionalFlowMatching,
    *,
    device: torch.device,
    ax0: int,
    ax1: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    gen: torch.Generator,
    n_gen_per_combo: int | None,
    control_pert: int | None = None,
    skip_perturbations: frozenset[int] | None = None,
    skip_contexts: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
) -> dict[str, Any] | None:
    if skip_combos is not None and (ci, pj) in skip_combos:
        ax.set_title(f"c={ci}, p={pj} (held out)")
        ax.text(0.5, 0.5, "not trained", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        return None
    if skip_contexts is not None and ci in skip_contexts:
        ax.set_title(f"c={ci}, p={pj} (held out)")
        ax.text(0.5, 0.5, "not trained", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        return None
    if skip_perturbations is not None and pj in skip_perturbations:
        ax.set_title(f"c={ci}, p={pj} (held out)")
        ax.text(0.5, 0.5, "not trained", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        return None
    focal = (c == ci) & (p == pj)
    xt = x[focal]
    n_true = xt.shape[0]
    if n_true == 0:
        ax.set_title(f"c={ci}, p={pj} (no data)")
        ax.axis("off")
        return None

    other = ~focal
    if np.any(other):
        xo = x[other]
        # Background: all real cells from other (c, p). Keep strong contrast (rasterized+low alpha
        # made this layer disappear in many PNG viewers).
        ax.scatter(
            xo[:, ax0],
            xo[:, ax1],
            s=10,
            alpha=0.55,
            c="#6e6e6e",
            edgecolors="none",
            label="true (other c,p)",
            zorder=1,
        )

    n_gen = n_gen_per_combo if n_gen_per_combo is not None else n_true
    n_gen = min(n_gen, n_true)
    xt_sub = xt[:n_gen]

    cb = torch.full((n_gen,), ci, dtype=torch.long, device=device)
    pb = torch.full((n_gen,), pj, dtype=torch.long, device=device)
    with torch.no_grad():
        if control_pert is not None:
            ctrl = (c == ci) & (p == control_pert)
            xc = x[ctrl]
            if xc.shape[0] == 0:
                ax.set_title(f"c={ci}, p={pj} (no control)")
                ax.axis("off")
                return None
            xc_t = torch.from_numpy(xc).float().to(device)
            pick = torch.randint(xc_t.shape[0], (n_gen,), device=device, generator=gen)
            x0 = xc_t[pick]
            xg = model.sample_from_x0(
                x0,
                cb,
                pb,
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
            )
        else:
            xg = model.sample(
                cb,
                pb,
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
                generator=gen,
            )
    xg_np = xg.detach().cpu().numpy()
    st = _per_condition_stats(xt_sub, xg_np, proj_axes=(ax0, ax1))

    ax.scatter(
        xt_sub[:, ax0],
        xt_sub[:, ax1],
        s=18,
        alpha=0.85,
        c="tab:blue",
        edgecolors="white",
        linewidths=0.2,
        label=f"true (c={ci}, p={pj})",
        zorder=3,
    )
    ax.scatter(
        xg_np[:, ax0],
        xg_np[:, ax1],
        s=18,
        alpha=0.85,
        c="tab:orange",
        edgecolors="white",
        linewidths=0.2,
        label=f"generated (c={ci}, p={pj})",
        zorder=4,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(_panel_title(ci, pj, st), fontsize=8)
    return {"c": ci, "p": pj, **st}


def save_true_scatter_by_cp(
    out_path: Path,
    x: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    *,
    proj_axes: tuple[int, int] = (0, 1),
    title: str | None = None,
) -> Path:
    """Single-panel scatter of true x: color = context c, marker = perturbation p (same as exemplify notebook)."""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(c, dtype=np.int64).ravel()
    p = np.asarray(p, dtype=np.int64).ravel()
    if x.shape[0] != c.shape[0] or x.shape[0] != p.shape[0]:
        raise ValueError("x, c, p must have the same number of rows.")
    d = x.shape[1]
    ax0, ax1 = proj_axes
    if d < 2 or ax0 < 0 or ax1 < 0 or ax0 >= d or ax1 >= d:
        raise ValueError(f"Need d>=2 and valid proj_axes; got d={d}, proj_axes=({ax0}, {ax1}).")

    n_c = int(c.max()) + 1
    n_p = int(p.max()) + 1
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for ci in range(n_c):
        for pj in range(n_p):
            m = (c == ci) & (p == pj)
            if not np.any(m):
                continue
            ax.scatter(
                x[m, ax0],
                x[m, ax1],
                s=16,
                alpha=0.55,
                color=cmap(ci % 10),
                marker=markers[pj % len(markers)],
                edgecolors="none",
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"feature {ax0}")
    ax.set_ylabel(f"feature {ax1}")
    if title is None:
        title = (
            f"True data by (c, p), projection ({ax0}, {ax1}), d={d}"
            if d > 2
            else "True data by (c, p)"
        )
    ax.set_title(title)
    c_handles = [
        mlines.Line2D(
            [],
            [],
            color=cmap(i % 10),
            marker="o",
            linestyle="None",
            markersize=8,
            label=f"c={i}",
        )
        for i in range(n_c)
    ]
    p_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.35",
            marker=markers[j % len(markers)],
            linestyle="None",
            markersize=8,
            label=f"p={j}",
        )
        for j in range(n_p)
    ]
    leg_c = ax.legend(handles=c_handles, title="context c", loc="upper left")
    ax.add_artist(leg_c)
    ax.legend(handles=p_handles, title="perturbation p", loc="lower left")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_true_vs_generated_figures(
    out_path_stem: Path,
    x: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    model: ConditionalFlowMatching,
    *,
    device: torch.device,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    plot_seed: int,
    n_gen_per_combo: int | None,
    proj_axes: tuple[int, int] = (0, 1),
    max_panels_per_figure: int = 48,
    control_pert: int | None = None,
    skip_perturbations: frozenset[int] | None = None,
    skip_contexts: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """One or more PNGs: each panel shows all true cells (gray) plus focal true (blue) and focal generated (orange)."""
    model.eval()
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(c, dtype=np.int64).ravel()
    p = np.asarray(p, dtype=np.int64).ravel()
    if c.shape[0] != x.shape[0] or p.shape[0] != x.shape[0]:
        raise ValueError("c and p must match number of rows in x.")
    d = x.shape[1]
    ax0, ax1 = proj_axes
    if ax0 >= d or ax1 >= d or ax0 < 0 or ax1 < 0:
        raise ValueError(f"proj_axes ({ax0}, {ax1}) invalid for data dim {d}.")
    if d < 2:
        raise ValueError("Need data dimension >= 2 for 2D scatter plots.")

    n_c = int(c.max()) + 1
    n_p = int(p.max()) + 1
    pairs: list[tuple[int, int]] = [(ci, pj) for ci in range(n_c) for pj in range(n_p)]

    gen = torch.Generator(device=device)
    gen.manual_seed(plot_seed)

    per_condition: list[dict[str, Any]] = []
    saved_paths: list[Path] = []

    for page_start in range(0, len(pairs), max_panels_per_figure):
        chunk = pairs[page_start : page_start + max_panels_per_figure]
        n_pan = len(chunk)
        ncol = min(8, n_pan)
        nrow = math.ceil(n_pan / ncol)
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(2.7 * ncol, 2.5 * nrow),
            squeeze=False,
        )
        flat = axes.ravel()
        first_legend = True
        for k in range(n_pan):
            ci, pj = chunk[k]
            st = _render_one_panel(
                flat[k],
                ci,
                pj,
                x,
                c,
                p,
                model,
                device=device,
                ax0=ax0,
                ax1=ax1,
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_options=ode_options,
                gen=gen,
                n_gen_per_combo=n_gen_per_combo,
                control_pert=control_pert,
                skip_perturbations=skip_perturbations,
                skip_contexts=skip_contexts,
                skip_combos=skip_combos,
            )
            if st is not None:
                per_condition.append(st)
                if first_legend:
                    flat[k].legend(loc="upper right", fontsize=7, markerscale=1.0)
                    first_legend = False
        for k in range(n_pan, len(flat)):
            flat[k].axis("off")

        if d > 2:
            title = (
                f"True vs generated (proj dims {ax0},{ax1}; d={d})\n"
                "Gray: true cells from other (c,p); blue/orange: focal true vs generated."
            )
        else:
            title = (
                "True vs generated\n"
                "Gray: true cells from other (c,p); blue/orange: focal true vs generated."
            )
        if len(pairs) > max_panels_per_figure:
            title += f" — page {page_start // max_panels_per_figure + 1}"
        fig.suptitle(title, fontsize=11, y=1.01)
        fig.tight_layout()
        suffix = "" if len(pairs) <= max_panels_per_figure else f"_{page_start // max_panels_per_figure:03d}"
        out_file = Path(f"{out_path_stem}{suffix}.png")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_file)

    return saved_paths, per_condition


def save_heldout_compare_overlay(
    out_path: Path,
    x: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    x_true_held: np.ndarray,
    x_gen_d2d: np.ndarray,
    x_gen_noise: np.ndarray,
    *,
    held_context: int,
    held_pert: int,
    proj_axes: tuple[int, int] = (0, 1),
    control_pert: int | None = None,
    metrics_d2d: dict[str, float] | None = None,
    metrics_noise: dict[str, float] | None = None,
    x_gen_d2d_alt: np.ndarray | None = None,
    label_d2d: str = "generated data-to-data",
    label_d2d_alt: str = "generated d2d (no identity train)",
    metrics_d2d_alt: dict[str, float] | None = None,
) -> Path:
    """Single-panel overlay: held-out true vs data-to-data vs noise-to-data generations."""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(c, dtype=np.int64).ravel()
    p = np.asarray(p, dtype=np.int64).ravel()
    d = x.shape[1]
    ax0, ax1 = proj_axes
    if ax0 >= d or ax1 >= d:
        raise ValueError(f"proj_axes ({ax0}, {ax1}) invalid for data dim {d}.")
    if d < 2:
        raise ValueError("Need data dimension >= 2 for 2D scatter plots.")

    focal = (c == held_context) & (p == held_pert)
    other = ~focal

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if np.any(other):
        xo = x[other]
        ax.scatter(
            xo[:, ax0],
            xo[:, ax1],
            s=8,
            alpha=0.45,
            c="#6e6e6e",
            edgecolors="none",
            label="true (other c,p)",
            zorder=1,
        )

    ax.scatter(
        x_true_held[:, ax0],
        x_true_held[:, ax1],
        s=22,
        alpha=0.9,
        c="tab:blue",
        edgecolors="white",
        linewidths=0.25,
        label=f"true held-out (c={held_context}, p={held_pert})",
        zorder=3,
    )
    ax.scatter(
        x_gen_d2d[:, ax0],
        x_gen_d2d[:, ax1],
        s=22,
        alpha=0.9,
        c="tab:orange",
        edgecolors="white",
        linewidths=0.25,
        label=label_d2d,
        zorder=4,
    )
    if x_gen_d2d_alt is not None:
        ax.scatter(
            x_gen_d2d_alt[:, ax0],
            x_gen_d2d_alt[:, ax1],
            s=22,
            alpha=0.9,
            c="tab:red",
            edgecolors="white",
            linewidths=0.25,
            label=label_d2d_alt,
            zorder=5,
        )
    ax.scatter(
        x_gen_noise[:, ax0],
        x_gen_noise[:, ax1],
        s=22,
        alpha=0.9,
        c="tab:green",
        edgecolors="white",
        linewidths=0.25,
        label="generated noise-to-data",
        zorder=6,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7)

    subtitle_parts: list[str] = []
    if control_pert is not None:
        subtitle_parts.append(f"control p={control_pert}")
    if metrics_d2d is not None:
        subtitle_parts.append(f"{label_d2d} E={metrics_d2d['energy']:.3f}")
    if metrics_d2d_alt is not None:
        subtitle_parts.append(f"{label_d2d_alt} E={metrics_d2d_alt['energy']:.3f}")
    if metrics_noise is not None:
        subtitle_parts.append(f"noise E={metrics_noise['energy']:.3f}")
    subtitle = "  |  ".join(subtitle_parts)
    title = f"Held-out (c={held_context}, p={held_pert})"
    if d > 2:
        title += f" — proj dims {ax0},{ax1}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path