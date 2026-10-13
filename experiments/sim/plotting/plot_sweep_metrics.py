#!/usr/bin/env python3
"""Aggregate finetuning-sweep metrics and plot OOD fit vs in-distribution forgetting.

Output layout (per coupling: additive / rotated / hadamard):
  ood_heatmap_*.png          — vf×emb cells; rows=holdout; cols=epochs×data
  seen_delta_heatmap_*.png   — same layout for forgetting (Δ energy on seen strata)
  ood_vs_epochs_*.png        — lines over epochs; rows=holdout; cols=data; 4 vf×emb lines
  seen_delta_vs_epochs_*.png — same for forgetting
  ood_main_effects_*.png     — marginal mean±std for each factor
  seen_delta_main_effects_*.png
  ood_by_vf_emb_*.png        — rows=holdout; cols=vf×emb; bars=epochs×data
  seen_delta_by_vf_emb_*.png
  overview/                  — legacy all-config bar charts
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COUPLINGS = ("additive", "rotated", "hadamard")
HOLDOUT_TAGS = ("hold_ctx", "hold_pert", "hold_combo")
HOLDOUT_LABELS = {
    "hold_ctx": "hold out context",
    "hold_pert": "hold out perturbation",
    "hold_combo": "hold out (c, p)",
}
HOLDOUT_MODES = {
    "hold_ctx": "context",
    "hold_pert": "perturbation",
    "hold_combo": "combo",
}
VF_LABELS = ("vf−", "vf+")
EMB_LABELS = ("emb−", "emb+")
DATA_LABELS = ("all", "held_out", "replay")
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")


def _finetune_data_levels(runs: list[RunScores]) -> tuple[str, ...]:
    """``DATA_LABELS`` present in ``runs``, preserving canonical order."""
    present = {s.config.finetune_data for s in runs}
    return tuple(d for d in DATA_LABELS if d in present)

RUN_DIR_RE = re.compile(
    r"^hold_(?P<hold>ctx|pert|combo)_coup_(?P<coupling>additive|rotated|hadamard)"
    r"_ftep_(?P<ftep>\d+)_vf_(?P<vf>[01])_emb_(?P<emb>[01])_data_(?P<data>all|held_out|replay(?:_r\d+)?)$"
)


@dataclass(frozen=True)
class RunConfig:
    run_dir: str
    holdout_tag: str
    holdout_mode: str
    coupling: str
    finetune_epochs: int
    finetune_train_vf: bool
    finetune_train_old_embeddings: bool
    finetune_data: str
    replay_fraction: float | None
    held_out_pert: int
    held_out_context: int | None

    @property
    def data_label(self) -> str:
        if self.finetune_data == "replay" and self.replay_fraction is not None:
            pct = int(round(self.replay_fraction * 100))
            return f"replay r{pct}"
        return self.finetune_data

    @property
    def config_label(self) -> str:
        vf = "vf+" if self.finetune_train_vf else "vf-"
        emb = "emb+" if self.finetune_train_old_embeddings else "emb-"
        return f"f{self.finetune_epochs}|{vf}|{emb}|{self.data_label}"


@dataclass
class RunScores:
    config: RunConfig
    ood_energy_finetune: float
    seen_energy_pretrain: float
    seen_energy_finetune: float
    seen_energy_delta: float
    ood_n_strata: int
    seen_n_strata: int


def is_held_out_stratum(
    row: dict,
    *,
    holdout_mode: str,
    held_out_pert: int,
    held_out_context: int | None,
) -> bool:
    c, p = int(row["c"]), int(row["p"])
    if holdout_mode == "combo":
        assert held_out_context is not None
        return c == held_out_context and p == held_out_pert
    if holdout_mode == "context":
        assert held_out_context is not None
        return c == held_out_context
    return p == held_out_pert


def mean_metric(rows: list[dict], key: str = "energy") -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(r[key]) for r in rows]))


def parse_run_dir(name: str, metrics: dict) -> RunConfig | None:
    m = RUN_DIR_RE.match(name)
    if m is None:
        return None
    hold_tag = f"hold_{m.group('hold')}"
    holdout_mode = metrics.get("holdout_mode") or HOLDOUT_MODES[hold_tag]
    held_out_context = metrics.get("held_out_context")
    if held_out_context is not None:
        held_out_context = int(held_out_context)
    return RunConfig(
        run_dir=name,
        holdout_tag=hold_tag,
        holdout_mode=str(holdout_mode),
        coupling=m.group("coupling"),
        finetune_epochs=int(m.group("ftep")),
        finetune_train_vf=bool(metrics.get("finetune_train_vf", m.group("vf") == "1")),
        finetune_train_old_embeddings=bool(
            metrics.get("finetune_train_old_embeddings", m.group("emb") == "1")
        ),
        finetune_data=str(metrics.get("finetune_data", m.group("data").split("_")[0])),
        replay_fraction=(
            float(metrics["replay_fraction"])
            if metrics.get("replay_fraction") is not None
            else None
        ),
        held_out_pert=int(metrics.get("held_out_pert", 3)),
        held_out_context=held_out_context,
    )


def score_run(config: RunConfig, metrics: dict) -> RunScores:
    pretrain_eval = metrics["pretrain_ode_eval"]
    finetune_eval = metrics["finetune_ode_eval"]
    kw = {
        "holdout_mode": config.holdout_mode,
        "held_out_pert": config.held_out_pert,
        "held_out_context": config.held_out_context,
    }

    finetune_ood = [r for r in finetune_eval if is_held_out_stratum(r, **kw)]
    finetune_seen = [r for r in finetune_eval if not is_held_out_stratum(r, **kw)]
    # Pretrain eval skips held-out strata; all rows are in-distribution.
    pretrain_seen = list(pretrain_eval)

    seen_pre = mean_metric(pretrain_seen)
    seen_ft = mean_metric(finetune_seen)
    return RunScores(
        config=config,
        ood_energy_finetune=mean_metric(finetune_ood),
        seen_energy_pretrain=seen_pre,
        seen_energy_finetune=seen_ft,
        seen_energy_delta=seen_ft - seen_pre,
        ood_n_strata=len(finetune_ood),
        seen_n_strata=len(finetune_seen),
    )


def load_sweep_scores(sweep_dir: Path) -> list[RunScores]:
    scores: list[RunScores] = []
    for run_path in sorted(sweep_dir.iterdir()):
        if not run_path.is_dir():
            continue
        metrics_path = run_path / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open() as f:
            metrics = json.load(f)
        config = parse_run_dir(run_path.name, metrics)
        if config is None:
            print(f"skip (unrecognized name): {run_path.name}")
            continue
        scores.append(score_run(config, metrics))
    if not scores:
        raise FileNotFoundError(f"No metrics.json found under {sweep_dir}")
    return scores


def write_summary_csv(scores: list[RunScores], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "coupling",
        "holdout_tag",
        "holdout_mode",
        "finetune_epochs",
        "finetune_train_vf",
        "finetune_train_old_embeddings",
        "finetune_data",
        "ood_energy_finetune",
        "seen_energy_pretrain",
        "seen_energy_finetune",
        "seen_energy_delta",
        "ood_n_strata",
        "seen_n_strata",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in scores:
            c = s.config
            w.writerow(
                {
                    "run_dir": c.run_dir,
                    "coupling": c.coupling,
                    "holdout_tag": c.holdout_tag,
                    "holdout_mode": c.holdout_mode,
                    "finetune_epochs": c.finetune_epochs,
                    "finetune_train_vf": c.finetune_train_vf,
                    "finetune_train_old_embeddings": c.finetune_train_old_embeddings,
                    "finetune_data": c.finetune_data,
                    "ood_energy_finetune": s.ood_energy_finetune,
                    "seen_energy_pretrain": s.seen_energy_pretrain,
                    "seen_energy_finetune": s.seen_energy_finetune,
                    "seen_energy_delta": s.seen_energy_delta,
                    "ood_n_strata": s.ood_n_strata,
                    "seen_n_strata": s.seen_n_strata,
                }
            )


def _subset_coupling(runs: list[RunScores], coupling: str) -> list[RunScores]:
    return [s for s in runs if s.config.coupling == coupling]


def _lookup(
    runs: list[RunScores],
    *,
    holdout_tag: str,
    finetune_epochs: int,
    finetune_train_vf: bool,
    finetune_train_old_embeddings: bool,
    finetune_data: str,
) -> RunScores | None:
    for s in runs:
        c = s.config
        if (
            c.holdout_tag == holdout_tag
            and c.finetune_epochs == finetune_epochs
            and c.finetune_train_vf == finetune_train_vf
            and c.finetune_train_old_embeddings == finetune_train_old_embeddings
            and c.finetune_data == finetune_data
        ):
            return s
    return None


def _vf_emb_matrix(
    runs: list[RunScores],
    *,
    holdout_tag: str,
    finetune_epochs: int,
    finetune_data: str,
    value_fn,
) -> np.ndarray:
    mat = np.full((2, 2), np.nan)
    for vi, vf in enumerate((False, True)):
        for ei, emb in enumerate((False, True)):
            hit = _lookup(
                runs,
                holdout_tag=holdout_tag,
                finetune_epochs=finetune_epochs,
                finetune_train_vf=vf,
                finetune_train_old_embeddings=emb,
                finetune_data=finetune_data,
            )
            if hit is not None:
                mat[vi, ei] = value_fn(hit)
    return mat


def _finite_vlim(mats: list[np.ndarray], *, percentile: float = 98) -> tuple[float, float]:
    vals = np.concatenate([m[np.isfinite(m)] for m in mats if np.any(np.isfinite(m))])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(vals)), float(np.percentile(vals, percentile))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _draw_vf_emb_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    title: str,
    fmt: str = ".3f",
    diverging: bool = False,
) -> None:
    if diverging:
        vabs = max(abs(vmin), abs(vmax), 1e-9)
        vmin, vmax = -vabs, vabs
        cmap = "RdBu_r"
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(EMB_LABELS)
    ax.set_yticklabels(VF_LABELS)
    ax.set_title(title, fontsize=9)
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if abs(val - vmin) > 0.55 * (vmax - vmin) else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, color=color)
    return im


def _sorted_runs(runs: list[RunScores]) -> list[RunScores]:
    return sorted(
        runs,
        key=lambda s: (
            HOLDOUT_TAGS.index(s.config.holdout_tag)
            if s.config.holdout_tag in HOLDOUT_TAGS
            else 99,
            s.config.finetune_epochs,
            s.config.finetune_train_vf,
            s.config.finetune_train_old_embeddings,
            s.config.finetune_data,
        ),
    )


def plot_config_heatmaps(
    runs: list[RunScores],
    coupling: str,
    out_path: Path,
    sweep_name: str,
    *,
    value_fn,
    suptitle: str,
    cbar_label: str,
    cmap: str = "viridis_r",
    diverging: bool = False,
) -> None:
    """Grid: rows = holdout type; cols = (epochs × finetune data); cell = vf×emb heatmap."""
    subset = _subset_coupling(runs, coupling)
    if not subset:
        return

    epochs_levels = sorted({s.config.finetune_epochs for s in subset})
    data_levels = _finetune_data_levels(subset)
    col_specs = [(ep, data) for ep in epochs_levels for data in data_levels]
    n_rows, n_cols = len(HOLDOUT_TAGS), len(col_specs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"{suptitle} — {coupling}\n({sweep_name})", fontsize=13)

    row_mats: list[list[np.ndarray]] = []
    for hold in HOLDOUT_TAGS:
        row: list[np.ndarray] = []
        for ep, data in col_specs:
            row.append(
                _vf_emb_matrix(
                    subset,
                    holdout_tag=hold,
                    finetune_epochs=ep,
                    finetune_data=data,
                    value_fn=value_fn,
                )
            )
        row_mats.append(row)

    im_last = None
    for row, hold in enumerate(HOLDOUT_TAGS):
        if diverging:
            vals = np.concatenate(
                [m[np.isfinite(m)] for m in row_mats[row] if np.any(np.isfinite(m))]
            )
            vabs = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
            vmin, vmax = -vabs, vabs
        else:
            vmin, vmax = _finite_vlim(row_mats[row])
        for col, (ep, data) in enumerate(col_specs):
            ax = axes[row, col]
            mat = row_mats[row][col]
            title = f"epochs={ep}, data={data}"
            im_last = _draw_vf_emb_heatmap(
                ax,
                mat,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                title=title,
                diverging=diverging,
            )
        axes[row, 0].set_ylabel(HOLDOUT_LABELS[hold], fontsize=10)

    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.08, wspace=0.35, hspace=0.45)
    if im_last is not None:
        fig.colorbar(
            im_last,
            ax=axes.ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label=cbar_label,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_data_lines(
    runs: list[RunScores],
    coupling: str,
    out_path: Path,
    sweep_name: str,
    *,
    value_fn,
    ylabel: str,
    suptitle: str,
) -> None:
    """Rows = holdout; cols = finetune data; lines = vf×emb; x = finetune epochs."""
    subset = _subset_coupling(runs, coupling)
    if not subset:
        return

    epochs_levels = sorted({s.config.finetune_epochs for s in subset})
    data_levels = _finetune_data_levels(subset)
    if not data_levels:
        return

    n_rows, n_cols = len(HOLDOUT_TAGS), len(data_levels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.3 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(f"{suptitle} — {coupling}\n({sweep_name})", fontsize=13)

    combos = [(vf, emb) for vf in (False, True) for emb in (False, True)]
    legend_col = min(1, n_cols - 1)
    for row, hold in enumerate(HOLDOUT_TAGS):
        for col, data in enumerate(data_levels):
            ax = axes[row, col]
            for k, (vf, emb) in enumerate(combos):
                ys: list[float] = []
                for ep in epochs_levels:
                    hit = _lookup(
                        subset,
                        holdout_tag=hold,
                        finetune_epochs=ep,
                        finetune_train_vf=vf,
                        finetune_train_old_embeddings=emb,
                        finetune_data=data,
                    )
                    ys.append(value_fn(hit) if hit else float("nan"))
                label = f"{VF_LABELS[int(vf)]}, {EMB_LABELS[int(emb)]}"
                ax.plot(
                    epochs_levels,
                    ys,
                    marker=MARKERS[k],
                    linestyle=LINE_STYLES[k],
                    label=label,
                    linewidth=1.8,
                    markersize=7,
                )
            ax.set_title(f"{HOLDOUT_LABELS[hold]} · data={data}", fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(epochs_levels)
            ax.set_xticklabels([str(e) for e in epochs_levels])
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0 and col == legend_col:
                ax.legend(fontsize=7, loc="best", ncol=1)
    for col in range(n_cols):
        axes[-1, col].set_xlabel("finetune epochs")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _group_means(
    runs: list[RunScores], key_fn, levels: tuple[str, ...], value_fn
) -> dict[str, tuple[float, float]]:
    """Return level -> (mean, std) aggregating over all other factors."""
    out: dict[str, list[float]] = {lv: [] for lv in levels}
    for s in runs:
        out[key_fn(s.config)].append(value_fn(s))
    return {
        lv: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0)
        for lv, v in out.items()
        if v
    }


def plot_main_effects(
    runs: list[RunScores],
    coupling: str,
    out_path: Path,
    sweep_name: str,
    *,
    value_fn,
    ylabel: str,
    suptitle: str,
) -> None:
    """Marginal means (± std) for each sweep factor, averaged over the others."""
    subset = _subset_coupling(runs, coupling)
    if not subset:
        return

    epochs_levels = [str(e) for e in sorted({s.config.finetune_epochs for s in subset})]
    data_levels = _finetune_data_levels(subset)
    factors: list[tuple[str, object, tuple[str, ...]]] = [
        (
            "holdout",
            lambda c: c.holdout_tag,
            HOLDOUT_TAGS,
        ),
        ("finetune epochs", lambda c: str(c.finetune_epochs), tuple(epochs_levels)),
        ("train vector field", lambda c: VF_LABELS[int(c.finetune_train_vf)], VF_LABELS),
        ("train old embeddings", lambda c: EMB_LABELS[int(c.finetune_train_old_embeddings)], EMB_LABELS),
        ("finetune data", lambda c: c.finetune_data, data_levels),
    ]

    fig, axes = plt.subplots(1, len(factors), figsize=(3.2 * len(factors), 4.2), squeeze=False)
    fig.suptitle(f"{suptitle} — {coupling}\n({sweep_name}; mean ± std over other factors)", fontsize=12)
    ax_row = axes[0]

    for ax, (fname, key_fn, levels) in zip(ax_row, factors):
        stats = _group_means(subset, key_fn, levels, value_fn)
        x = np.arange(len(levels))
        means = [stats.get(lv, (float("nan"), 0.0))[0] for lv in levels]
        stds = [stats.get(lv, (float("nan"), 0.0))[1] for lv in levels]
        ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        tick_labels = [HOLDOUT_LABELS.get(lv, lv) if fname == "holdout" else lv for lv in levels]
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(fname, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vf_emb_panels(
    runs: list[RunScores],
    coupling: str,
    out_path: Path,
    sweep_name: str,
    *,
    value_fn,
    ylabel: str,
    suptitle: str,
) -> None:
    """Rows = holdout; cols = vf×emb (4 panels); grouped bars: epochs × data."""
    subset = _subset_coupling(runs, coupling)
    if not subset:
        return

    epochs_levels = sorted({s.config.finetune_epochs for s in subset})
    data_levels = _finetune_data_levels(subset)
    bar_groups = [(ep, data) for ep in epochs_levels for data in data_levels]
    n_groups = len(bar_groups)
    x = np.arange(n_groups)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), squeeze=False)
    fig.suptitle(f"{suptitle} — {coupling}\n({sweep_name})", fontsize=13)

    for row, hold in enumerate(HOLDOUT_TAGS):
        for col, (vf, emb) in enumerate(
            (vf, emb) for vf in (False, True) for emb in (False, True)
        ):
            ax = axes[row, col]
            vals = []
            for ep, data in bar_groups:
                hit = _lookup(
                    subset,
                    holdout_tag=hold,
                    finetune_epochs=ep,
                    finetune_train_vf=vf,
                    finetune_train_old_embeddings=emb,
                    finetune_data=data,
                )
                vals.append(value_fn(hit) if hit else float("nan"))
            colors = ["#C44E52" if (np.isfinite(v) and v > 0) else "#55A868" for v in vals]
            ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.3, width=0.75)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"e{ep}\n{d}" for ep, d in bar_groups],
                fontsize=7,
                rotation=45,
                ha="right",
            )
            ax.set_title(f"{VF_LABELS[int(vf)]}, {EMB_LABELS[int(emb)]}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{HOLDOUT_LABELS[hold]}\n{ylabel}", fontsize=9)
            ax.axhline(0.0, color="gray", linewidth=0.6)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0.1, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ood_energy(runs: list[RunScores], coupling: str, out_path: Path, sweep_name: str) -> None:
    subset = _sorted_runs([s for s in runs if s.config.coupling == coupling])
    if not subset:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f"OOD energy distance after finetuning — {coupling}\n({sweep_name})",
        fontsize=13,
    )
    holdout_titles = {
        "hold_ctx": "hold out context",
        "hold_pert": "hold out perturbation",
        "hold_combo": "hold out (c, p) combo",
    }

    for ax, hold_tag in zip(axes, HOLDOUT_TAGS):
        group = [s for s in subset if s.config.holdout_tag == hold_tag]
        if not group:
            ax.set_visible(False)
            continue
        labels = [s.config.config_label for s in group]
        values = [s.ood_energy_finetune for s in group]
        x = np.arange(len(group))
        bars = ax.bar(x, values, color="#4C72B0", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(holdout_titles[hold_tag])
        ax.set_ylabel("mean energy distance")
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seen_forgetting(
    runs: list[RunScores], coupling: str, out_path: Path, sweep_name: str
) -> None:
    subset = _sorted_runs([s for s in runs if s.config.coupling == coupling])
    if not subset:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    fig.suptitle(
        f"In-distribution fit (seen strata) — {coupling}\n({sweep_name})",
        fontsize=13,
    )
    holdout_titles = {
        "hold_ctx": "hold out context",
        "hold_pert": "hold out perturbation",
        "hold_combo": "hold out (c, p) combo",
    }

    for col, hold_tag in enumerate(HOLDOUT_TAGS):
        group = [s for s in subset if s.config.holdout_tag == hold_tag]
        if not group:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        labels = [s.config.config_label for s in group]
        x = np.arange(len(group))
        width = 0.36

        pre = [s.seen_energy_pretrain for s in group]
        post = [s.seen_energy_finetune for s in group]
        delta = [s.seen_energy_delta for s in group]

        ax_top = axes[0, col]
        ax_top.bar(x - width / 2, pre, width, label="pretrain", color="#55A868")
        ax_top.bar(x + width / 2, post, width, label="finetuned", color="#C44E52")
        ax_top.set_title(holdout_titles[hold_tag])
        ax_top.set_ylabel("mean energy distance")
        if col == 0:
            ax_top.legend(fontsize=8)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)

        ax_bot = axes[1, col]
        colors = ["#C44E52" if d > 0 else "#55A868" for d in delta]
        ax_bot.bar(x, delta, color=colors, edgecolor="black", linewidth=0.4)
        ax_bot.axhline(0.0, color="black", linewidth=0.8)
        ax_bot.set_ylabel("Δ energy (finetune − pretrain)")
        ax_bot.set_xlabel("configuration")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)

    axes[0, 0].annotate(
        "lower is better",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        color="dimgray",
    )
    axes[1, 0].annotate(
        "positive = forgetting",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        color="dimgray",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot OOD and in-distribution metrics for a finetuning sweep folder."
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Sweep root (e.g. results_dir/sweep_ft_small_pert_effect)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for plots and summary CSV",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not sweep_dir.is_dir():
        raise FileNotFoundError(sweep_dir)

    scores = load_sweep_scores(sweep_dir)
    sweep_name = sweep_dir.name
    print(f"Loaded {len(scores)} runs from {sweep_dir}")

    write_summary_csv(scores, output_dir / "sweep_summary.csv")

    for coupling in COUPLINGS:
        n = sum(1 for s in scores if s.config.coupling == coupling)
        if not n:
            print(f"  {coupling}: no runs")
            continue

        # Factor-disentangled views (primary).
        plot_config_heatmaps(
            scores,
            coupling,
            output_dir / f"ood_heatmap_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.ood_energy_finetune,
            suptitle="OOD energy after finetuning (lower = better)",
            cbar_label="mean energy distance",
            cmap="viridis_r",
        )
        plot_config_heatmaps(
            scores,
            coupling,
            output_dir / f"seen_delta_heatmap_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.seen_energy_delta,
            suptitle="Seen-strata forgetting (Δ energy, finetune − pretrain)",
            cbar_label="Δ energy",
            diverging=True,
        )
        plot_epoch_data_lines(
            scores,
            coupling,
            output_dir / f"ood_vs_epochs_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.ood_energy_finetune,
            ylabel="OOD mean energy",
            suptitle="OOD energy vs finetune epochs",
        )
        plot_epoch_data_lines(
            scores,
            coupling,
            output_dir / f"seen_delta_vs_epochs_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.seen_energy_delta,
            ylabel="Δ energy on seen strata",
            suptitle="Forgetting vs finetune epochs",
        )
        plot_main_effects(
            scores,
            coupling,
            output_dir / f"ood_main_effects_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.ood_energy_finetune,
            ylabel="OOD mean energy",
            suptitle="OOD energy — main effects",
        )
        plot_main_effects(
            scores,
            coupling,
            output_dir / f"seen_delta_main_effects_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.seen_energy_delta,
            ylabel="Δ energy (seen)",
            suptitle="Forgetting — main effects",
        )
        plot_vf_emb_panels(
            scores,
            coupling,
            output_dir / f"ood_by_vf_emb_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.ood_energy_finetune,
            ylabel="OOD energy",
            suptitle="OOD energy by trainable components (vf × emb)",
        )
        plot_vf_emb_panels(
            scores,
            coupling,
            output_dir / f"seen_delta_by_vf_emb_{coupling}.png",
            sweep_name,
            value_fn=lambda s: s.seen_energy_delta,
            ylabel="Δ energy",
            suptitle="Forgetting by trainable components (vf × emb)",
        )

        # Compact overview (legacy layout).
        legacy_dir = output_dir / "overview"
        plot_ood_energy(scores, coupling, legacy_dir / f"ood_energy_finetuned_{coupling}.png", sweep_name)
        plot_seen_forgetting(
            scores, coupling, legacy_dir / f"seen_forgetting_{coupling}.png", sweep_name
        )
        print(
            f"  {coupling}: {n} runs → heatmaps, main effects, epoch lines, vf×emb panels"
            f" (+ overview/)"
        )

    print(f"Wrote summary and plots to {output_dir}")


if __name__ == "__main__":
    main()
