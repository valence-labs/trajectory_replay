#!/usr/bin/env python3
"""Plot seen (non-held-out) strata metrics vs replay settings for a replay finetune sweep.

Expects runs under ``sweep_ft_replay``-style names with ``finetune_data=replay``.
X-axis uses ``replay_buffer_size`` when it varies across runs; otherwise
``replay_fraction`` (per-batch fraction from the fixed buffer).

Panels follow the replay sweep design:
  vf+ emb+  →  hold_ctx, hold_pert, hold_combo
  vf- emb-  →  hold_combo only (embedding-row finetune on held-out c and p)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.sim.plotting.plot_sweep_metrics import (  # noqa: E402
    COUPLINGS,
    HOLDOUT_LABELS,
    HOLDOUT_TAGS,
    RunScores,
    parse_run_dir,
    score_run,
)


@dataclass(frozen=True)
class ReplayRunPoint:
    scores: RunScores
    replay_fraction: float
    replay_buffer_size: int | None
    replay_max_samples: int | None

    @property
    def x_buffer_size(self) -> float:
        """Primary x-axis: buffer size (falls back to fraction if size missing)."""
        if self.replay_buffer_size is not None:
            return float(self.replay_buffer_size)
        return self.replay_fraction

    @property
    def x_fraction(self) -> float:
        return self.replay_fraction


def _replay_config_active(vf: bool, emb: bool, holdout_tag: str) -> bool:
    """Match submit_sweep_ft_replay.sh: vf+emb+ on all holdouts; vf−emb− on combo only."""
    if vf and emb:
        return holdout_tag in HOLDOUT_TAGS
    if not vf and not emb:
        return holdout_tag == "hold_combo"
    return False


def _include_run(config) -> bool:
    return _replay_config_active(
        config.finetune_train_vf,
        config.finetune_train_old_embeddings,
        config.holdout_tag,
    )


def _run_metrics_path(sweep_dir: Path, run_dir_name: str) -> Path:
    """``run_dir`` in metrics is the leaf folder name under ``sweep_dir``."""
    leaf = run_dir_name.split("/")[-1]
    return sweep_dir / leaf / "metrics.json"


def load_replay_points(sweep_dir: Path) -> list[ReplayRunPoint]:
    points: list[ReplayRunPoint] = []
    for run_path in sorted(sweep_dir.iterdir()):
        if not run_path.is_dir():
            continue
        metrics_path = run_path / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open() as f:
            metrics = json.load(f)
        if metrics.get("finetune_data") != "replay":
            continue
        config = parse_run_dir(run_path.name, metrics)
        if config is None:
            print(f"skip (unrecognized name): {run_path.name}")
            continue
        if not _include_run(config):
            continue
        frac = metrics.get("replay_fraction")
        if frac is None:
            print(f"skip (no replay_fraction): {run_path.name}")
            continue
        buf = metrics.get("replay_buffer_size")
        if buf is not None:
            buf = int(buf)
        max_s = metrics.get("replay_max_samples")
        if max_s is not None:
            max_s = int(max_s)
        points.append(
            ReplayRunPoint(
                scores=score_run(config, metrics),
                replay_fraction=float(frac),
                replay_buffer_size=buf,
                replay_max_samples=max_s,
            )
        )
    if not points:
        raise FileNotFoundError(f"No replay metrics.json under {sweep_dir}")
    return points


def _x_axis_spec(points: list[ReplayRunPoint]) -> tuple[str, str, callable]:
    sizes = {p.replay_buffer_size for p in points if p.replay_buffer_size is not None}
    if len(sizes) > 1:
        return (
            "Replay buffer size (samples)",
            "buffer",
            lambda p: p.x_buffer_size,
        )
    fracs = sorted({p.replay_fraction for p in points})
    if len(fracs) > 1 or not sizes:
        return (
            "Replay batch fraction",
            "fraction",
            lambda p: p.x_fraction,
        )
    return (
        "Replay batch fraction",
        "fraction",
        lambda p: p.x_fraction,
    )


def _config_style(vf: bool, emb: bool) -> tuple[str, str, str]:
    if vf and emb:
        return "vf+ emb+", "#C44E52", "-"
    return "vf− emb−", "#4C72B0", "--"


def _legend_if_lines(ax, *, fontsize: float = 8) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fontsize, loc="best")


def _maybe_log_yscale(ax, values: list[float], *, log_y: bool) -> None:
    """Log y when enabled and all plotted values are strictly positive."""
    if not log_y:
        return
    finite = [float(v) for v in values if np.isfinite(v)]
    if finite and min(finite) > 0:
        ax.set_yscale("log")


def _subset(
    points: list[ReplayRunPoint],
    *,
    coupling: str,
    holdout_tag: str,
    vf: bool,
    emb: bool,
) -> list[ReplayRunPoint]:
    out = [
        p
        for p in points
        if p.scores.config.coupling == coupling
        and p.scores.config.holdout_tag == holdout_tag
        and p.scores.config.finetune_train_vf == vf
        and p.scores.config.finetune_train_old_embeddings == emb
    ]
    key_fn = _x_axis_spec(points)[2]
    return sorted(out, key=key_fn)


def write_summary_csv(points: list[ReplayRunPoint], path: Path, x_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "coupling",
        "holdout_tag",
        "finetune_train_vf",
        "finetune_train_old_embeddings",
        "replay_fraction",
        "replay_buffer_size",
        "replay_max_samples",
        x_key,
        "seen_energy_pretrain",
        "seen_energy_finetune",
        "seen_energy_delta",
        "ood_energy_finetune",
    ]
    key_fn = _x_axis_spec(points)[2]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in points:
            s = p.scores
            c = s.config
            w.writerow(
                {
                    "run_dir": c.run_dir,
                    "coupling": c.coupling,
                    "holdout_tag": c.holdout_tag,
                    "finetune_train_vf": c.finetune_train_vf,
                    "finetune_train_old_embeddings": c.finetune_train_old_embeddings,
                    "replay_fraction": p.replay_fraction,
                    "replay_buffer_size": p.replay_buffer_size,
                    "replay_max_samples": p.replay_max_samples,
                    x_key: key_fn(p),
                    "seen_energy_pretrain": s.seen_energy_pretrain,
                    "seen_energy_finetune": s.seen_energy_finetune,
                    "seen_energy_delta": s.seen_energy_delta,
                    "ood_energy_finetune": s.ood_energy_finetune,
                }
            )


def _replay_subtitle(points: list[ReplayRunPoint]) -> str:
    sizes = {p.replay_buffer_size for p in points if p.replay_buffer_size is not None}
    max_s = {p.replay_max_samples for p in points if p.replay_max_samples is not None}
    parts: list[str] = []
    if len(sizes) == 1:
        parts.append(f"buffer={next(iter(sizes))} samples")
    elif sizes:
        parts.append(f"buffer sizes={sorted(sizes)}")
    if len(max_s) == 1:
        parts.append(f"cap={next(iter(max_s))}")
    return "; ".join(parts)


def plot_seen_vs_replay(
    points: list[ReplayRunPoint],
    coupling: str,
    out_path: Path,
    sweep_name: str,
    *,
    log_y: bool = True,
) -> None:
    xlabel, x_key, key_fn = _x_axis_spec(points)
    subtitle = _replay_subtitle(points)
    n_rows = len(HOLDOUT_TAGS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.4 * n_rows), squeeze=False)
    title_extra = f"\n({subtitle})" if subtitle else ""
    y_note = "left: energy (log y)" if log_y else "y: linear"
    fig.suptitle(
        f"Seen (non-held-out) strata vs replay — {coupling}\n"
        f"({sweep_name}{title_extra}; two metrics — {y_note}, right: Δ linear)",
        fontsize=12,
    )

    metric_specs = (
        ("seen_energy_finetune", "Mean energy (seen strata)", "lower → better fit"),
        ("seen_energy_delta", "Δ energy (finetune − pretrain)", "0 → no forgetting"),
    )
    col_titles = (
        "Absolute finetune energy (seen strata)",
        "Forgetting: Δ energy (finetune − pretrain)",
    )

    for col, (metric_attr, ylabel, note) in enumerate(metric_specs):
        use_log_panel = log_y and metric_attr == "seen_energy_finetune"
        for row, hold in enumerate(HOLDOUT_TAGS):
            ax = axes[row, col]
            panel_ys: list[float] = []
            for vf, emb in ((True, True), (False, False)):
                if not _replay_config_active(vf, emb, hold):
                    continue
                sub = _subset(points, coupling=coupling, holdout_tag=hold, vf=vf, emb=emb)
                if not sub:
                    continue
                label, color, ls = _config_style(vf, emb)
                xs = [key_fn(p) for p in sub]
                ys = [getattr(p.scores, metric_attr) for p in sub]
                panel_ys.extend(ys)
                pre = sub[0].scores.seen_energy_pretrain
                ax.plot(xs, ys, marker="o", color=color, linestyle=ls, linewidth=2, label=label)
                if col == 0:
                    panel_ys.append(pre)
                    ax.axhline(pre, color=color, linestyle=":", alpha=0.5, linewidth=1)
            ylabel_ax = f"{ylabel} (log)" if use_log_panel else ylabel
            ax.set_title(f"{HOLDOUT_LABELS[hold]} — {ylabel_ax}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel_ax)
            ax.grid(True, alpha=0.3, which="both" if use_log_panel else "major")
            if metric_attr == "seen_energy_delta":
                ax.axhline(0.0, color="gray", linewidth=0.8)
            if use_log_panel:
                _maybe_log_yscale(ax, panel_ys, log_y=True)
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom", alpha=0.8)
            _legend_if_lines(ax)

    for col, title in enumerate(col_titles):
        axes[0, col].annotate(
            title,
            xy=(0.5, 1.12),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_stratum_vs_replay(
    points: list[ReplayRunPoint],
    coupling: str,
    sweep_dir: Path,
    out_path: Path,
    sweep_name: str,
    *,
    log_y: bool = True,
) -> None:
    """One line per seen (c,p) stratum: finetune energy vs replay x."""
    from experiments.sim.plotting.plot_sweep_metrics import is_held_out_stratum

    xlabel, _, key_fn = _x_axis_spec(points)
    configs = [
        ("hold_ctx", True, True),
        ("hold_pert", True, True),
        ("hold_combo", True, True),
        ("hold_combo", False, False),
    ]
    panels = [
        (hold, vf, emb)
        for hold, vf, emb in configs
        if _subset(points, coupling=coupling, holdout_tag=hold, vf=vf, emb=emb)
    ]
    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), squeeze=False)

    for ax, (hold, vf, emb) in zip(axes[:, 0], panels):
        sub = _subset(points, coupling=coupling, holdout_tag=hold, vf=vf, emb=emb)
        label_cfg, _color, ls = _config_style(vf, emb)
        panel_ys: list[float] = []
        ref = sub[0]
        kw_ref = {
            "holdout_mode": ref.scores.config.holdout_mode,
            "held_out_pert": ref.scores.config.held_out_pert,
            "held_out_context": ref.scores.config.held_out_context,
        }
        with _run_metrics_path(sweep_dir, ref.scores.config.run_dir).open() as f:
            ref_metrics = json.load(f)
        strata = sorted(
            {
                (int(r["c"]), int(r["p"]))
                for r in ref_metrics["finetune_ode_eval"]
                if not is_held_out_stratum(r, **kw_ref)
            }
        )

        for ci, pi in strata:
            xs, ys = [], []
            for p in sub:
                with _run_metrics_path(sweep_dir, p.scores.config.run_dir).open() as f:
                    m = json.load(f)
                kw_run = {
                    "holdout_mode": p.scores.config.holdout_mode,
                    "held_out_pert": p.scores.config.held_out_pert,
                    "held_out_context": p.scores.config.held_out_context,
                }
                row = next(
                    (
                        r
                        for r in m["finetune_ode_eval"]
                        if int(r["c"]) == ci
                        and int(r["p"]) == pi
                        and not is_held_out_stratum(r, **kw_run)
                    ),
                    None,
                )
                if row is None:
                    continue
                xs.append(key_fn(p))
                ys.append(float(row["energy"]))
                panel_ys.extend(ys)
            if xs:
                ax.plot(xs, ys, marker=".", linestyle=ls, alpha=0.85, label=f"c={ci}, p={pi}")

        ax.set_title(f"{HOLDOUT_LABELS[hold]} · {label_cfg}", fontsize=10)
        ax.set_xlabel(xlabel)
        ylabel = "energy (seen stratum, log)" if log_y else "energy (seen stratum)"
        ax.set_ylabel(ylabel)
        _maybe_log_yscale(ax, panel_ys, log_y=log_y)
        ax.grid(True, alpha=0.3, which="both" if log_y else "major")
        ax.legend(fontsize=6, ncol=4, loc="upper right")

    fig.suptitle(
        f"Per seen (c,p) energy vs replay — {coupling}\n({sweep_name})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seen_improvement_vs_replay(
    points: list[ReplayRunPoint],
    coupling: str,
    out_path: Path,
    sweep_name: str,
) -> None:
    """Seen improvement score: pretrain energy − finetune energy (positive = recovered fit)."""
    xlabel, _, key_fn = _x_axis_spec(points)
    fig, axes = plt.subplots(len(HOLDOUT_TAGS), 1, figsize=(9, 3.2 * len(HOLDOUT_TAGS)), squeeze=False)
    fig.suptitle(
        f"Seen-strata improvement (pretrain − finetune energy) vs replay — {coupling}\n"
        f"({sweep_name}; higher = closer to pretrain on seen data)",
        fontsize=12,
    )

    for row, hold in enumerate(HOLDOUT_TAGS):
        ax = axes[row, 0]
        for vf, emb in ((True, True), (False, False)):
            if not _replay_config_active(vf, emb, hold):
                continue
            sub = _subset(points, coupling=coupling, holdout_tag=hold, vf=vf, emb=emb)
            if not sub:
                continue
            label, color, ls = _config_style(vf, emb)
            xs = [key_fn(p) for p in sub]
            improve = [p.scores.seen_energy_pretrain - p.scores.seen_energy_finetune for p in sub]
            ax.plot(xs, improve, marker="o", color=color, linestyle=ls, linewidth=2, label=label)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(HOLDOUT_LABELS[hold], fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pretrain − finetune energy")
        ax.grid(True, alpha=0.3)
        _legend_if_lines(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="Replay sweep root (e.g. results_dir/sweep_ft_replay)")
    parser.add_argument("output_dir", type=Path, help="Directory for plots and CSV")
    parser.add_argument(
        "--no-per-stratum",
        action="store_true",
        help="Skip per-(c,p) line plots (faster).",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Linear y-axis (default: log scale for positive energy panels).",
    )
    args = parser.parse_args()
    log_y = not args.linear_y

    sweep_dir = args.sweep_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not sweep_dir.is_dir():
        raise FileNotFoundError(sweep_dir)

    points = load_replay_points(sweep_dir)
    sweep_name = sweep_dir.name
    _, x_key, _ = _x_axis_spec(points)
    print(f"Loaded {len(points)} replay runs from {sweep_dir} (x-axis: {x_key}, y-scale: {'log' if log_y else 'linear'})")

    write_summary_csv(points, output_dir / "replay_seen_summary.csv", x_key)

    for coupling in COUPLINGS:
        sub = [p for p in points if p.scores.config.coupling == coupling]
        if not sub:
            print(f"  {coupling}: no runs")
            continue
        plot_seen_vs_replay(
            sub,
            coupling,
            output_dir / f"seen_vs_replay_{coupling}.png",
            sweep_name,
            log_y=log_y,
        )
        plot_seen_improvement_vs_replay(
            sub, coupling, output_dir / f"seen_improvement_vs_replay_{coupling}.png", sweep_name
        )
        if not args.no_per_stratum:
            plot_per_stratum_vs_replay(
                sub,
                coupling,
                sweep_dir,
                output_dir / f"seen_per_stratum_vs_replay_{coupling}.png",
                sweep_name,
                log_y=log_y,
            )
        print(f"  {coupling}: {len(sub)} runs → plots")

    print(f"Wrote plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()
