#!/usr/bin/env python3
"""Aggregate combo hold-out comparison sweep metrics and plot energy + overlays."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COUPLINGS = ("additive", "rotated", "hadamard")
METHODS = (
    ("noise_to_data", "noise-to-data"),
    ("data_to_data_with_identity", "d2d (+ identity)"),
    ("data_to_data_no_identity", "d2d (no identity)"),
)
RUN_DIR_RE = re.compile(
    r"^(?P<regime>\w+)_coup_(?P<coupling>additive|rotated|hadamard)"
    r"(?:_hc(?P<holdout_c>\d+))?"
    r"(?:_cf(?P<cf_tag>[\dp]+))?$"
)


def _cf_tag_to_float(tag: str) -> float:
    return float(tag.replace("p", "."))
REGIME_ORDER = ("simple", "medium", "hard", "1", "2", "easy", "hard")
REGIME_LABELS = {
    "simple": "simple",
    "medium": "medium",
    "hard": "hard",
    "1": "easy",
    "2": "hard",
    "easy": "easy",
}


@dataclass
class SweepRun:
    run_dir: Path
    regime: str
    regime_label: str
    coupling: str
    held_out_context: int
    control_subsample_frac: float | None
    energies: dict[str, float]
    overlay_path: Path | None


def _heldout_energy(metrics: dict, key: str) -> float:
    block = metrics.get(key)
    if not isinstance(block, dict):
        return float("nan")
    hm = block.get("heldout_metrics")
    if not isinstance(hm, dict):
        return float("nan")
    return float(hm.get("energy", float("nan")))


def parse_run_dir(run_dir: Path) -> SweepRun | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("experiment") != "combo_holdout_compare":
        return None

    regime = str(metrics.get("sweep_regime", ""))
    coupling = None
    held_c = metrics.get("held_out_context")
    args = metrics.get("args") or {}
    if isinstance(args, dict):
        coupling = args.get("coupling")
        if held_c is None:
            held_c = args.get("held_out_context")

    m = RUN_DIR_RE.match(run_dir.name)
    if m:
        regime = regime or m.group("regime")
        coupling = coupling or m.group("coupling")
        if held_c is None and m.group("holdout_c") is not None:
            held_c = int(m.group("holdout_c"))

    if coupling not in COUPLINGS:
        return None
    if held_c is None:
        held_c = -1
    held_c = int(held_c)

    ctrl_frac: float | None = None
    pool = metrics.get("control_pool")
    if isinstance(pool, dict) and pool.get("control_subsampled"):
        ctrl_frac = float(pool.get("control_subsample_frac", float("nan")))
    if m and m.group("cf_tag"):
        ctrl_frac = _cf_tag_to_float(m.group("cf_tag"))
    if ctrl_frac is not None and not np.isfinite(ctrl_frac):
        ctrl_frac = None

    energies: dict[str, float] = {}
    for key, _ in METHODS:
        energies[key] = _heldout_energy(metrics, key)

    overlay = run_dir / "heldout_compare.png"
    if not overlay.is_file():
        plots = metrics.get("plot_files") or []
        for p in plots:
            if str(p).endswith("heldout_compare.png"):
                overlay = Path(p)
                break
        else:
            overlay = None

    regime_label = REGIME_LABELS.get(regime, regime)
    return SweepRun(
        run_dir=run_dir,
        regime=regime,
        regime_label=regime_label,
        coupling=coupling,
        held_out_context=held_c,
        control_subsample_frac=ctrl_frac,
        energies=energies,
        overlay_path=overlay if overlay is not None and Path(overlay).is_file() else None,
    )


def collect_runs(sweep_dir: Path) -> list[SweepRun]:
    runs: list[SweepRun] = []
    if not sweep_dir.is_dir():
        return runs
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        run = parse_run_dir(child)
        if run is not None:
            runs.append(run)
    return runs


def _regimes_ordered(runs: list[SweepRun]) -> list[str]:
    seen: list[str] = []
    for r in runs:
        if r.regime not in seen:
            seen.append(r.regime)
    def _key(r: str) -> tuple[int, str]:
        if r in REGIME_ORDER:
            return (REGIME_ORDER.index(r), r)
        return (len(REGIME_ORDER), r)

    return sorted(seen, key=_key)


def write_summary_csv(runs: list[SweepRun], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "regime",
        "regime_label",
        "coupling",
        "held_out_context",
        "control_subsample_frac",
    ] + [k for k, _ in METHODS]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            row = {
                "run_dir": str(r.run_dir),
                "regime": r.regime,
                "regime_label": r.regime_label,
                "coupling": r.coupling,
                "held_out_context": r.held_out_context,
                "control_subsample_frac": r.control_subsample_frac,
            }
            for key, _ in METHODS:
                row[key] = r.energies.get(key, float("nan"))
            w.writerow(row)


def plot_energy_by_holdout_context(runs: list[SweepRun], out_path: Path) -> None:
    """Lines over held-out context c, one panel per regime × method (mean over couplings)."""
    regimes = _regimes_ordered(runs)
    fig, axes = plt.subplots(len(regimes), len(METHODS), figsize=(4 * len(METHODS), 3.5 * len(regimes)), squeeze=False)
    for i, regime in enumerate(regimes):
        for j, (key, label) in enumerate(METHODS):
            ax = axes[i, j]
            for coupling in COUPLINGS:
                sub = [r for r in runs if r.regime == regime and r.coupling == coupling]
                if not sub:
                    continue
                sub = sorted(sub, key=lambda r: r.held_out_context)
                xs = [r.held_out_context for r in sub]
                ys = [r.energies.get(key, float("nan")) for r in sub]
                ax.plot(xs, ys, marker="o", label=coupling)
            ax.set_title(f"{REGIME_LABELS.get(regime, regime)} — {label}")
            ax.set_xlabel("held-out context c")
            ax.set_ylabel("energy")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Held-out energy vs held-out context (by coupling)", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_energy_mean_bars(runs: list[SweepRun], out_path: Path) -> None:
    """Mean energy over held-out contexts, grouped by coupling × method, per regime."""
    regimes = _regimes_ordered(runs)
    n_reg = len(regimes)
    fig, axes_grid = plt.subplots(1, n_reg, figsize=(5 * n_reg, 4.5), sharey=True, squeeze=False)
    axes = list(np.ravel(axes_grid))
    x = np.arange(len(COUPLINGS))
    width = 0.25
    colors = ("tab:green", "tab:orange", "tab:red")

    for ax, regime in zip(axes, regimes):
        for j, (key, label) in enumerate(METHODS):
            vals = []
            for coup in COUPLINGS:
                sub = [r for r in runs if r.regime == regime and r.coupling == coup]
                e = [r.energies.get(key, float("nan")) for r in sub]
                vals.append(float(np.nanmean(e)) if e else float("nan"))
            offset = (j - 1) * width
            ax.bar(x + offset, vals, width, label=label, color=colors[j], edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(COUPLINGS)
        ax.set_title(f"{REGIME_LABELS.get(regime, regime)} (mean over hold-out c)")
        ax.set_ylabel("energy distance")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out (c,p) fit — mean over held-out contexts", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _energy_for_key(run: SweepRun, key: str) -> float:
    return float(run.energies.get(key, float("nan")))


def _active_frac_methods(panel_runs: list[SweepRun]) -> list[tuple[str, str]]:
    """Method keys that have at least one finite held-out energy in this panel."""
    active: list[tuple[str, str]] = []
    for key, label in METHODS:
        if any(np.isfinite(_energy_for_key(r, key)) for r in panel_runs):
            active.append((key, label))
    if not any(k == "data_to_data_with_identity" for k, _ in active):
        if any(np.isfinite(_energy_for_key(r, "data_to_data")) for r in panel_runs):
            active.append(("data_to_data", "data-to-data"))
    return active


def _energies_per_frac(
    panel_runs: list[SweepRun],
    fracs: list[float],
    key: str,
) -> list[np.ndarray]:
    """One array of held-out energies per control fraction (one value per held-out context)."""
    out: list[np.ndarray] = []
    for frac in fracs:
        vals = [
            _energy_for_key(r, key)
            for r in panel_runs
            if r.control_subsample_frac == frac and np.isfinite(_energy_for_key(r, key))
        ]
        out.append(np.asarray(vals, dtype=float))
    return out


def plot_energy_vs_control_frac(runs: list[SweepRun], out_path: Path) -> None:
    """Boxplots of held-out energy vs control fraction (one point per held-out context)."""
    sub = [r for r in runs if r.control_subsample_frac is not None]
    if not sub:
        return
    fracs = sorted({r.control_subsample_frac for r in sub})
    if len(fracs) < 2:
        return

    regimes = _regimes_ordered(sub)
    method_colors = {
        "noise_to_data": "tab:green",
        "data_to_data_with_identity": "tab:orange",
        "data_to_data_no_identity": "tab:red",
        "data_to_data": "tab:orange",
    }

    fig, axes_grid = plt.subplots(
        len(regimes), len(COUPLINGS), figsize=(4.8 * len(COUPLINGS), 3.8 * len(regimes)), squeeze=False
    )
    frac_x = np.arange(len(fracs), dtype=float)

    for i, regime in enumerate(regimes):
        for j, coupling in enumerate(COUPLINGS):
            ax = axes_grid[i, j]
            panel = [r for r in sub if r.regime == regime and r.coupling == coupling]
            methods = _active_frac_methods(panel)
            if not methods:
                continue

            n_m = len(methods)
            group_w = min(0.22, 0.7 / max(n_m, 1))
            box_w = group_w * 0.85

            for mi, (key, label) in enumerate(methods):
                offset = (mi - (n_m - 1) / 2.0) * group_w
                positions = [xf + offset for xf in frac_x]
                data = _energies_per_frac(panel, fracs, key)
                if not any(d.size > 0 for d in data):
                    continue
                color = method_colors.get(key, "tab:blue")
                bp = ax.boxplot(
                    data,
                    positions=positions,
                    widths=box_w,
                    patch_artist=True,
                    showfliers=True,
                    medianprops={"color": "black", "linewidth": 1.2},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                    patch.set_edgecolor(color)
                for el in ("whiskers", "caps"):
                    for line in bp[el]:
                        line.set_color(color)
                # jittered points = individual held-out contexts
                for pos, vals in zip(positions, data):
                    if vals.size == 0:
                        continue
                    jitter = (np.random.default_rng(0).random(vals.size) - 0.5) * box_w * 0.35
                    ax.scatter(
                        np.full(vals.size, pos) + jitter,
                        vals,
                        s=12,
                        c=color,
                        alpha=0.75,
                        edgecolors="white",
                        linewidths=0.3,
                        zorder=3,
                    )

            ax.set_xticks(frac_x)
            ax.set_xticklabels([str(f) for f in fracs])
            ax.set_xlabel("control fraction")
            ax.set_ylabel("energy distance")
            ax.set_title(f"{REGIME_LABELS.get(regime, regime)} — {coupling}")
            ax.grid(axis="y", alpha=0.3)
            if j == len(COUPLINGS) - 1:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors.get(k, "tab:blue"),
                        lw=10,
                        alpha=0.55,
                        label=lab,
                    )
                    for k, lab in methods
                ]
                ax.legend(handles=handles, fontsize=11, loc="upper left", framealpha=0.92)

    fig.suptitle(
        "Held-out energy vs control fraction (box = held-out contexts c; points = each split)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def copy_overlay_gallery(runs: list[SweepRun], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        if r.overlay_path is None:
            continue
        dest = out_dir / f"{r.regime}_coup_{r.coupling}_hc{r.held_out_context}.png"
        shutil.copy2(r.overlay_path, dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep_dir", type=Path, help="e.g. results_dir/sweep_holdout_cmp")
    p.add_argument("output_dir", type=Path, help="Where to write plots and summary.csv")
    args = p.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    out_dir = args.output_dir.resolve()
    runs = collect_runs(sweep_dir)
    if not runs:
        raise SystemExit(f"No completed runs found under {sweep_dir}")

    print(f"Found {len(runs)} runs under {sweep_dir}")
    write_summary_csv(runs, out_dir / "summary.csv")
    plot_energy_mean_bars(runs, out_dir / "energy_mean_by_regime.png")
    plot_energy_by_holdout_context(runs, out_dir / "energy_vs_holdout_context.png")
    ctrl_plot = out_dir / "energy_vs_control_frac.png"
    plot_energy_vs_control_frac(runs, ctrl_plot)
    if ctrl_plot.is_file():
        print(f"Wrote {ctrl_plot}")
    copy_overlay_gallery(runs, out_dir / "overlays")
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
