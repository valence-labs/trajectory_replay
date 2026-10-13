#!/usr/bin/env python3
"""Stage A λ curves: fit / forget (and Pareto) for traj vs gen (+ naive).

Aggregates across seeds with median + IQR bands when multiple seeds exist.

Example:
  uv run python experiments/sim_conditional/stage_a_leaveout/plot_lambda_fit_forget.py
  uv run python experiments/sim_conditional/stage_a_leaveout/plot_lambda_fit_forget.py --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_conditional.common import REPO_ROOT, stage_a_runs_root

_RUN_RE = re.compile(
    r"^ft_(?P<strategy>naive|generative|trajectory)"
    r"(?:_lam(?P<lam>[0-9.]+))?_e(?P<epochs>\d+)_seed(?P<seed>\d+)$"
)

METHOD_STYLES = {
    "trajectory": ("Trajectory replay", "#457b9d", "o"),
    "generative": ("Generative replay", "#c1121f", "s"),
    "naive": ("Naive FT", "#6c757d", "D"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(
        "--metric",
        choices=("energy", "mean_l2"),
        default="energy",
        help="Base metric for fit / forget.",
    )
    p.add_argument(
        "--forget",
        choices=("absolute", "delta"),
        default="delta",
        help="Plot forget_energy_mean or delta_forget_* vs pretrained.",
    )
    p.add_argument("--seed", type=int, default=None, help="If set, only this seed.")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _fit_key(metric: str) -> str:
    return "fit_energy" if metric == "energy" else "fit_mean_l2"


def _forget_key(metric: str, forget: str) -> str:
    if forget == "delta":
        return (
            "delta_forget_energy_mean"
            if metric == "energy"
            else "delta_forget_mean_l2_mean"
        )
    return "forget_energy_mean" if metric == "energy" else "forget_mean_l2_mean"


def collect_rows(runs_root: Path, *, seed: int | None) -> list[dict]:
    rows: list[dict] = []
    ft_root = runs_root / "finetune"
    if not ft_root.is_dir():
        return rows
    for run_dir in sorted(ft_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _RUN_RE.match(run_dir.name)
        if not m:
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = payload.get("config") or {}
        strat = m.group("strategy")
        seed_i = int(m.group("seed"))
        if seed is not None and seed_i != seed:
            continue
        lam_s = m.group("lam")
        lam = float(lam_s) if lam_s is not None else cfg.get("lambda_mix")
        rows.append(
            {
                "run": run_dir.name,
                "strategy": strat,
                "lambda": None if strat == "naive" else float(lam),
                "seed": seed_i,
                "epochs": int(m.group("epochs")),
                "val": payload["val"],
                "test": payload["test"],
            }
        )
    return rows


def _agg_stats(vals: list[float]) -> tuple[float, float, float, int]:
    arr = np.asarray(vals, dtype=np.float64)
    n = int(arr.size)
    mid = float(np.median(arr))
    if n >= 2:
        lo = float(np.percentile(arr, 25))
        hi = float(np.percentile(arr, 75))
    else:
        lo = hi = mid
    return mid, lo, hi, n


def aggregate_split(
    rows: list[dict], split: str, metric: str, forget: str
) -> dict[str, list[tuple]]:
    """strategy -> list of (lam|None, fit_mid, fit_lo, fit_hi, forg_mid, forg_lo, forg_hi, n)."""
    fit_k = _fit_key(metric)
    forg_k = _forget_key(metric, forget)
    buckets: dict[tuple[str, float | None], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        block = r[split]
        key = (r["strategy"], r["lambda"])
        buckets[key].append((float(block[fit_k]), float(block[forg_k])))

    by_strat: dict[str, list[tuple]] = {
        "trajectory": [],
        "generative": [],
        "naive": [],
    }
    for (strat, lam), pairs in buckets.items():
        fits = [p[0] for p in pairs]
        forgs = [p[1] for p in pairs]
        f_mid, f_lo, f_hi, n = _agg_stats(fits)
        g_mid, g_lo, g_hi, _ = _agg_stats(forgs)
        by_strat[strat].append((lam, f_mid, f_lo, f_hi, g_mid, g_lo, g_hi, n))
    for k in by_strat:
        by_strat[k].sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else -1))
    return by_strat


def main() -> None:
    args = parse_args()
    runs_root = stage_a_runs_root(REPO_ROOT, dim=args.dim)
    rows = collect_rows(runs_root, seed=args.seed)
    if not rows:
        raise SystemExit(f"No FT metrics under {runs_root / 'finetune'}")

    out_dir = args.out_dir or (runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split
    metric = args.metric
    forget = args.forget
    by = aggregate_split(rows, split, metric, forget)

    fit_label = r"Fit energy $\downarrow$" if metric == "energy" else r"Fit mean-L2 $\downarrow$"
    if forget == "delta":
        forg_label = (
            r"$\Delta$Forget energy $\downarrow$"
            if metric == "energy"
            else r"$\Delta$Forget mean-L2 $\downarrow$"
        )
    else:
        forg_label = (
            r"Forget energy $\downarrow$"
            if metric == "energy"
            else r"Forget mean-L2 $\downarrow$"
        )

    n_seeds = sorted({r["seed"] for r in rows})
    seed_note = f"median+IQR (n≤{len(n_seeds)} seeds)" if len(n_seeds) > 1 else "seed0"

    # --- curve panel: fit & forget vs λ ---
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)
    for strat in ("trajectory", "generative"):
        pts = [t for t in by[strat] if t[0] is not None]
        if not pts:
            continue
        label, color, marker = METHOD_STYLES[strat]
        lams = np.asarray([t[0] for t in pts], dtype=np.float64)
        fits = np.asarray([t[1] for t in pts], dtype=np.float64)
        fit_lo = np.asarray([t[2] for t in pts], dtype=np.float64)
        fit_hi = np.asarray([t[3] for t in pts], dtype=np.float64)
        forgs = np.asarray([t[4] for t in pts], dtype=np.float64)
        forg_lo = np.asarray([t[5] for t in pts], dtype=np.float64)
        forg_hi = np.asarray([t[6] for t in pts], dtype=np.float64)
        axes[0].fill_between(lams, fit_lo, fit_hi, color=color, alpha=0.18, linewidth=0)
        axes[1].fill_between(lams, forg_lo, forg_hi, color=color, alpha=0.18, linewidth=0)
        axes[0].plot(lams, fits, marker=marker, color=color, label=label, lw=1.8)
        axes[1].plot(lams, forgs, marker=marker, color=color, label=label, lw=1.8)

    if by["naive"]:
        _, n_fit, n_flo, n_fhi, n_forg, n_glo, n_ghi, n_n = by["naive"][0]
        label, color, marker = METHOD_STYLES["naive"]
        axes[0].axhline(n_fit, color=color, ls="--", lw=1.4, label=label)
        if n_n >= 2:
            axes[0].axhspan(n_flo, n_fhi, color=color, alpha=0.12, linewidth=0)
        axes[1].text(
            0.02,
            0.98,
            rf"naive $\Delta$Forget$\approx${n_forg:.2f}",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color=color,
        )

    axes[0].set_xlabel(r"$\lambda$")
    axes[1].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(fit_label)
    axes[1].set_ylabel(forg_label)
    axes[0].set_title(rf"Stage A fit vs $\lambda$ ({split})")
    axes[1].set_title(rf"Stage A forget vs $\lambda$ ({split})")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    stem = f"lambda_fit_forget_{metric}_{forget}_{split}_dim{args.dim}"
    png1 = out_dir / f"{stem}_curves.png"
    fig.savefig(png1, dpi=160)
    plt.close(fig)

    def _draw_pareto(ax, *, include_naive: bool) -> None:
        for strat in ("trajectory", "generative"):
            pts = [t for t in by[strat] if t[0] is not None]
            if not pts:
                continue
            label, color, marker = METHOD_STYLES[strat]
            fits = [t[1] for t in pts]
            forgs = [t[4] for t in pts]
            lams = [t[0] for t in pts]
            ax.plot(fits, forgs, marker=marker, color=color, label=label, lw=1.6)
            for x, y, lam in zip(fits, forgs, lams):
                ax.annotate(
                    f"{lam:g}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )
        if include_naive and by["naive"]:
            _, n_fit, _, _, n_forg, _, _, _ = by["naive"][0]
            label, color, marker = METHOD_STYLES["naive"]
            ax.scatter(
                [n_fit], [n_forg], marker=marker, color=color, s=70, label=label, zorder=5
            )
            ax.annotate(
                "naive",
                (n_fit, n_forg),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=8,
                color=color,
            )
        ax.set_xlabel(fit_label)
        ax.set_ylabel(forg_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_pareto(ax, include_naive=True)
    ax.set_title(rf"Stage A Pareto ({split}, labels = $\lambda$)")
    fig.tight_layout()
    png2 = out_dir / f"{stem}_pareto.png"
    fig.savefig(png2, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_pareto(ax, include_naive=False)
    ax.set_title(rf"Stage A Pareto zoom ({split}, no naive)")
    fig.tight_layout()
    png2b = out_dir / f"{stem}_pareto_zoom.png"
    fig.savefig(png2b, dpi=160)
    plt.close(fig)

    csv_path = out_dir / f"{stem}.csv"
    fit_k = _fit_key(metric)
    forg_k = _forget_key(metric, forget)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "strategy",
                "lambda",
                "seed",
                "split",
                "fit",
                "forget",
                "fit_key",
                "forget_key",
                "run",
            ]
        )
        for r in rows:
            block = r[split]
            w.writerow(
                [
                    r["strategy"],
                    "" if r["lambda"] is None else r["lambda"],
                    r["seed"],
                    split,
                    block[fit_k],
                    block[forg_k],
                    fit_k,
                    forg_k,
                    r["run"],
                ]
            )

    print(f"Wrote {png1}")
    print(f"Wrote {png2}")
    print(f"Wrote {png2b}")
    print(f"Wrote {csv_path}")
    print(f"n_runs={len(rows)} seeds={n_seeds} ({seed_note})")


if __name__ == "__main__":
    main()
