#!/usr/bin/env python3
"""Post-hoc cumulative ref-NFE vs mixture quality (traj vs gen).

NFE is *estimated* from the fixed Euler protocol (not logged at train time),
unless ``eval_curve`` already stores ``cum_ref_nfe`` (preferred):

  steps/epoch ≈ floor(n_train / batch) = 100000 // 256 = 390  (drop_last)
  gen:  N_ref/step = 1 / ode_step_size = 20
  traj: N_ref/step ≈ E[t]/ode_step_size + 1 = 0.5/0.05 + 1 = 11

  cumulative_ref_nfe = epochs * steps_per_epoch * N_ref_per_step

Uses finished ``metrics.json`` only (dense ``eval_curve`` when present).

Example:
  uv run python experiments/sim_mixture/plot_nfe_vs_quality.py --dims 64 128 256
  uv run python experiments/sim_mixture/plot_nfe_vs_quality.py --dims 64 128 256 --metric energy_mixture_active --roll-window 5
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

from experiments.sim_mixture.common import REPO_ROOT

# Protocol defaults matching the 100k-sample sim_mixture FT jobs.
N_TRAIN = 100_000
BATCH_SIZE = 256
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE  # 390 with drop_last
ODE_STEP_SIZE = 0.05
NFE_PER_STEP = {
    "generative": int(round(1.0 / ODE_STEP_SIZE)),  # 20
    "trajectory": int(round(0.5 / ODE_STEP_SIZE)) + 1,  # 11
}

METHOD_STYLES = {
    "trajectory": ("Trajectory replay", "#457b9d", "o"),
    "generative": ("Generative replay", "#c1121f", "s"),
}

# Slurm-style run dirs: ft_{strategy}_lam0.5_e{epochs}_seed{seed}
# Skip lr / kappa tags so we compare the main λ=0.5, lr=1e-3, κ≡1 line.
_RUN_RE = re.compile(
    r"^ft_(?P<strategy>trajectory|generative)_lam0\.5_e(?P<epochs>\d+)_seed(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Dimensions to include (default: 64 128 256).",
    )
    p.add_argument(
        "--run-filter",
        default="e513_eval10",
        help="Substring that FT run dirs must contain (default: e513_eval10). Empty = any _eval run.",
    )
    p.add_argument(
        "--metric",
        choices=("energy_mixture_active", "energy_mixture", "pi0_abs_error"),
        default="energy_mixture_active",
    )
    p.add_argument("--lambda-mix", type=float, default=0.5)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: results_dir/sim_mixture/figures/nfe_vs_{metric}.png)",
    )
    p.add_argument(
        "--center",
        choices=("median", "mean"),
        default="median",
        help="Seed aggregation: median+IQR band (default) or mean±std band.",
    )
    p.add_argument(
        "--roll-window",
        type=int,
        default=1,
        help="Trailing rolling mean window over eval points (1 = no smoothing).",
    )
    p.add_argument(
        "--fig-width",
        type=float,
        default=2.4,
        help="Figure width in inches (panels are stacked, so this is one column slot).",
    )
    p.add_argument(
        "--panel-height",
        type=float,
        default=1.35,
        help="Height in inches of each stacked dimension panel.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV of per-seed points (default: next to PNG).",
    )
    return p.parse_args()


def _cum_ref_nfe(strategy: str, epochs: int) -> float:
    return float(epochs * STEPS_PER_EPOCH * NFE_PER_STEP[strategy])


def _metric_value(eval_block: dict, metric: str, lambda_mix: float) -> float | None:
    if metric == "pi0_abs_error":
        if "pi0_abs_error" in eval_block:
            return float(eval_block["pi0_abs_error"])
        if "pi0_hat" in eval_block:
            return abs(float(eval_block["pi0_hat"]) - float(lambda_mix))
        return None
    if metric not in eval_block:
        return None
    return float(eval_block[metric])


def collect_eval_curve_rows(
    *,
    dims: list[int],
    metric: str,
    lambda_mix: float,
    run_filter: str = "e513_eval10",
) -> list[dict]:
    """Rows from densified ``metrics.json`` ``eval_curve`` (``--eval-every`` runs)."""
    rows: list[dict] = []
    for dim in dims:
        ft_root = REPO_ROOT / "results_dir" / "sim_mixture" / f"dim{dim}" / "finetune"
        if not ft_root.is_dir():
            continue
        for run_dir in sorted(ft_root.iterdir()):
            if not run_dir.is_dir() or "_eval" not in run_dir.name:
                continue
            if run_filter and run_filter not in run_dir.name:
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            payload = json.loads(metrics_path.read_text())
            cfg = payload.get("config") or {}
            strategy = cfg.get("strategy")
            if strategy not in METHOD_STYLES:
                continue
            seed = int(cfg.get("train_seed", 0))
            for pt in payload.get("eval_curve") or []:
                y = _metric_value(pt, metric, lambda_mix)
                if y is None or "cum_ref_nfe" not in pt:
                    continue
                rows.append(
                    {
                        "dim": dim,
                        "strategy": strategy,
                        "epochs": int(pt["epoch"]),
                        "seed": seed,
                        "steps": float(pt.get("steps", 0.0)),
                        "nfe_per_step": float(
                            pt.get("nfe_per_step", NFE_PER_STEP[strategy])
                        ),
                        "cum_ref_nfe": float(pt["cum_ref_nfe"]),
                        "metric": metric,
                        "value": y,
                        "pi0_hat": pt.get("pi0_hat"),
                        "run": run_dir.name,
                    }
                )
    return rows


def collect_rows(
    *,
    dims: list[int],
    metric: str,
    lambda_mix: float,
    run_filter: str = "e513_eval10",
) -> list[dict]:
    curve_rows = collect_eval_curve_rows(
        dims=dims, metric=metric, lambda_mix=lambda_mix, run_filter=run_filter
    )
    if curve_rows:
        return curve_rows
    rows: list[dict] = []
    for dim in dims:
        ft_root = REPO_ROOT / "results_dir" / "sim_mixture" / f"dim{dim}" / "finetune"
        if not ft_root.is_dir():
            continue
        for run_dir in sorted(ft_root.iterdir()):
            if not run_dir.is_dir():
                continue
            m = _RUN_RE.match(run_dir.name)
            if m is None:
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            payload = json.loads(metrics_path.read_text())
            cfg = payload.get("config") or {}
            ev = payload.get("eval") or {}
            lr = cfg.get("lr")
            if lr is not None and abs(float(lr) - 1e-4) < 1e-9:
                continue
            if m.group("strategy") == "trajectory":
                kappa = cfg.get("traj_kappa")
                if kappa not in (None, "none"):
                    continue
            y = _metric_value(ev, metric, lambda_mix)
            if y is None:
                continue
            strategy = m.group("strategy")
            epochs = int(m.group("epochs"))
            seed = int(m.group("seed"))
            rows.append(
                {
                    "dim": dim,
                    "strategy": strategy,
                    "epochs": epochs,
                    "seed": seed,
                    "steps": epochs * STEPS_PER_EPOCH,
                    "nfe_per_step": NFE_PER_STEP[strategy],
                    "cum_ref_nfe": _cum_ref_nfe(strategy, epochs),
                    "metric": metric,
                    "value": y,
                    "pi0_hat": ev.get("pi0_hat"),
                    "run": run_dir.name,
                }
            )
    return rows


def _aggregate(
    rows: list[dict],
    *,
    center: str = "median",
) -> dict[tuple[int, str, int], tuple[float, float, float, float, int]]:
    """(dim, strategy, epochs) -> (cum_nfe, center_y, lo, hi, n).

    ``center='median'``: lo/hi = 25/75 percentiles (IQR band).
    ``center='mean'``: lo/hi = mean ± std (population).
    """
    buckets: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    nfe_of: dict[tuple[int, str, int], float] = {}
    for r in rows:
        key = (r["dim"], r["strategy"], r["epochs"])
        buckets[key].append(float(r["value"]))
        nfe_of[key] = float(r["cum_ref_nfe"])
    out: dict[tuple[int, str, int], tuple[float, float, float, float, int]] = {}
    for key, vals in buckets.items():
        arr = np.asarray(vals, dtype=np.float64)
        n = int(arr.size)
        if center == "median":
            mid = float(np.median(arr))
            if n >= 2:
                lo = float(np.percentile(arr, 25))
                hi = float(np.percentile(arr, 75))
            else:
                lo = hi = mid
        elif center == "mean":
            mid = float(arr.mean())
            std = float(arr.std(ddof=0))
            lo, hi = mid - std, mid + std
        else:
            raise ValueError(f"Unknown center={center!r}; expected 'median' or 'mean'")
        out[key] = (nfe_of[key], mid, lo, hi, n)
    return out


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean (causal): at i, mean of y[max(0,i-w+1):i+1]."""
    if window <= 1:
        return np.asarray(y, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    csum = np.cumsum(y)
    for i in range(len(y)):
        j0 = max(0, i - window + 1)
        total = csum[i] - (csum[j0 - 1] if j0 > 0 else 0.0)
        out[i] = total / float(i - j0 + 1)
    return out


def plot_rows(
    rows: list[dict],
    *,
    metric: str,
    out: Path,
    center: str = "median",
    roll_window: int = 1,
    fig_width: float = 2.4,
    panel_height: float = 1.35,
) -> None:
    # Match the compact, stacked style used by the existing d=128/256 figure.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.8,
        }
    )
    dims = sorted({r["dim"] for r in rows}, reverse=True)
    agg = _aggregate(rows, center=center)
    n_panels = len(dims)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(2.4, 1.35 * n_panels),
        sharex=True,
        squeeze=False,
    )
    ylabel = {
        "energy_mixture_active": r"Energy $\downarrow$",
        "energy_mixture": r"Energy $\downarrow$",
        "pi0_abs_error": r"$|\hat{\pi}_0 - \lambda| \downarrow$",
    }[metric]

    for ax, dim in zip(axes[:, 0], dims):
        for strategy, (label, color, marker) in METHOD_STYLES.items():
            pts = [
                (epochs, agg[(dim, strategy, epochs)])
                for epochs in sorted({e for (d, s, e) in agg if d == dim and s == strategy})
            ]
            if not pts:
                continue
            xs = np.asarray([p[1][0] for p in pts], dtype=np.float64)
            ys = np.asarray([p[1][1] for p in pts], dtype=np.float64)
            lo = np.asarray([p[1][2] for p in pts], dtype=np.float64)
            hi = np.asarray([p[1][3] for p in pts], dtype=np.float64)
            if roll_window > 1:
                ys = _rolling_mean(ys, roll_window)
                lo = _rolling_mean(lo, roll_window)
                hi = _rolling_mean(hi, roll_window)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
            markevery = max(1, len(pts) // 6) if len(pts) > 8 else 1
            ax.plot(
                xs,
                ys,
                label=label,
                color=color,
                marker=marker,
                linewidth=1.3,
                markersize=3.4 if len(pts) > 8 else 5,
                markevery=markevery,
                alpha=0.95,
            )
            if len(pts) <= 8:
                for (epochs, (_x, _y, _lo, _hi, n)), x, y in zip(pts, xs, ys):
                    ax.annotate(
                        f"e{epochs}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                        color=color,
                        alpha=0.85,
                    )
        ax.text(
            0.975,
            0.93,
            rf"$d={dim}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.25",
        )
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel(r"Cumulative teacher NFEs")
    axes[0, 0].legend(
        frameon=False,
        loc="lower left",
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.0,
        labelspacing=0.3,
    )

    fig.tight_layout(pad=0.3, h_pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dim",
        "strategy",
        "epochs",
        "seed",
        "steps",
        "nfe_per_step",
        "cum_ref_nfe",
        "metric",
        "value",
        "pi0_hat",
        "run",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    args = parse_args()
    if args.roll_window < 1:
        raise SystemExit("--roll-window must be >= 1")
    rows = collect_rows(
        dims=args.dims,
        metric=args.metric,
        lambda_mix=args.lambda_mix,
        run_filter=args.run_filter,
    )
    if not rows:
        raise SystemExit(
            "No matching runs found. Expected dirs like "
            "ft_trajectory_lam0.5_e513_eval10_seed0."
        )

    fig_dir = REPO_ROOT / "results_dir" / "sim_mixture" / "figures"
    dim_tag = "-".join(map(str, args.dims))
    roll_tag = f"_roll{args.roll_window}" if args.roll_window > 1 else ""
    # Distinguish 100k protocol figures from older 10k ones.
    out = args.out or fig_dir / f"nfe_vs_{args.metric}_dims{dim_tag}{roll_tag}_n100k.png"
    csv_path = args.csv or out.with_suffix(".csv")

    plot_rows(
        rows,
        metric=args.metric,
        out=out,
        center=args.center,
        roll_window=args.roll_window,
        fig_width=args.fig_width,
        panel_height=args.panel_height,
    )
    write_csv(rows, csv_path)

    print(f"NFE/step: generative={NFE_PER_STEP['generative']}, trajectory≈{NFE_PER_STEP['trajectory']}")
    print(f"steps/epoch={STEPS_PER_EPOCH}  center={args.center}  roll_window={args.roll_window}")
    print(f"Collected {len(rows)} seed-runs → {out}")
    print(f"CSV → {csv_path}")
    agg = _aggregate(rows, center=args.center)
    label = "median [q25,q75]" if args.center == "median" else "mean±std"
    print(f"\n{label} (pre-roll):")
    for key in sorted(agg):
        dim, strategy, epochs = key
        nfe, mid, lo, hi, n = agg[key]
        print(
            f"  dim{dim} {strategy:11s} e{epochs:<4d}  "
            f"NFE={nfe:.3e}  {args.metric}={mid:.4f} [{lo:.4f},{hi:.4f}]  (n={n})"
        )


if __name__ == "__main__":
    main()
