#!/usr/bin/env python3
"""Stage B λ curves: val fit / Δforget + heldout transfer (and Pareto).

1×3 panel:
  left   — fit energy on FT cells (val)
  middle — Δforget energy on pretrain cells (val)
  right  — heldout energy on (c*, p*) (test)

Plus Pareto (fit vs Δforget) with λ labels.

Example:
  uv run python experiments/sim_conditional/stage_b_shift/plot_lambda_fit_forget.py
  uv run python experiments/sim_conditional/stage_b_shift/plot_lambda_fit_forget.py --metric mean_l2
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

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.default"] = "it"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.7,
        "lines.markersize": 5.0,
    }
)

from experiments.sim_conditional.stage_b_shift.common import REPO_ROOT, stage_b_runs_root

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
    p.add_argument("--n-contexts", type=int, default=6)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument(
        "--metric",
        choices=("energy", "mean_l2"),
        default="energy",
    )
    p.add_argument(
        "--forget",
        choices=("absolute", "delta"),
        default="delta",
    )
    p.add_argument("--seed", type=int, default=None, help="If set, only this seed.")
    p.add_argument(
        "--heldout-full-range",
        action="store_true",
        help="Keep the naive heldout reference inside the right axis limits "
        "instead of zooming on the replay curves.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Only FT runs with this epoch count; pass 0 to aggregate all.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def collect_rows(
    runs_root: Path, *, seed: int | None, epochs: int | None
) -> list[dict]:
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
        epochs_i = int(m.group("epochs"))
        if seed is not None and seed_i != seed:
            continue
        if epochs is not None and epochs_i != epochs:
            continue
        lam_s = m.group("lam")
        lam = float(lam_s) if lam_s is not None else cfg.get("lambda_mix")
        rows.append(
            {
                "run": run_dir.name,
                "strategy": strat,
                "lambda": None if strat == "naive" else float(lam),
                "seed": seed_i,
                "epochs": epochs_i,
                "val": payload["val"],
                "test": payload["test"],
            }
        )
    return rows


def _triplet(
    val_block: dict,
    test_block: dict,
    *,
    metric: str,
    forget: str,
) -> tuple[float, float, float]:
    if metric == "energy":
        fit = float(val_block["fit_energy_mean"])
        forg_k = (
            "delta_forget_energy_mean" if forget == "delta" else "forget_energy_mean"
        )
        forg = float(val_block[forg_k])
        # Prefer test heldout; fall back to val (Stage B stores test readout there too).
        held = float(
            test_block.get(
                "heldout_energy", val_block.get("heldout_energy", float("nan"))
            )
        )
    else:
        fit = float(val_block["fit_mean_l2_mean"])
        forg_k = (
            "delta_forget_mean_l2_mean"
            if forget == "delta"
            else "forget_mean_l2_mean"
        )
        forg = float(val_block[forg_k])
        held = float(
            test_block.get(
                "heldout_mean_l2", val_block.get("heldout_mean_l2", float("nan"))
            )
        )
    return fit, forg, held


def _agg_stats(vals: list[float]) -> tuple[float, float, float, int]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mid = float(np.median(arr))
    if n >= 2:
        lo = float(np.percentile(arr, 25))
        hi = float(np.percentile(arr, 75))
    else:
        lo = hi = mid
    return mid, lo, hi, n


def aggregate(
    rows: list[dict],
    *,
    metric: str,
    forget: str,
) -> dict[str, list[tuple]]:
    """strategy -> list of (lam|None, fit_mid,lo,hi, forg_mid,lo,hi, held_mid,lo,hi, n)."""
    buckets: dict[tuple[str, float | None], list[tuple[float, float, float]]] = (
        defaultdict(list)
    )
    for r in rows:
        fit, forg, held = _triplet(
            r["val"], r["test"], metric=metric, forget=forget
        )
        buckets[(r["strategy"], r["lambda"])].append((fit, forg, held))

    by_strat: dict[str, list[tuple]] = {
        "trajectory": [],
        "generative": [],
        "naive": [],
    }
    for (strat, lam), triples in buckets.items():
        fits = [t[0] for t in triples]
        forgs = [t[1] for t in triples]
        helds = [t[2] for t in triples]
        f_mid, f_lo, f_hi, n = _agg_stats(fits)
        g_mid, g_lo, g_hi, _ = _agg_stats(forgs)
        h_mid, h_lo, h_hi, _ = _agg_stats(helds)
        by_strat[strat].append(
            (lam, f_mid, f_lo, f_hi, g_mid, g_lo, g_hi, h_mid, h_lo, h_hi, n)
        )
    for k in by_strat:
        by_strat[k].sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else -1))
    return by_strat


def main() -> None:
    args = parse_args()
    if args.epochs is not None and int(args.epochs) <= 0:
        args.epochs = None
    runs_root = stage_b_runs_root(
        REPO_ROOT,
        dim=args.dim,
        n_contexts=args.n_contexts,
        n_perturbations=args.n_perturbations,
    )
    rows = collect_rows(runs_root, seed=args.seed, epochs=args.epochs)
    if not rows:
        raise SystemExit(f"No FT metrics under {runs_root / 'finetune'}")

    out_dir = args.out_dir or (runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = args.metric
    forget = args.forget
    by = aggregate(rows, metric=metric, forget=forget)

    if metric == "energy":
        fit_label = r"Fit energy $\downarrow$"
        forg_label = (
            r"$\Delta$Forget energy $\downarrow$"
            if forget == "delta"
            else r"Forget energy $\downarrow$"
        )
        held_label = r"Heldout energy $\downarrow$"
    else:
        fit_label = r"Fit mean-L2 $\downarrow$"
        forg_label = (
            r"$\Delta$Forget mean-L2 $\downarrow$"
            if forget == "delta"
            else r"Forget mean-L2 $\downarrow$"
        )
        held_label = r"Heldout mean-L2 $\downarrow$"

    ep_tag = f"_e{args.epochs}" if args.epochs is not None else ""
    stem = f"lambda_fit_forget_heldout_{metric}_{forget}_dim{args.dim}{ep_tag}"
    n_seeds = sorted({r["seed"] for r in rows})
    n_epochs = sorted({r["epochs"] for r in rows})
    seed_note = (
        f"median+IQR (n≤{len(n_seeds)} seeds)" if len(n_seeds) > 1 else f"seed{n_seeds[0]}"
    )
    if len(n_epochs) == 1:
        seed_note = f"FT {n_epochs[0]} ep  |  {seed_note}"
    else:
        seed_note = f"FT epochs={n_epochs}  |  {seed_note}"

    # --- 1×3 curves ---
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.85), sharex=True)
    for strat in ("trajectory", "generative"):
        pts = [t for t in by[strat] if t[0] is not None]
        if not pts:
            continue
        label, color, marker = METHOD_STYLES[strat]
        lams = np.asarray([t[0] for t in pts], dtype=np.float64)
        for ax_i, (y_i, lo_i, hi_i) in enumerate(
            ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        ):
            y = np.asarray([t[y_i] for t in pts], dtype=np.float64)
            lo = np.asarray([t[lo_i] for t in pts], dtype=np.float64)
            hi = np.asarray([t[hi_i] for t in pts], dtype=np.float64)
            axes[ax_i].fill_between(lams, lo, hi, color=color, alpha=0.18, linewidth=0)
            axes[ax_i].plot(
                lams, y, marker=marker, color=color, label=label, lw=1.8
            )

    # Replay-only span of the heldout axis, used to decide whether the naive
    # reference can share the scale without flattening the λ curves.
    replay_held = [
        v
        for s in ("trajectory", "generative")
        for t in by[s]
        if t[0] is not None
        for v in (t[7], t[8], t[9])
        if np.isfinite(v)
    ]

    naive_forget_note = None
    if by["naive"]:
        t = by["naive"][0]
        _, n_fit, n_flo, n_fhi, n_forg, _, _, n_held, n_hlo, n_hhi, n_n = t
        label, color, marker = METHOD_STYLES["naive"]
        if np.isfinite(n_fit):
            axes[0].axhline(n_fit, color=color, ls="--", lw=1.4, label=label)
            if n_n >= 2 and np.isfinite(n_flo):
                axes[0].axhspan(n_flo, n_fhi, color=color, alpha=0.12, linewidth=0)
        if np.isfinite(n_held):
            zoom_held = None
            if replay_held and not args.heldout_full_range:
                r_lo, r_hi = min(replay_held), max(replay_held)
                pad = 0.10 * max(r_hi - r_lo, 1e-12)
                if n_held > r_hi + pad:
                    zoom_held = (max(0.0, r_lo - pad), r_hi + pad)
            if zoom_held is None:
                axes[2].axhline(n_held, color=color, ls="--", lw=1.4)
                if n_n >= 2 and np.isfinite(n_hlo):
                    axes[2].axhspan(n_hlo, n_hhi, color=color, alpha=0.12, linewidth=0)
            else:
                axes[2].set_ylim(*zoom_held)
                axes[2].annotate(
                    f"Naive ≈ {n_held:.2f} $\\uparrow$",
                    xy=(0.97, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=color,
                )
        if np.isfinite(n_forg):
            naive_forget_note = (
                f"Naive Δforget ≈ {n_forg:.2f}"
                if forget == "delta"
                else f"Naive forget ≈ {n_forg:.2f}"
            )

    axes[0].set_ylabel(fit_label)
    axes[1].set_ylabel(forg_label)
    axes[2].set_ylabel(held_label)
    axes[0].set_title(rf"Val fit vs $\lambda$")
    axes[1].set_title(rf"Val $\Delta$forget vs $\lambda$" if forget == "delta" else rf"Val forget vs $\lambda$")
    axes[2].set_title(r"Test heldout $(c^\star,y^\star)$ vs $\lambda$")
    for ax in axes:
        ax.set_xlabel(r"$\lambda$")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", framealpha=0.92)

    fig.suptitle(f"Stage B shift  |  {seed_note}", fontsize=11, y=1.02)
    if naive_forget_note is not None:
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
        fig.text(
            0.98,
            0.02,
            naive_forget_note + " (omitted from middle axis)",
            ha="right",
            va="bottom",
            fontsize=8,
            color=METHOD_STYLES["naive"][1],
        )
    else:
        fig.tight_layout()
    png1 = out_dir / f"{stem}_curves.png"
    pdf1 = out_dir / f"{stem}_curves.pdf"
    fig.savefig(png1, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf1, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # --- Pareto: fit vs Δforget ---
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
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
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.annotate(
                f"{lam:g}",
                (x, y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
                color=color,
            )
    if by["naive"]:
        _, n_fit, _, _, n_forg, _, _, _, _, _, _ = by["naive"][0]
        # Naive forget is huge; mark with arrow note instead of plotting on same scale
        # if it would dominate. Still plot if finite and within a reasonable range.
        if np.isfinite(n_fit) and np.isfinite(n_forg):
            label, color, marker = METHOD_STYLES["naive"]
            replay_forgs = [
                t[4]
                for s in ("trajectory", "generative")
                for t in by[s]
                if t[0] is not None and np.isfinite(t[4])
            ]
            max_replay = max(replay_forgs) if replay_forgs else 1.0
            if n_forg <= 5.0 * max_replay:
                ax.scatter(
                    [n_fit],
                    [n_forg],
                    marker=marker,
                    color=color,
                    s=55,
                    label=label,
                    zorder=5,
                )
            else:
                ax.annotate(
                    f"naive Δforget≈{n_forg:.1f} (off-scale)",
                    xy=(0.98, 0.02),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )
    ax.set_xlabel(fit_label)
    ax.set_ylabel(forg_label)
    ax.set_title(rf"Stage B Pareto (labels = $\lambda$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    png2 = out_dir / f"{stem}_pareto.png"
    pdf2 = out_dir / f"{stem}_pareto.pdf"
    fig.savefig(png2, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf2, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # --- CSV ---
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "strategy",
                "lambda",
                "n",
                "fit_mid",
                "fit_lo",
                "fit_hi",
                "forget_mid",
                "forget_lo",
                "forget_hi",
                "heldout_mid",
                "heldout_lo",
                "heldout_hi",
            ]
        )
        for strat in ("naive", "generative", "trajectory"):
            for t in by[strat]:
                lam, f_mid, f_lo, f_hi, g_mid, g_lo, g_hi, h_mid, h_lo, h_hi, n = t
                w.writerow(
                    [
                        strat,
                        "" if lam is None else lam,
                        n,
                        f_mid,
                        f_lo,
                        f_hi,
                        g_mid,
                        g_lo,
                        g_hi,
                        h_mid,
                        h_lo,
                        h_hi,
                    ]
                )

    print(f"Wrote {png1} and {pdf1}")
    print(f"Wrote {png2} and {pdf2}")
    print(f"Wrote {csv_path}")
    print(f"Aggregated {len(rows)} runs across seeds {n_seeds}")


if __name__ == "__main__":
    main()
