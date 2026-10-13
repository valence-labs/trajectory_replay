#!/usr/bin/env python3
"""Plot FMNIST trajectory replay quality and MF speedup across teacher NFEs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
NFE_GRID = (5, 10, 20, 40, 80, 100)
COLORS = {
    "ode": "#457b9d",
    "mf1": "#F4A261",
    "mf4": "#9B5DE5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "fmnist",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "figures",
    )
    return parser.parse_args()


def _ode_dir(root: Path, nfe: int, seed: int) -> Path:
    if nfe == 10:
        return root / f"trajectory_eq_s{seed}"
    return root / f"trajectory_eq_replaynfe{nfe}_s{seed}"


def _amortized_mf_time(root: Path, n: int, seed: int) -> float:
    path = root / f"trajectory_mf_n{n}_eq_tfix_s{seed}" / "timing.json"
    timing = json.loads(path.read_text(encoding="utf-8"))
    update = np.asarray(
        timing["single_ft_train_wall_seconds_all"][1:], dtype=float
    ).mean()
    setup = np.asarray(
        timing["mean_flow_distill_wall_seconds_all"], dtype=float
    ).sum()
    return float((update + setup / 4.0) / 3600.0)


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        mf1 = _amortized_mf_time(root, 1, seed)
        mf4 = _amortized_mf_time(root, 4, seed)
        for nfe in NFE_GRID:
            run_dir = _ode_dir(root, nfe, seed)
            metrics = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            )
            timing = json.loads(
                (run_dir / "timing.json").read_text(encoding="utf-8")
            )
            final = metrics["metrics_by_task"][-1]
            ode_time = float(
                np.asarray(
                    timing["single_ft_train_wall_seconds_all"][1:],
                    dtype=float,
                ).mean()
                / 3600.0
            )
            rows.append(
                {
                    "seed": seed,
                    "teacher_nfe": nfe,
                    "expected_trajectory_teacher_evals": (nfe + 3.0) / 2.0,
                    "final_fid": float(final["fid_stream_test"]),
                    "final_kld": float(final["stream_kld"]),
                    "ode_update_hours": ode_time,
                    "mf1_amortized_update_hours": mf1,
                    "mf4_amortized_update_hours": mf4,
                    "mf1_speedup": ode_time / mf1,
                    "mf4_speedup": ode_time / mf4,
                    "metrics_source": str(run_dir / "metrics.json"),
                    "timing_source": str(run_dir / "timing.json"),
                }
            )
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
        }
    )


def _aggregate(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean: list[float] = []
    se: list[float] = []
    for nfe in NFE_GRID:
        values = np.asarray(
            [float(row[field]) for row in rows if row["teacher_nfe"] == nfe],
            dtype=float,
        )
        mean.append(float(values.mean()))
        se.append(float(values.std(ddof=1) / np.sqrt(values.size)))
    return np.asarray(mean), np.asarray(se)


def _format_x(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(NFE_GRID, [str(value) for value in NFE_GRID])
    ax.set_xlim(4.2, 115)
    ax.tick_params(axis="x", labelsize=10, labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", color="0.87", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_quality(rows: list[dict[str, Any]], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.95))
    for ax, field, ylabel in (
        (axes[0], "final_fid", r"Final stream FID $\downarrow$"),
        (axes[1], "final_kld", r"Final stream KLD $\downarrow$"),
    ):
        mean, se = _aggregate(rows, field)
        ax.errorbar(
            NFE_GRID,
            mean,
            yerr=se,
            color=COLORS["ode"],
            marker="o",
            markersize=7.5,
            markerfacecolor=COLORS["ode"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            elinewidth=1.0,
            capsize=2.8,
        )
        ax.set_xlabel("Configured teacher NFE")
        ax.set_ylabel(ylabel)
        _format_x(ax)
    fig.tight_layout(w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_speedup(rows: list[dict[str, Any]], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95))
    ode_mean, ode_se = _aggregate(rows, "ode_update_hours")
    axes[0].errorbar(
        NFE_GRID,
        ode_mean,
        yerr=ode_se,
        color=COLORS["ode"],
        marker="o",
        markersize=7,
        linewidth=2.0,
        elinewidth=1.0,
        capsize=2.8,
        label="ODE teacher",
    )
    for key, label, linestyle in (
        ("mf1_amortized_update_hours", "MF-1", (0, (3.0, 2.0))),
        ("mf4_amortized_update_hours", "MF-4", (0, (1.0, 1.7))),
    ):
        mean, se = _aggregate(rows, key)
        color = COLORS["mf1" if "mf1" in key else "mf4"]
        axes[0].plot(
            NFE_GRID,
            mean,
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
        axes[0].fill_between(
            NFE_GRID,
            mean - se,
            mean + se,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    axes[0].set_ylabel("Amortized wall time (h)")

    for key, label, marker in (
        ("mf1_speedup", "MF-1", "^"),
        ("mf4_speedup", "MF-4", "s"),
    ):
        mean, se = _aggregate(rows, key)
        color = COLORS["mf1" if "mf1" in key else "mf4"]
        axes[1].errorbar(
            NFE_GRID,
            mean,
            yerr=se,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.0,
            elinewidth=1.0,
            capsize=2.8,
            label=label,
        )
    axes[1].axhline(
        1.0,
        color="#555555",
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.0,
    )
    axes[1].text(
        5.0,
        1.04,
        "Break-even",
        fontsize=8.5,
        color="#555555",
        va="bottom",
    )
    axes[1].set_ylabel(r"MF speedup over ODE $\uparrow$")

    for ax in axes:
        ax.set_xlabel("Configured teacher NFE")
        _format_x(ax)
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.results_root)
    quality_out = args.out_dir / "fmnist_teacher_nfe_quality.png"
    speedup_out = args.out_dir / "fmnist_meanflow_speedup_vs_teacher_nfe.png"
    render_quality(rows, quality_out)
    render_speedup(rows, speedup_out)
    csv_path = args.out_dir / "fmnist_teacher_nfe_sweep.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {quality_out} and {quality_out.with_suffix('.pdf')}")
    print(f"Wrote {speedup_out} and {speedup_out.with_suffix('.pdf')}")
    print(f"Wrote {csv_path} ({len(rows)} seed-level rows)")


if __name__ == "__main__":
    main()
