#!/usr/bin/env python3
"""Plot validation-selected lambda schedules for sequential Gaussian tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "generative": ("Generative replay", "#c1121f", "s", -0.055),
    "trajectory": ("Trajectory replay", "#457b9d", "o", 0.055),
}
SEEDS = (0, 1, 2)
TASK_IDS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "sim_sequential" / "runs_lr3e4",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "sim_sequential"
            / "figures"
            / "adaptive_lambda_schedule.png"
        ),
    )
    return parser.parse_args()


def collect(runs_root: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        for seed in SEEDS:
            path = runs_root / f"{method}_adaptive_s{seed}" / "stream_metrics.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for task in payload:
                task_id = int(task["task_id"])
                selected = task.get("selected_lambda")
                if task_id == 0 or selected is None:
                    continue
                selection = task.get("selection") or []
                metrics = {
                    candidate.get("selection_metric") for candidate in selection
                }
                metrics.discard(None)
                if metrics != {"energy_uniform_mixture"}:
                    raise ValueError(
                        f"Unexpected selection metric(s) in {path}: {sorted(metrics)}"
                    )
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "task_id": task_id,
                        "lambda": float(selected),
                        "selection_metric": "validation energy_uniform_mixture",
                        "source": str(path),
                    }
                )
    return rows


def render(rows: list[dict[str, float | int | str]], out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(4.15, 2.75))

    tasks = np.asarray(TASK_IDS, dtype=float)
    equal_task = tasks / (tasks + 1.0)
    ax.plot(
        tasks,
        equal_task,
        color="#555555",
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.8,
        marker="*",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="#555555",
        label=r"Equal-task $\lambda_k=k/(k+1)$",
        zorder=2,
    )

    for method, (label, color, marker, method_offset) in METHODS.items():
        method_rows = [row for row in rows if row["method"] == method]
        means: list[float] = []
        mean_tasks: list[float] = []
        for task_id in TASK_IDS:
            values = [
                float(row["lambda"])
                for row in method_rows
                if row["task_id"] == task_id
            ]
            if not values:
                continue
            mean_tasks.append(task_id + method_offset)
            means.append(float(np.mean(values)))
            if len(values) < len(SEEDS):
                ax.annotate(
                    rf"$n={len(values)}$",
                    (task_id + method_offset, max(values)),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=color,
                )
        ax.plot(
            mean_tasks,
            means,
            color=color,
            marker=marker,
            markersize=8.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=2.0,
            label=label,
            zorder=4,
        )

    ax.set_xticks(tasks, [rf"after $T_{task}$" for task in TASK_IDS])
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(0.44, 0.94)
    ax.set_yticks([0.5, 2 / 3, 0.75, 0.8, 0.9], ["0.50", "0.67", "0.75", "0.80", "0.90"])
    ax.set_xlabel("Sequential update")
    ax.set_ylabel(r"Selected $\lambda_k$")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=1,
        handlelength=2.0,
        borderpad=0.1,
        labelspacing=0.25,
    )
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fields = ["method", "seed", "task_id", "lambda", "selection_metric", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.runs_root)
    render(rows, args.out)
    write_csv(rows, args.out.with_suffix(".csv"))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.pdf')}")
    print(f"Wrote {args.out.with_suffix('.csv')} ({len(rows)} selections)")


if __name__ == "__main__":
    main()
