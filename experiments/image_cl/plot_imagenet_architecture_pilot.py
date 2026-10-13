#!/usr/bin/env python3
"""Plot the ImageNet task-0 UNet-size and optimizer-budget pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGETS = (10_000, 20_000, 50_000, 100_000, 200_000)
ARCHITECTURES = {
    "medium": ("Medium UNet", "#8C857B", "s"),
    "large": ("Large UNet", "#457b9d", "o"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "image_cl"
            / "imagenet100"
            / "pilot"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "image_cl"
            / "figures"
            / "imagenet_architecture_budget_pilot.png"
        ),
    )
    return parser.parse_args()


def collect(pilot_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for budget in BUDGETS:
            tag = f"{budget // 1000}k"
            run = f"pilot_cc_{architecture}_e3_lr1e4_{tag}_t1_s42"
            path = pilot_root / run / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            final = payload["metrics_by_task"][-1]
            rows.append(
                {
                    "architecture": architecture,
                    "budget": budget,
                    "fid": float(final["fid_stream_test"]),
                    "epochs": int(payload["epochs"]),
                    "seed": int(payload["seed"]),
                    "source": str(path),
                }
            )
    return rows


def render(rows: list[dict[str, object]], out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(4.15, 3.0))
    for architecture, (label, color, marker) in ARCHITECTURES.items():
        selected = sorted(
            [row for row in rows if row["architecture"] == architecture],
            key=lambda row: int(row["budget"]),
        )
        ax.plot(
            [int(row["budget"]) for row in selected],
            [float(row["fid"]) for row in selected],
            color=color,
            marker=marker,
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            label=label,
            zorder=3,
        )

    chosen = next(
        row
        for row in rows
        if row["architecture"] == "large" and int(row["budget"]) == 100_000
    )
    ax.scatter(
        [int(chosen["budget"])],
        [float(chosen["fid"])],
        s=155,
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.2,
        zorder=5,
    )
    ax.annotate(
        "Chosen",
        (int(chosen["budget"]), float(chosen["fid"])),
        xytext=(18, -8),
        textcoords="offset points",
        fontsize=9.5,
        color="#333333",
        ha="left",
        va="top",
    )
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, ["10k", "20k", "50k", "100k", "200k"])
    ax.set_xlabel("Optimizer steps on task 0")
    ax.set_ylabel(r"Task-0 stream FID $\downarrow$")
    ax.set_ylim(55, 145)
    ax.grid(axis="y", color="0.87", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = ["architecture", "budget", "fid", "epochs", "seed", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.pilot_root)
    render(rows, args.out)
    write_csv(rows, args.out.with_suffix(".csv"))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.pdf')}")
    print(f"Wrote {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
