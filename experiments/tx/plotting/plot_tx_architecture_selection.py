#!/usr/bin/env python3
"""Plot transcriptomics representation/architecture selection diagnostics."""

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


REPO_ROOT = Path(__file__).resolve().parents[3]
WIDTH_RUNS = {
    50: "oneshot_pca50_task0_hepg2_h1024_d4_st100000_s42",
    100: "oneshot_pca100_task0_hepg2_h1024_d4_st100000_s42",
    256: "oneshot_pca256_sidecar_hepg2_h1024_d4_st100000_s42",
    512: "oneshot_pca512_task0_hepg2_h1024_d4_st100000_s42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replogle-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "tx_results" / "replogle",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results_dir"
            / "tx_results"
            / "figures"
            / "replogle_pca_width_selection.png"
        ),
    )
    return parser.parse_args()


def collect(root: Path) -> list[dict[str, Any]]:
    oracle: dict[int, dict[str, float]] = {}
    for filename in (
        "diag_pca_width_oracle_50_100.json",
        "diag_pca_width_oracle.json",
    ):
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        for row in payload["widths"]:
            width = int(row["n_comps"])
            if "fit" in row:
                oracle[width] = {
                    "oracle_fit": float(
                        row["fit"]["oracle_pearson_delta_macro"]
                    ),
                    "oracle_heldout": float(
                        row["heldout"]["oracle_pearson_delta_macro"]
                    ),
                }
            else:
                oracle[width] = {
                    "oracle_fit": float(row["fit_oracle_pearson_delta"]),
                    "oracle_heldout": float(
                        row["heldout_oracle_pearson_delta"]
                    ),
                }

    rows: list[dict[str, Any]] = []
    for width in sorted(oracle):
        row: dict[str, Any] = {"pca_dim": width, **oracle[width]}
        run = WIDTH_RUNS.get(width)
        if run is not None:
            path = root / run / "hepg2_pearson.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            row.update(
                {
                    "cfm_fit": float(payload["fit"]["pearson_delta_macro"]),
                    "cfm_heldout": float(
                        payload["heldout"]["pearson_delta_macro"]
                    ),
                    "cfm_source": str(path),
                }
            )
        else:
            row.update(
                {"cfm_fit": None, "cfm_heldout": None, "cfm_source": None}
            )
        rows.append(row)
    return rows


def render(rows: list[dict[str, Any]], out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    widths = np.asarray([int(row["pca_dim"]) for row in rows], dtype=float)

    for ax, prefix, title in (
        (axes[0], "oracle", "Inverse-PCA ceiling"),
        (axes[1], "cfm", "One-shot CFM"),
    ):
        for split, linestyle, marker in (
            ("fit", "solid", "o"),
            ("heldout", (0, (3.0, 2.0)), "s"),
        ):
            values = np.asarray(
                [
                    np.nan
                    if row[f"{prefix}_{split}"] is None
                    else float(row[f"{prefix}_{split}"])
                    for row in rows
                ],
                dtype=float,
            )
            mask = np.isfinite(values)
            ax.plot(
                widths[mask],
                values[mask],
                color="#457b9d" if split == "fit" else "#c1121f",
                linestyle=linestyle,
                marker=marker,
                markersize=7,
                markerfacecolor=(
                    "#457b9d" if split == "fit" else "white"
                ),
                markeredgecolor=(
                    "#457b9d" if split == "fit" else "#c1121f"
                ),
                markeredgewidth=1.0,
                linewidth=2.0,
                label=(
                    "Fit"
                    if split == "fit"
                    else "STATE-test (diagnostic)"
                ),
            )
        ax.axvline(
            50,
            color="#555555",
            linestyle=(0, (2.0, 2.0)),
            linewidth=1.0,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(int(width)) for width in widths])
        ax.set_xlabel("PCA dimension")
        ax.set_title(title, pad=6)
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Pearson $\Delta$ $\uparrow$")
    axes[1].annotate(
        "Chosen",
        xy=(50, float(next(row["cfm_fit"] for row in rows if row["pca_dim"] == 50))),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=9,
        color="#444444",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.tight_layout(w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=400,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.03,
    )
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.03,
    )
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "pca_dim",
        "oracle_fit",
        "oracle_heldout",
        "cfm_fit",
        "cfm_heldout",
        "cfm_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.replogle_root)
    render(rows, args.out)
    write_csv(rows, args.out.with_suffix(".csv"))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.pdf')}")
    print(f"Wrote {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
