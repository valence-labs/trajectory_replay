#!/usr/bin/env python3
"""Plot all-strata PBMC Pearson Delta stratified by cell type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from experiments.tx.audit_final_cl_results import ROOT, SEEDS, SPECS, run_path


PBMC_SPEC = next(spec for spec in SPECS if spec.name == "pbmc")
METHODS = (
    ("Naive\nfull", "standard_p1c0"),
    ("Real\nreplay", "real_equal"),
    ("Gen. replay\n(ODE)", "generative_ode_equal"),
    ("Traj. replay\n(ODE)", "trajectory_ode_equal"),
)
CMAP = LinearSegmentedColormap.from_list(
    "tx_cell_type_blue",
    ("#F2F5F7", "#AFC9D7", "#5C91AD", "#003049"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def _cell_type_values(path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("eval_provenance") or {}
    if int(provenance.get("gen_per_stratum", -1)) != 500:
        raise ValueError(f"{path}: expected K=500 evaluation")
    if provenance.get("pred_control") != "decoded":
        raise ValueError(f"{path}: expected decoded controls")

    config_path = path.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    encoding = config["encoding"]
    if encoding.get("covariate_col_order") != ["cell_type"]:
        raise ValueError(f"{config_path}: unexpected covariate encoding")
    cell_types = list(encoding["covariate_categories"]["cell_type"])

    grouped: dict[int, list[float]] = {
        index: [] for index in range(len(cell_types))
    }
    for track in ("stream", "heldout"):
        for row in payload["final"][track]["rows"]:
            key = row["key"]
            if len(key) != 3:
                raise ValueError(
                    f"{path}: expected (context, perturbation, cell type)"
                )
            value = float(row["pearson_delta"])
            if np.isfinite(value):
                grouped[int(key[2])].append(value)
    missing = [
        cell_types[index] for index, values in grouped.items() if not values
    ]
    if missing:
        raise ValueError(f"{path}: no evaluated strata for {missing}")
    return cell_types, np.asarray(
        [np.mean(grouped[index]) for index in range(len(cell_types))],
        dtype=float,
    )


def collect() -> dict[str, Any]:
    method_values: list[np.ndarray] = []
    sources: dict[str, list[str]] = {}
    expected_cell_types: list[str] | None = None
    for label, run_key in METHODS:
        seed_values: list[np.ndarray] = []
        method_sources: list[str] = []
        for seed in SEEDS:
            source = run_path(PBMC_SPEC, run_key, seed) / "final_metrics.json"
            cell_types, values = _cell_type_values(source)
            if expected_cell_types is None:
                expected_cell_types = cell_types
            elif cell_types != expected_cell_types:
                raise ValueError(f"{source}: inconsistent cell-type order")
            seed_values.append(values)
            method_sources.append(str(source))
        method_values.append(np.asarray(seed_values, dtype=float))
        sources[label] = method_sources
    if expected_cell_types is None:
        raise RuntimeError("No PBMC results loaded")
    return {
        "cell_types": expected_cell_types,
        "values": np.stack(method_values, axis=2),
        "sources": sources,
    }


def _summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        values.mean(axis=0),
        values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]),
    )


def render(data: dict[str, Any], out: Path) -> None:
    _style()
    mean, _ = _summary(data["values"])
    fig, ax = plt.subplots(figsize=(4.45, 5.15))
    span = float(mean.max() - mean.min())
    image = ax.imshow(
        mean,
        cmap=CMAP,
        vmin=float(mean.min()) - 0.04 * span,
        vmax=float(mean.max()) + 0.04 * span,
        aspect="auto",
    )

    labels = [label for label, _ in METHODS]
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(data["cell_types"])), data["cell_types"])
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Parse-PBMC all-strata performance by cell type", pad=8)

    midpoint = (float(mean.min()) + float(mean.max())) / 2.0
    for row in range(mean.shape[0]):
        for column in range(mean.shape[1]):
            value = float(mean[row, column])
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value > midpoint else "#263238",
            )

    ax.set_xticks(
        np.arange(-0.5, len(labels), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(data["cell_types"]), 1),
        minor=True,
    )
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label(r"All-strata Pearson $\Delta$ $\uparrow$", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def write_summary(data: dict[str, Any], out: Path) -> None:
    mean, se = _summary(data["values"])
    payload = {
        "dataset": "pbmc",
        "stratification": "cell_type",
        "metric": "all-strata Pearson Delta",
        "aggregation": (
            "Within each seed and cell type, macro-mean over the union of final "
            "stream and heldout donor-cytokine strata; then mean and SE across "
            "seeds."
        ),
        "seeds": list(SEEDS),
        "evaluation": {
            "generated_cells_per_non_control_stratum": 500,
            "predicted_controls": "decoded",
        },
        "cell_types": data["cell_types"],
        "methods": [
            {
                "label": label,
                "run_key": run_key,
                "mean": mean[:, index].tolist(),
                "se": se[:, index].tolist(),
                "sources": data["sources"][label],
            }
            for index, (label, run_key) in enumerate(METHODS)
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results_dir" / "tx_results" / "figures",
    )
    args = parser.parse_args()
    data = collect()
    out = args.out_dir / "pbmc_all_strata_pearson_by_cell_type.png"
    render(data, out)
    write_summary(
        data,
        args.out_dir / "pbmc_all_strata_pearson_by_cell_type.json",
    )
    print(out)


if __name__ == "__main__":
    main()
