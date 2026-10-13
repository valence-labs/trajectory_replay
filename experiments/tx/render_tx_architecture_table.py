#!/usr/bin/env python3
"""Render TX architecture-selection diagnostics as a LaTeX table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_dir" / "tx_results"
PBMC_ARCHIVE = RESULTS / "pbmc" / "_old" / "20260904_before_final_res2048_d6"
OUT = ROOT / "experiments" / "tx" / "tables" / "tx_architecture_selection.tex"
SUMMARY = (
    ROOT
    / "results_dir"
    / "tx_results"
    / "figures"
    / "tx_architecture_selection.json"
)

REPLOGLE_WIDTH_RUNS = {
    50: "oneshot_pca50_task0_hepg2_h1024_d4_st100000_s42",
    100: "oneshot_pca100_task0_hepg2_h1024_d4_st100000_s42",
    256: "oneshot_pca256_sidecar_hepg2_h1024_d4_st100000_s42",
    512: "oneshot_pca512_task0_hepg2_h1024_d4_st100000_s42",
}
PBMC_WIDTH_RUNS = {
    width: f"oneshot_pca{width}_task0_pbmc_h1024_d4_st100000_s42"
    for width in (50, 100, 256, 512)
}
CAPACITY_RUNS = (
    (
        "Replogle",
        "MLP",
        2048,
        4,
        RESULTS
        / "replogle"
        / "trajectory_pca50_ode_equal_st100000_ft50000_cap_w2048_d4_s42",
        True,
    ),
    (
        "Replogle",
        "Residual MLP",
        2048,
        6,
        RESULTS
        / "replogle"
        / "trajectory_pca50_ode_equal_st100000_ft50000_cap_w2048_d6_res_s42",
        False,
    ),
    (
        "PBMC",
        "MLP",
        4096,
        4,
        PBMC_ARCHIVE
        / "trajectory_pca100_ode_equal_st300000_ft150000_cap_w4096_d4_s42",
        False,
    ),
    (
        "PBMC",
        "MLP",
        3072,
        4,
        PBMC_ARCHIVE
        / "trajectory_pca100_ode_equal_st300000_ft150000_cap_w3072_d4_s42",
        False,
    ),
    (
        "PBMC",
        "Residual MLP",
        3072,
        6,
        PBMC_ARCHIVE
        / "trajectory_pca100_ode_equal_st300000_ft150000_cap_w3072_d6_res_s42",
        False,
    ),
    (
        "PBMC",
        "Residual MLP",
        2048,
        6,
        PBMC_ARCHIVE
        / "trajectory_pca100_ode_equal_st300000_ft150000_cap_w2048_d6_res_s42",
        True,
    ),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replogle_reconstruction() -> dict[int, dict[str, float | None]]:
    rows: dict[int, dict[str, float | None]] = {}
    for filename in (
        "diag_pca_width_oracle_50_100.json",
        "diag_pca_width_oracle.json",
    ):
        payload = _read(RESULTS / "replogle" / filename)
        for row in payload["widths"]:
            width = int(row["n_comps"])
            if "heldout" in row:
                heldout = row["heldout"]
                rows[width] = {
                    "variance": (
                        None
                        if row.get("cumulative_variance_ratio") is None
                        else float(row["cumulative_variance_ratio"])
                    ),
                    "oracle_heldout": float(
                        heldout["oracle_pearson_delta_macro"]
                    ),
                    "reconstruction_pearson": float(
                        heldout["per_cell_recon_pearson"]
                    ),
                }
            else:
                rows[width] = {
                    "variance": (
                        None
                        if row.get("cumulative_variance_ratio") is None
                        else float(row["cumulative_variance_ratio"])
                    ),
                    "oracle_heldout": float(
                        row["heldout_oracle_pearson_delta"]
                    ),
                    "reconstruction_pearson": float(
                        row["heldout_recon_pearson"]
                    ),
                }
    return rows


def _pbmc_reconstruction() -> tuple[
    dict[int, float],
    dict[str, float],
]:
    width_payload = _read(PBMC_ARCHIVE / "diag_pca_width_oracle.json")
    variance = {
        int(row["n_comps"]): float(row["cumulative_variance_ratio"])
        for row in width_payload["widths"]
    }
    affine = _read(PBMC_ARCHIVE / "diag_pca_affine_bank_oracle.json")
    row = next(row for row in affine["widths"] if int(row["n_comps"]) == 100)
    heldout = [row[f"t{task}_heldout"] for task in range(1, 5)]
    summary = {
        "oracle_heldout": sum(
            float(block["oracle_pearson_delta_macro"]) for block in heldout
        )
        / len(heldout),
        "reconstruction_pearson": sum(
            float(block["per_cell_recon_pearson"]) for block in heldout
        )
        / len(heldout),
    }
    return variance, summary


def collect_widths() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replogle_recon = _replogle_reconstruction()
    for width, run_name in REPLOGLE_WIDTH_RUNS.items():
        source = RESULTS / "replogle" / run_name / "hepg2_pearson.json"
        score = _read(source)
        recon = replogle_recon[width]
        rows.append(
            {
                "dataset": "Replogle",
                "pca_dim": width,
                "variance": recon["variance"],
                "pilot_fit": float(score["fit"]["pearson_delta_macro"]),
                "pilot_heldout": float(
                    score["heldout"]["pearson_delta_macro"]
                ),
                "oracle_heldout": recon["oracle_heldout"],
                "reconstruction_pearson": recon[
                    "reconstruction_pearson"
                ],
                "chosen": width == 50,
                "source": str(source),
            }
        )

    pbmc_variance, pbmc_recon = _pbmc_reconstruction()
    for width, run_name in PBMC_WIDTH_RUNS.items():
        source = PBMC_ARCHIVE / run_name / "oneshot_pearson.json"
        score = _read(source)
        chosen = width == 100
        rows.append(
            {
                "dataset": "PBMC",
                "pca_dim": width,
                "variance": pbmc_variance[width],
                "pilot_fit": float(score["fit"]["pearson_delta_macro"]),
                "pilot_heldout": float(
                    score["heldout"]["pearson_delta_macro"]
                ),
                "oracle_heldout": (
                    pbmc_recon["oracle_heldout"] if chosen else None
                ),
                "reconstruction_pearson": (
                    pbmc_recon["reconstruction_pearson"] if chosen else None
                ),
                "chosen": chosen,
                "source": str(source),
            }
        )
    return rows


def collect_capacity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, backbone, width, depth, run, chosen in CAPACITY_RUNS:
        payload = _read(run / "final_metrics.json")
        provenance = payload["eval_provenance"]
        if int(provenance.get("gen_per_stratum", -1)) != 500:
            raise ValueError(f"{run}: capacity evaluation is not K=500")
        final = payload["final"]
        if dataset == "PBMC":
            fit = sum(
                float(final["stream"]["per_task"][str(task)]["pearson_delta"])
                for task in range(1, 5)
            ) / 4.0
            heldout = sum(
                float(final["heldout"]["per_task"][str(task)]["pearson_delta"])
                for task in range(1, 5)
            ) / 4.0
        else:
            fit = float(final["stream"]["per_task"]["1"]["pearson_delta"])
            heldout = float(final["heldout"]["per_task"]["1"]["pearson_delta"])
        retention = float(final["stream"]["per_task"]["0"]["pearson_delta"])
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "width": width,
                "depth": depth,
                "fit": fit,
                "retention": retention,
                "heldout": heldout,
                "chosen": chosen,
                "source": str(run / "final_metrics.json"),
            }
        )
    return rows


def _value(value: float | None, *, chosen: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if chosen else text


def render(widths: list[dict[str, Any]], capacity: list[dict[str, Any]]) -> str:
    lines = [
        r"% Requires \usepackage{booktabs} and \usepackage{graphicx}.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Transcriptomics architecture-selection diagnostics.} "
        r"Panel (a) reports single-seed selection-only PCA pilots and observed-latent "
        r"reconstruction ceilings; every PCA-width pilot fixes the CFM backbone to "
        r"an MLP with width 1024 and depth 4. These "
        r"exploratory scores predate the fixed-generation "
        r"benchmark protocol and are not directly comparable to the final continual-learning "
        r"tables. Variance is cumulative PCA variance explained, Oracle is Pearson "
        r"$\Delta$ after decoding observed PCA latents, and Recon.\ is mean per-cell "
        r"reconstruction correlation. Panel (b) compares model capacity under the final "
        r"sequential protocol with exactly $K=500$ generated cells per non-control stratum. "
        r"All rows use seed 42. Bold denotes the selected configuration.}",
        r"\label{tab:tx_architecture_selection}",
        r"\textbf{(a) PCA width and reconstruction ceiling "
        r"(fixed MLP $1024{\times}4$)}\\[2pt]",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Dataset & PCA width & Variance & Pilot fit $\Delta$ & "
        r"Pilot heldout $\Delta$ & Oracle heldout $\Delta$ & Recon.\ $r$ \\",
        r"\midrule",
    ]
    for index, row in enumerate(widths):
        if index and widths[index - 1]["dataset"] != row["dataset"]:
            lines.append(r"\midrule")
        chosen = bool(row["chosen"])
        width = (
            rf"\textbf{{{row['pca_dim']}}}"
            if chosen
            else str(row["pca_dim"])
        )
        lines.append(
            f"{row['dataset']} & {width} & "
            f"{_value(row['variance'])} & "
            f"{_value(row['pilot_fit'], chosen=chosen)} & "
            f"{_value(row['pilot_heldout'], chosen=chosen)} & "
            f"{_value(row['oracle_heldout'])} & "
            f"{_value(row['reconstruction_pearson'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{6pt}",
            r"\textbf{(b) Final-protocol model-capacity check ($K=500$)}\\[2pt]",
            r"\begin{tabular}{@{}llccccc@{}}",
            r"\toprule",
            r"Dataset & Backbone & Width & Depth & "
            r"$\uparrow$ Fit $\Delta$ & $\uparrow$ Retention $\Delta$ & "
            r"$\uparrow$ Heldout $\Delta$ \\",
            r"\midrule",
        ]
    )
    for index, row in enumerate(capacity):
        if index and capacity[index - 1]["dataset"] != row["dataset"]:
            lines.append(r"\midrule")
        chosen = bool(row["chosen"])
        backbone = (
            rf"\textbf{{{row['backbone']}}}"
            if chosen
            else str(row["backbone"])
        )
        width = (
            rf"\textbf{{{row['width']}}}" if chosen else str(row["width"])
        )
        depth = (
            rf"\textbf{{{row['depth']}}}" if chosen else str(row["depth"])
        )
        lines.append(
            f"{row['dataset']} & {backbone} & {width} & {depth} & "
            f"{_value(row['fit'], chosen=chosen)} & "
            f"{_value(row['retention'], chosen=chosen)} & "
            f"{_value(row['heldout'], chosen=chosen)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    widths = collect_widths()
    capacity = collect_capacity()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(widths, capacity), encoding="utf-8")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps(
            {
                "pca_width_selection": widths,
                "capacity_k500": capacity,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUT)
    print(SUMMARY)


if __name__ == "__main__":
    main()
