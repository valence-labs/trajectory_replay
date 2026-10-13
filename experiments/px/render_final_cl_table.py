#!/usr/bin/env python3
"""Render the final three-seed RxRx1 continual-learning table."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from experiments.px.plot_cl_experiment_fid import latest_complete_metrics


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results_dir" / "px" / "cl_final"
MASIP_ROOT = ROOT / "results_dir" / "px" / "masip_final"
OUT = ROOT / "experiments" / "px" / "tables" / "px_cl_final.tex"
SUMMARY = ROOT / "results_dir" / "px" / "figures" / "px_cl_final_table.json"
SEEDS = (42, 69, 1714)
EXPERIMENTS = ("U2OS-01", "U2OS-02", "U2OS-03")

# label, run stem, family, row style, memory-based reference
METHODS = (
    (
        r"\textbf{Joint training} (all data; upper bound)",
        "cumulative_ode_equal_s{seed}",
        "flow",
        "reference",
        True,
    ),
    (
        r"Real replay (stored data)",
        "real_ode_equal_s{seed}",
        "flow",
        "reference",
        True,
    ),
    (
        r"Naive fine-tuning",
        "naive_ode_equal_s{seed}",
        "flow",
        "baseline",
        False,
    ),
    (
        r"Generative distillation (diffusion) \citep{masip2025continual}",
        "generative_distillation_s{seed}",
        "masip",
        "baseline",
        False,
    ),
    (
        r"Generative replay (ODE, adaptive $\lambda$)",
        "generative_ode_grid_s{seed}",
        "flow",
        "generative",
        False,
    ),
    (
        r"Generative replay (ODE, equal-task $\lambda$)",
        "generative_ode_equal_s{seed}",
        "flow",
        "generative",
        False,
    ),
    (
        r"Trajectory replay (ODE, adaptive $\lambda$)",
        "trajectory_ode_grid_s{seed}",
        "flow",
        "trajectory",
        False,
    ),
    (
        r"Trajectory replay (ODE, equal-task $\lambda$)",
        "trajectory_ode_equal_s{seed}",
        "flow",
        "trajectory",
        False,
    ),
    (
        r"Trajectory replay (MF-1, equal-task $\lambda$)",
        "trajectory_mean_flow_n1_equal_s{seed}",
        "flow",
        "trajectory",
        False,
    ),
    (
        r"Trajectory replay (MF-4, equal-task $\lambda$)",
        "trajectory_mean_flow_n4_equal_s{seed}",
        "flow",
        "trajectory",
        False,
    ),
)


def _metric_row(payload: dict[str, Any]) -> tuple[float, ...]:
    blocks = payload.get("metrics_by_task") or []
    if not blocks or int(blocks[-1].get("task_id", -1)) != 2:
        raise ValueError("PX run does not contain a complete task-2 evaluation")
    final = blocks[-1]
    by_experiment = {
        str(row["batch_name"]): row for row in final["test_by_experiment"]
    }
    if set(by_experiment) != set(EXPERIMENTS):
        raise ValueError(f"Unexpected final experiments: {sorted(by_experiment)}")
    stream = final["test_stream"]
    return (
        *(float(by_experiment[name]["fid"]) for name in EXPERIMENTS),
        float(stream["fid"]),
        float(stream["fdd_dinov2"]),
    )


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, stem, family, style, reference in METHODS:
        values: list[tuple[float, ...]] = []
        sources: list[str] = []
        root = MASIP_ROOT if family == "masip" else RESULTS_ROOT
        for seed in SEEDS:
            found = latest_complete_metrics(root / stem.format(seed=seed))
            if found is None:
                raise FileNotFoundError(
                    f"No complete PX metrics for {stem.format(seed=seed)}"
                )
            path, payload = found
            values.append(_metric_row(payload))
            sources.append(str(path))
        summary = [
            {
                "mean": mean(column),
                "se": stdev(column) / math.sqrt(len(column)),
            }
            for column in zip(*values)
        ]
        rows.append(
            {
                "label": label,
                "stem": stem,
                "family": family,
                "style": style,
                "memory_based_reference": reference,
                "seeds": list(SEEDS),
                "metrics": {
                    "u2os01_fid": summary[0],
                    "u2os02_fid": summary[1],
                    "u2os03_fid": summary[2],
                    "stream_fid": summary[3],
                    "stream_fdd_dinov2": summary[4],
                },
                "sources": sources,
            }
        )
    return rows


def _value(value: float, se: float, *, bold: bool) -> str:
    number = rf"\mathbf{{{value:.1f}}}" if bold else f"{value:.1f}"
    return rf"${number}$ {{\scriptsize $(\pm\,{se:.1f})$}}"


def render(rows: list[dict[str, Any]]) -> str:
    keys = (
        "u2os01_fid",
        "u2os02_fid",
        "u2os03_fid",
        "stream_fid",
        "stream_fdd_dinov2",
    )
    eligible = [
        row for row in rows if not bool(row["memory_based_reference"])
    ]
    best = {
        key: min(float(row["metrics"][key]["mean"]) for row in eligible)
        for key in keys
    }
    lines = [
        r"% Requires \usepackage[table]{xcolor}, \usepackage{booktabs}, \usepackage{graphicx}, and natbib.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\definecolor{trajectoryrow}{HTML}{E5EEF3}",
        r"\definecolor{referencerow}{HTML}{F0F0F0}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\small",
        r"\caption{\textbf{Continual phenomic generation on RxRx1.} "
        r"Final-checkpoint performance after the U2OS-01$\rightarrow$U2OS-02"
        r"$\rightarrow$U2OS-03 experiment stream. Per-experiment FID compares "
        r"composition-matched generated and test images within each biological "
        r"experiment; stream metrics pool all three experiments. FDD denotes "
        r"Fr\'echet distance in DINOv2 feature space. Means with standard errors "
        r"are reported over seeds 42, 69, and 1714. Adaptive $\lambda$ is selected "
        r"using validation FID; equal-task uses $\lambda_k=k/(k+1)$. Gray rows "
        r"retain real historical data and are privileged references; blue rows "
        r"denote trajectory replay. Bold marks the best memory-free mean in each "
        r"column. Lower is better.}",
        r"\label{tab:px_cl_final}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & \multicolumn{3}{c}{Per-experiment FID $\downarrow$} & "
        r"\multicolumn{2}{c}{Pooled stream $\downarrow$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
        r"& U2OS-01 & U2OS-02 & U2OS-03 & FID & DINOv2 FDD \\",
        r"\midrule",
    ]
    previous_style: str | None = None
    for row in rows:
        style = str(row["style"])
        if previous_style is not None and style != previous_style:
            lines.append(r"\midrule")
        previous_style = style
        prefix = ""
        if style == "reference":
            prefix = r"\rowcolor{referencerow} "
        elif style == "trajectory":
            prefix = r"\rowcolor{trajectoryrow} "
        cells = []
        for key in keys:
            metric = row["metrics"][key]
            value = float(metric["mean"])
            cells.append(
                _value(
                    value,
                    float(metric["se"]),
                    bold=(
                        not bool(row["memory_based_reference"])
                        and abs(value - best[key]) < 1e-12
                    ),
                )
            )
        lines.append(prefix + str(row["label"]) + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows), encoding="utf-8")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "experiments": list(EXPERIMENTS),
                "rows": rows,
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
