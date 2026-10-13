#!/usr/bin/env python3
"""Render final Replogle/PBMC CL tables from audited three-seed outputs."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev

from experiments.tx.audit_final_cl_results import (
    METHODS,
    SEEDS,
    SPECS,
    run_metrics,
    run_path,
)


ROOT = Path(__file__).resolve().parents[2]


def display_label(label: str) -> str:
    return (
        label.replace(r"$\lambda$-grid", r"adaptive $\lambda$")
        .replace(r"$\lambda$-equal", r"equal-task $\lambda$")
        .replace(r"MeanFlow $N{=}1$", "MF-1")
        .replace(r"MeanFlow $N{=}4$", "MF-4")
    )


def aggregate(spec) -> list[tuple[str, tuple[tuple[float, float], ...]]]:
    rows = []
    for label, base in METHODS:
        values = [
            run_metrics(spec, run_path(spec, base, seed))
            for seed in SEEDS
        ]
        summary = tuple(
            (
                mean(column),
                stdev(column) / math.sqrt(len(SEEDS)),
            )
            for column in zip(*values)
        )
        rows.append((display_label(label), summary))
    return rows


def value_tex(value: float, se: float, *, bold: bool) -> str:
    number = rf"\mathbf{{{value:.3f}}}" if bold else f"{value:.3f}"
    return rf"${number}$ {{\scriptsize $(\pm\,{se:.3f})$}}"


def caption(spec) -> str:
    if spec.name == "replogle":
        return (
            r"\textbf{Continual perturbation-response modeling on "
            r"Replogle-Nadig.} Final-checkpoint performance after adaptation "
            r"from three task-0 cell lines to HepG2 using a task-0 PCA-50 "
            r"latent and an MLP $2048{\times}4$. "
        )
    return (
        r"\textbf{Continual perturbation-response modeling on Parse-PBMC.} "
        r"Final-checkpoint performance after four sequential donor updates "
        r"using a task-0 PCA-100 latent and a residual MLP $2048{\times}6$. "
    )


def render(spec) -> str:
    rows = aggregate(spec)
    # Real replay is a privileged memory-based reference; bold the strongest
    # memory-free mean in each metric column.
    eligible = [
        values
        for label, values in rows
        if not label.startswith("Real replay")
    ]
    best = [max(values[column][0] for values in eligible) for column in range(6)]

    lines = [
        r"% Requires \usepackage[table]{xcolor}, \usepackage{booktabs}, and \usepackage{graphicx}.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\definecolor{trajectoryrow}{HTML}{E5EEF3}",
        r"\definecolor{referencerow}{HTML}{F0F0F0}",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\small",
        r"\caption{" + caption(spec)
        + (
            r"Fit, retention, and heldout denote later-task adaptation, "
            r"task-0 preservation, and STATE-test perturbations, respectively. "
            r"Pearson $\Delta$ measures gene-space perturbation-effect "
            r"correlation; DE Overlap@N measures recovery of observed "
            r"significant top-effect genes. Means with standard errors are "
            r"reported over seeds 42, 69, and 1714. Adaptive $\lambda$ is "
            r"selected on validation strata; equal-task uses "
            r"$\lambda_k=k/(k+1)$. Gray denotes the privileged real-replay "
            r"reference and blue denotes trajectory replay. Bold is the best "
            r"memory-free mean in each column. Higher is better.}"
        ),
        (
            r"\label{tab:"
            + ("replogle_cl_full" if spec.name == "replogle" else "pbmc_cl_full")
            + "}"
        ),
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"Method & \multicolumn{3}{c}{Pearson $\Delta$} & "
        r"\multicolumn{3}{c}{DE Overlap@N} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"& $\uparrow$ Fit & $\uparrow$ Retention & $\uparrow$ Heldout "
        r"& $\uparrow$ Fit & $\uparrow$ Retention & $\uparrow$ Heldout \\",
        r"\midrule",
    ]

    for index, (label, values) in enumerate(rows):
        if index in {6, 13, 15, 21}:
            lines.append(r"\midrule")
        prefix = ""
        if label.startswith("Real replay"):
            prefix = r"\rowcolor{referencerow} "
        elif label.startswith("Trajectory replay"):
            prefix = r"\rowcolor{trajectoryrow} "
        cells = [
            value_tex(
                value,
                se,
                bold=(
                    not label.startswith("Real replay")
                    and abs(value - best[column]) < 1e-12
                ),
            )
            for column, (value, se) in enumerate(values)
        ]
        lines.append(prefix + label + " & " + " & ".join(cells) + r" \\")

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
    for spec in SPECS:
        path = ROOT / spec.table
        path.write_text(render(spec), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
