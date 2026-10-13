#!/usr/bin/env python3
"""Collect sequential Tx CL JSON outputs into CSV tables and summary plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TRACKS = ("fit", "forget", "stream", "heldout")
METRICS = (
    "pearson_delta_macro",
    "pearson_delta_residual_macro",
    "pearson_macro",
    "selection_score",
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "roots",
        type=Path,
        nargs="+",
        help="Dataset/run directories to scan recursively.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(roots: list[Path]) -> list[Path]:
    runs: set[Path] = set()
    for root in roots:
        if (root / "config.json").is_file():
            runs.add(root)
        runs.update(path.parent for path in root.rglob("final_metrics.json"))
    return sorted(runs)


def collect_final(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "final_metrics.json"
        if not config_path.is_file() or not metrics_path.is_file():
            continue
        cfg = _read(config_path)
        final = _read(metrics_path).get("final", {})
        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "dataset": cfg.get("dataset"),
            "strategy": cfg.get("strategy"),
            "finetune_scope": cfg.get("finetune_scope", "full"),
            "replay_backend": cfg.get("replay_backend", "ode"),
            "mean_flow_steps": cfg.get("mean_flow_steps", 0),
            "lambda_mode": cfg.get("lambda_mode"),
            "representation": cfg.get("representation", "se"),
            "pca_dim": cfg.get("pca_dim"),
            "seed": cfg.get("seed"),
        }
        for track in TRACKS:
            block = final.get(track, {})
            row[f"{track}_n_cells"] = block.get("n_cells")
            row[f"{track}_n_strata"] = block.get("n_strata")
            for metric in METRICS:
                row[f"{track}_{metric}"] = block.get(metric)
        rows.append(row)
    return rows


def collect_curve(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.json"
        curve_path = run_dir / "heldout_curve.json"
        if not cfg_path.is_file() or not curve_path.is_file():
            continue
        cfg = _read(cfg_path)
        for point in _read(curve_path).get("heldout_curve", []):
            block = point.get("heldout", {})
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "dataset": cfg.get("dataset"),
                    "strategy": cfg.get("strategy"),
                    "finetune_scope": cfg.get("finetune_scope", "full"),
                    "replay_backend": cfg.get("replay_backend", "ode"),
                    "mean_flow_steps": cfg.get("mean_flow_steps", 0),
                    "lambda_mode": cfg.get("lambda_mode"),
                    "representation": cfg.get("representation", "se"),
                    "pca_dim": cfg.get("pca_dim"),
                    "seed": cfg.get("seed"),
                    "task_id": point.get("task_id"),
                    "n_cells": block.get("n_cells"),
                    "n_strata": block.get("n_strata"),
                    "pearson_delta_macro": block.get("pearson_delta_macro"),
                    "pearson_delta_residual_macro": block.get(
                        "pearson_delta_residual_macro"
                    ),
                    "pearson_macro": block.get("pearson_macro"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def representation_tag(row: dict[str, Any]) -> str:
    rep = str(row.get("representation") or "se")
    if rep == "pca":
        dim = row.get("pca_dim")
        if dim not in (None, "", 0):
            return f"pca{int(dim)}"
        return "pca"
    if rep == "scimilarity":
        return "sci"
    return rep


def method_label(row: dict[str, Any]) -> str:
    strategy = str(row["strategy"])
    scope = str(row.get("finetune_scope", "full"))
    tag = representation_tag(row)
    if strategy in ("standard", "gft"):
        return f"{tag} {strategy} {scope}"
    if strategy == "real":
        return f"{tag} real {scope} {row['lambda_mode']}"
    backend = str(row["replay_backend"])
    if backend == "mean_flow":
        backend = f"MF-N{row['mean_flow_steps']}"
    else:
        backend = "ODE"
    return f"{tag} {strategy} {scope} {backend} {row['lambda_mode']}"


def _finite(values: list[Any]) -> np.ndarray:
    out = np.asarray(
        [float(x) for x in values if x is not None], dtype=np.float64
    )
    return out[np.isfinite(out)]


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), method_label(row))].append(row)
    lines = [
        "# Sequential Tx CL summary",
        "",
        "| Dataset | Method | Seeds | Stream Pearson-Δ | Heldout Pearson-Δ | Heldout residual-Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (dataset, method), group in sorted(groups.items()):
        stream = _finite([row["stream_pearson_delta_macro"] for row in group])
        heldout = _finite([row["heldout_pearson_delta_macro"] for row in group])
        residual = _finite(
            [row["heldout_pearson_delta_residual_macro"] for row in group]
        )

        def fmt(values: np.ndarray) -> str:
            if not values.size:
                return "n/a"
            return f"{values.mean():.4f} ± {values.std(ddof=1) if values.size > 1 else 0:.4f}"

        lines.append(
            f"| {dataset} | {method} | {len(group)} | {fmt(stream)} | {fmt(heldout)} | {fmt(residual)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(out_dir: Path, final_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV/Markdown tables were still written.")
        return

    for dataset in sorted({str(row["dataset"]) for row in final_rows}):
        selected = [row for row in final_rows if str(row["dataset"]) == dataset]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            groups[method_label(row)].append(row)
        labels = sorted(groups)
        if not labels:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(max(12, len(labels) * 1.4), 5))
        panels = (
            ("stream", "pearson_delta_macro", "Pearson-Δ"),
            ("heldout", "pearson_delta_macro", "Pearson-Δ"),
            ("heldout", "pearson_delta_residual_macro", "Residual-Δ"),
        )
        for axis, (track, metric, ylabel) in zip(axes, panels):
            means, errors = [], []
            for label in labels:
                values = _finite(
                    [row[f"{track}_{metric}"] for row in groups[label]]
                )
                means.append(float(values.mean()) if values.size else np.nan)
                errors.append(
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                )
            x = np.arange(len(labels))
            axis.errorbar(x, means, yerr=errors, fmt="o", capsize=4)
            axis.set_xticks(x, labels, rotation=45, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(f"Final {track}")
            axis.grid(axis="y", alpha=0.3)
        fig.suptitle(dataset)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_final.png", dpi=180)
        plt.close(fig)

    for dataset in sorted({str(row["dataset"]) for row in curve_rows}):
        selected = [row for row in curve_rows if str(row["dataset"]) == dataset]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            groups[method_label(row)].append(row)
        if not groups:
            continue
        fig, axis = plt.subplots(figsize=(8, 5))
        for label, group in sorted(groups.items()):
            by_task: dict[int, list[Any]] = defaultdict(list)
            for row in group:
                by_task[int(row["task_id"])].append(row["pearson_delta_macro"])
            tasks = sorted(by_task)
            means = [
                float(v.mean()) if (v := _finite(by_task[task])).size else np.nan
                for task in tasks
            ]
            axis.plot(tasks, means, marker="o", label=label)
        axis.set_xlabel("Task")
        axis.set_ylabel("Heldout Pearson-Δ")
        axis.set_title(f"{dataset}: sequential heldout")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_heldout_curve.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = make_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.roots)
    final_rows = collect_final(runs)
    curve_rows = collect_curve(runs)
    write_csv(args.out_dir / "final_summary.csv", final_rows)
    write_csv(args.out_dir / "heldout_curve.csv", curve_rows)
    write_markdown(args.out_dir / "summary.md", final_rows)
    write_plots(args.out_dir, final_rows, curve_rows)
    print(
        f"Collected {len(final_rows)} final runs and {len(curve_rows)} curve points "
        f"under {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
