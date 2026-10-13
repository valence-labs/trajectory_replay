#!/usr/bin/env python3
"""Audit and aggregate canonical image-CL result rows from three seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 69, 1714)
DATASETS = {
    "fmnist": {
        "dataset": "split_fmnist",
        "n_tasks": 5,
        "runs": (
            "naive",
            "generative",
            "generative_eq",
            "generative_mf_n1_tfix",
            "generative_mf_n1_eq_tfix",
            "generative_mf_n4_tfix",
            "generative_mf_n4_eq_tfix",
            "trajectory",
            "trajectory_eq",
            "trajectory_mf_n1_tfix",
            "trajectory_mf_n1_eq_tfix",
            "trajectory_mf_n4_tfix",
            "trajectory_mf_n4_eq_tfix",
        ),
    },
    "cifar10": {
        "dataset": "split_cifar10",
        "n_tasks": 5,
        "runs": (
            "naive",
            "generative_g3",
            "generative_eq",
            "generative_mf_n1_g3",
            "generative_mf_n1_eq",
            "generative_mf_n4_g3",
            "generative_mf_n4_eq",
            "trajectory_g3",
            "trajectory_eq",
            "trajectory_mf_n1_g3",
            "trajectory_mf_n1_eq",
            "trajectory_mf_n4_g3",
            "trajectory_mf_n4_eq",
        ),
    },
    "imagenet100": {
        "dataset": "split_imagenet100",
        "n_tasks": 3,
        "runs": (
            "naive",
            "cumulative",
            "generative_g3",
            "generative_eq_eq",
            "generative_mf_n1_g3",
            "generative_mf_n1_eq",
            "generative_mf_n4_g3",
            "generative_mf_n4_eq",
            "trajectory_g3",
            "trajectory_eq_eq",
            "trajectory_mf_n1_g3",
            "trajectory_mf_n1_eq",
            "trajectory_mf_n4_g3",
            "trajectory_mf_n4_eq",
        ),
    },
}

TABLE_LABELS = {
    "naive": r"Naive fine-tuning",
    "cumulative": r"Cumulative joint training",
    "generative": r"Generative replay (ODE, adaptive $\lambda$)",
    "generative_g3": r"Generative replay (ODE, adaptive $\lambda$)",
    "generative_eq": r"Generative replay (ODE, $\lambda$-equal)",
    "generative_eq_eq": r"Generative replay (ODE, $\lambda$-equal)",
    "trajectory": r"Trajectory replay (ODE, adaptive $\lambda$)",
    "trajectory_g3": r"Trajectory replay (ODE, adaptive $\lambda$)",
    "trajectory_eq": r"Trajectory replay (ODE, $\lambda$-equal)",
    "trajectory_eq_eq": r"Trajectory replay (ODE, $\lambda$-equal)",
}

EXTERNAL_RESULTS = {
    "joint": {
        "fmnist": {"fid": (12.5, 0.5), "kld": (0.04, 0.01)},
        "cifar10": {"fid": (25.9, 0.1), "kld": (0.10, 0.02)},
        "imagenet100": {"fid": (50.7, 0.7)},
    },
    "finetune": {
        "fmnist": {"fid": (67.7, 6.6), "kld": (4.14, 1.40)},
        "cifar10": {"fid": (53.6, 4.4), "kld": (1.10, 0.14)},
        "imagenet100": {"fid": (105.8, 3.1)},
    },
    "generative_replay": {
        "fmnist": {"fid": (184.9, 12.6), "kld": (3.68, 1.32)},
        "cifar10": {"fid": (84.3, 9.1), "kld": (0.59, 0.11)},
        "imagenet100": {"fid": (106.3, 2.5)},
    },
    "generative_distillation": {
        "fmnist": {"fid": (25.5, 1.1), "kld": (0.29, 0.04)},
        "cifar10": {"fid": (38.5, 4.4), "kld": (0.23, 0.08)},
        "imagenet100": {"fid": (85.5, 3.4)},
    },
}

UNIFIED_ROWS = (
    (
        r"\textbf{Joint training} (all data; upper bound)",
        {
            "fmnist": "external:joint",
            "cifar10": "external:joint",
            "imagenet100": "external:joint",
        },
        "baseline",
        False,
    ),
    (
        r"Fine-tuning",
        {
            "fmnist": "external:finetune",
            "cifar10": "external:finetune",
            "imagenet100": "external:finetune",
        },
        "baseline",
        False,
    ),
    (
        r"Diffusion replay",
        {
            "fmnist": "external:generative_replay",
            "cifar10": "external:generative_replay",
            "imagenet100": "external:generative_replay",
        },
        "baseline",
        False,
    ),
    (
        r"Diffusion distillation \citep{masip2025continual}",
        {
            "fmnist": "external:generative_distillation",
            "cifar10": "external:generative_distillation",
            "imagenet100": "external:generative_distillation",
        },
        "baseline",
        False,
    ),
    (
        r"FM fine-tuning",
        {"fmnist": "naive", "cifar10": "naive", "imagenet100": "naive"},
        "flow",
        False,
    ),
    (
        r"Generative replay (ODE, adaptive $\lambda$)",
        {
            "fmnist": "generative",
            "cifar10": "generative_g3",
            "imagenet100": "generative_g3",
        },
        "generative",
        False,
    ),
    (
        r"Generative replay (ODE, equal-task $\lambda$)",
        {
            "fmnist": "generative_eq",
            "cifar10": "generative_eq",
            "imagenet100": "generative_eq_eq",
        },
        "generative",
        False,
    ),
    (
        r"Generative replay (MF-1, adaptive $\lambda$)",
        {
            "fmnist": "generative_mf_n1_tfix",
            "cifar10": "generative_mf_n1_g3",
            "imagenet100": "generative_mf_n1_g3",
        },
        "generative",
        False,
    ),
    (
        r"Generative replay (MF-1, equal-task $\lambda$)",
        {
            "fmnist": "generative_mf_n1_eq_tfix",
            "cifar10": "generative_mf_n1_eq",
            "imagenet100": "generative_mf_n1_eq",
        },
        "generative",
        False,
    ),
    (
        r"Generative replay (MF-4, adaptive $\lambda$)",
        {
            "fmnist": "generative_mf_n4_tfix",
            "cifar10": "generative_mf_n4_g3",
            "imagenet100": "generative_mf_n4_g3",
        },
        "generative",
        False,
    ),
    (
        r"Generative replay (MF-4, equal-task $\lambda$)",
        {
            "fmnist": "generative_mf_n4_eq_tfix",
            "cifar10": "generative_mf_n4_eq",
            "imagenet100": "generative_mf_n4_eq",
        },
        "generative",
        False,
    ),
    (
        r"Trajectory replay (ODE, adaptive $\lambda$)",
        {
            "fmnist": "trajectory",
            "cifar10": "trajectory_g3",
            "imagenet100": "trajectory_g3",
        },
        "trajectory",
        True,
    ),
    (
        r"Trajectory replay (ODE, equal-task $\lambda$)",
        {
            "fmnist": "trajectory_eq",
            "cifar10": "trajectory_eq",
            "imagenet100": "trajectory_eq_eq",
        },
        "trajectory",
        True,
    ),
    (
        r"Trajectory replay (MF-1, adaptive $\lambda$)",
        {
            "fmnist": "trajectory_mf_n1_tfix",
            "cifar10": "trajectory_mf_n1_g3",
            "imagenet100": "trajectory_mf_n1_g3",
        },
        "trajectory",
        True,
    ),
    (
        r"Trajectory replay (MF-1, equal-task $\lambda$)",
        {
            "fmnist": "trajectory_mf_n1_eq_tfix",
            "cifar10": "trajectory_mf_n1_eq",
            "imagenet100": "trajectory_mf_n1_eq",
        },
        "trajectory",
        True,
    ),
    (
        r"Trajectory replay (MF-4, adaptive $\lambda$)",
        {
            "fmnist": "trajectory_mf_n4_tfix",
            "cifar10": "trajectory_mf_n4_g3",
            "imagenet100": "trajectory_mf_n4_g3",
        },
        "trajectory",
        True,
    ),
    (
        r"Trajectory replay (MF-4, equal-task $\lambda$)",
        {
            "fmnist": "trajectory_mf_n4_eq_tfix",
            "cifar10": "trajectory_mf_n4_eq",
            "imagenet100": "trajectory_mf_n4_eq",
        },
        "trajectory",
        True,
    ),
)


def _table_label(stem: str) -> str:
    if stem in TABLE_LABELS:
        return TABLE_LABELS[stem]
    strategy = "Generative" if stem.startswith("generative") else "Trajectory"
    n_steps = 1 if "_n1_" in stem else 4
    mode = "equal-task" if "_eq" in stem else "adaptive"
    return (
        rf"{strategy} replay (MeanFlow $N{{=}}{n_steps}$, "
        rf"{mode} $\lambda$)"
    )


def _row_group(stem: str) -> str:
    if stem in ("naive", "cumulative"):
        return "baseline"
    strategy = "generative" if stem.startswith("generative") else "trajectory"
    backend = "meanflow" if "_mf_" in stem else "ode"
    return f"{strategy}_{backend}"


def _render_table(slug: str, dataset_report: dict[str, Any]) -> str:
    titles = {
        "fmnist": "Split Fashion-MNIST",
        "cifar10": "Split CIFAR-10",
        "imagenet100": "Split ImageNet-100",
    }
    has_kld = slug in ("fmnist", "cifar10")
    column_spec = "lcc" if has_kld else "lc"
    header = (
        r"Strategy & $\downarrow$ FID & $\downarrow$ KLD \\"
        if has_kld
        else r"Strategy & $\downarrow$ FID \\"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            rf"\caption{{Continual learning on {titles[slug]} after the final task. "
            r"Reported is mean {\scriptsize$\pm$} standard error over three seeds; "
            r"lower is better.}"
        ),
        rf"\label{{tab:{slug}_cl_fm}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    previous_group = None
    for stem, aggregate in dataset_report.items():
        group = _row_group(stem)
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        mean = aggregate["mean"]
        se = aggregate["se"]
        if mean is None or se is None:
            value = r"\textit{missing}"
        else:
            value = rf"${mean:.1f}$ {{\scriptsize $(\pm\,{se:.1f})$}}"
        if has_kld:
            kld_mean = aggregate["kld_mean"]
            kld_se = aggregate["kld_se"]
            kld_value = (
                r"\textit{missing}"
                if kld_mean is None or kld_se is None
                else rf"${kld_mean:.3f}$ {{\scriptsize $(\pm\,{kld_se:.3f})$}}"
            )
            lines.append(rf"{_table_label(stem)} & {value} & {kld_value} \\")
        else:
            lines.append(rf"{_table_label(stem)} & {value} \\")
        previous_group = group
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    return "\n".join(lines)


def _unified_value(
    report: dict[str, Any],
    *,
    slug: str,
    stem: str | None,
    metric: str,
) -> tuple[float, float] | None:
    if stem is None:
        return None
    if stem.startswith("external:"):
        source = stem.split(":", 1)[1]
        value = EXTERNAL_RESULTS[source][slug].get(metric)
        return (
            (float(value[0]), float(value[1]))
            if value is not None
            else None
        )
    aggregate = report["datasets"][slug][stem]
    if metric == "fid":
        mean, se = aggregate["mean"], aggregate["se"]
    else:
        mean, se = aggregate["kld_mean"], aggregate["kld_se"]
    if mean is None or se is None:
        return None
    return float(mean), float(se)


def _format_unified_value(
    value: tuple[float, float] | None,
    *,
    metric: str,
    bold: bool,
) -> str:
    if value is None:
        return r"--"
    mean, se = value
    precision = 1 if metric == "fid" else 3
    mean_text = f"{mean:.{precision}f}"
    se_text = f"{se:.{precision}f}"
    if bold:
        return (
            rf"$\mathbf{{{mean_text}}}$ "
            rf"{{\scriptsize $(\pm\,{se_text})$}}"
        )
    return rf"${mean_text}$ {{\scriptsize $(\pm\,{se_text})$}}"


def _render_unified_table(report: dict[str, Any]) -> str:
    columns = (
        ("fmnist", "fid"),
        ("fmnist", "kld"),
        ("cifar10", "fid"),
        ("cifar10", "kld"),
        ("imagenet100", "fid"),
    )
    bold_rows: set[tuple[str, str, str]] = set()
    for slug, metric in columns:
        candidates = []
        for _, stems, _, _ in UNIFIED_ROWS:
            stem = stems[slug]
            if stem == "external:joint":
                continue
            value = _unified_value(
                report,
                slug=slug,
                stem=stem,
                metric=metric,
            )
            if value is not None:
                candidates.append((value[0], stem))
        best_mean = min(mean for mean, _ in candidates)
        for mean, stem in candidates:
            if math.isclose(mean, best_mean):
                bold_rows.add((slug, metric, stem))

    lines = [
        (
            r"% Requires \usepackage[table]{xcolor}, \usepackage{booktabs}, "
            r"\usepackage{graphicx}, and natbib."
        ),
        r"\begin{table*}[t]",
        r"\centering",
        r"\definecolor{trajectoryrow}{HTML}{E5EEF3}",
        r"\definecolor{upperboundrow}{HTML}{F0F0F0}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\small",
        (
            r"\caption{\textbf{Class-incremental image generation.} "
            r"Final-stream performance after five tasks on Fashion-MNIST and "
            r"CIFAR-10 and three tasks on ImageNet-100 (means with standard "
            r"errors over three seeds). FID compares generated samples with "
            r"pooled held-out data from all seen classes; KLD measures class-mass "
            r"recovery. Adaptive $\lambda$ minimizes held-out validation FID; "
            r"equal-task uses $\lambda_k=k/(k+1)$, and MF-$N$ uses $N$ MeanFlow "
            r"replay steps. "
            r"Blue rows denote trajectory replay; gray denotes joint training "
            r"with all data. Bold denotes the best mean in each column, "
            r"excluding joint training. Lower is better.}"
        ),
        r"\label{tab:image_cl_all}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        (
            r"Method & \multicolumn{2}{c}{Fashion-MNIST} "
            r"& \multicolumn{2}{c}{CIFAR-10} & ImageNet-100 \\"
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-6}",
        (
            r"& $\downarrow$ FID & $\downarrow$ KLD "
            r"& $\downarrow$ FID & $\downarrow$ KLD & $\downarrow$ FID \\"
        ),
        r"\midrule",
    ]
    previous_group = None
    for label, stems, group, is_ours in UNIFIED_ROWS:
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        values = []
        for slug, metric in columns:
            value = _unified_value(
                report,
                slug=slug,
                stem=stems[slug],
                metric=metric,
            )
            is_best = (slug, metric, stems[slug]) in bold_rows
            values.append(
                _format_unified_value(value, metric=metric, bold=is_best)
            )
        is_upper_bound = all(
            stem is None or stem == "external:joint" for stem in stems.values()
        )
        prefix = (
            r"\rowcolor{upperboundrow} "
            if is_upper_bound
            else r"\rowcolor{trajectoryrow} "
            if is_ours
            else ""
        )
        lines.append(prefix + label + " & " + " & ".join(values) + r" \\")
        previous_group = group
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        )
    )
    return "\n".join(lines)


def _write_tables(report: dict[str, Any], tex_dir: Path) -> list[Path]:
    tex_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug, dataset_report in report["datasets"].items():
        path = tex_dir / f"{slug}_cl_fm.tex"
        path.write_text(_render_table(slug, dataset_report), encoding="utf-8")
        written.append(path)
    if set(report["datasets"]) == set(DATASETS):
        path = tex_dir / "image_cl_all_datasets.tex"
        path.write_text(_render_unified_table(report), encoding="utf-8")
        written.append(path)
    return written


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_run(
    run_dir: Path,
    *,
    dataset: str,
    n_tasks: int,
    seed: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        return None, [f"missing config.json or metrics.json under {run_dir}"]

    config = _read_json(config_path)
    metrics = _read_json(metrics_path)
    if config.get("dataset") != dataset:
        issues.append(f"dataset={config.get('dataset')!r}, expected {dataset!r}")
    if int(config.get("seed", -1)) != seed:
        issues.append(f"seed={config.get('seed')!r}, expected {seed}")
    if bool(config.get("smoke", False)):
        issues.append("run is marked smoke")

    trace = metrics.get("metrics_by_task", [])
    if len(trace) != n_tasks:
        issues.append(f"metrics_by_task has {len(trace)} tasks, expected {n_tasks}")
    final_fid = metrics.get("final_fid_stream_test")
    if final_fid is None or not math.isfinite(float(final_fid)):
        issues.append(f"non-finite final_fid_stream_test={final_fid!r}")
    if trace and final_fid is not None:
        last_fid = trace[-1].get("fid_stream_test")
        if last_fid is None or not math.isclose(
            float(final_fid), float(last_fid), rel_tol=0.0, abs_tol=1e-9
        ):
            issues.append("top-level final FID does not match the last task")
    final_kld = metrics.get("final_stream_kld")
    expect_kld = dataset in ("split_fmnist", "split_cifar10")
    if expect_kld and (
        final_kld is None or not math.isfinite(float(final_kld))
    ):
        issues.append(f"non-finite final_stream_kld={final_kld!r}")
    if trace and final_kld is not None:
        last_kld = trace[-1].get("stream_kld")
        if last_kld is None or not math.isclose(
            float(final_kld), float(last_kld), rel_tol=0.0, abs_tol=1e-9
        ):
            issues.append("top-level final KLD does not match the last task")

    lambda_mode = config.get("lambda_mode")
    strategy = config.get("strategy")
    if strategy in ("generative", "trajectory"):
        for task in trace[1:]:
            chosen = task.get("chosen_lambda")
            candidates = task.get("candidates", [])
            if chosen is None or not candidates:
                issues.append(f"task {task.get('task_id')}: missing lambda selection")
                continue
            if lambda_mode == "grid":
                best = min(candidates, key=lambda row: float(row["fid_macro_val"]))
                if not math.isclose(float(chosen), float(best["lambda"])):
                    issues.append(
                        f"task {task.get('task_id')}: chosen lambda is not val-FID argmin"
                    )
            elif lambda_mode == "equal":
                task_id = int(task["task_id"])
                expected = task_id / (task_id + 1.0)
                if not math.isclose(float(chosen), expected):
                    issues.append(
                        f"task {task_id}: equal lambda {chosen} != {expected}"
                    )

    row = {
        "run_dir": str(run_dir),
        "seed": seed,
        "final_fid_stream_test": (
            float(final_fid) if final_fid is not None else None
        ),
        "final_stream_kld": (
            float(final_kld) if final_kld is not None else None
        ),
        "strategy": strategy,
        "lambda_mode": lambda_mode,
        "lambda_grid": config.get("lambda_grid"),
        "replay_backend": config.get("replay_backend"),
        "mean_flow_steps": config.get("mean_flow_steps"),
        "fid_test_samples": config.get("fid_test_samples"),
        "ode_step_size": config.get("ode_step_size"),
    }
    return row, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results_dir" / "image_cl" / "table_audit.json",
    )
    parser.add_argument(
        "--tex-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "image_cl" / "tables",
    )
    parser.add_argument(
        "--no-write-tex",
        action="store_true",
        help="Audit JSON only; do not regenerate LaTeX tables.",
    )
    args = parser.parse_args()

    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    report: dict[str, Any] = {"seeds": list(SEEDS), "datasets": {}, "issues": []}
    for slug, spec in selected.items():
        dataset_rows: dict[str, Any] = {}
        for stem in spec["runs"]:
            seed_rows: list[dict[str, Any]] = []
            for seed in SEEDS:
                run_dir = args.results_root / slug / f"{stem}_s{seed}"
                row, issues = _audit_run(
                    run_dir,
                    dataset=str(spec["dataset"]),
                    n_tasks=int(spec["n_tasks"]),
                    seed=seed,
                )
                report["issues"].extend(f"{slug}/{stem}_s{seed}: {x}" for x in issues)
                if row is not None and not issues:
                    seed_rows.append(row)
            values = np.asarray(
                [row["final_fid_stream_test"] for row in seed_rows],
                dtype=np.float64,
            )
            kld_values = np.asarray(
                [
                    row["final_stream_kld"]
                    for row in seed_rows
                    if row["final_stream_kld"] is not None
                ],
                dtype=np.float64,
            )
            aggregate = {
                "n_seeds": int(values.size),
                "mean": float(values.mean()) if values.size else None,
                "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                "se": (
                    float(values.std(ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else None
                ),
                "kld_mean": float(kld_values.mean()) if kld_values.size else None,
                "kld_sd": (
                    float(kld_values.std(ddof=1)) if kld_values.size > 1 else None
                ),
                "kld_se": (
                    float(kld_values.std(ddof=1) / np.sqrt(kld_values.size))
                    if kld_values.size > 1
                    else None
                ),
                "runs": seed_rows,
            }
            dataset_rows[stem] = aggregate
            if values.size:
                print(
                    f"{slug:11s} {stem:38s} "
                    f"{aggregate['mean']:7.2f} ± SE {aggregate['se']:6.2f} "
                    + (
                        f"KLD {aggregate['kld_mean']:.3f} ± SE "
                        f"{aggregate['kld_se']:.3f} "
                        if aggregate["kld_mean"] is not None
                        else ""
                    )
                    + f"(n={values.size})"
                )
        report["datasets"][slug] = dataset_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    if report["issues"]:
        print("\nAudit issues:")
        for issue in report["issues"]:
            print(f"- {issue}")
        raise SystemExit(1)
    if not args.no_write_tex:
        for path in _write_tables(report, args.tex_dir):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
