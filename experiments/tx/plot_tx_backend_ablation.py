#!/usr/bin/env python3
"""Plot TX replay-backend quality relative to matched ODE evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.tx.audit_final_cl_results import ROOT, SEEDS, SPECS, run_path


DATASET_ORDER = ("replogle", "pbmc")
DATASET_LABELS = {"replogle": "REPLOGLE", "pbmc": "PBMC"}
STRATEGIES = {
    "generative": ("Generative replay", "#C1121F"),
    "trajectory": ("Trajectory replay", "#457B9D"),
}
POLICIES = {
    "adaptive": (r"Adaptive $\lambda$", "grid", "solid", False),
    "equal": ("Equal-task", "equal", (0, (3.0, 2.0)), True),
}
BACKENDS = {
    "mf1": ("MF-1", "mf_n1", "^"),
    "mf4": ("MF-4", "mf_n4", "s"),
    "ode": ("ODE", "ode", "o"),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.85,
        }
    )


def _run_key(strategy: str, policy: str, backend: str) -> str:
    mode = POLICIES[policy][1]
    backend_key = BACKENDS[backend][1]
    return f"{strategy}_{backend_key}_{mode}"


def _replay_evals(strategy: str, backend: str, ode_step_size: float) -> float:
    if backend == "mf1":
        rollout = 1.0
    elif backend == "mf4":
        rollout = 4.0
    else:
        rollout = float(round(1.0 / ode_step_size))
    if strategy == "generative":
        return rollout
    if backend != "ode":
        return rollout + 1.0
    return (rollout + 3.0) / 2.0


def _heldout_pearson(dataset: str, path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("eval_provenance") or {}
    if int(provenance.get("gen_per_stratum", -1)) != 500:
        raise ValueError(f"{path}: expected K=500 evaluation")
    per_task = payload["final"]["heldout"]["per_task"]
    if dataset == "pbmc":
        value = float(
            np.mean(
                [float(per_task[str(task)]["pearson_delta"]) for task in range(1, 5)]
            )
        )
    else:
        value = float(per_task["1"]["pearson_delta"])
    if not np.isfinite(value):
        raise ValueError(f"{path}: non-finite held-out Pearson Delta")
    return value, float(provenance["ode_step_size"])


def collect() -> list[dict[str, Any]]:
    specs = {spec.name: spec for spec in SPECS}
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        spec = specs[dataset]
        for strategy in STRATEGIES:
            for policy in POLICIES:
                for backend in BACKENDS:
                    key = _run_key(strategy, policy, backend)
                    for seed in SEEDS:
                        source = run_path(spec, key, seed) / "final_metrics.json"
                        value, ode_step_size = _heldout_pearson(dataset, source)
                        rows.append(
                            {
                                "dataset": dataset,
                                "strategy": strategy,
                                "policy": policy,
                                "backend": backend,
                                "seed": seed,
                                "run_key": key,
                                "replay_field_evals": _replay_evals(
                                    strategy,
                                    backend,
                                    ode_step_size,
                                ),
                                "heldout_pearson_delta": value,
                                "source": str(source),
                            }
                        )
    return rows


def _group(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    strategy: str,
    policy: str,
    backend: str,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["strategy"] == strategy
        and row["policy"] == policy
        and row["backend"] == backend
    ]
    if len(selected) != len(SEEDS):
        raise ValueError(
            f"Expected {len(SEEDS)} rows for "
            f"{dataset}/{strategy}/{policy}/{backend}, got {len(selected)}"
        )
    return selected


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / np.sqrt(array.size)),
    )


def _legend(fig: plt.Figure) -> Any:
    handles = [
        Line2D([0], [0], color=color, linewidth=2.0, label=label)
        for label, color in STRATEGIES.values()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=linestyle,
                marker="o",
                markerfacecolor="white" if open_marker else "#555555",
                markeredgecolor="#555555",
                label=label,
            )
            for label, _, linestyle, open_marker in POLICIES.values()
        ]
    )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="none",
                marker=marker,
                label=label,
            )
            for label, _, marker in BACKENDS.values()
        ]
    )
    return fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        handlelength=1.9,
        columnspacing=0.8,
        handletextpad=0.35,
    )


def render(rows: list[dict[str, Any]], out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    all_bounds: list[tuple[float, float]] = []

    for panel, (ax, dataset) in enumerate(
        zip(axes, DATASET_ORDER, strict=True)
    ):
        for strategy, (_, color) in STRATEGIES.items():
            for policy, (_, _, linestyle, open_marker) in POLICIES.items():
                ode_rows = _group(
                    rows,
                    dataset=dataset,
                    strategy=strategy,
                    policy=policy,
                    backend="ode",
                )
                ode_by_seed = {
                    int(row["seed"]): float(row["heldout_pearson_delta"])
                    for row in ode_rows
                }
                points: list[tuple[float, float, float, str]] = []
                for backend, (_, _, marker) in BACKENDS.items():
                    selected = _group(
                        rows,
                        dataset=dataset,
                        strategy=strategy,
                        policy=policy,
                        backend=backend,
                    )
                    ratios = [
                        float(row["heldout_pearson_delta"])
                        / ode_by_seed[int(row["seed"])]
                        for row in selected
                    ]
                    mean, se = _mean_se(ratios)
                    points.append(
                        (
                            float(selected[0]["replay_field_evals"]),
                            mean,
                            se,
                            marker,
                        )
                    )
                    all_bounds.append((mean - se, mean + se))
                points.sort(key=lambda point: point[0])
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    zorder=2,
                )
                for x, mean, se, marker in points:
                    ax.errorbar(
                        x,
                        mean,
                        yerr=se,
                        color=color,
                        marker=marker,
                        markersize=7,
                        markerfacecolor="white" if open_marker else color,
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        linestyle="none",
                        elinewidth=1.0,
                        capsize=2.5,
                        zorder=3,
                    )
        ax.axhline(
            1.0,
            color="#666666",
            linestyle=(0, (2.0, 2.0)),
            linewidth=1.0,
            zorder=1,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            [1, 2, 4, 8, 16, 32],
            [r"$2^0$", r"$2^1$", r"$2^2$", r"$2^3$", r"$2^4$", r"$2^5$"],
        )
        ax.set_xlim(0.8, 36)
        ax.set_title(DATASET_LABELS[dataset], pad=5)
        ax.set_xlabel("Replay-field evaluations per update")
        if panel == 0:
            ax.set_ylabel(r"Held-out $\Delta$ ratio to ODE $\uparrow$")
        ax.grid(axis="y", color="0.87", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    lower = min(bound[0] for bound in all_bounds)
    upper = max(bound[1] for bound in all_bounds)
    padding = max(0.025, 0.12 * (upper - lower))
    axes[0].set_ylim(lower - padding, upper + padding)
    legend = _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.0)
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


def write_summary(rows: list[dict[str, Any]], out: Path) -> None:
    summaries: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for strategy in STRATEGIES:
            for policy in POLICIES:
                ode_rows = _group(
                    rows,
                    dataset=dataset,
                    strategy=strategy,
                    policy=policy,
                    backend="ode",
                )
                ode_by_seed = {
                    int(row["seed"]): float(row["heldout_pearson_delta"])
                    for row in ode_rows
                }
                for backend in BACKENDS:
                    selected = _group(
                        rows,
                        dataset=dataset,
                        strategy=strategy,
                        policy=policy,
                        backend=backend,
                    )
                    ratios = [
                        float(row["heldout_pearson_delta"])
                        / ode_by_seed[int(row["seed"])]
                        for row in selected
                    ]
                    mean, se = _mean_se(ratios)
                    summaries.append(
                        {
                            "dataset": dataset,
                            "strategy": strategy,
                            "policy": policy,
                            "backend": backend,
                            "replay_field_evals": selected[0][
                                "replay_field_evals"
                            ],
                            "ratio_mean": mean,
                            "ratio_se": se,
                        }
                    )
    payload = {
        "metric": "heldout Pearson Delta ratio to seed-matched ODE",
        "higher_is_better": True,
        "seeds": list(SEEDS),
        "evaluation": {
            "generated_cells_per_non_control_stratum": 500,
            "error_bars": "standard error of paired seed-level ratios",
        },
        "summary": summaries,
        "seed_level_rows": rows,
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
    rows = collect()
    out = args.out_dir / "tx_backend_relative_pearson_vs_replay_evals.png"
    render(rows, out)
    write_summary(
        rows,
        args.out_dir / "tx_backend_relative_pearson_vs_replay_evals.json",
    )
    print(out)


if __name__ == "__main__":
    main()
