#!/usr/bin/env python3
"""Status inventory for the image-CL LoRA / gradual-finetuning campaign.

Prints one row per planned run with its progress, queue state, and the GPU time
still owed, so the remaining work can be prioritized.

    python experiments/image_cl/inventory_lora_gft.py
    python experiments/image_cl/inventory_lora_gft.py --datasets fmnist cifar10
"""

from __future__ import annotations

import argparse
import getpass
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results_dir"

# Measured from the naive runs: hours per sequential update, and the number of
# updates after the shared task 0.
DATASETS: dict[str, dict[str, float | int | str]] = {
    "fmnist": {"cli": "split_fmnist", "updates": 4, "hours_per_update": 0.075, "eval_hours": 0.05},
    "cifar10": {"cli": "split_cifar10", "updates": 4, "hours_per_update": 0.318, "eval_hours": 0.21},
    "imagenet100": {"cli": "split_imagenet100", "updates": 2, "hours_per_update": 4.11, "eval_hours": 0.46},
}
# LoRA trains the same activations as naive; GFT adds one frozen teacher forward.
COST_MULTIPLIER = {"lora": 1.0, "gft": 1.2}
DEFAULT_RANKS = (8, 16, 32)
DEFAULT_SCHEDULES = ("sigmoid", "linear")
DEFAULT_SEEDS = (42, 69, 1714)
# Tier 1 selects the config, tier 2 carries it to the expensive dataset.
TIER = {"fmnist": 1, "cifar10": 1, "imagenet100": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    p.add_argument("--schedules", nargs="+", default=list(DEFAULT_SCHEDULES))
    p.add_argument(
        "--imagenet-config",
        nargs="+",
        default=None,
        help="Configs run on ImageNet-100. Defaults to the same grid as tier 1.",
    )
    return p.parse_args()


def squeue_states() -> dict[str, str]:
    """Map job name to queue state for the current user."""
    try:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j|%T"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except Exception:
        return {}
    states: dict[str, str] = {}
    for line in out.splitlines():
        if "|" in line:
            name, state = line.rsplit("|", 1)
            states[name.strip()] = state.strip()
    return states


def n_tasks_done(run_dir: Path) -> int:
    tasks = run_dir / "tasks"
    if not tasks.is_dir():
        return 0
    return sum(
        1
        for d in sorted(tasks.iterdir())
        if (d / "metrics.json").is_file() and (d / "checkpoint.pt").is_file()
    )


def job_name(stem: str, slug: str, seed: int) -> str:
    if stem.startswith("lora_r"):
        return f"icl-lora{stem.removeprefix('lora_r')}-{slug[:4]}-s{seed}"
    schedule = stem.removeprefix("gft_")
    return f"icl-gft-{schedule[:3]}-{slug[:4]}-s{seed}"


def main() -> None:
    args = parse_args()
    queued = squeue_states()

    rows = []
    for slug in args.datasets:
        spec = DATASETS[slug]
        stems = [f"lora_r{r}" for r in args.ranks] + [f"gft_{s}" for s in args.schedules]
        if TIER[slug] == 2 and args.imagenet_config is not None:
            stems = list(args.imagenet_config)
        for seed in args.seeds:
            for stem in stems:
                strategy = "lora" if stem.startswith("lora") else "gft"
                run_dir = RESULTS / "image_cl" / slug / f"{stem}_s{seed}"
                done_tasks = n_tasks_done(run_dir)
                complete = (run_dir / "metrics.json").is_file()
                state = queued.get(job_name(stem, slug, seed))
                if complete:
                    status = "done"
                elif state:
                    status = state.lower()
                elif done_tasks > 1:
                    status = "partial"
                else:
                    status = "todo"
                total = int(spec["updates"]) + 1
                # Task 0 is reused, so only the updates are ever charged.
                remaining_updates = 0 if complete else max(0, total - max(done_tasks, 1))
                hours = remaining_updates * float(spec["hours_per_update"]) * COST_MULTIPLIER[strategy]
                if not complete:
                    hours += float(spec["eval_hours"])
                rows.append(
                    {
                        "tier": TIER[slug],
                        "dataset": slug,
                        "config": stem,
                        "seed": seed,
                        "status": status,
                        "tasks": f"{done_tasks}/{total}",
                        "hours": hours,
                    }
                )

    order = {"todo": 0, "partial": 1, "pending": 2, "running": 3, "done": 4}
    rows.sort(key=lambda r: (r["tier"], order.get(r["status"], 9), r["dataset"], r["config"], r["seed"]))

    print(f"{'tier':<5}{'dataset':<13}{'config':<14}{'seed':<7}{'status':<10}{'tasks':<8}{'GPU-h left':>10}")
    print("-" * 67)
    for r in rows:
        print(
            f"{r['tier']:<5}{r['dataset']:<13}{r['config']:<14}{r['seed']:<7}"
            f"{r['status']:<10}{r['tasks']:<8}{r['hours']:>10.2f}"
        )

    print("-" * 67)
    for tier in sorted({r["tier"] for r in rows}):
        block = [r for r in rows if r["tier"] == tier]
        left = [r for r in block if r["status"] != "done"]
        print(
            f"tier {tier}: {len(block)} runs, {len(left)} outstanding, "
            f"{sum(r['hours'] for r in left):.1f} GPU-h remaining"
        )
    outstanding = [r for r in rows if r["status"] != "done"]
    print(
        f"total: {len(rows)} runs, {len(outstanding)} outstanding, "
        f"{sum(r['hours'] for r in outstanding):.1f} GPU-h remaining"
    )
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("by status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))


if __name__ == "__main__":
    main()
