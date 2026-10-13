#!/usr/bin/env python3
"""Re-evaluate existing sim_mixture checkpoints (adds mean_loglik / energy_mixture).

Example:
  uv run python experiments/sim_mixture/reeval_metrics.py --device cuda --epochs 400
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from trajectory_replay.io.results import dump_metrics
from trajectory_replay.utils.cli import resolve_device
from experiments.sim_mixture.common import (
    DEFAULT_LAMBDAS,
    REPO_ROOT,
    evaluate_model,
    load_checkpoint,
    load_mixture,
    load_tasks,
    mixture_data_root,
    mixture_runs_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--n-gen", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ode-step-size", type=float, default=0.05)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["trajectory", "generative", "oracle"],
        help="Which run families to re-evaluate.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _lam_from_name(name: str) -> float | None:
    m = re.search(r"lam(\d+)p(\d+)", name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"lam(\d+\.\d+)", name)
    if m:
        return float(m.group(1))
    return None


def _seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)$", name)
    return int(m.group(1)) if m else None


def _collect_runs(
    runs: Path,
    *,
    methods: list[str],
    epochs: int,
    seeds: set[int],
) -> list[tuple[str, float, int, Path]]:
    out: list[tuple[str, float, int, Path]] = []
    ft = runs / "finetune"
    if ft.is_dir():
        for d in sorted(ft.iterdir()):
            if not d.is_dir() or not (d / "checkpoint.pt").exists():
                continue
            name = d.name
            if f"_e{epochs}_" not in name:
                continue
            seed = _seed_from_name(name)
            if seed is None or seed not in seeds:
                continue
            for method in ("trajectory", "generative"):
                if method not in methods:
                    continue
                if f"ft_{method}_" not in name:
                    continue
                lam = _lam_from_name(name)
                if lam is None:
                    continue
                out.append((method, lam, seed, d))
    if "oracle" in methods:
        od = runs / "oracle_mixture"
        if od.is_dir():
            for d in sorted(od.iterdir()):
                if not d.is_dir() or not (d / "checkpoint.pt").exists():
                    continue
                name = d.name
                if f"_e{epochs}_" not in name:
                    continue
                seed = _seed_from_name(name)
                if seed is None or seed not in seeds:
                    continue
                lam = _lam_from_name(name)
                if lam is None:
                    continue
                out.append(("oracle", lam, seed, d))
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = mixture_data_root(REPO_ROOT, dim=args.dim)
    runs = mixture_runs_root(REPO_ROOT, dim=args.dim)
    ds, meta = load_tasks(data_dir)
    sigma = float(meta["sigma"])
    lambdas = tuple(float(x) for x in meta.get("lambdas", DEFAULT_LAMBDAS))
    mix_cache = {lam: load_mixture(data_dir, lam).x for lam in lambdas}

    jobs = _collect_runs(
        runs,
        methods=list(args.methods),
        epochs=args.epochs,
        seeds=set(args.seeds),
    )
    print(f"Re-eval {len(jobs)} runs on {device} (epochs={args.epochs})")
    for method, lam, seed, d in jobs:
        eval_seed = seed + (321 if method == "oracle" else 999)
        print(f"  {method} λ={lam} seed={seed} <- {d.name}")
        if args.dry_run:
            continue
        model, _ = load_checkpoint(d / "checkpoint.pt", device)
        metrics, _ = evaluate_model(
            model,
            ds,
            device=device,
            n_gen=args.n_gen,
            seed=eval_seed,
            lambda_target=lam,
            sigma=sigma,
            x_true_mixture=mix_cache[lam],
            ode_step_size=args.ode_step_size,
        )
        mpath = d / "metrics.json"
        payload: dict = {}
        if mpath.exists():
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        payload["eval"] = metrics
        dump_metrics(mpath, payload)
        print(
            f"    pi0={metrics['pi0_hat']:.3f}  "
            f"mean_loglik={metrics.get('mean_loglik')}  "
            f"energy_mix={metrics.get('energy_mixture')}  "
            f"energy_mix_active={metrics.get('energy_mixture_active')}"
        )


if __name__ == "__main__":
    main()
