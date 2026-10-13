#!/usr/bin/env python3
"""Print forget/fit metrics for MNIST finetune runs under results_dir/mnist/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(run_dir: Path) -> dict | None:
    for name in ("metrics_eval.json", "metrics.json"):
        path = run_dir / name
        if path.is_file():
            with path.open() as f:
                return json.load(f)
    return None


def row_from_metrics(name: str, m: dict) -> dict:
    forget = m.get("eval_forget", {}).get("summary", m)
    fit = m.get("eval_fit", {}).get("summary", m)
    frac = m.get("generative_replay_fraction")
    return {
        "run": name,
        "replay_frac": frac,
        "forget_mse": m.get("eval_mse_forget", forget.get("mean_mse_to_true")),
        "forget_energy": forget.get("mean_energy"),
        "fit_mse": m.get("eval_mse_fit", fit.get("mean_mse_to_true")),
        "fit_energy": fit.get("mean_energy"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results_dir" / "mnist",
    )
    p.add_argument("--glob", default="mnist_ft_*")
    args = p.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(args.results_root.glob(args.glob)):
        if not run_dir.is_dir():
            continue
        m = load_metrics(run_dir)
        if m is None:
            continue
        rows.append(row_from_metrics(run_dir.name, m))

    if not rows:
        print(f"No metrics found under {args.results_root}/{args.glob}")
        return

    rows.sort(key=lambda r: (r.get("replay_frac") or 0, r["run"]))
    hdr = ("run", "r", "forget_mse", "forget_E", "fit_mse", "fit_E")
    print("\t".join(hdr))
    for r in rows:
        frac = r.get("replay_frac")
        frac_s = f"{frac:.2f}" if isinstance(frac, (int, float)) else "?"
        print(
            "\t".join(
                [
                    r["run"],
                    frac_s,
                    f"{r['forget_mse']:.4f}" if r["forget_mse"] is not None else "?",
                    f"{r['forget_energy']:.3f}" if r["forget_energy"] is not None else "?",
                    f"{r['fit_mse']:.4f}" if r["fit_mse"] is not None else "?",
                    f"{r['fit_energy']:.3f}" if r["fit_energy"] is not None else "?",
                ]
            )
        )


if __name__ == "__main__":
    main()
