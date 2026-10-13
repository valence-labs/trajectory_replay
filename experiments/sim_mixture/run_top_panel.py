#!/usr/bin/env python3
"""Top-panel dim=2 sweep: build data, train oracle FM on each λ mixture, pretrain, finetune grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.sim_mixture.common import DEFAULT_LAMBDAS, REPO_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["naive", "generative", "trajectory"],
        choices=("naive", "generative", "trajectory", "real"),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--mixture-epochs", type=int, default=200)
    p.add_argument("--finetune-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--samples-per-task", type=int, default=10_000)
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--skip-oracle-mixture", action="store_true")
    p.add_argument("--skip-pretrain", action="store_true")
    p.add_argument("--skip-finetune", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    py = sys.executable
    exp = Path("experiments/sim_mixture")

    if not args.skip_datasets:
        _run(
            [
                py,
                str(exp / "build_datasets.py"),
                "--dim",
                str(args.dim),
                "--samples-per-task",
                str(args.samples_per_task),
                "--lambdas",
                *[str(x) for x in args.lambdas],
            ]
        )

    if not args.skip_oracle_mixture:
        for lam in args.lambdas:
            _run(
                [
                    py,
                    str(exp / "train_on_mixture.py"),
                    "--dim",
                    str(args.dim),
                    "--lambda-mix",
                    str(lam),
                    "--epochs",
                    str(args.mixture_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ]
            )

    if not args.skip_pretrain:
        _run(
            [
                py,
                str(exp / "train_pretrain.py"),
                "--dim",
                str(args.dim),
                "--epochs",
                str(args.pretrain_epochs),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
        )

    if not args.skip_finetune:
        # Naive once (λ ignored).
        if "naive" in args.strategies:
            _run(
                [
                    py,
                    str(exp / "train_finetune.py"),
                    "--dim",
                    str(args.dim),
                    "--strategy",
                    "naive",
                    "--epochs",
                    str(args.finetune_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ]
            )
        for strategy in args.strategies:
            if strategy == "naive":
                continue
            for lam in args.lambdas:
                _run(
                    [
                        py,
                        str(exp / "train_finetune.py"),
                        "--dim",
                        str(args.dim),
                        "--strategy",
                        strategy,
                        "--lambda-mix",
                        str(lam),
                        "--epochs",
                        str(args.finetune_epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--device",
                        args.device,
                    ]
                )

    print("\nTop-panel sweep finished.")


if __name__ == "__main__":
    main()
