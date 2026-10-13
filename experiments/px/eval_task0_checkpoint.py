#!/usr/bin/env python3
"""Evaluate a reusable task-0 PX checkpoint on the first experiment's test split.

The continual-learning runs reuse a shared task-0 checkpoint and start at task 1,
so no stage-0 FID is recorded. This reproduces the training-time evaluation for
that checkpoint: EMA weights, dopri5 sampling, and eval seed ``--eval-seed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.px.cl_dataset import build_experiment_tasks
from experiments.px.cl_evaluation import (
    aggregate_experiment_metrics,
    evaluate_experiment,
)
from experiments.px.dataset import (
    IMG_SIZE,
    METADATA_PATH,
    NUM_CELL_TYPES,
    NUM_CHANNELS,
    NUM_EXPERIMENTS,
    NUM_PERTURBATIONS,
)
from experiments.px.main import MODEL_NAMES
from experiments.px.model import get_model_cls

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Task-0 TrainState checkpoint, e.g. .../cl_final/task0_s42/checkpoint.ckpt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON; defaults to eval_task0.json beside the checkpoint.",
    )
    parser.add_argument("--model-name", choices=MODEL_NAMES, default="ADM")
    parser.add_argument("--metadata", type=Path, default=Path(METADATA_PATH))
    parser.add_argument("--eval-seed", type=int, default=2027)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def load_ema_model(
    checkpoint: Path,
    *,
    model_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    model = get_model_cls(model_name)(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=NUM_CELL_TYPES,
    )
    ema_state = {
        key.removeprefix("module."): value for key, value in state["ema"].items()
    }
    model.load_state_dict(ema_state)
    return model.to(device).eval(), int(state["global_step"])


def main() -> None:
    args = parse_args()
    out = args.out or args.checkpoint.parent / "eval_task0.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    tasks = build_experiment_tasks(
        metadata_path=args.metadata,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )[: args.max_tasks]
    task = tasks[0]

    model, global_step = load_ema_model(
        args.checkpoint,
        model_name=args.model_name,
        device=device,
    )
    row = evaluate_experiment(
        model=model,
        task=task,
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.eval_seed,
        return_features=True,
    )
    # With one experiment seen, the pooled stream is that experiment alone.
    stream = aggregate_experiment_metrics(
        [row],
        evaluation_name="seen-experiment stream",
    )
    row.pop("_features")

    payload = {
        "task_id": 0,
        "batch_name": task.batch_name,
        "checkpoint": str(args.checkpoint),
        "global_step": global_step,
        "test_by_experiment": [row],
        "test_stream": stream,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{task.batch_name} FID={row['fid']:.2f} FDD={row['fdd_dinov2']:.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
