#!/usr/bin/env python3
"""Matched Masip-style diffusion CL on the RxRx1 experiment stream."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time
from typing import Any

import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments.px.cl_dataset import (
    build_experiment_tasks,
    concatenate_task_frames,
    make_loader,
)
from experiments.px.cl_evaluation import aggregate_experiment_metrics
from experiments.px.masip_evaluation import evaluate_diffusion_experiment
from experiments.px.masip_trainer import MasipStrategy, train_masip_task
from experiments.px.model import ADMWrapper
from experiments.px.trainer import TrainState
from experiments.px.utils.distributed import Distributed
from experiments.px.utils.ema import KarrasEMA
from experiments.px.utils.evaluation import FeatureExtractor
from experiments.px.utils.launcher import launch
from experiments.px.utils.name_run import generate_random_name

from experiments.px.dataset import (
    IMG_SIZE,
    METADATA_PATH,
    NUM_CHANNELS,
    NUM_EXPERIMENTS,
    NUM_PERTURBATIONS,
)


def add_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("matched Masip diffusion CL")
    group.add_argument(
        "--strategy",
        choices=(
            "finetune",
            "cumulative",
            "generative_replay",
            "generative_distillation",
        ),
        default="generative_distillation",
    )
    group.add_argument("--metadata", type=pathlib.Path, default=pathlib.Path(METADATA_PATH))
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--eval-seed", type=int, default=2027)
    group.add_argument("--split-seed", type=int, default=0)
    group.add_argument("--val-fraction", type=float, default=0.1)
    group.add_argument("--steps-per-task", type=int, default=25_000)
    group.add_argument("--batch-size", type=int, default=64)
    group.add_argument("--eval-batch-size", type=int, default=16)
    group.add_argument("--num-workers", type=int, default=8)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max-grad-norm", type=float, default=1.0)
    group.add_argument("--replay-loss-scale", type=float, default=0.75)
    group.add_argument("--teacher-steps", type=int, default=20)
    group.add_argument("--eval-steps", type=int, default=50)
    group.add_argument("--max-tasks", type=int, default=3)
    group.add_argument(
        "--eval-fid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument("--no-compile", action="store_true")
    group.add_argument("--task0-checkpoint", type=pathlib.Path, default=None)
    group.add_argument("--export-task0-dir", type=pathlib.Path, default=None)


def dump(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def make_model() -> ADMWrapper:
    return ADMWrapper(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=1,
    )


def equal_lambda(task_id: int) -> float:
    return task_id / (task_id + 1.0)


def validate(args: argparse.Namespace) -> None:
    if args.steps_per_task < 1:
        raise ValueError("steps-per-task must be positive")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if not (1 <= args.max_tasks <= 3):
        raise ValueError("max-tasks must be between 1 and 3")
    if args.teacher_steps < 1 or args.eval_steps < 1:
        raise ValueError("DDIM step counts must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.replay_loss_scale <= 0:
        raise ValueError("replay-loss-scale must be positive")
    if args.task0_checkpoint is not None and not args.task0_checkpoint.is_file():
        raise FileNotFoundError(args.task0_checkpoint)


def main(args: argparse.Namespace) -> None:
    validate(args)
    with Distributed() as distributed:
        job = submitit.JobEnvironment()
        output = pathlib.Path(args.launch_dir) / job.job_id
        name = generate_random_name(str(output))
        if distributed.rank == 0:
            wandb.init(
                name=name,
                id=name,
                dir=str(output),
                project=os.environ.get("WANDB_PROJECT", "trajectory-replay-px-masip"),
                config=vars(args),
                resume="allow",
                save_code=False,
            )

        torch.manual_seed(args.seed + distributed.rank)
        torch.cuda.manual_seed(args.seed + distributed.rank)
        tasks = build_experiment_tasks(
            metadata_path=args.metadata,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
        )[: args.max_tasks]
        model = make_model().to(distributed.device)
        if not args.no_compile:
            model.compile()
        state = TrainState(
            ddp=DDP(model),
            ema=KarrasEMA(model),
            opt=torch.optim.Adam(model.parameters(), lr=args.lr),
            global_step=0,
        )
        start_task = 0
        if args.task0_checkpoint is not None:
            state.load_ckpt(str(args.task0_checkpoint), distributed.device)
            start_task = 1

        config = {
            **vars(args),
            "metadata": str(args.metadata),
            "task0_checkpoint": (
                str(args.task0_checkpoint)
                if args.task0_checkpoint is not None
                else None
            ),
            "export_task0_dir": (
                str(args.export_task0_dir)
                if args.export_task0_dir is not None
                else None
            ),
            "output_dir": str(output),
            "run_name": name,
            "model": "matched_config_b_adm",
            "objective": "ddim_epsilon_prediction",
            "conditioning": {
                "perturbation": True,
                "experiment": True,
                "cell_type": False,
            },
            "task_order": [task.batch_name for task in tasks],
        }
        if distributed.rank == 0:
            dump(output / "config.json", config)
        distributed.barrier()

        results: list[dict[str, Any]] = []
        started = time.time()
        for task in tasks[start_task:]:
            task_id = task.task_id
            effective: MasipStrategy = (
                "finetune" if task_id == 0 else args.strategy
            )
            frame = (
                concatenate_task_frames(
                    tasks,
                    through_task=task_id,
                    split="train",
                )
                if effective == "cumulative"
                else task.train
            )
            loader = make_loader(
                frame,
                train=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            teacher = None
            if effective in {
                "generative_replay",
                "generative_distillation",
            }:
                teacher = make_model().to(distributed.device)
                online = getattr(state.ddp.module, "_orig_mod", state.ddp.module)
                teacher.load_state_dict(online.state_dict())
                teacher.eval().requires_grad_(False)
            training = train_masip_task(
                state=state,
                strategy=effective,
                current_loader=loader,
                D=distributed,
                num_steps=args.steps_per_task,
                lr=args.lr,
                seed=args.seed + 10_000 * task_id,
                lambda_mix=equal_lambda(task_id) if task_id > 0 else None,
                replay_loss_scale=args.replay_loss_scale,
                teacher=teacher,
                old_frames=[old.train for old in tasks[:task_id]],
                teacher_steps=args.teacher_steps,
                max_grad_norm=args.max_grad_norm,
            )

            task_dir = output / "tasks" / f"{task_id:02d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            state.save_ckpt(str(task_dir / "checkpoint.ckpt"))
            distributed.barrier()
            if task_id == 0 and args.export_task0_dir is not None and distributed.rank == 0:
                args.export_task0_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    task_dir / "checkpoint.ckpt",
                    args.export_task0_dir / "checkpoint.ckpt",
                )
            distributed.barrier()

            test_rows: list[dict[str, Any]] = []
            stream = None
            if args.eval_fid and distributed.rank == 0:
                detector = FeatureExtractor(distributed.device)
                for old_task in tasks[: task_id + 1]:
                    test_rows.append(
                        evaluate_diffusion_experiment(
                            model=state.ema.module,
                            task=old_task,
                            device=distributed.device,
                            batch_size=args.eval_batch_size,
                            num_workers=args.num_workers,
                            seed=args.eval_seed + 100 * task_id + old_task.task_id,
                            num_steps=args.eval_steps,
                            detector=detector,
                            return_features=True,
                        )
                    )
                stream = aggregate_experiment_metrics(
                    test_rows,
                    evaluation_name="seen-experiment stream",
                )
                for row in test_rows:
                    row.pop("_features")
            distributed.barrier()
            result = {
                "task_id": task_id,
                "batch_name": task.batch_name,
                "training": training,
                "test_by_experiment": test_rows,
                "test_stream": stream,
            }
            if distributed.rank == 0:
                results.append(result)
                dump(task_dir / "metrics.json", result)
                dump(
                    output / "metrics.json",
                    {
                        "metrics_by_task": results,
                        "elapsed_seconds": time.time() - started,
                    },
                )

        if distributed.rank == 0:
            wandb.finish()


if __name__ == "__main__":
    launch(main, add_args=add_args)
