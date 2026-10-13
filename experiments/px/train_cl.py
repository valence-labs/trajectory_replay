#!/usr/bin/env python3
"""Experiment-incremental continual learning on RxRx1."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import time
from typing import Any

import submitit
import polars as pl
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments.px.cl_dataset import (
    build_experiment_tasks,
    concatenate_task_frames,
    make_loader,
)
from experiments.px.cl_evaluation import (
    aggregate_experiment_metrics,
    evaluate_experiment,
)
from experiments.px.cl_trainer import (
    CLStrategy,
    equal_task_lambda,
    train_cl_task,
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
from experiments.px.mean_flow import train_task0_mean_flow
from experiments.px.model import ADMMeanFlowWrapper, get_model_cls
from experiments.px.trainer import TrainState
from experiments.px.utils.distributed import Distributed
from experiments.px.utils.ema import KarrasEMA
from experiments.px.utils.evaluation import FeatureExtractor
from experiments.px.utils.launcher import launch
from experiments.px.utils.name_run import generate_random_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("phenomics continual learning")
    group.add_argument(
        "--strategy",
        choices=("naive", "cumulative", "real", "generative", "trajectory"),
        default="trajectory",
    )
    group.add_argument("--model-name", choices=MODEL_NAMES, default="ADM")
    group.add_argument("--metadata", type=pathlib.Path, default=pathlib.Path(METADATA_PATH))
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--eval-seed", type=int, default=2027)
    group.add_argument("--split-seed", type=int, default=0)
    group.add_argument("--val-fraction", type=float, default=0.1)
    group.add_argument("--steps-per-task", type=int, default=25_000)
    group.add_argument("--max-tasks", type=int, default=3)
    group.add_argument("--batch-size", type=int, default=64)
    group.add_argument("--eval-batch-size", type=int, default=16)
    group.add_argument("--num-workers", type=int, default=8)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max-grad-norm", type=float, default=0.5)
    group.add_argument("--replay-step-size", type=float, default=0.05)
    group.add_argument(
        "--replay-backend",
        choices=("ode", "mean_flow"),
        default="ode",
    )
    group.add_argument("--mean-flow-n-steps", type=int, choices=(1, 4), default=1)
    group.add_argument("--mean-flow-train-steps", type=int, default=None)
    group.add_argument("--mean-flow-batch-size", type=int, default=16)
    group.add_argument("--mean-flow-lr", type=float, default=1e-4)
    group.add_argument("--mean-flow-diag-fraction", type=float, default=0.25)
    group.add_argument(
        "--mean-flow-time-sampler",
        choices=("uniform", "logit_normal"),
        default="logit_normal",
    )
    group.add_argument("--mean-flow-time-mu", type=float, default=2.0)
    group.add_argument("--mean-flow-time-sigma", type=float, default=2.0)
    group.add_argument(
        "--lambda-mode",
        choices=("equal", "constant", "grid", "schedule"),
        default="equal",
    )
    group.add_argument("--lambda-constant", type=float, default=0.5)
    group.add_argument(
        "--lambda-schedule",
        type=float,
        nargs="+",
        default=None,
        help="Fixed λ values for T1,T2; used with --lambda-mode schedule.",
    )
    group.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 0.75),
    )
    group.add_argument(
        "--lambda-eval-step-size",
        type=float,
        default=0.05,
        help="Euler step size for validation-FID lambda selection.",
    )
    group.add_argument(
        "--eval-fid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument("--no-compile", action="store_true")
    group.add_argument(
        "--task0-checkpoint",
        type=pathlib.Path,
        default=None,
        help="Reuse a completed task-0 TrainState checkpoint and start at task 1.",
    )
    group.add_argument(
        "--task0-mean-flow",
        type=pathlib.Path,
        default=None,
        help="Task-0 MeanFlow state required with reused MeanFlow runs.",
    )
    group.add_argument(
        "--export-task0-dir",
        type=pathlib.Path,
        default=None,
        help="Stable directory receiving reusable task-0 checkpoint artifacts.",
    )


def _dump(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _make_model(args: argparse.Namespace) -> torch.nn.Module:
    model_cls = get_model_cls(args.model_name)
    return model_cls(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=NUM_CELL_TYPES,
    )


def _make_mean_flow() -> ADMMeanFlowWrapper:
    return ADMMeanFlowWrapper(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=NUM_CELL_TYPES,
    )


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _restore_state(
    state: TrainState,
    *,
    model_state: dict[str, torch.Tensor],
    ema_state: dict[str, torch.Tensor],
    global_step: int,
) -> None:
    state.ddp.module.load_state_dict(model_state)
    state.ema.module.load_state_dict(ema_state)
    state.global_step = int(global_step)


def _broadcast_float(value: float | None, device: torch.device) -> float:
    tensor = torch.tensor(
        0.0 if value is None else float(value),
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return float(tensor.item())


def _broadcast_module(module: torch.nn.Module) -> None:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=0)


def _validation_fid(
    *,
    state: TrainState,
    tasks: list,
    through_task: int,
    args: argparse.Namespace,
    D: Distributed,
    detector: FeatureExtractor | None,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    score = None
    if D.rank == 0:
        assert detector is not None
        for task in tasks[: through_task + 1]:
            validation_frame = task.val.filter(
                pl.col("is_eval_pert") & (pl.col("ANNOT") == "treated")
            )
            row = evaluate_experiment(
                model=state.ema.module,
                task=task,
                frame=validation_frame,
                split_name="validation",
                device=D.device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                seed=args.eval_seed + 10_000 + through_task,
                detector=detector,
                sampler="euler",
                euler_step_size=args.lambda_eval_step_size,
            )
            rows.append(row)
        score = float(sum(row["fid"] for row in rows) / len(rows))
    score = _broadcast_float(score, D.device)
    D.barrier()
    return score, rows


def _validate(args: argparse.Namespace) -> None:
    if args.steps_per_task < 1 or args.batch_size < 2:
        raise ValueError("steps-per-task must be positive and batch-size >= 2")
    if not (1 <= args.max_tasks <= 3):
        raise ValueError("max-tasks must be in [1,3]")
    if args.replay_step_size <= 0.0:
        raise ValueError("replay-step-size must be positive")
    if args.mean_flow_train_steps is not None and args.mean_flow_train_steps < 1:
        raise ValueError("mean-flow-train-steps must be positive")
    if args.mean_flow_batch_size < 2:
        raise ValueError("mean-flow-batch-size must be >= 2")
    if not (0.0 <= args.mean_flow_diag_fraction <= 1.0):
        raise ValueError("mean-flow-diag-fraction must be in [0,1]")
    if args.replay_backend == "mean_flow" and args.model_name != "ADM":
        raise ValueError("PX MeanFlow replay currently requires model-name ADM")
    if (
        args.task0_checkpoint is not None
        and args.replay_backend == "mean_flow"
        and args.strategy in {"generative", "trajectory"}
        and args.task0_mean_flow is None
    ):
        raise ValueError("reused MeanFlow runs require --task0-mean-flow")
    if args.lambda_mode == "constant" and not (
        0.0 < args.lambda_constant < 1.0
    ):
        raise ValueError("lambda-constant must be in (0,1)")
    if args.lambda_mode == "grid" and (
        not args.lambda_grid
        or any(not (0.0 < value < 1.0) for value in args.lambda_grid)
    ):
        raise ValueError("all lambda-grid values must be in (0,1)")
    if args.lambda_mode == "schedule" and (
        args.lambda_schedule is None
        or len(args.lambda_schedule) < args.max_tasks - 1
        or any(not (0.0 < value < 1.0) for value in args.lambda_schedule)
    ):
        raise ValueError(
            "lambda-schedule requires one value in (0,1) for every update task"
        )


def main(args: argparse.Namespace) -> None:
    _validate(args)
    with Distributed() as D:
        job_env = submitit.JobEnvironment()
        output_dir = pathlib.Path(args.launch_dir) / job_env.job_id
        run_name = generate_random_name(str(output_dir))
        if D.rank == 0:
            wandb.init(
                name=run_name,
                id=run_name,
                dir=str(output_dir),
                project=os.environ.get("WANDB_PROJECT", "trajectory-replay-px-cl"),
                resume="allow",
                save_code=False,
                config=vars(args),
            )

        torch.manual_seed(args.seed + D.rank)
        torch.cuda.manual_seed(args.seed + D.rank)
        torch.set_float32_matmul_precision("medium")
        tasks = build_experiment_tasks(
            metadata_path=args.metadata,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
        )[: args.max_tasks]

        net = _make_model(args).to(D.device)
        n_params = sum(parameter.numel() for parameter in net.parameters())
        if not args.no_compile:
            net.compile()
        state = TrainState(
            ddp=DDP(net),
            ema=KarrasEMA(net),
            opt=torch.optim.Adam(net.parameters(), lr=args.lr),
            global_step=0,
        )
        start_task = 0
        replay_mean_flow: ADMMeanFlowWrapper | None = None
        if args.task0_checkpoint is not None:
            state.load_ckpt(str(args.task0_checkpoint), D.device)
            start_task = 1
            if args.task0_mean_flow is not None:
                replay_mean_flow = _make_mean_flow().to(D.device)
                replay_mean_flow.load_state_dict(
                    torch.load(
                        args.task0_mean_flow,
                        weights_only=True,
                        map_location=D.device,
                    )
                )
                replay_mean_flow.eval().requires_grad_(False)

        config = {
            **vars(args),
            "metadata": str(args.metadata),
            "task0_checkpoint": (
                str(args.task0_checkpoint)
                if args.task0_checkpoint is not None
                else None
            ),
            "task0_mean_flow": (
                str(args.task0_mean_flow)
                if args.task0_mean_flow is not None
                else None
            ),
            "export_task0_dir": (
                str(args.export_task0_dir)
                if args.export_task0_dir is not None
                else None
            ),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "model_parameters": n_params,
            "task_order": [
                {
                    "task_id": task.task_id,
                    "experiment_id": task.experiment_id,
                    "batch_name": task.batch_name,
                    "n_train": task.train.height,
                    "n_val": task.val.height,
                    "n_test": task.test.height,
                }
                for task in tasks
            ],
            "flow": "noise_to_data",
            "interpolant": "linear",
            "minibatch_ot": False,
            "conditioning": {
                "perturbation": True,
                "experiment": True,
                "cell_type": False,
            },
        }
        if D.rank == 0:
            _dump(output_dir / "config.json", config)
        D.barrier()

        metrics_by_task: list[dict[str, Any]] = []
        if start_task == 1:
            metrics_by_task.append(
                {
                    "task_id": 0,
                    "batch_name": tasks[0].batch_name,
                    "reused_checkpoint": str(args.task0_checkpoint),
                    "test_by_experiment": [],
                }
            )
        started = time.time()
        for task in tasks[start_task:]:
            task_id = task.task_id
            effective_strategy: CLStrategy = (
                "naive" if task_id == 0 else args.strategy
            )
            if effective_strategy == "cumulative":
                train_frame = concatenate_task_frames(
                    tasks,
                    through_task=task_id,
                    split="train",
                )
            else:
                train_frame = task.train

            old_frames = [old_task.train for old_task in tasks[:task_id]]
            old_frame = (
                concatenate_task_frames(
                    tasks,
                    through_task=task_id - 1,
                    split="train",
                )
                if effective_strategy == "real"
                else None
            )

            def loaders() -> tuple[Any, Any]:
                current_loader = make_loader(
                    train_frame,
                    train=True,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                old_loader = (
                    make_loader(
                        old_frame,
                        train=True,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                    if old_frame is not None
                    else None
                )
                return current_loader, old_loader

            teacher = None
            if effective_strategy in {"generative", "trajectory"}:
                teacher = _make_model(args).to(D.device)
                teacher.load_state_dict(state.ema.module.state_dict())
                teacher.eval().requires_grad_(False)
                if not args.no_compile:
                    teacher.compile()
            joint_mean_flow = None
            base_mean_flow_state = None
            if (
                task_id > 0
                and args.replay_backend == "mean_flow"
                and effective_strategy in {"generative", "trajectory"}
            ):
                if replay_mean_flow is None:
                    raise RuntimeError("MeanFlow replay is missing from the previous task")
                replay_mean_flow.eval().requires_grad_(False)
                joint_mean_flow = _make_mean_flow().to(D.device)
                joint_mean_flow.load_state_dict(replay_mean_flow.state_dict())
                _broadcast_module(joint_mean_flow)
                base_mean_flow_state = _cpu_state(joint_mean_flow)

            uses_replay_lambda = task_id > 0 and effective_strategy in {
                "real",
                "generative",
                "trajectory",
            }
            lambda_candidates: list[float | None]
            if not uses_replay_lambda:
                lambda_candidates = [None]
            elif args.lambda_mode == "grid":
                lambda_candidates = [float(value) for value in args.lambda_grid]
            elif args.lambda_mode == "equal":
                lambda_candidates = [equal_task_lambda(task_id)]
            elif args.lambda_mode == "schedule":
                lambda_candidates = [
                    float(args.lambda_schedule[task_id - 1])
                ]
            else:
                lambda_candidates = [float(args.lambda_constant)]

            base_model_state = _cpu_state(state.ddp.module)
            base_ema_state = _cpu_state(state.ema.module)
            base_global_step = state.global_step
            best_score = float("inf")
            best_model_state = None
            best_ema_state = None
            best_mean_flow_state = None
            training = None
            chosen_lambda = lambda_candidates[0]
            candidate_rows: list[dict[str, Any]] = []
            detector = (
                FeatureExtractor(D.device)
                if args.lambda_mode == "grid"
                and uses_replay_lambda
                and D.rank == 0
                else None
            )

            for candidate_id, lambda_mix in enumerate(lambda_candidates):
                _restore_state(
                    state,
                    model_state=base_model_state,
                    ema_state=base_ema_state,
                    global_step=base_global_step,
                )
                if joint_mean_flow is not None:
                    assert base_mean_flow_state is not None
                    joint_mean_flow.load_state_dict(base_mean_flow_state)
                current_loader, old_loader = loaders()
                candidate_training = train_cl_task(
                    state=state,
                    strategy=effective_strategy,
                    current_loader=current_loader,
                    D=D,
                    num_steps=args.steps_per_task,
                    lr=args.lr,
                    seed=args.seed + 10_000 * task_id,
                    lambda_mix=lambda_mix,
                    teacher=teacher,
                    old_frames=old_frames,
                    old_loader=old_loader,
                    replay_step_size=args.replay_step_size,
                    max_grad_norm=args.max_grad_norm,
                    mean_flow=joint_mean_flow,
                    replay_mean_flow=(
                        replay_mean_flow
                        if args.replay_backend == "mean_flow"
                        else None
                    ),
                    mean_flow_n_steps=args.mean_flow_n_steps,
                    mean_flow_lr=args.mean_flow_lr,
                    mean_flow_diag_fraction=args.mean_flow_diag_fraction,
                    mean_flow_time_sampler=args.mean_flow_time_sampler,
                    mean_flow_time_mu=args.mean_flow_time_mu,
                    mean_flow_time_sigma=args.mean_flow_time_sigma,
                    mean_flow_batch_size=args.mean_flow_batch_size,
                )
                if len(lambda_candidates) > 1:
                    validation_score, validation_rows = _validation_fid(
                        state=state,
                        tasks=tasks,
                        through_task=task_id,
                        args=args,
                        D=D,
                        detector=detector,
                    )
                else:
                    validation_score = 0.0
                    validation_rows = []
                candidate_row = {
                    "lambda": lambda_mix,
                    "validation_fid_macro": validation_score,
                    "validation_by_experiment": validation_rows,
                    "training": candidate_training,
                }
                candidate_rows.append(candidate_row)
                if validation_score < best_score:
                    best_score = validation_score
                    chosen_lambda = lambda_mix
                    training = candidate_training
                    best_model_state = _cpu_state(state.ddp.module)
                    best_ema_state = _cpu_state(state.ema.module)
                    best_mean_flow_state = (
                        _cpu_state(joint_mean_flow)
                        if joint_mean_flow is not None
                        else None
                    )

            assert (
                training is not None
                and best_model_state is not None
                and best_ema_state is not None
            )
            _restore_state(
                state,
                model_state=best_model_state,
                ema_state=best_ema_state,
                global_step=base_global_step + args.steps_per_task,
            )
            if joint_mean_flow is not None:
                assert best_mean_flow_state is not None
                joint_mean_flow.load_state_dict(best_mean_flow_state)
                replay_mean_flow = joint_mean_flow

            mean_flow_training = None
            if (
                task_id == 0
                and args.replay_backend == "mean_flow"
                and args.strategy in {"generative", "trajectory"}
            ):
                replay_mean_flow = _make_mean_flow().to(D.device)
                _broadcast_module(replay_mean_flow)
                mean_flow_loader = make_loader(
                    train_frame,
                    train=True,
                    batch_size=args.mean_flow_batch_size,
                    num_workers=args.num_workers,
                )
                mean_flow_training = train_task0_mean_flow(
                    mean_flow=replay_mean_flow,
                    loader=mean_flow_loader,
                    D=D,
                    num_steps=(
                        args.mean_flow_train_steps
                        if args.mean_flow_train_steps is not None
                        else args.steps_per_task
                    ),
                    lr=args.mean_flow_lr,
                    seed=args.seed + 70_000,
                    diag_fraction=args.mean_flow_diag_fraction,
                    time_sampler=args.mean_flow_time_sampler,
                    time_mu=args.mean_flow_time_mu,
                    time_sigma=args.mean_flow_time_sigma,
                )
            task_dir = output_dir / "tasks" / f"{task_id:02d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            state.save_ckpt(str(task_dir / "checkpoint.ckpt"))
            if D.rank == 0 and replay_mean_flow is not None:
                torch.save(
                    replay_mean_flow.state_dict(),
                    task_dir / "mean_flow.pt",
                )
            D.barrier()
            if task_id == 0 and args.export_task0_dir is not None and D.rank == 0:
                args.export_task0_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    task_dir / "checkpoint.ckpt",
                    args.export_task0_dir / "checkpoint.ckpt",
                )
                if replay_mean_flow is not None:
                    shutil.copy2(
                        task_dir / "mean_flow.pt",
                        args.export_task0_dir / "mean_flow.pt",
                    )
                _dump(
                    args.export_task0_dir / "provenance.json",
                    {
                        "source_run": str(output_dir),
                        "seed": args.seed,
                        "steps": args.steps_per_task,
                        "batch_size": args.batch_size,
                        "model_name": args.model_name,
                    },
                )
            D.barrier()

            test_metrics: list[dict[str, Any]] = []
            stream_test_metrics = None
            if args.eval_fid and D.rank == 0:
                test_detector = FeatureExtractor(D.device)
                for evaluated_task in tasks[: task_id + 1]:
                    metric_row = evaluate_experiment(
                            model=state.ema.module,
                            task=evaluated_task,
                            device=D.device,
                            batch_size=args.eval_batch_size,
                            num_workers=args.num_workers,
                            seed=args.eval_seed
                            + 100 * task_id
                            + evaluated_task.task_id,
                            detector=test_detector,
                            return_features=True,
                        )
                    test_metrics.append(metric_row)
                stream_test_metrics = aggregate_experiment_metrics(
                    test_metrics,
                    evaluation_name="seen-experiment stream",
                )
                for metric_row in test_metrics:
                    metric_row.pop("_features")
            D.barrier()
            task_payload = {
                "task_id": task_id,
                "batch_name": task.batch_name,
                "chosen_lambda": chosen_lambda,
                "lambda_candidates": candidate_rows,
                "training": training,
                "mean_flow_training": mean_flow_training,
                "test_by_experiment": test_metrics,
                "test_stream": stream_test_metrics,
            }
            if D.rank == 0:
                metrics_by_task.append(task_payload)
                _dump(task_dir / "metrics.json", task_payload)
                _dump(
                    output_dir / "metrics.json",
                    {
                        "metrics_by_task": metrics_by_task,
                        "elapsed_seconds": time.time() - started,
                    },
                )

        if D.rank == 0:
            wandb.finish()


if __name__ == "__main__":
    launch(main, add_args=add_args)
