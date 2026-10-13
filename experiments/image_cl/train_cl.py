#!/usr/bin/env python3
"""Image continual CFM on Split-FMNIST/CIFAR10/ImageNet-100 with per-task λ via val FID.

Protocol (see paper §2.4 + Masip backbone matching):
  - Task 0: plain CFM on real data (unconditional, or class-conditional with CFG).
  - Tasks k>=1: from the shared previous checkpoint, try each λ in --lambda-grid;
    select the candidate with best macro-average validation FID over tasks 0..k;
    keep that model for the next transition.
  - Class-conditional FID (λ + reporting): for each task, generate with near-uniform
    labels over that task's classes, score one FID against that task's full val/test
    pool, then macro-average across tasks. Stream test FID uses balanced labels over
    all classes seen so far vs the concatenated test set.
  - Optional ``--replay-backend mean_flow``:
      * Task 0: train CFM, then post-hoc distill ``MF^0``.
      * Tasks k>=1: jointly train CFM + MeanFlow. Trajectory shares one
        ``MF^{k-1}`` rollout to ``r`` (``--mean-flow-steps`` jumps) for both
        CFM field matching and the MF identity; generative treats
        ``MF^{k-1}`` endpoints as data for both heads. Warm-start
        ``MF^k`` from ``MF^{k-1}``. FID/KLD still use ODE on the CFM UNet.
  - Test FID + stream KLD are reporting-only (never used for λ selection).
  - Resume: if ``tasks/XX/metrics.json`` already exists, skip that task and continue
    from the next incomplete one (loads CFM + optional MeanFlow checkpoints).
    Pass ``--no-resume`` to force a full restart.

Example:
  uv run python experiments/image_cl/train_cl.py \\
    --strategy trajectory --seed 42 --run-name image_cl/fmnist/traj_s42
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.image_cl.image_cl_common import (
    add_image_cl_data_args,
    add_image_cl_eval_args,
    add_image_cl_model_args,
    add_image_cl_ode_args,
    add_image_cl_train_args,
    apply_image_cl_dataset_defaults,
    build_image_cl_tasks,
    clone_train_state,
    image_cl_uses_hflip,
    default_image_cl_run_name,
    dump_json,
    equal_weight_lambda,
    image_cl_in_channels,
    image_cl_num_classes,
    image_cl_unet_kwargs,
    load_checkpoint,
    make_frozen_diffusers_copy,
    make_model,
    NO_REPLAY_STRATEGIES,
    normalize_image_cl_run_name,
    image_cl_kld_num_classes,
    resolve_cumulative_task_epochs,
    resolve_data_root,
    resolve_device,
    resolve_image_cl_kld_classifier,
    resolve_latent_dir,
    resolve_latent_to_pixels,
    run_results_dir,
    save_checkpoint,
    train_one_task,
)
from trajectory_replay.eval.image_fid import (
    fid_against_reals,
    generate_samples,
    macro_average_task_fid,
    mean_per_class_fid,
)
from trajectory_replay.eval.stream_kld import (
    ImageNet100KLDClassifier,
    stream_kld_from_model,
)
from trajectory_replay.model.diffusers_unet_fm import disable_fused_sdpa_if_needed
from trajectory_replay.model.image_mean_flow import (
    distill_mean_flow_from_image_teacher,
    load_spatial_mean_flow_replay,
    make_spatial_mean_flow_from_teacher,
    mean_flow_replay_fidelity,
)
from trajectory_replay.model.diffusers_unet_mean_flow import (
    save_diffusers_unet_mean_flow_checkpoint,
)
from trajectory_replay.training.mean_flow_distill import MEAN_FLOW_CKPT_NAME
from trajectory_replay.utils.timing import WallTimer, training_timing_record


def _joint_mf_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mf_clip = float(getattr(args, "mean_flow_max_grad_norm", 1.0) or 0.0)
    mf_lr = getattr(args, "mean_flow_lr", None)
    return {
        "mf_lr": float(mf_lr) if mf_lr is not None else None,
        "diag_fraction": float(args.mean_flow_diag_fraction),
        # Adaptive weighting stays off in the joint loss: it self-normalizes each
        # branch by its own per-sample error, so λ would stop mixing comparable
        # squared errors (tests/test_mean_flow_cl.py pins that identity).
        "adaptive_p": None,
        "time_sampler": str(args.mean_flow_time_sampler),
        "time_mu": float(args.mean_flow_time_mu),
        "time_sigma": float(args.mean_flow_time_sigma),
        "mf_max_grad_norm": mf_clip if mf_clip > 0.0 else None,
        "mf_use_adamw": bool(int(args.mean_flow_use_adamw)),
        "mf_betas": (0.9, float(args.mean_flow_adam_beta2)),
        "mf_weight_decay": float(args.mean_flow_weight_decay),
    }


def _init_joint_mean_flow(
    teacher: torch.nn.Module,
    replay_mean_flow: Any | None,
    *,
    device: torch.device,
) -> Any:
    """Warm-start from previous MeanFlow when available, else fresh from teacher."""
    if replay_mean_flow is not None:
        mf = copy.deepcopy(replay_mean_flow)
        mf.to(device)
    else:
        mf = make_spatial_mean_flow_from_teacher(teacher, device=device)
    mf.train()
    for p in mf.parameters():
        p.requires_grad_(True)
    return mf


def _save_joint_mean_flow(
    *,
    mean_flow: Any,
    teacher: torch.nn.Module,
    task_dir: Path,
    args: argparse.Namespace,
    train_seed: int,
    k: int,
    device: torch.device,
    class_conditional: bool,
    tasks: list,
) -> dict[str, Any]:
    mean_flow.eval()
    save_diffusers_unet_mean_flow_checkpoint(
        task_dir / MEAN_FLOW_CKPT_NAME,
        mean_flow,
        parent_checkpoint=str(task_dir / "checkpoint.pt"),
        extra={
            "joint_mean_flow": True,
            "diag_fraction": float(args.mean_flow_diag_fraction),
            "adaptive_p": None,
            "time_sampler": str(args.mean_flow_time_sampler),
            "time_mu": float(args.mean_flow_time_mu),
            "time_sigma": float(args.mean_flow_time_sigma),
            "mean_flow_steps": int(args.mean_flow_steps),
        },
    )
    distill_class_ids = (
        [c for i in range(k + 1) for c in tasks[i].classes] if class_conditional else None
    )
    mf_fidelity = mean_flow_replay_fidelity(
        mean_flow,
        teacher,
        device=device,
        n_steps=int(args.mean_flow_steps),
        ode_method=args.ode_method,
        ode_step_size=float(args.ode_step_size),
        seed=int(args.eval_seed) + 7000 + k,
        class_ids=distill_class_ids,
    )
    print(
        f"     joint MeanFlow fidelity ({args.mean_flow_steps}-step vs teacher ODE): "
        f"rel_err={mf_fidelity['rel_err_vs_teacher']:.4f} "
        f"std_ratio={mf_fidelity['std_ratio']:.3f}"
    )
    if mf_fidelity["rel_err_vs_teacher"] > 0.5:
        print(
            "     WARNING: MeanFlow replay samples deviate strongly from the "
            "teacher; replay is likely unusable"
        )
    return {
        "checkpoint": str(task_dir / MEAN_FLOW_CKPT_NAME),
        "replay_fidelity": mf_fidelity,
        "joint_mean_flow": True,
    }


def _task_is_complete(task_dir: Path, *, require_mean_flow: bool) -> bool:
    """A task is complete once reporting metrics are on disk (after train+eval[+MF])."""
    if not (task_dir / "metrics.json").is_file():
        return False
    if not (task_dir / "checkpoint.pt").is_file():
        return False
    if require_mean_flow and not (task_dir / MEAN_FLOW_CKPT_NAME).is_file():
        return False
    return True


def _load_resume_state(
    out_dir: Path,
    *,
    n_tasks: int,
    device: torch.device,
    use_mean_flow: bool,
) -> tuple[Any, Any, int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Load contiguous completed tasks from ``out_dir``, or ``None`` if none complete."""
    completed: list[int] = []
    for k in range(n_tasks):
        task_dir = out_dir / "tasks" / f"{k:02d}"
        # Task 0 has post-hoc MF distill; tasks k>=1 save jointly trained MF.
        if _task_is_complete(task_dir, require_mean_flow=use_mean_flow):
            completed.append(k)
        else:
            break
    if not completed:
        return None

    last = completed[-1]
    ckpt_path = out_dir / "tasks" / f"{last:02d}" / "checkpoint.pt"
    model, _cfg = load_checkpoint(ckpt_path, device)
    model.train()

    replay_mean_flow = None
    if use_mean_flow:
        mf_path = out_dir / "tasks" / f"{last:02d}" / MEAN_FLOW_CKPT_NAME
        if mf_path.is_file():
            replay_mean_flow = load_spatial_mean_flow_replay(mf_path, device)

    metrics_by_task: list[dict[str, Any]] = []
    lambda_trace: list[dict[str, Any]] = []
    timing_by_task: list[dict[str, Any]] = []
    for k in completed:
        task_dir = out_dir / "tasks" / f"{k:02d}"
        task_metrics = json.loads((task_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics_by_task.append(task_metrics)
        lambda_trace.append(
            {
                "task_id": k,
                "chosen_lambda": task_metrics.get("chosen_lambda"),
                "fid_stream_test": task_metrics.get("fid_stream_test"),
                "stream_kld": task_metrics.get("stream_kld"),
                "fid_per_class_mean": task_metrics.get("fid_per_class_mean"),
            }
        )
        timing_path = task_dir / "timing.json"
        if timing_path.is_file():
            timing_by_task.append(json.loads(timing_path.read_text(encoding="utf-8")))
        else:
            timing_by_task.append({"task_id": k, "resumed": True})

    start_task = last + 1
    print(
        f"Resume: loaded task {last:02d} checkpoint"
        f"{' + MeanFlow' if replay_mean_flow is not None else ''}; "
        f"skipping tasks 0..{last}, continuing at task {start_task}"
    )
    return model, replay_mean_flow, start_task, metrics_by_task, lambda_trace, timing_by_task


def _write_run_summary(
    *,
    out_dir: Path,
    base_cfg: dict[str, Any],
    metrics_by_task: list[dict[str, Any]],
    lambda_trace: list[dict[str, Any]],
    timing_by_task: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    """Write top-level metrics / λ-trace / timing aggregates for a finished run."""
    single_ft_walls = []
    for tblock in timing_by_task:
        if "finetune" in tblock:
            single_ft_walls.append(tblock["finetune"]["train_wall_seconds"])
        for c in tblock.get("lambda_candidates", []):
            single_ft_walls.append(c["train_wall_seconds"])
    distill_walls = [
        tblock["mean_flow_joint"]["wall_seconds"]
        for tblock in timing_by_task
        if "mean_flow_joint" in tblock
    ] + [
        tblock["mean_flow_distill"]["wall_seconds"]
        for tblock in timing_by_task
        if "mean_flow_distill" in tblock
    ]
    replay_rel_errs = [
        tblock.get("mean_flow_joint", tblock.get("mean_flow_distill", {})).get(
            "replay_fidelity", {}
        ).get("rel_err_vs_teacher")
        for tblock in timing_by_task
        if "mean_flow_joint" in tblock or "mean_flow_distill" in tblock
    ]
    replay_rel_errs = [x for x in replay_rel_errs if x is not None]
    summary = {
        **base_cfg,
        "elapsed_sec": elapsed,
        "lambda_trace": lambda_trace,
        "metrics_by_task": metrics_by_task,
        "timing_by_task": timing_by_task,
        "final_fid_stream_test": metrics_by_task[-1]["fid_stream_test"],
        "final_stream_kld": metrics_by_task[-1].get("stream_kld"),
        "timing_summary": {
            "total_wall_seconds": elapsed,
            "single_ft_train_wall_seconds_mean": (
                float(sum(single_ft_walls) / len(single_ft_walls)) if single_ft_walls else None
            ),
            "single_ft_train_wall_seconds_all": single_ft_walls,
            "mean_flow_distill_wall_seconds_mean": (
                float(sum(distill_walls) / len(distill_walls)) if distill_walls else None
            ),
            "mean_flow_distill_wall_seconds_all": distill_walls,
            "mean_flow_replay_rel_err_all": replay_rel_errs,
        },
    }
    dump_json(out_dir / "metrics.json", summary)
    dump_json(
        out_dir / "lambda_trace.json",
        {"seed": int(base_cfg["seed"]), "trace": lambda_trace},
    )
    dump_json(out_dir / "timing.json", summary["timing_summary"])
    print(f"\nDone in {elapsed / 60:.1f} min. Results: {out_dir}")
    print(f"Final stream test FID: {summary['final_fid_stream_test']:.3f}")
    if summary["final_stream_kld"] is not None:
        print(f"Final stream KLD: {summary['final_stream_kld']:.3f}")
    if single_ft_walls:
        print(
            f"Mean single-FT train wall: "
            f"{summary['timing_summary']['single_ft_train_wall_seconds_mean']:.1f}s"
        )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_image_cl_data_args(p)
    add_image_cl_model_args(p)
    add_image_cl_train_args(p)
    add_image_cl_ode_args(p)
    add_image_cl_eval_args(p)
    p.add_argument(
        "--run-name",
        default=None,
        help=(
            "Under results_dir/. Default: image_cl/<dataset-slug>/<strategy>[_mf][_eq]_s<seed>. "
            "Legacy image_cl/<leaf> names are rewritten under the dataset slug."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny epochs / FID samples for a pipeline smoke test.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing task metrics/checkpoints and retrain from task 0.",
    )
    return p.parse_args()


def _stream_test_fid(
    model: torch.nn.Module,
    tasks: list,
    up_to: int,
    *,
    device: torch.device,
    n_gen: int,
    batch_size: int,
    ode_method: str,
    ode_step_size: float,
    seed: int,
    to_pixels=None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    reals = torch.cat([tasks[i].x_test for i in range(up_to + 1)], dim=0)
    # Class-conditional: near-uniform labels over all classes seen so far, then
    # one pooled FID vs the concatenated test set (Masip-style stream metric).
    class_ids = None
    if bool(getattr(model, "use_conditioning", False)):
        class_ids = [int(c) for i in range(up_to + 1) for c in tasks[i].classes]
    return {
        "fid_stream_test": fid_against_reals(
            model,
            reals,
            device=device,
            n_gen=n_gen,
            batch_size=batch_size,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            seed=seed,
            to_pixels=to_pixels,
            class_ids=class_ids,
            guidance_scale=guidance_scale,
        ),
        "n_real_test": int(reals.shape[0]),
        "n_gen": int(n_gen),
    }


def main() -> None:
    args = parse_args()
    apply_image_cl_dataset_defaults(args)
    repo_root = Path(__file__).resolve().parents[2]
    device = resolve_device(args)
    train_seed = int(args.seed if args.train_seed is None else args.train_seed)
    # Seed before any model construction so initial weights are reproducible
    # across --seed values (must precede make_model() below).
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    disable_fused_sdpa_if_needed()

    epochs = 2 if args.smoke else int(args.epochs)
    distill_epochs = 2 if args.smoke else int(args.distill_epochs)
    fid_val = 256 if args.smoke else int(args.fid_val_samples)
    fid_test = 512 if args.smoke else int(args.fid_test_samples)
    lambda_grid = [float(x) for x in args.lambda_grid]
    lambda_mode = str(args.lambda_mode)
    if int(args.mean_flow_steps) < 1:
        raise ValueError(f"--mean-flow-steps must be >= 1, got {args.mean_flow_steps}")
    if args.strategy == "joint":
        args.strategy = "cumulative"
    class_conditional = bool(args.class_conditional)
    if class_conditional and args.strategy not in (
        "naive",
        "cumulative",
        "joint",
        "lora",
        "gft",
        "generative",
        "trajectory",
    ):
        raise ValueError(
            "--class-conditional currently supports "
            "--strategy naive|cumulative|lora|gft|generative|trajectory "
            f"(got {args.strategy!r})"
        )
    if args.strategy == "lora" and int(args.lora_rank) < 1:
        raise ValueError(f"--lora-rank must be >= 1, got {args.lora_rank}")
    # Stream KLD is a coverage metric for unconditional gens; with class-cond +
    # label-matched sampling it mostly reflects classifier agreement. Skip by default.
    if class_conditional and not bool(args.no_kld):
        print("class-conditional: disabling stream KLD (use --per-class-fid-samples instead)")
        args.no_kld = True
    num_classes = image_cl_num_classes(args.dataset, args.num_classes) if class_conditional else 0
    guidance_scale = float(args.guidance_scale) if class_conditional else None
    label_dropout = float(args.label_dropout)
    per_class_fid_samples = int(args.per_class_fid_samples)
    max_tasks = int(args.max_tasks) if args.max_tasks is not None else None
    if max_tasks is not None and max_tasks < 1:
        raise ValueError(f"--max-tasks must be >= 1, got {max_tasks}")
    use_mean_flow = (
        args.replay_backend == "mean_flow" and args.strategy in ("generative", "trajectory")
    )
    if args.strategy in ("generative", "trajectory") and args.replay_backend == "ode":
        print(
            f"ODE steps: replay={float(args.replay_ode_step_size):.6f} "
            f"(~{round(1.0 / float(args.replay_ode_step_size))} NFEs) | "
            f"eval={float(args.ode_step_size):.6f} "
            f"(~{round(1.0 / float(args.ode_step_size))} NFEs)"
        )

    if args.run_name is None:
        args.run_name = default_image_cl_run_name(
            args.dataset,
            args.strategy,
            int(args.seed),
            replay_backend=str(args.replay_backend),
            lambda_mode=str(args.lambda_mode),
            class_conditional=class_conditional,
            unet_size=str(getattr(args, "unet_size", "medium")),
            n_experiences=int(args.n_experiences),
            mean_flow_steps=int(args.mean_flow_steps),
            lora_rank=int(args.lora_rank),
            gft_beta_schedule=str(args.gft_beta_schedule),
        )
        if max_tasks is not None:
            # e.g. cumulative_cc_t1_s42 for a single-task pilot.
            parts = str(args.run_name).rsplit("_s", 1)
            if len(parts) == 2:
                args.run_name = f"{parts[0]}_t{max_tasks}_s{parts[1]}"
    else:
        args.run_name = normalize_image_cl_run_name(str(args.run_name), args.dataset)
    out_dir = run_results_dir(repo_root, args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks").mkdir(exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    kld_clf = resolve_image_cl_kld_classifier(args, repo_root, device)
    to_pixels = resolve_latent_to_pixels(args, device)
    kld_num_classes = image_cl_kld_num_classes(args.dataset)

    cumulative_schedule = str(args.cumulative_schedule)
    finetune_kwargs: dict[str, Any] = {
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_train_output_layer": bool(args.lora_train_output_layer),
        "gft_beta_start": float(args.gft_beta_start),
        "gft_beta_end": float(args.gft_beta_end),
        "gft_beta_schedule": str(args.gft_beta_schedule),
        "gft_sigmoid_midpoint": float(args.gft_sigmoid_midpoint),
        "gft_sigmoid_steepness": float(args.gft_sigmoid_steepness),
    }
    print(
        f"image_cl strategy={args.strategy} seed={args.seed} device={device} "
        f"epochs/task={epochs} λ-mode={lambda_mode} λ-grid={lambda_grid} "
        f"replay_backend={args.replay_backend} class_conditional={class_conditional}"
        + (
            f" cumulative_schedule={cumulative_schedule}"
            if args.strategy in ("cumulative", "joint")
            else ""
        )
    )
    tasks = build_image_cl_tasks(args, repo_root)
    if max_tasks is not None:
        if max_tasks > len(tasks):
            raise ValueError(
                f"--max-tasks={max_tasks} exceeds n_experiences={len(tasks)}"
            )
        tasks = tasks[:max_tasks]
        print(f"  (pilot) truncated to first {max_tasks} task(s)")
    for t in tasks:
        print(
            f"  task {t.task_id}: classes={list(t.classes)} "
            f"train={len(t.x_train)} val={len(t.x_val)} test={len(t.x_test)}"
        )

    steps_per_epoch = max(1, (len(tasks[0].x_train) + int(args.batch_size) - 1) // int(args.batch_size))

    base_cfg: dict[str, Any] = {
        "experiment": "image_cl",
        "dataset": args.dataset,
        "strategy": args.strategy,
        "seed": int(args.seed),
        "train_seed": train_seed,
        "eval_seed": int(args.eval_seed),
        "n_experiences": int(args.n_experiences),
        "data_root": (
            str(resolve_data_root(args, repo_root))
            if args.dataset != "split_imagenet100"
            else None
        ),
        "latent_dir": (
            str(resolve_latent_dir(args))
            if args.dataset == "split_imagenet100"
            else None
        ),
        "image_size": int(args.image_size),
        "in_channels": image_cl_in_channels(args.dataset),
        "unet_size": str(getattr(args, "unet_size", "medium")),
        "unet_kwargs": image_cl_unet_kwargs(
            args.dataset, unet_size=str(getattr(args, "unet_size", "medium"))
        ),
        "random_horizontal_flip": image_cl_uses_hflip(args.dataset),
        "latent_decode": to_pixels is not None,
        "vae_id": str(args.vae_id) if to_pixels is not None else None,
        "train_timesteps": int(args.train_timesteps),
        "backbone_type": "diffusers_unet2d",
        "use_conditioning": class_conditional,
        "num_classes": num_classes if class_conditional else None,
        "label_dropout": label_dropout if class_conditional else None,
        "guidance_scale": guidance_scale if class_conditional else None,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "max_grad_norm": (
            float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else None
        ),
        "epochs": epochs,
        "cumulative_schedule": (
            cumulative_schedule if args.strategy in ("cumulative", "joint") else None
        ),
        "lambda_mode": lambda_mode,
        "lambda_grid": lambda_grid if lambda_mode == "grid" else None,
        "lora_rank": int(args.lora_rank) if args.strategy == "lora" else None,
        "lora_alpha": float(args.lora_alpha) if args.strategy == "lora" else None,
        "lora_train_output_layer": (
            bool(args.lora_train_output_layer) if args.strategy == "lora" else None
        ),
        "gft_beta_start": (
            float(args.gft_beta_start) if args.strategy == "gft" else None
        ),
        "gft_beta_end": float(args.gft_beta_end) if args.strategy == "gft" else None,
        "gft_beta_schedule": (
            str(args.gft_beta_schedule) if args.strategy == "gft" else None
        ),
        "gft_sigmoid_midpoint": (
            float(args.gft_sigmoid_midpoint) if args.strategy == "gft" else None
        ),
        "gft_sigmoid_steepness": (
            float(args.gft_sigmoid_steepness) if args.strategy == "gft" else None
        ),
        "traj_kappa": args.traj_kappa if args.strategy == "trajectory" else None,
        "replay_backend": args.replay_backend,
        "mean_flow_steps": (
            int(args.mean_flow_steps) if use_mean_flow else None
        ),
        "joint_mean_flow": bool(use_mean_flow),
        "mean_flow_diag_fraction": (
            float(args.mean_flow_diag_fraction) if use_mean_flow else None
        ),
        "task0_mean_flow_adaptive_p": (
            float(args.mean_flow_adaptive_p) if use_mean_flow else None
        ),
        "joint_mean_flow_adaptive_p": None,
        "mean_flow_time_mu": float(args.mean_flow_time_mu) if use_mean_flow else None,
        "mean_flow_time_sigma": (
            float(args.mean_flow_time_sigma) if use_mean_flow else None
        ),
        "distill_epochs": int(distill_epochs) if use_mean_flow else None,
        "ode_method": args.ode_method,
        "ode_step_size": float(args.ode_step_size),
        "replay_ode_step_size": float(args.replay_ode_step_size),
        "val_fraction": float(args.val_fraction),
        "fid_val_samples": fid_val,
        "fid_test_samples": fid_test,
        "fid_batch_size": int(args.fid_batch_size),
        "per_class_fid_samples": per_class_fid_samples if per_class_fid_samples > 0 else None,
        "max_tasks": max_tasks,
        "kld_clf": (
            {
                "type": "resnet50",
                "weights": kld_clf.weights_name,
                "num_classes": int(kld_clf.num_classes),
            }
            if isinstance(kld_clf, ImageNet100KLDClassifier)
            else (
                {"type": "masip_resnet18", "num_classes": kld_num_classes}
                if kld_clf is not None
                else None
            )
        ),
        "kld_num_classes": kld_num_classes if kld_clf is not None else None,
        "task_classes": [list(t.classes) for t in tasks],
        "smoke": bool(args.smoke),
    }
    dump_json(out_dir / "config.json", base_cfg)

    replay_mean_flow = None  # DiffusersUNetMeanFlow from previous task
    lambda_trace: list[dict[str, Any]] = []
    metrics_by_task: list[dict[str, Any]] = []
    timing_by_task: list[dict[str, Any]] = []
    start_task = 0
    model = None
    if not args.no_resume:
        resumed = _load_resume_state(
            out_dir,
            n_tasks=len(tasks),
            device=device,
            use_mean_flow=use_mean_flow,
        )
        if resumed is not None:
            model, replay_mean_flow, start_task, metrics_by_task, lambda_trace, timing_by_task = (
                resumed
            )
    if model is None:
        model = make_model(args).to(device)
    t0 = time.time()
    if start_task >= len(tasks):
        print(f"Resume: all {len(tasks)} tasks already complete under {out_dir}")
        _write_run_summary(
            out_dir=out_dir,
            base_cfg=base_cfg,
            metrics_by_task=metrics_by_task,
            lambda_trace=lambda_trace,
            timing_by_task=timing_by_task,
            elapsed=0.0,
        )
        return

    for k, task in enumerate(tasks):
        if k < start_task:
            continue
        print(f"\n=== Task {k}/{len(tasks) - 1} classes={list(task.classes)} ===")
        task_dir = out_dir / "tasks" / f"{k:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_timing: dict[str, Any] = {"task_id": k}
        best_mf: Any | None = None
        best_history: list[dict[str, float]] = []

        if k == 0 or args.strategy in NO_REPLAY_STRATEGIES:
            chosen_lambda: float | None = None
            # cumulative/joint: CFM on the union of all real train data seen so far.
            if args.strategy in ("cumulative", "joint"):
                x_train = torch.cat([tasks[i].x_train for i in range(k + 1)], dim=0)
                y_train = (
                    torch.cat([tasks[i].y_train for i in range(k + 1)], dim=0)
                    if class_conditional
                    else None
                )
                ft_name = "cumulative"
            else:
                x_train = task.x_train
                y_train = task.y_train if class_conditional else None
                # lora/gft only differ from naive once there is a previous task.
                ft_name = "naive" if k == 0 else str(args.strategy)
            # GFT regularizes toward the field learned on the tasks seen so far.
            ft_teacher = (
                make_frozen_diffusers_copy(model) if ft_name == "gft" else None
            )
            steps_this = max(
                1, (len(x_train) + int(args.batch_size) - 1) // int(args.batch_size)
            )
            if ft_name == "cumulative":
                epochs_this, target_steps = resolve_cumulative_task_epochs(
                    base_epochs=epochs,
                    steps_per_epoch=steps_this,
                    reference_steps_per_epoch=steps_per_epoch,
                    cumulative_schedule=cumulative_schedule,
                )
                if cumulative_schedule == "steps" and epochs_this != epochs:
                    print(
                        f"  cumulative schedule=steps: epochs {epochs}→{epochs_this} "
                        f"(~{target_steps} Adam steps, {steps_this}/epoch)"
                    )
            else:
                epochs_this = epochs
                target_steps = epochs * steps_this
            with WallTimer() as t_train:
                history = train_one_task(
                    ft_name,
                    model,
                    ft_teacher,
                    x_train,
                    device=device,
                    epochs=epochs_this,
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    train_seed=train_seed + 1000 * k,
                    lambda_mix=None,
                    traj_kappa=args.traj_kappa,
                    ode_method=args.ode_method,
                    ode_step_size=float(args.ode_step_size),
                    ode_rtol=float(args.ode_rtol),
                    ode_atol=float(args.ode_atol),
                    random_horizontal_flip=image_cl_uses_hflip(args.dataset),
                    y_train=y_train,
                    label_dropout=label_dropout,
                    max_grad_norm=float(args.max_grad_norm),
                    **finetune_kwargs,
                )
            task_timing["finetune"] = training_timing_record(
                wall_seconds=t_train.elapsed_seconds,
                epochs=epochs_this,
                steps_per_epoch=steps_this,
                finetune_strategy=ft_name,
                replay_backend=None,
            )
            task_timing["finetune"]["target_train_steps"] = int(target_steps)
            if ft_name == "cumulative":
                task_timing["finetune"]["cumulative_schedule"] = cumulative_schedule
            dump_json(task_dir / "history.json", {"lambda": None, "history": history})
            cand_rows: list[dict[str, Any]] = []
        else:
            teacher = make_frozen_diffusers_copy(model)
            # Previous-task MF for replay (required after task-0 distill).
            mf_for_replay = replay_mean_flow if use_mean_flow else None
            if use_mean_flow and mf_for_replay is None:
                print(
                    "  WARNING: MeanFlow replay requested but MF^{k-1} is missing; "
                    "falling back to ODE Euler for the old branch"
                )
            mf_kwargs = _joint_mf_kwargs(args) if use_mean_flow else {}
            cand_rows: list[dict[str, Any]] = []
            cand_timings: list[dict[str, Any]] = []
            val_reals = [tasks[i].x_val for i in range(k + 1)]
            val_class_ids = (
                [list(tasks[i].classes) for i in range(k + 1)]
                if class_conditional
                else None
            )

            if lambda_mode == "equal":
                lam = equal_weight_lambda(k)
                print(f"  -- fixed equal-weight λ={lam:.6f} (n={k + 1} tasks)")
                if use_mean_flow:
                    print("  -- joint MeanFlow co-training enabled")
                student = clone_train_state(teacher, device)
                train_mf = (
                    _init_joint_mean_flow(teacher, mf_for_replay, device=device)
                    if use_mean_flow
                    else None
                )
                old_class_ids = (
                    [c for i in range(k) for c in tasks[i].classes]
                    if class_conditional
                    else None
                )
                with WallTimer() as t_train:
                    hist = train_one_task(
                        args.strategy,
                        student,
                        teacher,
                        task.x_train,
                        device=device,
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        train_seed=train_seed + 1000 * k + int(100 * lam),
                        lambda_mix=float(lam),
                        traj_kappa=args.traj_kappa,
                        ode_method=args.ode_method,
                        ode_step_size=float(args.replay_ode_step_size),
                        ode_rtol=float(args.ode_rtol),
                        ode_atol=float(args.ode_atol),
                        mean_flow=mf_for_replay,
                        mean_flow_n_steps=int(args.mean_flow_steps),
                        joint_mean_flow=train_mf,
                        random_horizontal_flip=image_cl_uses_hflip(args.dataset),
                        y_train=task.y_train if class_conditional else None,
                        old_class_ids=old_class_ids,
                        label_dropout=label_dropout,
                        max_grad_norm=float(args.max_grad_norm),
                        **mf_kwargs,
                    )
                with WallTimer() as t_val:
                    val_scores = macro_average_task_fid(
                        student,
                        val_reals,
                        device=device,
                        n_gen_per_task=fid_val,
                        batch_size=int(args.fid_batch_size),
                        ode_method=args.ode_method,
                        ode_step_size=float(args.ode_step_size),
                        seed=int(args.eval_seed) + 10 * k,
                        to_pixels=to_pixels,
                        task_class_ids=val_class_ids,
                        guidance_scale=guidance_scale,
                    )
                ft_timing = training_timing_record(
                    wall_seconds=t_train.elapsed_seconds,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    finetune_strategy=args.strategy,
                    replay_backend=args.replay_backend,
                    mean_flow_steps=int(args.mean_flow_steps) if use_mean_flow else None,
                )
                row = {
                    "lambda": float(lam),
                    "fid_macro_val": val_scores["fid_macro"],
                    "fid_per_task_val": val_scores["fid_per_task"],
                    "history_last_loss": hist[-1]["loss_mean"] if hist else None,
                    "train_wall_seconds": t_train.elapsed_seconds,
                    "val_fid_wall_seconds": t_val.elapsed_seconds,
                    "finetune": ft_timing,
                    "selection": "fixed_equal",
                }
                cand_rows.append(row)
                cand_timings.append(
                    {
                        "lambda": float(lam),
                        "train_wall_seconds": t_train.elapsed_seconds,
                        "val_fid_wall_seconds": t_val.elapsed_seconds,
                        "finetune": ft_timing,
                    }
                )
                model.load_state_dict(student.state_dict())
                best_mf = train_mf
                chosen_lambda = float(lam)
                best_history = hist
                dump_json(
                    task_dir / "lambda_sweep.json",
                    {
                        "lambda_mode": "equal",
                        "candidates": cand_rows,
                        "chosen_lambda": chosen_lambda,
                        "n_tasks_seen": k + 1,
                    },
                )
                dump_json(
                    task_dir / "history.json",
                    {"lambda": chosen_lambda, "history": best_history},
                )
                print(
                    f"  >> fixed λ={chosen_lambda:.6f}  "
                    f"val FID={val_scores['fid_macro']:.3f}  "
                    f"train={t_train.elapsed_seconds:.1f}s"
                )
            else:
                best_fid = float("inf")
                best_state: dict[str, torch.Tensor] | None = None
                best_lambda = lambda_grid[0]
                best_history = []

                for lam in lambda_grid:
                    print(f"  -- candidate λ={lam}")
                    if use_mean_flow:
                        print("     (joint MeanFlow)")
                    student = clone_train_state(teacher, device)
                    train_mf = (
                        _init_joint_mean_flow(teacher, mf_for_replay, device=device)
                        if use_mean_flow
                        else None
                    )
                    old_class_ids = (
                        [c for i in range(k) for c in tasks[i].classes]
                        if class_conditional
                        else None
                    )
                    with WallTimer() as t_train:
                        hist = train_one_task(
                            args.strategy,
                            student,
                            teacher,
                            task.x_train,
                            device=device,
                            epochs=epochs,
                            batch_size=int(args.batch_size),
                            lr=float(args.lr),
                            train_seed=train_seed + 1000 * k + int(100 * lam),
                            lambda_mix=float(lam),
                            traj_kappa=args.traj_kappa,
                            ode_method=args.ode_method,
                            ode_step_size=float(args.replay_ode_step_size),
                            ode_rtol=float(args.ode_rtol),
                            ode_atol=float(args.ode_atol),
                            mean_flow=mf_for_replay,
                            mean_flow_n_steps=int(args.mean_flow_steps),
                            joint_mean_flow=train_mf,
                            random_horizontal_flip=image_cl_uses_hflip(args.dataset),
                            y_train=task.y_train if class_conditional else None,
                            old_class_ids=old_class_ids,
                            label_dropout=label_dropout,
                            max_grad_norm=float(args.max_grad_norm),
                            **mf_kwargs,
                        )
                    with WallTimer() as t_val:
                        val_scores = macro_average_task_fid(
                            student,
                            val_reals,
                            device=device,
                            n_gen_per_task=fid_val,
                            batch_size=int(args.fid_batch_size),
                            ode_method=args.ode_method,
                            ode_step_size=float(args.ode_step_size),
                            seed=int(args.eval_seed) + 10 * k,
                            to_pixels=to_pixels,
                            task_class_ids=val_class_ids,
                            guidance_scale=guidance_scale,
                        )
                    ft_timing = training_timing_record(
                        wall_seconds=t_train.elapsed_seconds,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        finetune_strategy=args.strategy,
                        replay_backend=args.replay_backend,
                        mean_flow_steps=int(args.mean_flow_steps) if use_mean_flow else None,
                    )
                    row = {
                        "lambda": float(lam),
                        "fid_macro_val": val_scores["fid_macro"],
                        "fid_per_task_val": val_scores["fid_per_task"],
                        "history_last_loss": hist[-1]["loss_mean"] if hist else None,
                        "train_wall_seconds": t_train.elapsed_seconds,
                        "val_fid_wall_seconds": t_val.elapsed_seconds,
                        "finetune": ft_timing,
                        "selection": "grid",
                    }
                    cand_rows.append(row)
                    cand_timings.append(
                        {
                            "lambda": float(lam),
                            "train_wall_seconds": t_train.elapsed_seconds,
                            "val_fid_wall_seconds": t_val.elapsed_seconds,
                            "finetune": ft_timing,
                        }
                    )
                    print(
                        f"     val FID macro={val_scores['fid_macro']:.3f}  "
                        f"train={t_train.elapsed_seconds:.1f}s  valFID={t_val.elapsed_seconds:.1f}s"
                    )
                    if val_scores["fid_macro"] < best_fid:
                        best_fid = float(val_scores["fid_macro"])
                        best_lambda = float(lam)
                        best_state = copy.deepcopy(student.state_dict())
                        best_history = hist
                        best_mf = train_mf

                assert best_state is not None
                model.load_state_dict(best_state)
                chosen_lambda = best_lambda
                dump_json(
                    task_dir / "lambda_sweep.json",
                    {
                        "lambda_mode": "grid",
                        "candidates": cand_rows,
                        "chosen_lambda": chosen_lambda,
                    },
                )
                dump_json(
                    task_dir / "history.json",
                    {"lambda": chosen_lambda, "history": best_history},
                )
                print(f"  >> selected λ={chosen_lambda} (val FID={best_fid:.3f})")

            task_timing["lambda_mode"] = lambda_mode
            task_timing["lambda_candidates"] = cand_timings
            task_timing["chosen_lambda"] = chosen_lambda
            task_timing["chosen_train_wall_seconds"] = next(
                c["train_wall_seconds"] for c in cand_timings if c["lambda"] == chosen_lambda
            )

        ckpt_cfg = {**base_cfg, "task_id": k, "chosen_lambda": chosen_lambda}
        save_checkpoint(task_dir / "checkpoint.pt", model, ckpt_cfg)
        save_checkpoint(out_dir / "checkpoint.pt", model, ckpt_cfg)

        # Task 0 + MeanFlow: post-hoc distill MF^0 after CFM checkpoint is on disk.
        if use_mean_flow and k == 0:
            adapt = float(getattr(args, "mean_flow_adaptive_p", 0.0) or 0.0)
            mf_kw = _joint_mf_kwargs(args)
            distill_lr = float(
                args.mean_flow_lr if args.mean_flow_lr is not None else args.distill_lr
            )
            print(
                f"  -- task-0 data-conditioned MeanFlow "
                f"(OT v=x1-x0, epochs={distill_epochs}, "
                f"lr={distill_lr:g}, batch={int(args.distill_batch_size)}, "
                f"adaptive_p={adapt:g}, adamw={int(mf_kw['mf_use_adamw'])}, "
                f"clip={mf_kw['mf_max_grad_norm']})"
            )
            with WallTimer() as t_distill:
                best_mf, _, distill_hist = distill_mean_flow_from_image_teacher(
                    model,
                    task.x_train,
                    device=device,
                    out_dir=task_dir,
                    epochs=distill_epochs,
                    batch_size=int(args.distill_batch_size),
                    lr=distill_lr,
                    train_seed=train_seed + 7000,
                    y_train=task.y_train if class_conditional else None,
                    diag_fraction=float(args.mean_flow_diag_fraction),
                    adaptive_p=None if adapt <= 0.0 else adapt,
                    time_sampler=str(args.mean_flow_time_sampler),
                    time_mu=float(args.mean_flow_time_mu),
                    time_sigma=float(args.mean_flow_time_sigma),
                    parent_checkpoint=str(task_dir / "checkpoint.pt"),
                    unet_kwargs=image_cl_unet_kwargs(
                        args.dataset,
                        unet_size=str(getattr(args, "unet_size", "medium")),
                    ),
                    use_adamw=bool(mf_kw["mf_use_adamw"]),
                    betas=mf_kw["mf_betas"],
                    weight_decay=float(mf_kw["mf_weight_decay"]),
                    max_grad_norm=mf_kw["mf_max_grad_norm"],
                )
            distill_class_ids = list(task.classes) if class_conditional else None
            mf_fidelity = mean_flow_replay_fidelity(
                best_mf,
                model,
                device=device,
                n_steps=int(args.mean_flow_steps),
                ode_method=args.ode_method,
                ode_step_size=float(args.ode_step_size),
                seed=int(args.eval_seed) + 7000,
                class_ids=distill_class_ids,
            )
            print(
                f"     distill fidelity ({args.mean_flow_steps}-step vs teacher ODE): "
                f"rel_err={mf_fidelity['rel_err_vs_teacher']:.4f} "
                f"std_ratio={mf_fidelity['std_ratio']:.3f}"
            )
            replay_mean_flow = best_mf
            task_timing["mean_flow_distill"] = {
                "wall_seconds": t_distill.elapsed_seconds,
                "checkpoint": str(task_dir / MEAN_FLOW_CKPT_NAME),
                "replay_fidelity": mf_fidelity,
                "joint_mean_flow": False,
                "mean_flow_steps": int(args.mean_flow_steps),
            }
            dump_json(
                task_dir / "mean_flow_history.json",
                {
                    "history": distill_hist,
                    "joint_mean_flow": False,
                    "post_hoc_distill": True,
                    "velocity": "ot",
                },
            )
            print(f"     MeanFlow distill wall={t_distill.elapsed_seconds:.1f}s")

        # Persist jointly trained MeanFlow for next-task replay (tasks k>=1).
        if use_mean_flow and k >= 1:
            if best_mf is None:
                raise RuntimeError("joint MeanFlow training did not produce a MeanFlow")
            print("  -- saving jointly trained MeanFlow for next-task replay")
            with WallTimer() as t_mf:
                mf_info = _save_joint_mean_flow(
                    mean_flow=best_mf,
                    teacher=model,
                    task_dir=task_dir,
                    args=args,
                    train_seed=train_seed,
                    k=k,
                    device=device,
                    class_conditional=class_conditional,
                    tasks=tasks,
                )
            replay_mean_flow = best_mf
            task_timing["mean_flow_joint"] = {
                "wall_seconds": t_mf.elapsed_seconds,
                **mf_info,
            }
            dump_json(
                task_dir / "mean_flow_history.json",
                {"history": best_history, "joint_mean_flow": True},
            )
            print(f"     MeanFlow save wall={t_mf.elapsed_seconds:.1f}s")

        with WallTimer() as t_test:
            test_metrics = _stream_test_fid(
                model,
                tasks,
                k,
                device=device,
                n_gen=fid_test,
                batch_size=int(args.fid_batch_size),
                ode_method=args.ode_method,
                ode_step_size=float(args.ode_step_size),
                seed=int(args.eval_seed) + 1000 + k,
                to_pixels=to_pixels,
                guidance_scale=guidance_scale,
            )
            per_task_test = macro_average_task_fid(
                model,
                [tasks[i].x_test for i in range(k + 1)],
                device=device,
                n_gen_per_task=min(fid_test, 2048) if args.smoke else min(fid_test, 5000),
                batch_size=int(args.fid_batch_size),
                ode_method=args.ode_method,
                ode_step_size=float(args.ode_step_size),
                seed=int(args.eval_seed) + 2000 + k,
                to_pixels=to_pixels,
                task_class_ids=(
                    [list(tasks[i].classes) for i in range(k + 1)]
                    if class_conditional
                    else None
                ),
                guidance_scale=guidance_scale,
            )
        task_timing["test_fid_wall_seconds"] = t_test.elapsed_seconds

        kld_metrics: dict[str, Any] = {}
        if kld_clf is not None:
            with WallTimer() as t_kld:
                y_true = torch.cat([tasks[i].y_test for i in range(k + 1)], dim=0)
                kld_metrics = stream_kld_from_model(
                    model,
                    y_true,
                    clf=kld_clf,
                    device=device,
                    n_gen=fid_test,
                    batch_size=int(args.fid_batch_size),
                    ode_method=args.ode_method,
                    ode_step_size=float(args.ode_step_size),
                    seed=int(args.eval_seed) + 4000 + k,
                    num_classes=kld_num_classes,
                    to_pixels=to_pixels,
                    guidance_scale=guidance_scale,
                )
            task_timing["test_kld_wall_seconds"] = t_kld.elapsed_seconds
            print(
                f"  test stream FID={test_metrics['fid_stream_test']:.3f}  "
                f"KLD={kld_metrics['stream_kld']:.3f}"
            )
        else:
            print(f"  test stream FID={test_metrics['fid_stream_test']:.3f}")

        per_class_metrics: dict[str, Any] = {}
        n_pc = 64 if args.smoke and per_class_fid_samples > 0 else per_class_fid_samples
        if n_pc > 0:
            with WallTimer() as t_pc:
                seen_y = torch.cat([tasks[i].y_test for i in range(k + 1)], dim=0)
                seen_x = torch.cat([tasks[i].x_test for i in range(k + 1)], dim=0)
                seen_classes = [c for i in range(k + 1) for c in tasks[i].classes]
                per_class_metrics = mean_per_class_fid(
                    model,
                    seen_x,
                    seen_y,
                    device=device,
                    class_ids=seen_classes,
                    n_gen_per_class=n_pc,
                    batch_size=int(args.fid_batch_size),
                    ode_method=args.ode_method,
                    ode_step_size=float(args.ode_step_size),
                    seed=int(args.eval_seed) + 5000 + k,
                    to_pixels=to_pixels,
                    guidance_scale=guidance_scale,
                )
            task_timing["per_class_fid_wall_seconds"] = t_pc.elapsed_seconds
            print(
                f"  mean per-class FID={per_class_metrics['fid_per_class_mean']:.3f}  "
                f"({per_class_metrics['n_classes_evaluated']} classes × {n_pc} gens)"
            )

        task_metrics = {
            "task_id": k,
            "classes": list(task.classes),
            "chosen_lambda": chosen_lambda,
            "candidates": cand_rows,
            **test_metrics,
            "fid_macro_test_per_task_mean": per_task_test["fid_macro"],
            "fid_per_task_test": per_task_test["fid_per_task"],
            **kld_metrics,
            **per_class_metrics,
        }
        metrics_by_task.append(task_metrics)
        lambda_trace.append(
            {
                "task_id": k,
                "chosen_lambda": chosen_lambda,
                "fid_stream_test": test_metrics["fid_stream_test"],
                "stream_kld": kld_metrics.get("stream_kld"),
                "fid_per_class_mean": per_class_metrics.get("fid_per_class_mean"),
            }
        )
        timing_by_task.append(task_timing)
        dump_json(task_dir / "metrics.json", task_metrics)
        dump_json(task_dir / "timing.json", task_timing)

        seen_classes = [c for i in range(k + 1) for c in tasks[i].classes]
        gens = generate_samples(
            model,
            64,
            device=device,
            batch_size=64,
            ode_method=args.ode_method,
            ode_step_size=float(args.ode_step_size),
            seed=int(args.eval_seed) + 3000 + k,
            to_pixels=to_pixels,
            class_ids=seen_classes if class_conditional else None,
            guidance_scale=guidance_scale,
        )
        torch.save(gens, out_dir / "samples" / f"task{k:02d}_gen.pt")

    _write_run_summary(
        out_dir=out_dir,
        base_cfg=base_cfg,
        metrics_by_task=metrics_by_task,
        lambda_trace=lambda_trace,
        timing_by_task=timing_by_task,
        elapsed=time.time() - t0,
    )


if __name__ == "__main__":
    main()
