#!/usr/bin/env python3
"""Train sequential CL methods on the five-task Gaussian ring."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.sim_mixture.common import (
    build_unconditional_model,
    clone_trainable,
    make_x_loader,
    run_finetune_strategy,
    sample_unconditional,
    save_checkpoint,
)
from experiments.sim_sequential.common import (
    REPO_ROOT,
    load_dataset_bundle,
    sequential_data_root,
    sequential_results_root,
)
from experiments.sim_sequential.metrics import (
    evaluate_sequential_samples,
    update_policy_weights,
)
from trajectory_replay.training.mixture_fm import train_unconditional_fm
from trajectory_replay.utils.cli import resolve_device

DEFAULT_LAMBDA_GRID = (0.25, 0.5, 2.0 / 3.0, 0.75, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("naive", "generative", "trajectory"),
        required=True,
    )
    parser.add_argument(
        "--lambda-mode",
        choices=("none", "constant", "equal", "adaptive"),
        required=True,
    )
    parser.add_argument("--lambda-constant", type=float, default=None)
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_GRID),
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--steps-per-task", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient-norm clipping threshold; <=0 disables clipping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--ode-step-size", type=float, default=0.05)
    parser.add_argument("--n-gen-select", type=int, default=4_000)
    parser.add_argument("--n-gen-test", type=int, default=10_000)
    parser.add_argument("--mass-pseudocount", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Output root; defaults to results_dir/sim_sequential/runs.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one epoch/task, 64 evaluation samples, and two adaptive candidates.",
    )
    return parser.parse_args()


def _lambda_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    if args.strategy == "naive":
        policy = "none"
    elif args.lambda_mode == "constant":
        policy = f"constant_lam{_lambda_tag(float(args.lambda_constant))}"
    else:
        policy = args.lambda_mode
    return f"{args.strategy}_{policy}_s{args.seed}"


def _validate_args(args: argparse.Namespace) -> None:
    if args.strategy == "naive" and args.lambda_mode != "none":
        raise ValueError("naive requires --lambda-mode none")
    if args.strategy != "naive" and args.lambda_mode == "none":
        raise ValueError("replay strategies require constant, equal, or adaptive lambda mode")
    if args.lambda_mode == "constant" and args.lambda_constant is None:
        raise ValueError("--lambda-mode constant requires --lambda-constant")
    values = list(args.lambda_grid)
    if args.lambda_constant is not None:
        values.append(args.lambda_constant)
    if any(not 0.0 < float(value) < 1.0 for value in values):
        raise ValueError("all lambda values must lie in (0, 1)")
    if args.steps_per_task < 1 or args.batch_size < 2:
        raise ValueError("steps-per-task must be positive and batch-size >= 2")


def _epochs_for_steps(n_examples: int, batch_size: int, requested_steps: int) -> tuple[int, int]:
    steps_per_epoch = n_examples // batch_size
    if steps_per_epoch < 1:
        raise ValueError("batch size exceeds the incoming task dataset")
    epochs = math.ceil(requested_steps / steps_per_epoch)
    return epochs, epochs * steps_per_epoch


def _candidate_lambdas(args: argparse.Namespace, task_id: int) -> list[float | None]:
    if args.strategy == "naive":
        return [None]
    if args.lambda_mode == "constant":
        return [float(args.lambda_constant)]
    if args.lambda_mode == "equal":
        # task_id is zero-based: after adding it, task_id+1 tasks have equal mass.
        return [task_id / (task_id + 1.0)]
    return sorted(set(float(value) for value in args.lambda_grid))


def _evaluate_model(
    model,
    test,
    *,
    device: torch.device,
    n_seen: int,
    policy_weights: np.ndarray,
    n_gen: int,
    seed: int,
    ode_step_size: float,
    mass_pseudocount: float,
) -> tuple[dict, np.ndarray]:
    x_gen = sample_unconditional(
        model,
        n_gen,
        device=device,
        seed=seed,
        ode_step_size=ode_step_size,
    )
    metrics = evaluate_sequential_samples(
        x_gen,
        test,
        n_seen=n_seen,
        policy_weights=policy_weights,
        mass_pseudocount=mass_pseudocount,
    )
    return metrics, x_gen


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _add_forgetting_metrics(metrics: dict, stream: list[dict]) -> None:
    """Attach per-task error increases relative to each acquisition checkpoint."""
    distribution_delta: list[float] = []
    geometry_delta: list[float] = []
    for task_id in range(int(metrics["n_seen"])):
        if task_id == len(stream):
            distribution_baseline = metrics["per_task_distribution_energy"][task_id]
            geometry_baseline = metrics["per_mode_energy"][task_id]
        else:
            acquisition = stream[task_id]["metrics"]
            distribution_baseline = acquisition["per_task_distribution_energy"][task_id]
            geometry_baseline = acquisition["per_mode_energy"][task_id]
        distribution_delta.append(
            float(
                metrics["per_task_distribution_energy"][task_id]
                - distribution_baseline
            )
        )
        current_geometry = metrics["per_mode_energy"][task_id]
        geometry_delta.append(
            float(current_geometry - geometry_baseline)
            if np.isfinite(current_geometry) and np.isfinite(geometry_baseline)
            else float("nan")
        )
    metrics["per_task_distribution_energy_increase"] = distribution_delta
    metrics["per_mode_energy_increase"] = geometry_delta
    old_distribution_delta = distribution_delta[:-1]
    old_geometry_delta = [x for x in geometry_delta[:-1] if np.isfinite(x)]
    metrics["forgetting_distribution_mean"] = (
        float(np.mean(old_distribution_delta))
        if old_distribution_delta
        else float("nan")
    )
    metrics["forgetting_geometry_mean"] = (
        float(np.mean(old_geometry_delta)) if old_geometry_delta else float("nan")
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps_per_task = 1
        args.n_gen_select = 64
        args.n_gen_test = 64
        args.lambda_grid = [0.5, 0.8]
        args.hidden_dim = 64
        args.depth = 2
    _validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir or sequential_data_root(REPO_ROOT, dim=args.dim)
    splits, data_meta = load_dataset_bundle(data_dir)
    train, val, test = splits["train"], splits["val"], splits["test"]
    if train.dim != args.dim:
        raise ValueError(f"dataset dim {train.dim} != requested dim {args.dim}")

    run_name = args.run_name or _default_run_name(args)
    runs_dir = args.runs_dir or sequential_results_root(REPO_ROOT) / "runs"
    run_dir = runs_dir / run_name
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "strategy": args.strategy,
        "lambda_mode": args.lambda_mode,
        "lambda_constant": args.lambda_constant,
        "lambda_grid": [float(x) for x in args.lambda_grid],
        "steps_per_task_requested": int(args.steps_per_task),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "max_grad_norm": float(args.max_grad_norm),
        "seed": int(args.seed),
        "data_dim": int(args.dim),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "time_emb_dim": int(args.time_emb_dim),
        "ode_step_size": float(args.ode_step_size),
        "n_gen_select": int(args.n_gen_select),
        "n_gen_test": int(args.n_gen_test),
        "mass_pseudocount": float(args.mass_pseudocount),
        "data_dir": str(data_dir),
        "runs_dir": str(runs_dir),
        "data_meta": data_meta,
        "device": str(device),
        "smoke": bool(args.smoke),
    }
    _write_json(run_dir / "config.json", config)

    stream: list[dict] = []
    policy_weights = np.asarray([1.0], dtype=np.float64)

    # Task 0 pretraining.
    task_dir = run_dir / "tasks" / "00"
    x_task = train.subset(0).x
    if args.smoke:
        x_task = x_task[: args.batch_size]
    loader = make_x_loader(x_task, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    epochs, actual_steps = _epochs_for_steps(
        len(x_task), args.batch_size, args.steps_per_task
    )
    model = build_unconditional_model(
        args.dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    print(
        f"Task 0/{train.n_tasks - 1}: pretrain, "
        f"{epochs} epochs = {actual_steps} optimizer steps"
    )
    history = train_unconditional_fm(
        model,
        loader,
        device=device,
        epochs=epochs,
        lr=args.lr,
        train_seed=args.seed,
        max_grad_norm=args.max_grad_norm,
    )
    task_cfg = {
        **config,
        "task_id": 0,
        "n_seen": 1,
        "epochs": epochs,
        "actual_optimizer_steps": actual_steps,
        "selected_lambda": None,
    }
    save_checkpoint(task_dir / "checkpoint.pt", model, config=task_cfg, history=history)
    metrics, x_gen = _evaluate_model(
        model,
        test,
        device=device,
        n_seen=1,
        policy_weights=policy_weights,
        n_gen=args.n_gen_test,
        seed=100_000 + args.seed,
        ode_step_size=args.ode_step_size,
        mass_pseudocount=args.mass_pseudocount,
    )
    _add_forgetting_metrics(metrics, stream)
    task_record = {
        "task_id": 0,
        "n_seen": 1,
        "selected_lambda": None,
        "selection": [],
        "actual_optimizer_steps": actual_steps,
        "metrics": metrics,
    }
    stream.append(task_record)
    _write_json(task_dir / "metrics.json", task_record)
    np.savez_compressed(task_dir / "samples.npz", x=x_gen)
    _write_json(run_dir / "stream_metrics.json", stream)

    # Sequential updates.
    for task_id in range(1, train.n_tasks):
        teacher = model
        x_task = train.subset(task_id).x
        if args.smoke:
            x_task = x_task[: args.batch_size]
        loader_seed = args.seed + 10_000 * task_id
        epochs, actual_steps = _epochs_for_steps(
            len(x_task), args.batch_size, args.steps_per_task
        )
        candidates: list[dict] = []
        candidate_models = []
        lambdas = _candidate_lambdas(args, task_id)
        print(
            f"Task {task_id}/{train.n_tasks - 1}: {args.strategy}/{args.lambda_mode}, "
            f"{epochs} epochs = {actual_steps} steps, λ={lambdas}"
        )
        for candidate_id, lam in enumerate(lambdas):
            # Recreate the loader so every λ candidate sees the same minibatch order.
            loader = make_x_loader(
                x_task,
                batch_size=args.batch_size,
                shuffle=True,
                seed=loader_seed,
            )
            candidate = clone_trainable(teacher, device)
            candidate_policy = (
                np.full(task_id + 1, 1.0 / (task_id + 1), dtype=np.float64)
                if lam is None
                else update_policy_weights(policy_weights, lam)
            )
            train_seed = loader_seed
            candidate_history = run_finetune_strategy(
                args.strategy,
                candidate,
                teacher,
                loader,
                x_support=torch.as_tensor(x_task, dtype=torch.float32),
                device=device,
                epochs=epochs,
                lr=args.lr,
                train_seed=train_seed,
                lambda_mix=0.5 if lam is None else lam,
                ode_method="euler",
                ode_step_size=args.ode_step_size,
                traj_kappa="none",
                max_grad_norm=args.max_grad_norm,
            )
            val_metrics, _ = _evaluate_model(
                candidate,
                val,
                device=device,
                n_seen=task_id + 1,
                policy_weights=candidate_policy,
                n_gen=args.n_gen_select,
                seed=200_000 + args.seed + 1_000 * task_id,
                ode_step_size=args.ode_step_size,
                mass_pseudocount=args.mass_pseudocount,
            )
            candidate_record = {
                "candidate_id": candidate_id,
                "lambda": lam,
                "policy_target_mass": candidate_policy.tolist(),
                "selection_metric": "energy_uniform_mixture",
                "selection_value": val_metrics["energy_uniform_mixture"],
                "validation_metrics": val_metrics,
            }
            candidates.append(candidate_record)
            candidate_models.append((candidate, candidate_history, candidate_policy))

            if args.lambda_mode == "adaptive":
                candidate_dir = (
                    run_dir
                    / "tasks"
                    / f"{task_id:02d}"
                    / "candidates"
                    / f"lambda_{_lambda_tag(float(lam))}"
                )
                candidate_cfg = {
                    **config,
                    "task_id": task_id,
                    "n_seen": task_id + 1,
                    "epochs": epochs,
                    "actual_optimizer_steps": actual_steps,
                    "candidate_lambda": lam,
                }
                save_checkpoint(
                    candidate_dir / "checkpoint.pt",
                    candidate,
                    config=candidate_cfg,
                    history=candidate_history,
                )
                _write_json(candidate_dir / "metrics.json", candidate_record)

        selected_id = int(
            np.argmin([candidate["selection_value"] for candidate in candidates])
        )
        selected = candidates[selected_id]
        model, history, policy_weights = candidate_models[selected_id]
        selected_lambda = selected["lambda"]
        task_dir = run_dir / "tasks" / f"{task_id:02d}"
        task_cfg = {
            **config,
            "task_id": task_id,
            "n_seen": task_id + 1,
            "epochs": epochs,
            "actual_optimizer_steps": actual_steps,
            "selected_lambda": selected_lambda,
            "policy_target_mass": policy_weights.tolist(),
        }
        save_checkpoint(task_dir / "checkpoint.pt", model, config=task_cfg, history=history)
        metrics, x_gen = _evaluate_model(
            model,
            test,
            device=device,
            n_seen=task_id + 1,
            policy_weights=policy_weights,
            n_gen=args.n_gen_test,
            seed=300_000 + args.seed + 1_000 * task_id,
            ode_step_size=args.ode_step_size,
            mass_pseudocount=args.mass_pseudocount,
        )
        _add_forgetting_metrics(metrics, stream)
        task_record = {
            "task_id": task_id,
            "n_seen": task_id + 1,
            "selected_lambda": selected_lambda,
            "selection": candidates,
            "actual_optimizer_steps": actual_steps,
            "metrics": metrics,
        }
        stream.append(task_record)
        _write_json(task_dir / "metrics.json", task_record)
        np.savez_compressed(task_dir / "samples.npz", x=x_gen)
        _write_json(run_dir / "stream_metrics.json", stream)
        print(
            f"  selected λ={selected_lambda}; "
            f"test KL={metrics['mass_kl_uniform']:.4f}, "
            f"energy={metrics['energy_uniform_mixture']:.4f}"
        )

    print(f"Completed {run_dir}")


if __name__ == "__main__":
    main()
