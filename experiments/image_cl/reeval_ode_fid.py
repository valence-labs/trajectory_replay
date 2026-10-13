#!/usr/bin/env python3
"""Re-evaluate a finished image_cl run's checkpoint under alternate ODE settings.

Does **not** retrain. Writes ``reeval_ode.json`` (and optional sample grids) under
the run directory. Safe to run while other training jobs are in flight.

Example:
  uv run python experiments/image_cl/reeval_ode_fid.py \\
    --run-dir results_dir/image_cl/imagenet100/pilot/pilot_cc_t1_s42 \\
    --settings euler:0.05 euler:0.02 euler:0.01 dopri5 \\
    --save-grids
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import make_grid, save_image

from trajectory_replay.eval.image_fid import fid_against_reals, generate_samples
from trajectory_replay.model.diffusers_unet_fm import disable_fused_sdpa_if_needed

from experiments.image_cl.image_cl_common import (
    build_image_cl_tasks,
    dump_json,
    load_checkpoint,
    resolve_latent_to_pixels,
)


def _parse_setting(spec: str) -> dict[str, Any]:
    """``euler:0.05`` → method+step; ``dopri5`` → adaptive (step ignored)."""
    spec = spec.strip()
    if ":" in spec:
        method, step_s = spec.split(":", 1)
        return {
            "name": spec,
            "ode_method": method,
            "ode_step_size": float(step_s),
        }
    return {"name": spec, "ode_method": spec, "ode_step_size": 0.05}


def _approx_nfe(method: str, step: float) -> str:
    if method in ("euler", "midpoint", "rk4"):
        return f"~{int(round(1.0 / float(step)))}"
    return "adaptive"


def _infer_n_experiences(cfg: dict[str, Any]) -> int:
    value = cfg.get("n_experiences")
    if value is not None:
        return int(value)
    task_classes = cfg.get("task_classes")
    if task_classes:
        return len(task_classes)
    return 3 if cfg["dataset"] == "split_imagenet100" else 5


def _canonical_eval_seed(cfg: dict[str, Any], *, task_id: int, stream: bool) -> int:
    base = int(cfg.get("eval_seed", 0))
    if stream:
        return base + 1000 + task_id
    return base + 2000 + task_id + 17 * task_id


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more finished image_cl run directories.",
    )
    p.add_argument(
        "--settings",
        nargs="+",
        default=["euler:0.05", "euler:0.02", "euler:0.01", "dopri5"],
        help="ODE settings as method or method:step_size.",
    )
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument(
        "--stream",
        action="store_true",
        help="Masip-style stream FID: concat test sets / labels for tasks 0..task-id "
        "(matches train_cl final_fid_stream_test). Default is single-task FID.",
    )
    p.add_argument("--n-gen", type=int, default=None, help="Default: run config fid_test_samples.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Exact generation seed override. Default reproduces train_cl's "
        "stream or per-task test-FID seed.",
    )
    p.add_argument(
        "--save-grids",
        action="store_true",
        help="Save 64-sample PNG grids per setting under samples/reeval_*.png.",
    )
    p.add_argument("--grid-n", type=int, default=64)
    p.add_argument(
        "--out-name",
        type=str,
        default="reeval_ode.json",
        help="Output JSON filename under each run dir.",
    )
    args = p.parse_args()

    disable_fused_sdpa_if_needed()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[2]
    settings = [_parse_setting(s) for s in args.settings]

    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        ckpt = run_dir / "tasks" / f"{int(args.task_id):02d}" / "checkpoint.pt"
        cfg_path = run_dir / "config.json"
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config: {cfg_path}")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mode = "stream" if args.stream else "task"
        print(f"\n=== Re-eval {run_dir.name} task {args.task_id:02d} ({mode}) ===")
        model, _ = load_checkpoint(ckpt, device)
        model.eval()

        # Build a minimal args namespace for task + VAE loading from saved config.
        ns = argparse.Namespace(
            dataset=str(cfg["dataset"]),
            seed=int(cfg["seed"]),
            n_experiences=_infer_n_experiences(cfg),
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            latent_dir=cfg.get("latent_dir"),
            vae_id=cfg.get("vae_id") or "stabilityai/sd-vae-ft-mse",
            data_root=cfg.get("data_root"),
            image_size=int(cfg.get("image_size", 32)),
            fid_batch_size=int(cfg.get("fid_batch_size", 128)),
        )
        tasks = build_image_cl_tasks(ns, repo_root)
        tid = int(args.task_id)
        if tid < 0 or tid >= len(tasks):
            raise ValueError(f"task-id={tid} out of range for {len(tasks)} tasks")

        class_cond = bool(getattr(model, "use_conditioning", False))
        if args.stream:
            reals = torch.cat([tasks[i].x_test for i in range(tid + 1)], dim=0)
            label_source = None
            class_ids = (
                [int(c) for i in range(tid + 1) for c in tasks[i].classes]
                if class_cond
                else None
            )
        else:
            task = tasks[tid]
            reals = task.x_test
            label_source = task.y_test if class_cond else None
            class_ids = None

        guidance = (
            float(cfg["guidance_scale"])
            if cfg.get("guidance_scale") is not None and class_cond
            else None
        )
        to_pixels = None
        if bool(cfg.get("latent_decode", False)):
            to_pixels = resolve_latent_to_pixels(ns, device)

        n_gen = int(args.n_gen if args.n_gen is not None else cfg.get("fid_test_samples", 2048))
        batch_size = int(
            args.batch_size
            if args.batch_size is not None
            else cfg.get("fid_batch_size", 128)
        )
        eval_seed_base = int(cfg.get("eval_seed", 0))
        if args.seed is not None:
            eval_seed = int(args.seed)
        else:
            eval_seed = _canonical_eval_seed(cfg, task_id=tid, stream=args.stream)

        baseline = None
        metrics_path = run_dir / "tasks" / f"{tid:02d}" / "metrics.json"
        if metrics_path.is_file():
            task_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if args.stream:
                baseline = task_metrics.get("fid_stream_test")
            else:
                per_task = task_metrics.get("fid_per_task_test", [])
                if tid < len(per_task):
                    baseline = per_task[tid]

        rows: list[dict[str, Any]] = []
        for setting in settings:
            method = str(setting["ode_method"])
            step = float(setting["ode_step_size"])
            name = str(setting["name"])
            t0 = time.perf_counter()
            fid = fid_against_reals(
                model,
                reals,
                device=device,
                n_gen=n_gen,
                batch_size=batch_size,
                ode_method=method,
                ode_step_size=step,
                seed=eval_seed,
                to_pixels=to_pixels,
                label_source=label_source,
                class_ids=class_ids,
                guidance_scale=guidance,
            )
            elapsed = time.perf_counter() - t0
            row = {
                "setting": name,
                "ode_method": method,
                "ode_step_size": step if method in ("euler", "midpoint", "rk4") else None,
                "approx_nfe": _approx_nfe(method, step),
                "fid_stream_test": float(fid),
                "n_gen": n_gen,
                "n_real": int(reals.shape[0]),
                "wall_seconds": float(elapsed),
            }
            rows.append(row)
            print(
                f"  {name:16s}  FID={fid:8.3f}  nfe={row['approx_nfe']:>8s}  "
                f"wall={elapsed:6.1f}s"
            )

            if args.save_grids:
                samples_dir = run_dir / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                safe = name.replace(":", "_")
                gens = generate_samples(
                    model,
                    int(args.grid_n),
                    device=device,
                    batch_size=min(batch_size, int(args.grid_n)),
                    ode_method=method,
                    ode_step_size=step,
                    seed=eval_seed + 7,
                    to_pixels=to_pixels,
                    label_source=label_source,
                    class_ids=class_ids,
                    guidance_scale=guidance,
                )
                pt_path = samples_dir / f"reeval_{safe}_gen.pt"
                png_path = samples_dir / f"reeval_{safe}_gen.png"
                torch.save(gens.detach().cpu(), pt_path)
                grid = make_grid(
                    gens.detach().float().cpu().clamp(-1, 1),
                    nrow=8,
                    normalize=True,
                    value_range=(-1.0, 1.0),
                    padding=2,
                )
                save_image(grid, png_path)
                print(f"    wrote {png_path.relative_to(run_dir)}")

        out = {
            "run_dir": str(run_dir),
            "task_id": tid,
            "stream": bool(args.stream),
            "checkpoint": str(ckpt),
            "baseline_fid": baseline,
            "baseline_fid_stream_test": baseline if args.stream else None,
            "eval_seed_base": eval_seed_base,
            "eval_seed": eval_seed,
            "n_gen": n_gen,
            "batch_size": batch_size,
            "guidance_scale": guidance,
            "results": rows,
        }
        out_path = run_dir / str(args.out_name)
        dump_json(out_path, out)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
