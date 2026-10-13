#!/usr/bin/env python3
"""Generate larger fixed pools from final FMNIST checkpoints for appendix grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.image_cl.image_cl_common import load_checkpoint
from trajectory_replay.eval.image_fid import generate_samples
from trajectory_replay.eval.stream_kld import (
    classify_images,
    load_fmnist_kld_classifier,
)
from trajectory_replay.data.split_fmnist import class_order_for_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "generative": "generative_s{seed}",
    "trajectory": "trajectory_mf_n4_eq_tfix_s{seed}",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--ode-step-size", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    output_dir = (
        REPO_ROOT / "results_dir" / "image_cl" / "images" / "fmnist" / "pools"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = load_fmnist_kld_classifier(
        REPO_ROOT
        / "baselines"
        / "diffusion_continual_learning"
        / "weights"
        / "cnn_fmnist",
        device=device,
    )
    task_classes = class_order_for_seed(int(args.seed))

    sampling_seed = (
        int(args.sampling_seed)
        if args.sampling_seed is not None
        else 60_000 + int(args.seed)
    )
    suffix = f"_{args.tag}" if args.tag else ""
    metadata = {
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "sampling_seed": sampling_seed,
        "tag": str(args.tag),
        "methods": {},
    }
    for method, stem_template in METHODS.items():
        stem = stem_template.format(seed=int(args.seed))
        checkpoint = (
            REPO_ROOT
            / "results_dir"
            / "image_cl"
            / "fmnist"
            / stem
            / "tasks"
            / "04"
            / "checkpoint.pt"
        )
        print(f"Loading {method}: {checkpoint}")
        model, config = load_checkpoint(checkpoint, device)
        model.eval()
        samples = generate_samples(
            model,
            int(args.n_samples),
            device=device,
            batch_size=int(args.batch_size),
            ode_method=str(config.get("ode_method", "euler")),
            ode_step_size=(
                float(args.ode_step_size)
                if args.ode_step_size is not None
                else float(config.get("ode_step_size", 0.1))
            ),
            seed=sampling_seed,
        )
        output = (
            output_dir
            / f"{method}_seed{args.seed}_n{args.n_samples}{suffix}.pt"
        )
        torch.save(samples, output)
        predictions = classify_images(
            classifier,
            samples,
            device=device,
            batch_size=int(args.batch_size),
        )
        counts = [
            sum(int(label) in set(classes) for label in predictions.tolist())
            for classes in task_classes
        ]
        metadata["methods"][method] = {
            "checkpoint": str(checkpoint),
            "output": str(output),
            "ode_method": str(config.get("ode_method", "euler")),
            "ode_step_size": (
                float(args.ode_step_size)
                if args.ode_step_size is not None
                else float(config.get("ode_step_size", 0.1))
            ),
            "task_counts": counts,
        }
        print(f"{method}: task counts {counts}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata_path = (
        output_dir / f"seed{args.seed}_n{args.n_samples}{suffix}.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
