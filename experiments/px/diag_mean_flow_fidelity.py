#!/usr/bin/env python3
"""Does the distilled task-0 MeanFlow reproduce the ODE teacher on RxRx1?

Guards the MeanFlow replay path. A mis-conditioned average-velocity field can fit
the distillation objective while producing degenerate few-step samples, and the
distillation loss alone cannot detect it. This rolls both models from shared
noise under shared conditions and reports the relative error and the spread ratio,
mirroring ``mean_flow_replay_fidelity`` on the image side.

    python experiments/px/diag_mean_flow_fidelity.py --seeds 42 69 1714
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch

from experiments.px.dataset import (
    IMG_SIZE,
    NUM_CELL_TYPES,
    NUM_CHANNELS,
    NUM_EXPERIMENTS,
    NUM_PERTURBATIONS,
)
from experiments.px.model import ADMMeanFlowWrapper, get_model_cls

CL_FINAL = REPO_ROOT / "results_dir" / "px" / "cl_final"
OUT = REPO_ROOT / "results_dir" / "px" / "figures" / "px_mean_flow_fidelity.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 69, 1714])
    p.add_argument("--jumps", nargs="+", type=int, default=[1, 4])
    p.add_argument("--n", type=int, default=256, help="Samples per comparison.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--teacher-step-size",
        type=float,
        default=0.05,
        help="Euler step for the reference ODE teacher (matches replay).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--noise-seed", type=int, default=2027)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """CFM checkpoints are saved through DDP/EMA wrappers, MeanFlow ones are not."""
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def _build_cfm(device: torch.device) -> torch.nn.Module:
    model = get_model_cls("ADM")(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=NUM_CELL_TYPES,
    )
    return model.to(device).eval()


def _build_mean_flow(device: torch.device) -> ADMMeanFlowWrapper:
    model = ADMMeanFlowWrapper(
        in_channels=NUM_CHANNELS,
        input_size=IMG_SIZE,
        y_dim=NUM_PERTURBATIONS,
        e_dim=NUM_EXPERIMENTS,
        c_dim=NUM_CELL_TYPES,
    )
    return model.to(device).eval()


@torch.no_grad()
def _euler_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    *,
    y: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Teacher ODE from ``x0`` to t=1, matching the replay integrator."""
    n_steps = max(1, round(1.0 / float(step_size)))
    dt = 1.0 / n_steps
    x = x0
    for step in range(n_steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        x = x + dt * model(x=x, t=t, y=y, e=e, c=c)
    return x


@torch.no_grad()
def compare_seed(
    seed: int,
    *,
    jumps: list[int],
    n: int,
    batch_size: int,
    teacher_step_size: float,
    device: torch.device,
    noise_seed: int,
) -> dict[str, object]:
    run = CL_FINAL / f"task0_s{seed}"
    ckpt_path = run / "checkpoint.ckpt"
    mf_path = run / "mean_flow.pt"
    if not ckpt_path.is_file() or not mf_path.is_file():
        raise FileNotFoundError(f"missing task-0 artifacts under {run}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    teacher = _build_cfm(device)
    # Replay uses the EMA weights, so the reference must too.
    teacher.load_state_dict(_strip_module_prefix(ckpt["ema"]))
    mean_flow = _build_mean_flow(device)
    mean_flow.load_state_dict(
        _strip_module_prefix(
            torch.load(mf_path, map_location="cpu", weights_only=False)
        )
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(noise_seed + seed)
    # Conditions spread over the full perturbation and experiment vocabulary.
    y_all = torch.randint(NUM_PERTURBATIONS, (n,), generator=gen)
    e_all = torch.randint(NUM_EXPERIMENTS, (n,), generator=gen)
    c_all = torch.zeros(n, dtype=torch.long)
    x0_all = torch.randn((n, NUM_CHANNELS, IMG_SIZE, IMG_SIZE), generator=gen)

    sq_err = {k: 0.0 for k in jumps}
    sq_ref = 0.0
    mf_sq = {k: 0.0 for k in jumps}
    ref_sq = 0.0
    mf_sum = {k: 0.0 for k in jumps}
    ref_sum = 0.0
    count = 0

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x0 = x0_all[start:stop].to(device)
        y = y_all[start:stop].to(device)
        e = e_all[start:stop].to(device)
        c = c_all[start:stop].to(device)

        x_ref = _euler_rollout(
            teacher, x0, y=y, e=e, c=c, step_size=teacher_step_size
        )
        sq_ref += float((x_ref**2).sum())
        ref_sq += float((x_ref**2).sum())
        ref_sum += float(x_ref.sum())
        count += x_ref.numel()

        for k in jumps:
            x_mf = mean_flow.sample_from_x0(x0, y=y, e=e, c=c, n_steps=int(k))
            sq_err[k] += float(((x_mf - x_ref) ** 2).sum())
            mf_sq[k] += float((x_mf**2).sum())
            mf_sum[k] += float(x_mf.sum())

    def _std(total_sq: float, total: float) -> float:
        mean = total / count
        return max(total_sq / count - mean * mean, 0.0) ** 0.5

    ref_std = _std(ref_sq, ref_sum)
    rows = []
    for k in jumps:
        rows.append(
            {
                "jumps": int(k),
                "rel_err_vs_teacher": sq_err[k] / max(sq_ref, 1e-12),
                "std_ratio": _std(mf_sq[k], mf_sum[k]) / max(ref_std, 1e-12),
            }
        )
    return {
        "seed": int(seed),
        "n": int(n),
        "teacher_nfe": max(1, round(1.0 / float(teacher_step_size))),
        "reference_std": ref_std,
        "per_jump": rows,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}", flush=True)

    results = []
    for seed in args.seeds:
        row = compare_seed(
            seed,
            jumps=list(args.jumps),
            n=int(args.n),
            batch_size=int(args.batch_size),
            teacher_step_size=float(args.teacher_step_size),
            device=device,
            noise_seed=int(args.noise_seed),
        )
        results.append(row)
        for entry in row["per_jump"]:
            print(
                f"  seed {seed}  N={entry['jumps']}  "
                f"rel_err={entry['rel_err_vs_teacher']:.4f}  "
                f"std_ratio={entry['std_ratio']:.4f}",
                flush=True,
            )

    payload = {
        "teacher_step_size": float(args.teacher_step_size),
        "n_samples": int(args.n),
        "noise_seed": int(args.noise_seed),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
