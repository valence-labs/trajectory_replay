#!/usr/bin/env python3
"""Train conditional flow matching with data-to-data paths (control p=0 -> target p)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
from trajectory_replay.utils.cli import parse_plot_proj, resolve_device
from trajectory_replay.data.simulate import sample_gaussian_sc
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import (
    eval_ode_strata_from_control,
    train_data_to_data_flow_matching,
)
from trajectory_replay.io.results import (
    dump_metrics,
    namespace_to_jsonable,
    run_results_dir,
    save_true_scatter_by_cp,
    save_true_vs_generated_figures,
)
from experiments.sim.sim_common import build_coupling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coupling",
        choices=("additive", "rotated", "hadamard"),
        default="additive",
        help="Simulator mean coupling (see data/simulate.py).",
    )
    p.add_argument("--n-contexts", type=int, default=3)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument("--dim", type=int, default=2, help="Data dimension d.")
    p.add_argument("--samples-per-combo", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.18, help="Gaussian noise scale (isotropic).")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument(
        "--control-pert",
        type=int,
        default=0,
        help="Perturbation index used as control (x_0 drawn from this p, matched context).",
    )
    p.add_argument(
        "--d2d-exclude-identity-targets",
        action="store_true",
        help="Only use non-control cells as x_1 targets (skip control→control identity paths).",
    )

    p.add_argument("--context-scale", type=float, default=1.2)
    p.add_argument("--pert-scale", type=float, default=0.9)
    p.add_argument("--interaction-scale", type=float, default=0.35, help="Hadamard coupling only.")

    p.add_argument("--cond-emb-dim", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4, help="Number of linear layers in the MLP trunk.")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu or cuda")

    p.add_argument("--save", type=Path, default=None, help="Optional path to save state_dict + config.")
    p.add_argument(
        "--eval-ode",
        action="store_true",
        help="After training, integrate from control x_0 per (c,p) and report MSE vs true stratum.",
    )
    p.add_argument(
        "--ode-method",
        default="dopri5",
        help="torchdiffeq solver (e.g. dopri5, rk4, euler). For euler, set --ode-options step_size.",
    )
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument(
        "--ode-step-size",
        type=float,
        default=None,
        help="If set, passed as options {'step_size': ...} (needed for fixed-step methods like euler).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="If set, write metrics, checkpoint, and plots under results_dir/<run-name>/ (repo root).",
    )
    p.add_argument(
        "--plot-seed",
        type=int,
        default=None,
        help="RNG seed for ODE samples in plots; default train_seed + 4242.",
    )
    p.add_argument(
        "--plot-n-per-combo",
        type=int,
        default=None,
        help="Max points per (c,p) for true/gen overlay; default = all available in that stratum.",
    )
    p.add_argument(
        "--plot-proj",
        default="0,1",
        help="Comma-separated indices (i,j) for 2D scatter projection when d>2.",
    )
    p.add_argument(
        "--plot-max-panels",
        type=int,
        default=48,
        help="Max (c,p) panels per PNG file; extra panels go to additional numbered PNGs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.control_pert < args.n_perturbations):
        raise ValueError(
            f"--control-pert={args.control_pert} must be in [0, {args.n_perturbations})"
        )
    device = resolve_device(args.device)

    data_rng = np.random.default_rng(args.data_seed)
    coupling = build_coupling(
        data_rng,
        args.coupling,
        n_contexts=args.n_contexts,
        n_perturbations=args.n_perturbations,
        dim=args.dim,
        context_scale=args.context_scale,
        pert_scale=args.pert_scale,
        interaction_scale=args.interaction_scale,
    )
    ds = sample_gaussian_sc(
        coupling, samples_per_combo=args.samples_per_combo, rng=data_rng, sigma=args.sigma
    )

    x = torch.from_numpy(ds.x).float()
    c = torch.from_numpy(ds.c).long()
    p = torch.from_numpy(ds.p).long()

    model = ConditionalFlowMatching(
        data_dim=coupling.dim,
        n_contexts=coupling.n_contexts,
        n_perturbations=coupling.n_perturbations,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    epoch_history = train_data_to_data_flow_matching(
        model,
        x,
        c,
        p,
        control_pert=args.control_pert,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_seed=args.train_seed,
        exclude_identity_targets=args.d2d_exclude_identity_targets,
    )

    ode_informal_batch_mse: float | None = None
    ode_eval_per_combo: list[dict[str, float | int]] = []
    if args.eval_ode:
        ode_options = None
        if args.ode_step_size is not None:
            ode_options = {"step_size": float(args.ode_step_size)}
        ode_eval_per_combo, ode_informal_batch_mse = eval_ode_strata_from_control(
            model,
            x,
            c,
            p,
            control_pert=args.control_pert,
            n_contexts=coupling.n_contexts,
            n_perturbations=coupling.n_perturbations,
            device=device,
            eval_seed=args.train_seed + 999,
            ode_method=args.ode_method,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_options=ode_options,
        )
        print(f"ode_eval ({args.ode_method}) — control -> (c,p), MSE vs true:")
        for row in ode_eval_per_combo:
            print(f"  c={row['c']}  p={row['p']}  n={row['n']}  mse={row['mse_to_true']:.6f}")
        print(f"ode_eval mean MSE over strata (equal weight): {ode_informal_batch_mse:.6f}")

    if args.run_name is not None:
        rdir = run_results_dir(_REPO_ROOT, args.run_name)
        rdir.mkdir(parents=True, exist_ok=True)
        plot_seed = args.plot_seed if args.plot_seed is not None else args.train_seed + 4242
        ode_options = None
        if args.ode_step_size is not None:
            ode_options = {"step_size": float(args.ode_step_size)}

        plot_files: list[str] = []
        per_condition_plot: list[dict[str, float | int]] = []
        plot_note = ""
        if coupling.dim < 2:
            plot_note = "skipped_data_dim_lt_2"
        else:
            proj = parse_plot_proj(args.plot_proj)
            paths, per_condition_plot = save_true_vs_generated_figures(
                rdir / "true_vs_generated",
                ds.x,
                ds.c,
                ds.p,
                model,
                device=device,
                ode_method=args.ode_method,
                ode_rtol=args.ode_rtol,
                ode_atol=args.ode_atol,
                ode_options=ode_options,
                plot_seed=plot_seed,
                n_gen_per_combo=args.plot_n_per_combo,
                proj_axes=proj,
                max_panels_per_figure=args.plot_max_panels,
                control_pert=args.control_pert,
            )
            plot_files = [str(p) for p in paths]
            true_cp = save_true_scatter_by_cp(
                rdir / "true_data_by_cp.png",
                ds.x,
                ds.c,
                ds.p,
                proj_axes=proj,
            )
            plot_files.insert(0, str(true_cp))

        metrics: dict[str, object] = {
            "args": namespace_to_jsonable(args),
            "flow_type": "data_to_data",
            "control_pert": args.control_pert,
            "d2d_exclude_identity_targets": args.d2d_exclude_identity_targets,
            "epoch_history": epoch_history,
            "final_loss_mean": epoch_history[-1]["loss_mean"] if epoch_history else None,
            "ode_informal_batch_mse": ode_informal_batch_mse,
            "ode_eval_per_combo": ode_eval_per_combo if args.eval_ode else None,
            "plot_files": plot_files,
            "per_condition_plot_stats": per_condition_plot,
            "plot_projection": args.plot_proj if coupling.dim >= 2 else None,
            "plot_note": plot_note or None,
        }
        dump_metrics(rdir / "metrics.json", metrics)

        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "data_dim": coupling.dim,
                "n_contexts": coupling.n_contexts,
                "n_perturbations": coupling.n_perturbations,
                "cond_emb_dim": args.cond_emb_dim,
                "time_emb_dim": args.time_emb_dim,
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "coupling": args.coupling,
                "flow_type": "data_to_data",
                "control_pert": args.control_pert,
            },
        }
        torch.save(ckpt, rdir / "checkpoint.pt")
        print(f"run results written under {rdir}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "config": {
                "data_dim": coupling.dim,
                "n_contexts": coupling.n_contexts,
                "n_perturbations": coupling.n_perturbations,
                "cond_emb_dim": args.cond_emb_dim,
                "time_emb_dim": args.time_emb_dim,
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "coupling": args.coupling,
                "flow_type": "data_to_data",
                "control_pert": args.control_pert,
            },
        }
        torch.save(payload, args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
