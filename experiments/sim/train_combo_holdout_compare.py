#!/usr/bin/env python3
"""Compare noise-to-data vs data-to-data on a held-out (context, perturbation) combo.

Simulates Gaussian SC data, trains both models on all strata except one (c, p),
then evaluates generation only on the held-out stratum (energy distance + overlay plot).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
from trajectory_replay.utils.cli import parse_plot_proj, resolve_device
from trajectory_replay.data.simulate import sample_gaussian_sc
from trajectory_replay.eval.metrics import distribution_metrics
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import (
    filter_dataset_control_subsample,
    sample_combo_from_control,
    sample_combo_from_noise,
    train_data_to_data_flow_matching,
    train_flow_matching,
)
from trajectory_replay.holdout.data import mask_excluding_combo, mask_only_combo
from trajectory_replay.io.results import (
    dump_metrics,
    namespace_to_jsonable,
    run_results_dir,
    save_heldout_compare_overlay,
    save_true_scatter_by_cp,
)
from experiments.sim.sim_common import build_coupling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coupling",
        choices=("additive", "rotated", "hadamard"),
        default="additive",
    )
    p.add_argument("--n-contexts", type=int, default=3)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--samples-per-combo", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.18)
    p.add_argument("--data-seed", type=int, default=0)

    p.add_argument(
        "--held-out-context",
        type=int,
        default=1,
        help="Held-out context index c (excluded from training).",
    )
    p.add_argument(
        "--held-out-pert",
        type=int,
        default=3,
        help="Held-out perturbation index p (excluded from training).",
    )
    p.add_argument(
        "--control-pert",
        type=int,
        default=0,
        help="Control perturbation for data-to-data paths (x_0 source, matched context).",
    )

    p.add_argument("--context-scale", type=float, default=1.2)
    p.add_argument("--pert-scale", type=float, default=0.9)
    p.add_argument("--interaction-scale", type=float, default=0.35)

    p.add_argument("--cond-emb-dim", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--ode-method", default="dopri5")
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument("--ode-step-size", type=float, default=None)
    p.add_argument("--noise-scale", type=float, default=1.0, help="x_0 scale for noise-to-data sampling.")

    p.add_argument(
        "--run-name",
        default=None,
        help="Write metrics and plots under results_dir/<run-name>/ (repo root).",
    )
    p.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="If set, write checkpoint_*.pt under the run directory (default: off).",
    )
    p.add_argument(
        "--sweep-regime",
        default=None,
        help="Optional sweep label (e.g. easy, hard) stored in metrics.json.",
    )
    p.add_argument("--plot-seed", type=int, default=None, help="Default: train_seed + 4242.")
    p.add_argument("--plot-proj", default="0,1")
    p.add_argument(
        "--n-gen",
        type=int,
        default=None,
        help="Number of held-out samples to generate per model; default = all held-out cells.",
    )
    p.add_argument(
        "--d2d-exclude-identity-targets",
        action="store_true",
        help=(
            "Data-to-data training: only use non-control cells as x_1 targets "
            "(skip control→control identity paths)."
        ),
    )
    p.add_argument(
        "--train-both-d2d-variants",
        action="store_true",
        help=(
            "Train two data-to-data models (with and without identity targets) and "
            "compare both on the held-out stratum. Overrides --d2d-exclude-identity-targets."
        ),
    )
    p.add_argument(
        "--control-subsample-frac",
        type=float,
        default=None,
        help=(
            "If set in (0, 1], drop control cells from the dataset before train/eval: "
            "same retained count per context (see subsample_control_pools). Affects "
            "both noise-to-data and data-to-data; perturbation cells are kept."
        ),
    )
    p.add_argument(
        "--control-subsample-max",
        type=int,
        default=None,
        help="Optional cap on control cells per context after subsampling.",
    )
    p.add_argument(
        "--control-subsample-seed",
        type=int,
        default=None,
        help="RNG seed for control subsampling; default train_seed + 777.",
    )
    return p.parse_args()


def _maybe_filter_controls(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    if args.control_subsample_frac is None:
        return x, c, p, {"control_subsampled": 0}
    seed = (
        args.control_subsample_seed
        if args.control_subsample_seed is not None
        else args.train_seed + 777
    )
    return filter_dataset_control_subsample(
        x,
        c,
        p,
        args.control_pert,
        frac=float(args.control_subsample_frac),
        seed=seed,
        max_per_context=args.control_subsample_max,
    )


def _model_config(coupling_dim: int, n_c: int, n_p: int, args: argparse.Namespace) -> dict[str, object]:
    return {
        "data_dim": coupling_dim,
        "n_contexts": n_c,
        "n_perturbations": n_p,
        "cond_emb_dim": args.cond_emb_dim,
        "time_emb_dim": args.time_emb_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "coupling": args.coupling,
    }


def _eval_heldout(
    x_true: np.ndarray,
    x_gen: np.ndarray,
    *,
    proj_axes: tuple[int, int],
) -> dict[str, float]:
    return distribution_metrics(x_true, x_gen, dims=list(proj_axes))


def main() -> None:
    args = parse_args()
    held_c = args.held_out_context
    held_p = args.held_out_pert
    n_c = args.n_contexts
    n_p = args.n_perturbations

    if not (0 <= held_c < n_c):
        raise ValueError(f"--held-out-context must be in [0, {n_c}), got {held_c}")
    if not (0 <= held_p < n_p):
        raise ValueError(f"--held-out-pert must be in [0, {n_p}), got {held_p}")
    if not (0 <= args.control_pert < n_p):
        raise ValueError(f"--control-pert must be in [0, {n_p}), got {args.control_pert}")

    device = resolve_device(args.device) 
    ode_options = None 
    if args.ode_step_size is not None:
        ode_options = {"step_size": float(args.ode_step_size)}
    eval_seed = args.train_seed + 999
    plot_seed = args.plot_seed if args.plot_seed is not None else args.train_seed + 4242

    data_rng = np.random.default_rng(args.data_seed)
    coupling = build_coupling(
        data_rng,
        args.coupling,
        n_contexts=n_c,
        n_perturbations=n_p,
        dim=args.dim,
        context_scale=args.context_scale,
        pert_scale=args.pert_scale,
        interaction_scale=args.interaction_scale,
    )
    ds = sample_gaussian_sc(
        coupling, samples_per_combo=args.samples_per_combo, rng=data_rng, sigma=args.sigma
    )

    x_all = torch.from_numpy(ds.x).float()
    c_all = torch.from_numpy(ds.c).long()
    p_all = torch.from_numpy(ds.p).long()

    x_all, c_all, p_all, control_pool_meta = _maybe_filter_controls(x_all, c_all, p_all, args)
    if control_pool_meta.get("control_subsampled"):
        print(
            f"Control subsample (dataset): frac={args.control_subsample_frac} "
            f"→ {control_pool_meta['n_control_keep_per_context']} controls/context, "
            f"{control_pool_meta['n_rows_after']} rows "
            f"(dropped {control_pool_meta['n_control_dropped']} control cells; "
            f"noise + d2d both use this data)"
        )

    x_train, c_train, p_train = mask_excluding_combo(
        x_all, c_all, p_all, held_c, held_p
    )
    x_held, c_held, p_held = mask_only_combo(x_all, c_all, p_all, held_c, held_p)
    n_held = int(x_held.shape[0])
    if n_held == 0:
        raise ValueError(f"No cells in held-out combo (c={held_c}, p={held_p})")

    n_gen = args.n_gen if args.n_gen is not None else n_held
    n_gen = min(n_gen, n_held)
    x_true_np = x_held[:n_gen].numpy()

    print(
        f"Held-out (c={held_c}, p={held_p}): {n_held} cells; "
        f"train on {len(x_train)} cells; eval n_gen={n_gen}"
    )

    loader = DataLoader(
        TensorDataset(x_train, c_train, p_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(x_train) >= args.batch_size,
    )

    def make_model() -> ConditionalFlowMatching:
        return ConditionalFlowMatching(
            data_dim=coupling.dim,
            n_contexts=n_c,
            n_perturbations=n_p,
            cond_emb_dim=args.cond_emb_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        ).to(device)

    print("Training noise-to-data (held-out combo excluded)...")
    model_noise = make_model()
    hist_noise = train_flow_matching(
        model_noise,
        loader,
        data_dim=coupling.dim,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        train_seed=args.train_seed,
    )

    train_both_d2d = args.train_both_d2d_variants
    d2d_exclude_identity = args.d2d_exclude_identity_targets and not train_both_d2d

    model_d2d: ConditionalFlowMatching | None = None
    model_d2d_no_id: ConditionalFlowMatching | None = None
    hist_d2d: list[dict[str, float]] = []
    hist_d2d_no_id: list[dict[str, float]] = []

    if train_both_d2d:
        print("Training data-to-data WITH identity targets (held-out combo excluded)...")
        model_d2d = make_model()
        hist_d2d = train_data_to_data_flow_matching(
            model_d2d,
            x_train,
            c_train,
            p_train,
            control_pert=args.control_pert,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            train_seed=args.train_seed,
            exclude_identity_targets=False,
        )
        print("Training data-to-data WITHOUT identity targets (held-out combo excluded)...")
        model_d2d_no_id = make_model()
        hist_d2d_no_id = train_data_to_data_flow_matching(
            model_d2d_no_id,
            x_train,
            c_train,
            p_train,
            control_pert=args.control_pert,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            train_seed=args.train_seed,
            exclude_identity_targets=True,
        )
    else:
        mode = "no identity targets" if d2d_exclude_identity else "all targets"
        print(f"Training data-to-data ({mode}; held-out combo excluded)...")
        model_d2d = make_model()
        hist_d2d = train_data_to_data_flow_matching(
            model_d2d,
            x_train,
            c_train,
            p_train,
            control_pert=args.control_pert,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            train_seed=args.train_seed,
            exclude_identity_targets=d2d_exclude_identity,
        )

    model_noise.eval()
    if model_d2d is not None:
        model_d2d.eval()
    if model_d2d_no_id is not None:
        model_d2d_no_id.eval()
    ode_kw = dict(
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options,
    )

    print("Sampling held-out stratum...")
    with torch.no_grad():
        xh_noise = sample_combo_from_noise(
            model_noise,
            context=held_c,
            perturbation=held_p,
            n=n_gen,
            device=device,
            eval_seed=eval_seed,
            noise_scale=args.noise_scale,
            **ode_kw,
        )
        xh_d2d = sample_combo_from_control(
            model_d2d,
            x_all,
            c_all,
            p_all,
            context=held_c,
            perturbation=held_p,
            control_pert=args.control_pert,
            n=n_gen,
            device=device,
            eval_seed=eval_seed + 1,
            **ode_kw,
        )
        xh_d2d_no_id = None
        if model_d2d_no_id is not None:
            xh_d2d_no_id = sample_combo_from_control(
                model_d2d_no_id,
                x_all,
                c_all,
                p_all,
                context=held_c,
                perturbation=held_p,
                control_pert=args.control_pert,
                n=n_gen,
                device=device,
                eval_seed=eval_seed + 2,
                **ode_kw,
            )

    x_gen_noise = xh_noise.cpu().numpy()
    x_gen_d2d = xh_d2d.cpu().numpy()
    x_gen_d2d_no_id = (
        xh_d2d_no_id.cpu().numpy() if xh_d2d_no_id is not None else None
    )
    proj = parse_plot_proj(args.plot_proj) if coupling.dim >= 2 else (0, 1)
    metrics_noise = _eval_heldout(x_true_np, x_gen_noise, proj_axes=proj)
    metrics_d2d = _eval_heldout(x_true_np, x_gen_d2d, proj_axes=proj)
    metrics_d2d_no_id: dict[str, float] | None = None
    if x_gen_d2d_no_id is not None:
        metrics_d2d_no_id = _eval_heldout(x_true_np, x_gen_d2d_no_id, proj_axes=proj)

    print(f"Held-out energy — noise-to-data: {metrics_noise['energy']:.6f}")
    label_d2d = "d2d (+ identity train)" if train_both_d2d else "data-to-data"
    print(f"Held-out energy — {label_d2d}: {metrics_d2d['energy']:.6f}")
    if metrics_d2d_no_id is not None:
        print(
            "Held-out energy — d2d (no identity train): "
            f"{metrics_d2d_no_id['energy']:.6f}"
        )

    plot_files: list[str] = []
    if args.run_name is not None and coupling.dim >= 2:
        rdir = run_results_dir(_REPO_ROOT, args.run_name)
        rdir.mkdir(parents=True, exist_ok=True)

        true_cp = save_true_scatter_by_cp(
            rdir / "true_data_by_cp.png",
            ds.x,
            ds.c,
            ds.p,
            proj_axes=proj,
            title="Simulated data (held-out combo marked in compare panel)",
        )
        plot_files.append(str(true_cp))

        label_d2d_plot = "d2d (+ identity train)" if train_both_d2d else (
            "d2d (no identity train)"
            if d2d_exclude_identity
            else "generated data-to-data"
        )
        overlay = save_heldout_compare_overlay(
            rdir / "heldout_compare.png",
            ds.x,
            ds.c,
            ds.p,
            x_true_np,
            x_gen_d2d,
            x_gen_noise,
            held_context=held_c,
            held_pert=held_p,
            proj_axes=proj,
            control_pert=args.control_pert,
            metrics_d2d=metrics_d2d,
            metrics_noise=metrics_noise,
            x_gen_d2d_alt=x_gen_d2d_no_id,
            label_d2d=label_d2d_plot,
            label_d2d_alt="d2d (no identity train)",
            metrics_d2d_alt=metrics_d2d_no_id,
        )
        plot_files.append(str(overlay))

        checkpoint_files: list[str] = []
        if args.save_checkpoints:
            base_cfg = _model_config(coupling.dim, n_c, n_p, args)
            noise_path = rdir / "checkpoint_noise_to_data.pt"
            torch.save(
                {
                    "state_dict": model_noise.state_dict(),
                    "config": {**base_cfg, "flow_type": "noise_to_data"},
                },
                noise_path,
            )
            checkpoint_files.append(str(noise_path))
            d2d_ckpt_cfg = {
                **base_cfg,
                "flow_type": "data_to_data",
                "control_pert": args.control_pert,
            }
            ckpt_name = (
                "checkpoint_data_to_data_no_identity.pt"
                if d2d_exclude_identity and not train_both_d2d
                else "checkpoint_data_to_data.pt"
            )
            d2d_path = rdir / ckpt_name
            torch.save(
                {
                    "state_dict": model_d2d.state_dict(),
                    "config": {
                        **d2d_ckpt_cfg,
                        "exclude_identity_targets": d2d_exclude_identity and not train_both_d2d,
                    },
                },
                d2d_path,
            )
            checkpoint_files.append(str(d2d_path))
            if model_d2d_no_id is not None:
                no_id_path = rdir / "checkpoint_data_to_data_no_identity.pt"
                torch.save(
                    {
                        "state_dict": model_d2d_no_id.state_dict(),
                        "config": {**d2d_ckpt_cfg, "exclude_identity_targets": True},
                    },
                    no_id_path,
                )
                checkpoint_files.append(str(no_id_path))

        metrics: dict[str, object] = {
            "args": namespace_to_jsonable(args),
            "experiment": "combo_holdout_compare",
            "sweep_regime": args.sweep_regime,
            "held_out_context": held_c,
            "held_out_pert": held_p,
            "control_pert": args.control_pert,
            "control_pool": control_pool_meta,
            "train_both_d2d_variants": train_both_d2d,
            "d2d_exclude_identity_targets": d2d_exclude_identity,
            "n_train": len(x_train),
            "n_held_out": n_held,
            "n_gen": n_gen,
            "noise_to_data": {
                "epoch_history": hist_noise,
                "final_loss_mean": hist_noise[-1]["loss_mean"] if hist_noise else None,
                "heldout_metrics": metrics_noise,
            },
            "plot_files": plot_files,
            "checkpoint_files": checkpoint_files if args.save_checkpoints else None,
            "plot_projection": args.plot_proj,
        }
        if train_both_d2d:
            metrics["data_to_data_with_identity"] = {
                "exclude_identity_targets": False,
                "epoch_history": hist_d2d,
                "final_loss_mean": hist_d2d[-1]["loss_mean"] if hist_d2d else None,
                "heldout_metrics": metrics_d2d,
            }
            metrics["data_to_data_no_identity"] = {
                "exclude_identity_targets": True,
                "epoch_history": hist_d2d_no_id,
                "final_loss_mean": hist_d2d_no_id[-1]["loss_mean"] if hist_d2d_no_id else None,
                "heldout_metrics": metrics_d2d_no_id,
            }
        elif d2d_exclude_identity:
            metrics["data_to_data_no_identity"] = {
                "exclude_identity_targets": True,
                "epoch_history": hist_d2d,
                "final_loss_mean": hist_d2d[-1]["loss_mean"] if hist_d2d else None,
                "heldout_metrics": metrics_d2d,
            }
        else:
            metrics["data_to_data"] = {
                "exclude_identity_targets": False,
                "epoch_history": hist_d2d,
                "final_loss_mean": hist_d2d[-1]["loss_mean"] if hist_d2d else None,
                "heldout_metrics": metrics_d2d,
            }
        dump_metrics(rdir / "metrics.json", metrics)
        print(f"run results written under {rdir}")


if __name__ == "__main__":
    main()
