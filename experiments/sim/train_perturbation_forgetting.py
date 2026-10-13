#!/usr/bin/env python3
"""Hold out a perturbation, context, or (context, perturbation) pair, then finetune."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
from trajectory_replay.utils.cli import parse_plot_proj, resolve_device
from trajectory_replay.data.simulate import sample_gaussian_sc
from trajectory_replay.model.flow_matching import ConditionalFlowMatching
from trajectory_replay.training.flow_matching import (
    eval_ode_strata,
    train_flow_matching,
    train_flow_matching_replay,
)
from trajectory_replay.holdout.data import (
    mask_excluding_combo,
    mask_excluding_context,
    mask_excluding_perturbation,
    mask_only_combo,
    mask_only_context,
    mask_only_perturbation,
    mask_replay_pool,
    stratified_subsample,
)
from trajectory_replay.holdout.models import RemappedContextModel, RemappedPerturbationModel
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
        "--holdout-mode",
        choices=("perturbation", "context", "combo"),
        default="perturbation",
        help=(
            "perturbation: hold out one p in all contexts (pretrain without p; new pert row at finetune). "
            "context: hold out one c in all perturbations (pretrain without c; new context row at finetune). "
            "combo: hold out one (c,p); finetune held-out c and p rows."
        ),
    )
    p.add_argument("--held-out-pert", type=int, default=3, help="Held-out perturbation index p.")
    p.add_argument(
        "--held-out-context",
        type=int,
        default=0,
        help="Held-out context index c (required for --holdout-mode context or combo).",
    )
    p.add_argument("--coupling", choices=("additive", "rotated", "hadamard"), default="additive")
    p.add_argument("--n-contexts", type=int, default=3)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--samples-per-combo", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.18)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--context-scale", type=float, default=1.2)
    p.add_argument("--pert-scale", type=float, default=0.9)
    p.add_argument("--interaction-scale", type=float, default=0.35)

    p.add_argument("--cond-emb-dim", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)

    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--finetune-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-2)
    p.add_argument(
        "--finetune-train-vf",
        action="store_true",
        help="Unfreeze the vector field (MLP) during finetune.",
    )
    p.add_argument(
        "--finetune-train-old-embeddings",
        action="store_true",
        help=(
            "Unfreeze context and perturbation embedding tables. "
            "Unfreeze all embedding rows. Held-out row(s) for the active holdout mode are always "
            "trained when this flag is off."
        ),
    )
    p.add_argument(
        "--finetune-data",
        choices=("held_out", "all", "replay"),
        default="held_out",
        help=(
            "held_out: finetune on held-out strata only (all c for perturbation; all p for context; "
            "one (c,p) for combo). all: finetune on every (c,p). "
            "replay: every step, mixed batch with (1-replay_fraction) OOD and replay_fraction "
            "from a stratified buffer of pretrain support."
        ),
    )
    p.add_argument(
        "--replay-fraction",
        type=float,
        default=0.25,
        help="Fraction of each finetune batch drawn from the replay buffer (replay mode only).",
    )
    p.add_argument(
        "--replay-max-samples",
        type=int,
        default=None,
        help="Cap replay buffer size with stratified subsampling (default: use full pretrain pool).",
    )
    p.add_argument(
        "--replay-seed",
        type=int,
        default=None,
        help="RNG seed for replay buffer subsampling (default: train_seed + 17).",
    )
    p.add_argument("--new-emb-init-std", type=float, default=0.02)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--ode-method", default="dopri5")
    p.add_argument("--ode-rtol", type=float, default=1e-5)
    p.add_argument("--ode-atol", type=float, default=1e-5)
    p.add_argument("--ode-step-size", type=float, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--plot-seed", type=int, default=None)
    p.add_argument("--plot-n-per-combo", type=int, default=None)
    p.add_argument("--plot-proj", default="0,1")
    p.add_argument("--plot-max-panels", type=int, default=48)

    p.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Write checkpoint_finetuned.pt under the run results directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    holdout_mode = args.holdout_mode
    held_out_p = args.held_out_pert
    held_out_c = args.held_out_context
    n_p = args.n_perturbations
    n_c = args.n_contexts
    if held_out_p < 0 or held_out_p >= n_p:
        raise ValueError(f"--held-out-pert must be in [0, {n_p - 1}], got {held_out_p}")
    if holdout_mode in ("context", "combo") and (held_out_c < 0 or held_out_c >= n_c):
        raise ValueError(f"--held-out-context must be in [0, {n_c - 1}], got {held_out_c}")
    if args.finetune_data == "replay":
        if not (0.0 < args.replay_fraction < 1.0):
            raise ValueError(
                f"--replay-fraction must be in (0, 1) for replay finetune, got {args.replay_fraction}"
            )
        n_ood_slots = int(args.batch_size * (1.0 - args.replay_fraction))
        if n_ood_slots < 1:
            raise ValueError(
                f"--batch-size={args.batch_size} and --replay-fraction={args.replay_fraction} "
                f"leave no OOD slots per batch"
            )
    replay_seed = (
        args.replay_seed if args.replay_seed is not None else args.train_seed + 17
    )

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

    skip_perturbations: frozenset[int] | None = None
    skip_contexts: frozenset[int] | None = None
    skip_combos: frozenset[tuple[int, int]] | None = None
    n_ctx_model = n_c
    n_pert_model = n_p
    if holdout_mode == "perturbation":
        x_pre, c_pre, p_pre = mask_excluding_perturbation(x_all, c_all, p_all, held_out_p)
        skip_perturbations = frozenset({held_out_p})
        n_pert_model = n_p - 1
    elif holdout_mode == "context":
        x_pre, c_pre, p_pre = mask_excluding_context(x_all, c_all, p_all, held_out_c)
        skip_contexts = frozenset({held_out_c})
        n_ctx_model = n_c - 1
    else:
        x_pre, c_pre, p_pre = mask_excluding_combo(x_all, c_all, p_all, held_out_c, held_out_p)
        skip_combos = frozenset({(held_out_c, held_out_p)})

    pre_loader = DataLoader(
        TensorDataset(x_pre, c_pre, p_pre),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(x_pre) >= args.batch_size,
    )
    if args.finetune_data == "all":
        x_ft, c_ft, p_ft = x_all, c_all, p_all
    elif holdout_mode == "perturbation":
        x_ft, c_ft, p_ft = mask_only_perturbation(x_all, c_all, p_all, held_out_p)
    elif holdout_mode == "context":
        x_ft, c_ft, p_ft = mask_only_context(x_all, c_all, p_all, held_out_c)
    else:
        x_ft, c_ft, p_ft = mask_only_combo(x_all, c_all, p_all, held_out_c, held_out_p)

    x_replay = c_replay = p_replay = None
    x_pool: torch.Tensor | None = None
    if args.finetune_data == "replay":
        x_pool, c_pool, p_pool = mask_replay_pool(
            x_all,
            c_all,
            p_all,
            holdout_mode=holdout_mode,
            held_out_pert=held_out_p,
            held_out_context=held_out_c,
        )
        x_replay, c_replay, p_replay = stratified_subsample(
            x_pool,
            c_pool,
            p_pool,
            max_samples=args.replay_max_samples,
            seed=replay_seed,
        )

    ft_loader = None
    if args.finetune_data != "replay":
        ft_loader = DataLoader(
            TensorDataset(x_ft, c_ft, p_ft),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=len(x_ft) >= args.batch_size,
        )

    model = ConditionalFlowMatching(
        data_dim=coupling.dim,
        n_contexts=n_ctx_model,
        n_perturbations=n_pert_model,
        cond_emb_dim=args.cond_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)

    if holdout_mode == "perturbation":
        print(
            f"Pretrain: hold out p={held_out_p} in all contexts "
            f"({len(x_pre)} points, {n_pert_model} pert embeddings)"
        )
    elif holdout_mode == "context":
        print(
            f"Pretrain: hold out c={held_out_c} in all perturbations "
            f"({len(x_pre)} points, {n_ctx_model} context embeddings)"
        )
    else:
        print(
            f"Pretrain: hold out (c={held_out_c}, p={held_out_p}) only; "
            f"p={held_out_p} seen in other contexts ({len(x_pre)} points)"
        )
    pretrain_history = train_flow_matching(
        model,
        pre_loader,
        data_dim=coupling.dim,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        train_seed=args.train_seed,
    )

    ode_options = None
    if args.ode_step_size is not None:
        ode_options = {"step_size": float(args.ode_step_size)}

    if holdout_mode == "perturbation":
        plot_model_pre: nn.Module = RemappedPerturbationModel(model, held_out_p)
    elif holdout_mode == "context":
        plot_model_pre = RemappedContextModel(model, held_out_c)
    else:
        plot_model_pre = model

    print("Pretrain ODE eval (held-out strata skipped):")
    pretrain_eval, pretrain_mse = eval_ode_strata(
        plot_model_pre,
        x_all,
        c_all,
        p_all,
        n_contexts=coupling.n_contexts,
        n_perturbations=n_p,
        device=device,
        eval_seed=args.train_seed + 999,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options,
        skip_perturbations=skip_perturbations,
        skip_contexts=skip_contexts,
        skip_combos=skip_combos,
    )
    for row in pretrain_eval:
        print(f"  c={row['c']}  p={row['p']}  mse={row['mse_to_true']:.6f}")
    print(f"pretrain mean MSE (evaluated strata): {pretrain_mse:.6f}")

    rdir: Path | None = None
    plot_files: list[str] = []
    per_cond: list[dict[str, float | int]] = []
    plot_seed = args.plot_seed if args.plot_seed is not None else args.train_seed + 4242
    proj = parse_plot_proj(args.plot_proj) if coupling.dim >= 2 else (0, 1)

    if args.run_name is not None:
        rdir = run_results_dir(_REPO_ROOT, args.run_name)
        rdir.mkdir(parents=True, exist_ok=True)
        if coupling.dim >= 2:
            true_cp = save_true_scatter_by_cp(
                rdir / "true_data_by_cp.png",
                ds.x,
                ds.c,
                ds.p,
                proj_axes=proj,
            )
            plot_files.append(str(true_cp))
            pre_paths, _ = save_true_vs_generated_figures(
                rdir / "true_vs_generated_pretrain",
                ds.x,
                ds.c,
                ds.p,
                plot_model_pre,  # type: ignore[arg-type]
                device=device,
                ode_method=args.ode_method,
                ode_rtol=args.ode_rtol,
                ode_atol=args.ode_atol,
                ode_options=ode_options,
                plot_seed=plot_seed,
                n_gen_per_combo=args.plot_n_per_combo,
                proj_axes=proj,
                max_panels_per_figure=args.plot_max_panels,
                skip_perturbations=skip_perturbations,
                skip_contexts=skip_contexts,
                skip_combos=skip_combos,
            )
            plot_files.extend(str(p) for p in pre_paths)

    vf_status = "train" if args.finetune_train_vf else "frozen"
    old_emb_status = "train" if args.finetune_train_old_embeddings else "frozen"
    if holdout_mode == "perturbation":
        print(
            f"Finetune: new pert embedding at p={held_out_p} ({len(x_ft)} points, "
            f"data={args.finetune_data})"
        )
        print(
            f"  vector field: {vf_status}; old embeddings: {old_emb_status}; "
            f"held-out pert row: train"
        )
        model.expand_perturbation_embedding(held_out_p, init_std=args.new_emb_init_std)
    elif holdout_mode == "context":
        print(
            f"Finetune: new context embedding at c={held_out_c} ({len(x_ft)} points, "
            f"data={args.finetune_data})"
        )
        print(
            f"  vector field: {vf_status}; old embeddings: {old_emb_status}; "
            f"held-out context row: train"
        )
        model.expand_context_embedding(held_out_c, init_std=args.new_emb_init_std)
    else:
        emb_target = (
            "all embedding rows"
            if args.finetune_train_old_embeddings
            else f"context c={held_out_c} and perturbation p={held_out_p} rows"
        )
        print(
            f"Finetune: held-out (c={held_out_c}, p={held_out_p}) via {emb_target} "
            f"({len(x_ft)} points, data={args.finetune_data})"
        )
        print(f"  vector field: {vf_status}; old embeddings: {old_emb_status}")
    finetune_params, finetune_pert_mask, finetune_context_mask = model.configure_finetune(
        holdout_mode=holdout_mode,
        train_vector_field=args.finetune_train_vf,
        train_old_embeddings=args.finetune_train_old_embeddings,
        held_out_pert=held_out_p,
        held_out_context=held_out_c if holdout_mode in ("context", "combo") else None,
    )
    if args.finetune_data == "replay":
        assert x_replay is not None and c_replay is not None and p_replay is not None
        assert x_pool is not None
        n_replay = int(args.batch_size * args.replay_fraction)
        n_ood = args.batch_size - n_replay
        print(
            f"  replay: buffer={len(x_replay)} (pool={len(x_pool)}), "
            f"fraction={args.replay_fraction}, batch={n_ood} OOD + {n_replay} replay"
        )
        finetune_history = train_flow_matching_replay(
            model,
            x_ft,
            c_ft,
            p_ft,
            x_replay,
            c_replay,
            p_replay,
            data_dim=coupling.dim,
            device=device,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=args.batch_size,
            replay_fraction=args.replay_fraction,
            train_seed=args.train_seed + 1,
            params=finetune_params,
            finetune_pert_index=finetune_pert_mask,
            finetune_context_index=finetune_context_mask,
        )
    else:
        assert ft_loader is not None
        finetune_history = train_flow_matching(
            model,
            ft_loader,
            data_dim=coupling.dim,
            device=device,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            train_seed=args.train_seed + 1,
            params=finetune_params,
            finetune_pert_index=finetune_pert_mask,
            finetune_context_index=finetune_context_mask,
        )

    print("Post-finetune ODE eval (all perturbations):")
    finetune_eval, finetune_mse = eval_ode_strata(
        model,
        x_all,
        c_all,
        p_all,
        n_contexts=coupling.n_contexts,
        n_perturbations=n_p,
        device=device,
        eval_seed=args.train_seed + 1999,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_options=ode_options,
    )
    for row in finetune_eval:
        if holdout_mode == "combo":
            held = row["c"] == held_out_c and row["p"] == held_out_p
        elif holdout_mode == "context":
            held = row["c"] == held_out_c
        else:
            held = row["p"] == held_out_p
        tag = "held-out→finetuned" if held else "seen@pretrain"
        print(f"  c={row['c']}  p={row['p']}  n={row['n']}  mse={row['mse_to_true']:.6f}  ({tag})")
    print(f"mean MSE (all strata): {finetune_mse:.6f}")

    if rdir is not None:
        if coupling.dim >= 2:
            ft_paths, per_cond = save_true_vs_generated_figures(
                rdir / "true_vs_generated_finetuned",
                ds.x,
                ds.c,
                ds.p,
                model,
                device=device,
                ode_method=args.ode_method,
                ode_rtol=args.ode_rtol,
                ode_atol=args.ode_atol,
                ode_options=ode_options,
                plot_seed=plot_seed + 1,
                n_gen_per_combo=args.plot_n_per_combo,
                proj_axes=proj,
                max_panels_per_figure=args.plot_max_panels,
            )
            plot_files.extend(str(p) for p in ft_paths)

        metrics: dict[str, object] = {
            "args": namespace_to_jsonable(args),
            "holdout_mode": holdout_mode,
            "held_out_pert": held_out_p,
            "held_out_context": held_out_c if holdout_mode in ("context", "combo") else None,
            "finetune_train_vf": args.finetune_train_vf,
            "finetune_train_old_embeddings": args.finetune_train_old_embeddings,
            "finetune_data": args.finetune_data,
            "replay_fraction": args.replay_fraction if args.finetune_data == "replay" else None,
            "replay_max_samples": args.replay_max_samples if args.finetune_data == "replay" else None,
            "replay_buffer_size": len(x_replay) if x_replay is not None else None,
            "pretrain_history": pretrain_history,
            "finetune_history": finetune_history,
            "pretrain_ode_eval": pretrain_eval,
            "pretrain_mean_mse_seen": pretrain_mse,
            "finetune_ode_eval": finetune_eval,
            "finetune_mean_mse": finetune_mse,
            "plot_files": plot_files,
            "per_condition_plot_finetuned": per_cond,
        }
        dump_metrics(rdir / "metrics.json", metrics)

        if args.save_checkpoint:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "data_dim": coupling.dim,
                        "n_contexts": coupling.n_contexts,
                        "n_perturbations": n_p,
                        "cond_emb_dim": args.cond_emb_dim,
                        "time_emb_dim": args.time_emb_dim,
                        "hidden_dim": args.hidden_dim,
                        "depth": args.depth,
                        "holdout_mode": holdout_mode,
                        "held_out_pert": held_out_p,
                        "held_out_context": held_out_c if holdout_mode in ("context", "combo") else None,
                    },
                },
                rdir / "checkpoint_finetuned.pt",
            )

        print(f"results written under {rdir}")


if __name__ == "__main__":
    main()
