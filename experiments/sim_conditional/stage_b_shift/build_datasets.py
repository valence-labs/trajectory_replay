#!/usr/bin/env python3
"""Build Stage B additive (c, p) data with pretrain / finetune / heldout pools.

Protocol (defaults: 6 contexts × 4 perturbations, dim=2; results live under
``…/dim{{D}}/c{{C}}_p{{P}}/``):
  - Pretrain pool: all contexts except ``ft_context``, all perturbations
    (train + val per cell).
  - Finetune pool: only ``ft_context``, all perturbations except ``heldout_pert``
    (train + val per cell).
  - Heldout pool: only ``(ft_context, heldout_pert)`` (test only).

Additive coupling ``μ_{c,p} = μ_c + δ_p`` with control ``δ_0 = 0``.

The defaults reproduce the paper configuration (``c6_p4``, dim=2, radius=4,
data_seed=0). The published data was built with ``--context-separation 4``,
which is the same layout at 6 contexts up to float round-off (~1e-15):
  uv run python experiments/sim_conditional/stage_b_shift/build_datasets.py

Other configurations need explicit flags, e.g.:
  uv run python experiments/sim_conditional/stage_b_shift/build_datasets.py --n-contexts 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.sim_conditional.stage_b_shift.common import (
    POOL_FINETUNE,
    POOL_HELDOUT,
    POOL_PRETRAIN,
    REPO_ROOT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    stage_b_data_root,
)
from trajectory_replay.data.mixture_mog import make_polygon_means
from trajectory_replay.data.simulate import AdditiveCoupling, make_additive_coupling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-contexts", type=int, default=6)
    p.add_argument("--n-perturbations", type=int, default=4)
    p.add_argument(
        "--ft-context",
        type=int,
        default=None,
        help="Context left out of pretrain (default: n_contexts - 1).",
    )
    p.add_argument(
        "--heldout-pert",
        type=int,
        default=None,
        help="Perturbation held out of finetune within FT context "
        "(default: n_perturbations - 1). Must not be the control (0).",
    )
    p.add_argument("--control-pert", type=int, default=0)
    p.add_argument("--context-scale", type=float, default=4.0,
                   help="Unused under polygon layout (radius / separation).")
    p.add_argument(
        "--context-radius",
        type=float,
        default=None,
        help="Distance from origin (noise prior) to each context center. "
        "Preferred when sweeping n_contexts: keeps ODE transport length fixed. "
        "Default 4.0 (paper c6_p4; equals separation=4 at 6 contexts). "
        "Legacy c4_p4 used 2*sqrt(2)≈2.828.",
    )
    p.add_argument(
        "--context-separation",
        type=float,
        default=None,
        help="Legacy: nearest-neighbor chord length on the polygon "
        "(radius grows with n_contexts). Mutually exclusive with --context-radius.",
    )
    p.add_argument("--pert-scale", type=float, default=1.5)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument(
        "--min-margin-sigma",
        type=float,
        default=3.5,
        help="Reject a perturbation draw if any two (c,p) cell centers land closer than "
        "this many sigma; redraw --pert-scale (context layout unchanged) until satisfied. "
        "Prevents cross-context cell overlap at a small --context-radius / large n_contexts.",
    )
    p.add_argument(
        "--margin-max-tries",
        type=int,
        default=500,
        help="Max redraw attempts before giving up on --min-margin-sigma.",
    )
    p.add_argument("--n-train", type=int, default=800, help="Train samples per (c,p) cell.")
    p.add_argument("--n-val", type=int, default=200, help="Val samples per (c,p) cell.")
    p.add_argument(
        "--n-heldout-test",
        type=int,
        default=1000,
        help="Test samples for the single heldout (c*, p*) cell.",
    )
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _min_cross_cell_margin(mu_context: np.ndarray, delta_pert: np.ndarray) -> float:
    """Smallest center-to-center distance among all (c,p) cells (all c, all p)."""
    n_c, n_p = mu_context.shape[0], delta_pert.shape[0]
    centers = np.array(
        [mu_context[ci] + delta_pert[pj] for ci in range(n_c) for pj in range(n_p)]
    )
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _sample_cell(
    coupling: AdditiveCoupling,
    *,
    c: int,
    p: int,
    n: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = coupling.mean(c, p)
    cov = float(sigma) ** 2 * np.eye(coupling.dim, dtype=np.float64)
    return rng.multivariate_normal(mu, cov, size=n)


def _append_cell(
    buckets: dict[str, list[np.ndarray]],
    *,
    x: np.ndarray,
    c: int,
    p: int,
    pool: int,
    split: int,
) -> None:
    n = x.shape[0]
    buckets["x"].append(x)
    buckets["c"].append(np.full(n, c, dtype=np.int64))
    buckets["p"].append(np.full(n, p, dtype=np.int64))
    buckets["pool"].append(np.full(n, pool, dtype=np.int64))
    buckets["split"].append(np.full(n, split, dtype=np.int64))


def main() -> None:
    args = parse_args()
    n_c, n_p = int(args.n_contexts), int(args.n_perturbations)
    if n_c < 2 or n_p < 3:
        raise SystemExit("Need n_contexts >= 2 and n_perturbations >= 3 (control + FT + heldout).")

    ft_context = int(args.ft_context) if args.ft_context is not None else n_c - 1
    heldout_pert = int(args.heldout_pert) if args.heldout_pert is not None else n_p - 1
    control = int(args.control_pert)
    if not (0 <= ft_context < n_c):
        raise SystemExit(f"--ft-context must be in [0, {n_c - 1}]")
    if not (0 <= heldout_pert < n_p):
        raise SystemExit(f"--heldout-pert must be in [0, {n_p - 1}]")
    if heldout_pert == control:
        raise SystemExit("--heldout-pert must differ from --control-pert")
    if control < 0 or control >= n_p:
        raise SystemExit(f"--control-pert must be in [0, {n_p - 1}]")

    n_train, n_val = int(args.n_train), int(args.n_val)
    n_held = int(args.n_heldout_test)
    if min(n_train, n_val, n_held) < 1:
        raise SystemExit("n_train / n_val / n_heldout_test must all be >= 1")

    out_dir = args.out_dir or stage_b_data_root(
        REPO_ROOT, dim=args.dim, n_contexts=n_c, n_perturbations=n_p
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.data_seed)
    # Prefer fixed radius so transport length does not grow with n_contexts.
    # Default radius = 4.0, the paper c6_p4 geometry (separation=4 at C=6).
    if args.context_radius is not None and args.context_separation is not None:
        raise SystemExit("Pass only one of --context-radius / --context-separation.")
    if args.context_radius is not None:
        context_radius = float(args.context_radius)
        mu_context = make_polygon_means(n_c, dim=args.dim, radius=context_radius)
        context_separation = float(
            2.0 * context_radius * np.sin(np.pi / float(n_c))
        )
    elif args.context_separation is not None:
        context_separation = float(args.context_separation)
        mu_context = make_polygon_means(
            n_c, dim=args.dim, separation=context_separation
        )
        context_radius = float(np.linalg.norm(mu_context[0]))
    else:
        context_radius = 4.0
        mu_context = make_polygon_means(n_c, dim=args.dim, radius=context_radius)
        context_separation = float(
            2.0 * context_radius * np.sin(np.pi / float(n_c))
        )
    min_margin = float(args.min_margin_sigma) * float(args.sigma)
    margin_tries = 0
    achieved_margin = float("-inf")
    while True:
        base = make_additive_coupling(
            rng,
            n_contexts=n_c,
            n_perturbations=n_p,
            dim=args.dim,
            context_scale=args.context_scale,
            pert_scale=args.pert_scale,
        )
        delta = np.array(base.delta_pert, dtype=np.float64, copy=True)
        delta[control] = 0.0
        margin_tries += 1
        achieved_margin = _min_cross_cell_margin(mu_context, delta)
        if achieved_margin >= min_margin or margin_tries >= int(args.margin_max_tries):
            break
    if achieved_margin < min_margin:
        raise SystemExit(
            f"Could not draw a --pert-scale={args.pert_scale} perturbation set with "
            f"min cross-cell margin >= {args.min_margin_sigma}σ after {margin_tries} tries "
            f"(best {achieved_margin / args.sigma:.2f}σ). Try a larger --context-radius or "
            "a smaller --pert-scale."
        )
    coupling = AdditiveCoupling(mu_context=mu_context, delta_pert=delta)

    pretrain_contexts = [c for c in range(n_c) if c != ft_context]
    ft_perts = [p for p in range(n_p) if p != heldout_pert]
    pretrain_cells = [(c, p) for c in pretrain_contexts for p in range(n_p)]
    finetune_cells = [(ft_context, p) for p in ft_perts]
    heldout_cell = (ft_context, heldout_pert)

    buckets: dict[str, list[np.ndarray]] = {
        "x": [],
        "c": [],
        "p": [],
        "pool": [],
        "split": [],
    }
    cell_rng = np.random.default_rng(args.data_seed + 101)

    for c, p in pretrain_cells:
        x_all = _sample_cell(
            coupling, c=c, p=p, n=n_train + n_val, sigma=args.sigma, rng=cell_rng
        )
        order = cell_rng.permutation(n_train + n_val)
        _append_cell(
            buckets,
            x=x_all[order[:n_train]],
            c=c,
            p=p,
            pool=POOL_PRETRAIN,
            split=SPLIT_TRAIN,
        )
        _append_cell(
            buckets,
            x=x_all[order[n_train:]],
            c=c,
            p=p,
            pool=POOL_PRETRAIN,
            split=SPLIT_VAL,
        )

    for c, p in finetune_cells:
        x_all = _sample_cell(
            coupling, c=c, p=p, n=n_train + n_val, sigma=args.sigma, rng=cell_rng
        )
        order = cell_rng.permutation(n_train + n_val)
        _append_cell(
            buckets,
            x=x_all[order[:n_train]],
            c=c,
            p=p,
            pool=POOL_FINETUNE,
            split=SPLIT_TRAIN,
        )
        _append_cell(
            buckets,
            x=x_all[order[n_train:]],
            c=c,
            p=p,
            pool=POOL_FINETUNE,
            split=SPLIT_VAL,
        )

    x_h = _sample_cell(
        coupling,
        c=heldout_cell[0],
        p=heldout_cell[1],
        n=n_held,
        sigma=args.sigma,
        rng=cell_rng,
    )
    _append_cell(
        buckets,
        x=x_h,
        c=heldout_cell[0],
        p=heldout_cell[1],
        pool=POOL_HELDOUT,
        split=SPLIT_TEST,
    )

    arrays = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    means = np.stack(
        [coupling.mean(c, p) for c in range(n_c) for p in range(n_p)],
        axis=0,
    ).reshape(n_c, n_p, args.dim)

    meta = {
        "stage": "b_shift",
        "coupling": "additive",
        "dim": int(args.dim),
        "n_contexts": n_c,
        "n_perturbations": n_p,
        "ft_context": ft_context,
        "heldout_pert": heldout_pert,
        "heldout_combo": [ft_context, heldout_pert],
        "control_pert": control,
        "pretrain_contexts": pretrain_contexts,
        "finetune_perturbations": ft_perts,
        "pretrain_cells": pretrain_cells,
        "finetune_cells": finetune_cells,
        "heldout_cell": list(heldout_cell),
        "context_scale": float(args.context_scale),
        "context_radius": float(context_radius),
        "context_separation": float(context_separation),
        "pert_scale": float(args.pert_scale),
        "sigma": float(args.sigma),
        "min_margin_sigma": float(args.min_margin_sigma),
        "achieved_margin_sigma": float(achieved_margin / args.sigma),
        "margin_draw_tries": int(margin_tries),
        "data_seed": int(args.data_seed),
        "n_train": n_train,
        "n_val": n_val,
        "n_heldout_test": n_held,
        "split_names": ["train", "val", "test"],
        "split_id": {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST},
        "pool_names": ["pretrain", "finetune", "heldout"],
        "pool_id": {
            "pretrain": POOL_PRETRAIN,
            "finetune": POOL_FINETUNE,
            "heldout": POOL_HELDOUT,
        },
        "mu_context": coupling.mu_context.tolist(),
        "delta_pert": coupling.delta_pert.tolist(),
        "protocol": {
            "pretrain": "all c ≠ ft_context, all p; train+val",
            "finetune": "c = ft_context, p ≠ heldout_pert; train+val",
            "heldout": "(ft_context, heldout_pert); test only",
        },
    }

    np.savez_compressed(
        out_dir / "cp.npz",
        x=arrays["x"],
        c=arrays["c"],
        p=arrays["p"],
        pool=arrays["pool"],
        split=arrays["split"],
        mu_context=coupling.mu_context,
        delta_pert=coupling.delta_pert,
        means=means,
        cov=(float(args.sigma) ** 2) * np.eye(args.dim, dtype=np.float64),
        ft_context=np.asarray([ft_context], dtype=np.int64),
        heldout_pert=np.asarray([heldout_pert], dtype=np.int64),
        control_pert=np.asarray([control], dtype=np.int64),
        pretrain_contexts=np.asarray(pretrain_contexts, dtype=np.int64),
        finetune_perturbations=np.asarray(ft_perts, dtype=np.int64),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _count(pool: int, split: int | None = None) -> int:
        m = arrays["pool"] == pool
        if split is not None:
            m &= arrays["split"] == split
        return int(m.sum())

    print(f"Wrote Stage B data under {out_dir}")
    print(f"coupling=additive  dim={args.dim}  C={n_c}  P={n_p}")
    print(
        f"context_radius={context_radius:.6f}  "
        f"context_separation={context_separation:.6f}  "
        f"(4σ={4.0 * float(args.sigma):.2f})"
    )
    print(
        f"cross-cell margin: {achieved_margin:.4f} ({achieved_margin / args.sigma:.2f}σ) "
        f">= target {args.min_margin_sigma}σ after {margin_tries} pert-draw "
        f"{'try' if margin_tries == 1 else 'tries'}"
    )
    print(f"ft_context={ft_context}  heldout_pert={heldout_pert}  control={control}")
    print(f"pretrain_contexts={pretrain_contexts}  finetune_perts={ft_perts}")
    print(
        "counts: "
        f"pretrain train/val={_count(POOL_PRETRAIN, SPLIT_TRAIN)}/{_count(POOL_PRETRAIN, SPLIT_VAL)}  "
        f"finetune train/val={_count(POOL_FINETUNE, SPLIT_TRAIN)}/{_count(POOL_FINETUNE, SPLIT_VAL)}  "
        f"heldout test={_count(POOL_HELDOUT, SPLIT_TEST)}"
    )
    print(f"cells: pretrain={len(pretrain_cells)}  finetune={len(finetune_cells)}  heldout=1")
    print(f"delta_pert (control pinned 0)=\n{coupling.delta_pert}")


if __name__ == "__main__":
    main()
