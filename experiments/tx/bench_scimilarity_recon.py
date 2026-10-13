#!/usr/bin/env python3
"""SCimilarity → HVG reconstruction: linear vs small MLP, no CFM.

Fits decoders on the Replogle ``train`` split (task-0 lines; HepG2 is
``finetune``/``test``) and scores gene-space reconstruction of ``adata.X``.

Models
------
* ``ridge`` — sklearn Ridge (best affine map)
* ``linear_adam`` — ``nn.Linear``, same optimizer as TaskDecoderBank
* ``mlp_adam`` — one hidden layer GELU MLP

Example
-------
  python experiments/tx/bench_scimilarity_recon.py
  python experiments/tx/bench_scimilarity_recon.py --max-fit-cells 100000 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from trajectory_replay.data.tx_cl_data import load_gene_rows, load_se_rows
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE

DecodeFn = Callable[[np.ndarray], np.ndarray]


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def pearson_rows(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=1, keepdims=True)
    r = recon - recon.mean(axis=1, keepdims=True)
    num = (t * r).sum(axis=1)
    den = np.sqrt((t * t).sum(axis=1) * (r * r).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def pearson_cols(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=0, keepdims=True)
    r = recon - recon.mean(axis=0, keepdims=True)
    num = (t * r).sum(axis=0)
    den = np.sqrt((t * t).sum(axis=0) * (r * r).sum(axis=0))
    return num / np.maximum(den, 1e-12)


def subsample(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if max_n <= 0 or idx.size <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(max_n), replace=False))


def iter_batches(idx: np.ndarray, batch_size: int):
    for start in range(0, idx.size, batch_size):
        yield idx[start : start + batch_size]


def score_pool(
    adata: ad.AnnData,
    z: np.ndarray,
    indices: np.ndarray,
    decode: DecodeFn,
    *,
    batch_size: int,
    max_cells: int,
    seed: int,
) -> dict[str, float]:
    idx = subsample(indices, max_cells, seed)
    if idx.size == 0:
        return {
            "n_cells": 0,
            "mean_mse": float("nan"),
            "r2": float("nan"),
            "mean_pearson_cell": float("nan"),
            "median_pearson_cell": float("nan"),
            "mean_pearson_gene": float("nan"),
            "median_pearson_gene": float("nan"),
        }
    n_genes = int(adata.n_vars)
    sum_sq_err = 0.0
    gene_sum = np.zeros(n_genes, dtype=np.float64)
    gene_sumsq = np.zeros(n_genes, dtype=np.float64)
    gene_true_recon = np.zeros(n_genes, dtype=np.float64)
    gene_recon_sum = np.zeros(n_genes, dtype=np.float64)
    gene_recon_sumsq = np.zeros(n_genes, dtype=np.float64)
    cell_pearson = np.empty(idx.size, dtype=np.float64)
    cell_mse = np.empty(idx.size, dtype=np.float64)
    offset = 0
    for batch in iter_batches(idx, batch_size):
        x = load_gene_rows(adata, batch).astype(np.float64, copy=False)
        recon = np.asarray(decode(load_se_rows(z, batch)), dtype=np.float64)
        err = x - recon
        n = int(batch.size)
        sum_sq_err += float((err * err).sum())
        gene_sum += x.sum(axis=0)
        gene_sumsq += (x * x).sum(axis=0)
        gene_recon_sum += recon.sum(axis=0)
        gene_recon_sumsq += (recon * recon).sum(axis=0)
        gene_true_recon += (x * recon).sum(axis=0)
        cell_pearson[offset : offset + n] = pearson_rows(x, recon)
        cell_mse[offset : offset + n] = (err * err).mean(axis=1)
        offset += n
    n_cells = int(idx.size)
    gene_mean = gene_sum / n_cells
    ss_tot = float(np.sum(gene_sumsq - n_cells * (gene_mean**2)))
    gene_num = gene_true_recon - n_cells * (gene_sum / n_cells) * (gene_recon_sum / n_cells)
    gene_den = np.sqrt(
        (gene_sumsq - n_cells * (gene_sum / n_cells) ** 2)
        * (gene_recon_sumsq - n_cells * (gene_recon_sum / n_cells) ** 2)
    )
    gene_pearson = gene_num / np.maximum(gene_den, 1e-12)
    return {
        "n_cells": n_cells,
        "mean_mse": float(sum_sq_err / max(n_cells * n_genes, 1)),
        "median_mse": float(np.median(cell_mse)),
        "r2": float(1.0 - sum_sq_err / max(ss_tot, 1e-12)),
        "mean_pearson_cell": float(np.mean(cell_pearson)),
        "median_pearson_cell": float(np.median(cell_pearson)),
        "mean_pearson_gene": float(np.mean(gene_pearson)),
        "median_pearson_gene": float(np.median(gene_pearson)),
    }


def fit_ridge(z_fit: np.ndarray, x_fit: np.ndarray, *, alpha: float) -> DecodeFn:
    print(
        f"  ridge alpha={alpha} on {z_fit.shape[0]:,} x {z_fit.shape[1]} → {x_fit.shape[1]}",
        flush=True,
    )
    t0 = time.time()
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(z_fit, x_fit)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)

    def decode(z: np.ndarray) -> np.ndarray:
        return model.predict(np.asarray(z, dtype=np.float64))

    return decode


def parse_mlp_specs(spec: str) -> list[tuple[int, int]]:
    """Parse ``256x1,1024x2`` or ``256,512`` (depth 1)."""
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" in part:
            hidden_s, depth_s = part.split("x", 1)
            hidden, depth = int(hidden_s), int(depth_s)
        else:
            hidden, depth = int(part), 1
        if hidden < 1 or depth < 1:
            raise ValueError(f"Invalid MLP spec {part!r}; hidden and depth must be >= 1")
        out.append((hidden, depth))
    return out


def _mlp(se_dim: int, n_genes: int, hidden: int, depth: int = 1) -> nn.Module:
    if hidden <= 0:
        return nn.Linear(se_dim, n_genes)
    layers: list[nn.Module] = []
    in_dim = se_dim
    for _ in range(int(depth)):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.GELU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, n_genes))
    return nn.Sequential(*layers)


def fit_torch(
    z_fit: np.ndarray,
    x_fit: np.ndarray,
    *,
    hidden: int,
    depth: int = 1,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> DecodeFn:
    name = "linear" if hidden <= 0 else f"mlp h={hidden} d={depth}"
    print(
        f"  {name} adam lr={lr} epochs={epochs} batch={batch_size} "
        f"on {z_fit.shape[0]:,} cells",
        flush=True,
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    layer = _mlp(int(z_fit.shape[1]), int(x_fit.shape[1]), hidden, depth).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    n = int(z_fit.shape[0])
    bs = min(int(batch_size), n)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    t0 = time.time()
    z_t = torch.as_tensor(np.ascontiguousarray(z_fit, dtype=np.float32))
    x_t = torch.as_tensor(np.ascontiguousarray(x_fit, dtype=np.float32))
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        n_seen = 0
        layer.train()
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            zb = z_t[idx].to(device, non_blocking=True)
            xb = x_t[idx].to(device, non_blocking=True)
            loss = F.mse_loss(layer(zb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * int(idx.numel())
            n_seen += int(idx.numel())
        if epoch == 0 or (epoch + 1) % max(epochs // 5, 1) == 0 or epoch + 1 == epochs:
            print(
                f"    epoch {epoch + 1}/{epochs}  mse {epoch_loss / max(n_seen, 1):.6f}",
                flush=True,
            )
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    layer.eval()

    def decode(z: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            arr = np.asarray(z, dtype=np.float32)
            for start in range(0, arr.shape[0], 8192):
                chunk = torch.as_tensor(arr[start : start + 8192], device=device)
                out.append(layer(chunk).cpu().numpy())
        return np.concatenate(out, axis=0)

    return decode


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad-path", type=Path, default=REPO_ROOT / "data_dir/replogle/adata.h5ad")
    p.add_argument(
        "--sci-path", type=Path, default=REPO_ROOT / "data_dir/replogle/scimilarity.npy"
    )
    p.add_argument(
        "--se-path",
        type=Path,
        default=None,
        help="Optional STATE sidecar; Ridge-only reference on the same cells.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir/tx_results/replogle/sci_recon_bench",
    )
    p.add_argument("--max-fit-cells", type=int, default=100_000)
    p.add_argument("--eval-max-cells", type=int, default=0, help="0 = all cells in each pool.")
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--mlp-specs",
        default="256x1,512x1,1024x1,1024x2",
        help="Comma-separated MLP decoder specs: hidden x depth (GELU). "
        "1024x2 is the closest small inverse of the SCimilarity encoder trunk.",
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-adam-linear", action="store_true")
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = make_parser().parse_args()
    if args.smoke:
        args.max_fit_cells = min(args.max_fit_cells or 2048, 2048)
        args.epochs = min(args.epochs, 5)
        args.eval_max_cells = min(args.eval_max_cells or 1024, 1024)
    device_name = args.device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    z_sci = np.load(args.sci_path, mmap_mode="r")
    if z_sci.shape[0] != adata.n_obs:
        raise ValueError(f"sci n_cells={z_sci.shape[0]} != adata n_obs={adata.n_obs}")
    split = np.asarray(adata.obs["split"]).astype(str)
    context = np.asarray(adata.obs["cell_line"]).astype(str)
    hepg2 = REPLOGLE_FEWSHOT_CELL_LINE
    train = np.flatnonzero(split == "train")
    finetune = np.flatnonzero(split == "finetune")
    test = np.flatnonzero(split == "test")
    hepg2_fit = np.flatnonzero((context == hepg2) & (split != "test"))
    hepg2_test = np.flatnonzero((context == hepg2) & (split == "test"))
    fit_idx = subsample(train, args.max_fit_cells, args.seed)
    train_heldout = np.setdiff1d(train, fit_idx, assume_unique=False)
    print(
        f"adata {adata.n_obs:,} x {adata.n_vars}  sci {tuple(z_sci.shape)}  device={device}",
        flush=True,
    )
    print(
        f"fit on train n={fit_idx.size:,}  "
        f"train leftover={train_heldout.size:,}  test={test.size:,}  "
        f"hepg2 fit={hepg2_fit.size:,}  hepg2 test={hepg2_test.size:,}",
        flush=True,
    )

    z_fit = load_se_rows(z_sci, fit_idx)
    x_fit = load_gene_rows(adata, fit_idx)
    models: dict[str, DecodeFn] = {
        "sci_ridge": fit_ridge(z_fit, x_fit, alpha=args.ridge_alpha),
    }
    if not args.skip_adam_linear:
        models["sci_linear_adam"] = fit_torch(
            z_fit,
            x_fit,
            hidden=0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
    mlp_specs = [] if args.skip_mlp else parse_mlp_specs(args.mlp_specs)
    for hidden, depth in mlp_specs:
        models[f"sci_mlp_h{hidden}_d{depth}"] = fit_torch(
            z_fit,
            x_fit,
            hidden=hidden,
            depth=depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
    z_se: np.ndarray | None = None
    if args.se_path is not None:
        z_se = np.load(args.se_path, mmap_mode="r")
        if z_se.shape[0] != adata.n_obs:
            raise ValueError(f"se n_cells={z_se.shape[0]} != adata n_obs={adata.n_obs}")
        models["se_ridge"] = fit_ridge(
            load_se_rows(z_se, fit_idx), x_fit, alpha=args.ridge_alpha
        )

    pools = {
        "train_fit": fit_idx,
        "train_heldout": train_heldout,
        "test": test,
        "hepg2_fit": hepg2_fit,
        "hepg2_test": hepg2_test,
    }
    results: dict[str, Any] = {}
    for model_name, decode in models.items():
        results[model_name] = {}
        print(f"scoring {model_name}", flush=True)
        for pool_name, idx in pools.items():
            if model_name.startswith("se_"):
                if z_se is None:
                    raise RuntimeError("STATE sidecar was not loaded")
                z_eval = z_se
            else:
                z_eval = z_sci
            block = score_pool(
                adata,
                z_eval,
                idx,
                decode,
                batch_size=args.eval_batch_size,
                max_cells=args.eval_max_cells,
                seed=args.seed + 17,
            )
            results[model_name][pool_name] = block
            print(
                f"  {pool_name:14s}  n={block['n_cells']:<7,d}  "
                f"R2={block['r2']:.4f}  "
                f"cellP={block['mean_pearson_cell']:.4f}  "
                f"geneP={block['mean_pearson_gene']:.4f}  "
                f"mse={block['mean_mse']:.5f}",
                flush=True,
            )

    payload = {
        "h5ad_path": str(args.h5ad_path),
        "sci_path": str(args.sci_path),
        "se_path": str(args.se_path) if args.se_path else None,
        "n_fit": int(fit_idx.size),
        "sci_dim": int(z_sci.shape[1]),
        "n_genes": int(adata.n_vars),
        "epochs": int(args.epochs),
        "mlp_specs": [f"{h}x{d}" for h, d in mlp_specs],
        "ridge_alpha": float(args.ridge_alpha),
        "seed": int(args.seed),
        "device": str(device),
        "smoke": bool(args.smoke),
        "metrics": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.out_dir / "metrics.json", payload)
    print(f"Wrote {args.out_dir / 'metrics.json'}", flush=True)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
