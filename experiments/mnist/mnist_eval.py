"""MNIST ODE eval with conditional strata and KNN-stratified unconditional eval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

from trajectory_replay.eval.metrics import ode_stratum_metrics
from trajectory_replay.training.flow_matching import _state_as_matrix


@dataclass(frozen=True)
class MnistEvalResult:
    forget_strata: list[dict[str, float | int]]
    fit_strata: list[dict[str, float | int]]
    knn_meta: dict[str, Any]

    def to_metrics_dict(self) -> dict[str, Any]:
        forget = summarize_strata(self.forget_strata)
        fit = summarize_strata(self.fit_strata)
        out: dict[str, Any] = {
            "eval_knn": self.knn_meta,
            "eval_forget": {"summary": forget, "strata": self.forget_strata},
            "eval_fit": {"summary": fit, "strata": self.fit_strata},
        }
        # Legacy / convenience fields
        out["eval_mse_forget"] = forget.get("mean_mse_to_true")
        out["eval_mse_fit"] = fit.get("mean_mse_to_true")
        out["eval_strata_forget"] = self.forget_strata
        out["eval_strata_fit"] = self.fit_strata
        return out


def add_mnist_eval_args(p) -> None:
    p.add_argument(
        "--eval-uncond-n-samples",
        type=int,
        default=5000,
        help="Unconditional images to generate before KNN stratification.",
    )
    p.add_argument(
        "--eval-uncond-batch-size",
        type=int,
        default=256,
        help="Batch size for unconditional ODE generation at eval.",
    )
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--knn-train-max", type=int, default=20_000)
    p.add_argument(
        "--eval-min-gen-per-class",
        type=int,
        default=32,
        help="Min KNN-assigned gen images per digit to compute class metrics.",
    )


def flatten_x_numpy(x: torch.Tensor) -> np.ndarray:
    return _state_as_matrix(x).cpu().numpy().astype(np.float64)


def subsample_indices(n: int, k: int, seed: int) -> torch.Tensor:
    if n <= k:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[:k]


def fit_knn_classifier(
    x: torch.Tensor,
    p: torch.Tensor,
    *,
    k: int,
    train_max: int,
    seed: int,
) -> tuple[KNeighborsClassifier, dict[str, Any]]:
    idx = subsample_indices(len(x), train_max, seed)
    x_tr = flatten_x_numpy(x[idx])
    y_tr = p[idx].numpy().astype(np.int64)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=1, metric="euclidean")
    clf.fit(x_tr, y_tr)
    meta = {"k": k, "n_train": int(len(y_tr)), "train_max": train_max, "seed": seed}
    return clf, meta


def summarize_strata(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {"n_strata": 0}
    keys = ("mse_to_true", "energy", "mse_mean_vectors")
    out: dict[str, float | int] = {"n_strata": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
        if vals:
            out[f"mean_{key}"] = float(np.mean(vals))
            out[f"median_{key}"] = float(np.median(vals))
    out["total_n_gen"] = int(sum(int(r.get("n_gen", 0)) for r in rows))
    out["total_n_true"] = int(sum(int(r.get("n_true", r.get("n", 0))) for r in rows))
    return out


def compare_clouds(
    x_true: torch.Tensor,
    x_gen: torch.Tensor,
    *,
    digit: int,
    seed: int,
    max_pairs: int | None = None,
) -> dict[str, float | int]:
    n_true = int(x_true.shape[0])
    n_gen = int(x_gen.shape[0])
    if n_true == 0 or n_gen == 0:
        return {
            "p": digit,
            "n_true": n_true,
            "n_gen": n_gen,
            "mse_to_true": float("nan"),
            "energy": float("nan"),
            "mse_mean_vectors": float("nan"),
        }
    n = min(n_true, n_gen)
    if max_pairs is not None:
        n = min(n, max_pairs)
    idx_t = subsample_indices(n_true, n, seed)
    idx_g = subsample_indices(n_gen, n, seed + 1)
    xt = x_true[idx_t].to(x_gen.device)
    xg = x_gen[idx_g]
    paired_mse = F.mse_loss(xg, xt).item()
    # Cap cloud size for energy (O(n^2) memory); paired MSE already uses ``n`` pairs.
    max_cloud = max(n, 128)
    idx_t_m = subsample_indices(n_true, min(n_true, max_cloud), seed + 2)
    idx_g_m = subsample_indices(n_gen, min(n_gen, max_cloud), seed + 3)
    metrics = ode_stratum_metrics(
        flatten_x_numpy(x_true[idx_t_m]),
        flatten_x_numpy(x_gen[idx_g_m]),
        paired_mse=paired_mse,
    )
    return {"p": digit, "n_true": n_true, "n_gen": n_gen, **metrics}


@torch.no_grad()
def sample_conditional_digit(
    model: nn.Module,
    *,
    digit: int,
    n: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> torch.Tensor:
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(eval_seed + int(digit) * 10_007)
    cb = torch.zeros(n, dtype=torch.long, device=device)
    pb = torch.full((n,), digit, dtype=torch.long, device=device)
    return model.sample(
        cb,
        pb,
        method=ode_method,
        rtol=ode_rtol,
        atol=ode_atol,
        options=ode_options,
        generator=gen,
    ).cpu()


@torch.no_grad()
def sample_unconditional_batched(
    model: nn.Module,
    *,
    n: int,
    batch_size: int,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    remaining = n
    batch_idx = 0
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    while remaining > 0:
        b = min(batch_size, remaining)
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(eval_seed + batch_idx)
        xh = model.sample(
            batch_size=b,
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            options=ode_options,
            generator=gen,
        ).cpu()
        chunks.append(xh)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        remaining -= b
        batch_idx += 1
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def save_uncond_knn_digit_sample_grids(
    model: nn.Module,
    x_forget: torch.Tensor,
    p_forget: torch.Tensor,
    x_fit: torch.Tensor,
    p_fit: torch.Tensor,
    *,
    out_dir: Path,
    device: torch.device,
    plot_n_per_digit: int,
    plot_digits: list[int] | None = None,
    max_digits: int = 3,
    knn_k: int = 5,
    knn_train_max: int = 20_000,
    eval_seed: int = 0,
    uncond_n_samples: int = 3000,
    uncond_batch_size: int = 256,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> list[int]:
    """Save real/gen grids for digits classified by KNN on unconditional samples."""
    from experiments.mnist.mnist_common import save_digit_grid

    if plot_digits is None:
        plot_digits = sorted({int(v) for v in p_forget.unique().tolist()})[:max_digits]

    x_knn = torch.cat([x_forget, x_fit], dim=0)
    p_knn = torch.cat([p_forget, p_fit], dim=0)
    clf, _ = fit_knn_classifier(
        x_knn,
        p_knn,
        k=knn_k,
        train_max=knn_train_max,
        seed=eval_seed,
    )
    x_gen = sample_unconditional_batched(
        model,
        n=uncond_n_samples,
        batch_size=uncond_batch_size,
        device=device,
        eval_seed=eval_seed + 50_000,
        ode_method=ode_method,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        ode_options=ode_options,
    )
    pred = clf.predict(flatten_x_numpy(x_gen))

    plotted: list[int] = []
    for digit in plot_digits:
        real_idx = (p_forget == digit).nonzero(as_tuple=True)[0]
        if len(real_idx) == 0:
            continue
        real = x_forget[real_idx[:plot_n_per_digit]]

        pool = np.nonzero(pred == digit)[0]
        if len(pool) == 0:
            print(f"  skip gen_{digit}: no unconditional samples KNN-assigned to p={digit}")
            save_digit_grid(out_dir / f"real_p{digit}.png", real, title=f"Real p{digit}")
            continue

        n_pick = min(plot_n_per_digit, len(pool))
        pick = subsample_indices(len(pool), n_pick, eval_seed + digit + 1000).numpy()
        gen = x_gen[torch.as_tensor(pool[pick], dtype=torch.long)]

        tag = f"p{digit}"
        save_digit_grid(out_dir / f"real_{tag}.png", real, title=f"Real {tag}")
        save_digit_grid(
            out_dir / f"gen_{tag}.png",
            gen,
            title=f"Uncond gen → KNN {tag} (n={len(pool)})",
        )
        plotted.append(digit)
    return plotted


def real_digit_tensors(
    x: torch.Tensor,
    p: torch.Tensor,
    digit: int,
    *,
    max_true: int,
    seed: int,
) -> torch.Tensor:
    idx = (p == digit).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return x[:0]
    idx = idx[subsample_indices(len(idx), max_true, seed)]
    return x[idx]


def eval_conditional_digits(
    model: nn.Module,
    *,
    digits: list[int],
    x_true_pool: torch.Tensor,
    p_true_pool: torch.Tensor,
    device: torch.device,
    eval_seed: int,
    max_true_per_digit: int,
    max_gen_per_digit: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for digit in digits:
        x_true = real_digit_tensors(
            x_true_pool,
            p_true_pool,
            digit,
            max_true=max_true_per_digit,
            seed=eval_seed + digit,
        )
        if len(x_true) == 0:
            continue
        n_gen = min(max_gen_per_digit, len(x_true))
        x_gen = sample_conditional_digit(
            model,
            digit=digit,
            n=n_gen,
            device=device,
            eval_seed=eval_seed + digit,
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_options=ode_options,
        )
        rows.append(
            compare_clouds(
                x_true,
                x_gen,
                digit=digit,
                seed=eval_seed + digit + 17,
                max_pairs=max_true_per_digit,
            )
        )
    return rows


def _uncond_rows_from_predictions(
    x_gen_all: torch.Tensor,
    pred: np.ndarray,
    *,
    digits: list[int],
    x_true_pool: torch.Tensor,
    p_true_pool: torch.Tensor,
    max_true_per_digit: int,
    min_gen_per_class: int,
    eval_seed: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for digit in digits:
        gen_idx = torch.tensor(np.nonzero(pred == digit)[0], dtype=torch.long)
        x_gen = x_gen_all[gen_idx] if len(gen_idx) > 0 else x_gen_all[:0]
        if len(x_gen) < min_gen_per_class:
            rows.append(
                {
                    "p": digit,
                    "n_true": int((p_true_pool == digit).sum()),
                    "n_gen": int(len(x_gen)),
                    "n_gen_assigned": int(len(x_gen)),
                    "mse_to_true": float("nan"),
                    "energy": float("nan"),
                    "mse_mean_vectors": float("nan"),
                }
            )
            continue
        x_true = real_digit_tensors(
            x_true_pool,
            p_true_pool,
            digit,
            max_true=max_true_per_digit,
            seed=eval_seed + digit,
        )
        row = compare_clouds(
            x_true,
            x_gen,
            digit=digit,
            seed=eval_seed + digit + 17,
            max_pairs=max_true_per_digit,
        )
        row["n_gen_assigned"] = int(len(x_gen))
        rows.append(row)
    return rows


def eval_unconditional_knn_digits(
    model: nn.Module,
    *,
    digits: list[int],
    x_true_pool: torch.Tensor,
    p_true_pool: torch.Tensor,
    clf: KNeighborsClassifier,
    device: torch.device,
    eval_seed: int,
    n_samples: int,
    batch_size: int,
    max_true_per_digit: int,
    min_gen_per_class: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
) -> tuple[list[dict[str, float | int]], dict[int, int]]:
    x_gen_all = sample_unconditional_batched(
        model,
        n=n_samples,
        batch_size=batch_size,
        device=device,
        eval_seed=eval_seed,
        ode_method=ode_method,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        ode_options=ode_options,
    )
    pred = clf.predict(flatten_x_numpy(x_gen_all))
    pred_counts = {int(d): int((pred == d).sum()) for d in np.unique(pred)}
    rows = _uncond_rows_from_predictions(
        x_gen_all,
        pred,
        digits=digits,
        x_true_pool=x_true_pool,
        p_true_pool=p_true_pool,
        max_true_per_digit=max_true_per_digit,
        min_gen_per_class=min_gen_per_class,
        eval_seed=eval_seed,
    )
    return rows, pred_counts


def run_mnist_ode_eval(
    model: nn.Module,
    *,
    use_conditioning: bool,
    held_out: int,
    x_forget: torch.Tensor,
    p_forget: torch.Tensor,
    x_fit: torch.Tensor,
    p_fit: torch.Tensor,
    x_knn: torch.Tensor,
    p_knn: torch.Tensor,
    device: torch.device,
    eval_seed: int,
    max_true_per_digit: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    knn_k: int = 5,
    knn_train_max: int = 20_000,
    uncond_n_samples: int = 5000,
    uncond_batch_size: int = 256,
    min_gen_per_class: int = 32,
) -> MnistEvalResult:
    forget_digits = sorted({int(v) for v in p_forget.unique().tolist() if int(v) != held_out})
    fit_digits = [held_out]

    knn_meta: dict[str, Any]
    if use_conditioning:
        knn_meta = {"mode": "conditional", "knn_used": False}
        forget_rows = eval_conditional_digits(
            model,
            digits=forget_digits,
            x_true_pool=x_forget,
            p_true_pool=p_forget,
            device=device,
            eval_seed=eval_seed,
            max_true_per_digit=max_true_per_digit,
            max_gen_per_digit=max_true_per_digit,
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_options=ode_options,
        )
        fit_rows = eval_conditional_digits(
            model,
            digits=fit_digits,
            x_true_pool=x_fit,
            p_true_pool=p_fit,
            device=device,
            eval_seed=eval_seed + 50_000,
            max_true_per_digit=max_true_per_digit,
            max_gen_per_digit=max_true_per_digit,
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_options=ode_options,
        )
    else:
        clf, knn_fit = fit_knn_classifier(
            x_knn,
            p_knn,
            k=knn_k,
            train_max=knn_train_max,
            seed=eval_seed,
        )
        x_gen_all = sample_unconditional_batched(
            model,
            n=uncond_n_samples,
            batch_size=uncond_batch_size,
            device=device,
            eval_seed=eval_seed,
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_options=ode_options,
        )
        pred = clf.predict(flatten_x_numpy(x_gen_all))
        pred_counts = {int(d): int((pred == d).sum()) for d in np.unique(pred)}
        knn_meta = {
            "mode": "unconditional_knn",
            "knn_used": True,
            **knn_fit,
            "uncond_n_samples": uncond_n_samples,
            "uncond_batch_size": uncond_batch_size,
            "min_gen_per_class": min_gen_per_class,
            "pred_counts_all": pred_counts,
        }
        forget_rows = _uncond_rows_from_predictions(
            x_gen_all,
            pred,
            digits=forget_digits,
            x_true_pool=x_forget,
            p_true_pool=p_forget,
            max_true_per_digit=max_true_per_digit,
            min_gen_per_class=min_gen_per_class,
            eval_seed=eval_seed,
        )
        fit_rows = _uncond_rows_from_predictions(
            x_gen_all,
            pred,
            digits=fit_digits,
            x_true_pool=x_fit,
            p_true_pool=p_fit,
            max_true_per_digit=max_true_per_digit,
            min_gen_per_class=min_gen_per_class,
            eval_seed=eval_seed + 50_000,
        )

    return MnistEvalResult(
        forget_strata=forget_rows,
        fit_strata=fit_rows,
        knn_meta=knn_meta,
    )


def print_eval_summary(result: MnistEvalResult, *, held_out: int) -> None:
    forget = summarize_strata(result.forget_strata)
    fit = summarize_strata(result.fit_strata)

    def _f(key: str) -> str:
        v = forget.get(key, float("nan"))
        return f"{v:.6f}" if isinstance(v, float) and v == v else "nan"

    def _fe(key: str) -> str:
        v = forget.get(key, float("nan"))
        return f"{v:.4f}" if isinstance(v, float) and v == v else "nan"

    print(
        f"Eval forget (digits != {held_out}): mean_mse={_f('mean_mse_to_true')}  "
        f"mean_energy={_fe('mean_energy')}  n_strata={forget.get('n_strata', 0)}"
    )
    if result.fit_strata:
        fv = fit.get("mean_mse_to_true", float("nan"))
        fe = fit.get("mean_energy", float("nan"))
        mse_s = f"{fv:.6f}" if isinstance(fv, float) and fv == fv else "nan"
        en_s = f"{fe:.4f}" if isinstance(fe, float) and fe == fe else "nan"
        print(
            f"Eval fit (digit {held_out}): mean_mse={mse_s}  "
            f"mean_energy={en_s}  n_strata={fit.get('n_strata', 0)}"
        )
    if result.knn_meta.get("knn_used"):
        print(f"  KNN k={result.knn_meta.get('k')}  uncond_n={result.knn_meta.get('uncond_n_samples')}")
        if "pred_counts_all" in result.knn_meta:
            print(f"  pred histogram: {result.knn_meta['pred_counts_all']}")
