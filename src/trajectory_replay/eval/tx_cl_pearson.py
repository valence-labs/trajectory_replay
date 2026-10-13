"""Fast gene-space Pearson-Δ selection metric for Tx CL λ sweeps.

This is the lightweight stage-C scorer.  Perturbed cells are generated with
the CFM ODE then decoded.  Control basal on the predicted side is
``D_k(observed SE)`` by default (``pred_control='decoded'``), not an ODE
sample of the control condition.  ``generated`` ODEs the control condition
from noise (STATE-style).  Real basal is observed gene expression.
Macro-average over ``(context, perturbation[, covariates])`` strata.  Full
CellEval ``minimal`` is wired in the following stage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from trajectory_replay.data.tx_cl_data import TxClArrays
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank
from trajectory_replay.model.flow_matching import ConditionalFlowMatching

PRED_CONTROL_MODES = ("decoded", "observed", "generated")
FLOW_SOURCES = ("noise", "control")
# Residual Pearson-Δ subtracts the pool mean over scored perturbations.
# One perturbation cannot define a residual; two still define a contrast.
MIN_RESIDUAL_POOL = 2


def pearson_vector(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def _stratum_keys(arrays: TxClArrays, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    cols = [
        arrays.c[idx],
        arrays.p[idx],
        *(arrays.z[col][idx] for col in arrays.covariate_col_order),
    ]
    return np.stack(cols, axis=1).astype(np.int64, copy=False)


def stratum_generation_plan(
    arrays: TxClArrays,
    indices: np.ndarray,
    task_ids: np.ndarray,
    *,
    samples_per_stratum: int,
    control_pert_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows to predict, decoupling the sample count from the observed count.

    With ``samples_per_stratum <= 0`` the plan is the identity: one prediction
    per observed cell, so a stratum with 53 cells yields a 53-sample estimate
    of the model mean and one with 500 yields 500. That makes stratum-level
    Monte Carlo error a function of assay depth rather than of the model.

    Otherwise every non-control stratum gets exactly ``samples_per_stratum``
    rows. A stratum key *is* ``(c, p[, z...])``, so one representative row
    repeated K times reproduces that stratum's condition exactly. Controls are
    left as the observed cells: the predicted basal is a decoded
    reconstruction of observed latents, not a sample, so replicating it would
    add no information.

    Returns the row indices to predict, their task ids, and for each predicted
    row the position in ``indices`` it was taken from (``-1`` for generated
    rows, which have no single observed counterpart).
    """
    idx = np.asarray(indices, dtype=np.int64)
    tids = np.asarray(task_ids)
    if idx.size != tids.shape[0]:
        raise ValueError("indices and task_ids must have equal length")
    k = int(samples_per_stratum)
    if k <= 0 or idx.size == 0:
        return idx, tids, np.arange(idx.size, dtype=np.int64)

    is_ctrl = (
        arrays.p[idx] == int(control_pert_index)
        if control_pert_index is not None
        else np.zeros(idx.size, dtype=bool)
    )
    pert_pos = np.flatnonzero(~is_ctrl)
    ctrl_pos = np.flatnonzero(is_ctrl)

    gen_idx = np.empty(0, dtype=np.int64)
    gen_tids = tids[:0]
    if pert_pos.size:
        keys = _stratum_keys(arrays, idx[pert_pos])
        _, first = np.unique(keys, axis=0, return_index=True)
        representatives = pert_pos[np.sort(first)]
        gen_idx = np.repeat(idx[representatives], k)
        gen_tids = np.repeat(tids[representatives], k)

    pred_idx = np.concatenate((gen_idx, idx[ctrl_pos]))
    pred_tids = np.concatenate((gen_tids, tids[ctrl_pos]))
    pred_src = np.concatenate(
        (np.full(gen_idx.size, -1, dtype=np.int64), ctrl_pos)
    )
    return pred_idx, pred_tids, pred_src


@torch.no_grad()
def generate_se_for_indices(
    model: ConditionalFlowMatching,
    arrays: TxClArrays,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    ode_method: str,
    ode_step_size: float | None,
    ode_rtol: float,
    ode_atol: float,
) -> np.ndarray:
    """Generate one SE row per supplied real-reference row's condition."""
    idx = np.asarray(indices, dtype=np.int64)
    chunks: list[np.ndarray] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    options = {"step_size": float(ode_step_size)} if ode_step_size is not None else None
    model.eval()
    for start in range(0, idx.size, batch_size):
        take = idx[start : start + batch_size]
        c = torch.from_numpy(arrays.c[take]).long().to(device)
        p = torch.from_numpy(arrays.p[take]).long().to(device)
        z = (
            {
                col: torch.from_numpy(arrays.z[col][take]).long().to(device)
                for col in arrays.covariate_col_order
            }
            if arrays.covariate_col_order
            else None
        )
        pred = model.sample(
            c,
            p,
            z,
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            options=options,
            generator=gen,
        )
        chunks.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, arrays.se_dim), np.float32)


def sample_matched_control_indices(
    arrays: TxClArrays,
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Resample controls matching each target's context and covariates."""
    target = np.asarray(target_indices, dtype=np.int64)
    source = np.asarray(source_indices, dtype=np.int64)
    control = arrays.encoding.control_pert_index
    if control is None:
        raise ValueError("data-to-data generation requires a control perturbation")
    source = source[arrays.p[source] == int(control)]
    if source.size == 0:
        raise ValueError("data-to-data generation source has no control cells")

    def keys(indices: np.ndarray) -> list[tuple[int, ...]]:
        return [
            tuple(
                [int(arrays.c[i])]
                + [int(arrays.z[col][i]) for col in arrays.covariate_col_order]
            )
            for i in indices
        ]

    pools: dict[tuple[int, ...], list[int]] = {}
    for i, key in zip(source, keys(source), strict=True):
        pools.setdefault(key, []).append(int(i))
    rng = np.random.default_rng(seed)
    sampled = np.empty(target.size, dtype=np.int64)
    for pos, key in enumerate(keys(target)):
        pool = pools.get(key)
        if not pool:
            raise ValueError(
                "No matched control source for context/covariates "
                f"{key}; source controls must cover every target stratum"
            )
        sampled[pos] = pool[int(rng.integers(len(pool)))]
    return sampled


@torch.no_grad()
def generate_se_from_controls_for_indices(
    model: ConditionalFlowMatching,
    arrays: TxClArrays,
    indices: np.ndarray,
    source_indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    ode_method: str,
    ode_step_size: float | None,
    ode_rtol: float,
    ode_atol: float,
) -> np.ndarray:
    """Generate targets by integrating matched observed controls to ``t=1``."""
    idx = np.asarray(indices, dtype=np.int64)
    source = sample_matched_control_indices(
        arrays, idx, source_indices, seed=seed
    )
    chunks: list[np.ndarray] = []
    options = {"step_size": float(ode_step_size)} if ode_step_size is not None else None
    model.eval()
    for start in range(0, idx.size, batch_size):
        take = idx[start : start + batch_size]
        src = source[start : start + batch_size]
        x0 = torch.from_numpy(np.asarray(arrays.se[src], dtype=np.float32)).to(device)
        c = torch.from_numpy(arrays.c[take]).long().to(device)
        p = torch.from_numpy(arrays.p[take]).long().to(device)
        z = (
            {
                col: torch.from_numpy(arrays.z[col][take]).long().to(device)
                for col in arrays.covariate_col_order
            }
            if arrays.covariate_col_order
            else None
        )
        pred = model.sample_from_x0(
            x0,
            c,
            p,
            z,
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            options=options,
        )
        chunks.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, arrays.se_dim), np.float32)


def decode_observed_se(
    arrays: TxClArrays,
    decoders: TaskDecoderBank,
    indices: np.ndarray,
    task_ids: np.ndarray,
) -> np.ndarray:
    """Map stored SE through the task decoder (reconstruction, not generation)."""
    idx = np.asarray(indices, dtype=np.int64)
    return decoders.decode(np.asarray(arrays.se[idx]), np.asarray(task_ids))


def predicted_genes_for_indices(
    model: ConditionalFlowMatching,
    arrays: TxClArrays,
    decoders: TaskDecoderBank,
    indices: np.ndarray,
    task_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    ode_method: str,
    ode_step_size: float | None,
    ode_rtol: float,
    ode_atol: float,
    pred_control: str = "decoded",
    flow_source: str = "noise",
    source_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Generate perturbed cells; fill controls per ``pred_control``.

    ``decoded`` (default): reconstruct observed control latents.
    ``generated``: ODE the control condition from noise, like STATE.
    ``observed``: still reconstruct here; ``apply_pred_control_genes`` then
    copies real ``X`` onto control rows.
    """
    idx = np.asarray(indices, dtype=np.int64)
    tids = np.asarray(task_ids)
    if idx.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    mode = str(pred_control)
    if mode not in PRED_CONTROL_MODES:
        raise ValueError(
            f"pred_control must be one of {PRED_CONTROL_MODES}, got {mode!r}"
        )
    source_mode = str(flow_source)
    if source_mode not in FLOW_SOURCES:
        raise ValueError(f"flow_source must be one of {FLOW_SOURCES}, got {source_mode!r}")
    if source_mode == "control" and mode == "generated":
        raise ValueError("pred_control='generated' is only defined for noise-to-data")
    if source_mode == "control" and source_indices is None:
        raise ValueError("control-source generation requires source_indices")
    control = arrays.encoding.control_pert_index
    is_ctrl = (
        arrays.p[idx] == int(control)
        if control is not None
        else np.zeros(idx.size, dtype=bool)
    )
    generate_all = mode == "generated" or control is None
    if generate_all:
        pred_se = generate_se_for_indices(
            model,
            arrays,
            idx,
            device=device,
            batch_size=batch_size,
            seed=seed,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
        )
        return decoders.decode(pred_se, tids)

    pert_pos = np.flatnonzero(~is_ctrl)
    ctrl_pos = np.flatnonzero(is_ctrl)
    out: np.ndarray | None = None
    if pert_pos.size:
        generate = (
            generate_se_from_controls_for_indices
            if source_mode == "control"
            else generate_se_for_indices
        )
        source_kw = (
            {"source_indices": np.asarray(source_indices, dtype=np.int64)}
            if source_mode == "control"
            else {}
        )
        pred_se = generate(
            model,
            arrays,
            idx[pert_pos],
            device=device,
            batch_size=batch_size,
            seed=seed,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            **source_kw,
        )
        pert_genes = decoders.decode(pred_se, tids[pert_pos])
        out = np.empty((idx.size, pert_genes.shape[1]), dtype=np.float64)
        out[pert_pos] = pert_genes
    if ctrl_pos.size:
        recon = decode_observed_se(arrays, decoders, idx[ctrl_pos], tids[ctrl_pos])
        if out is None:
            out = np.empty((idx.size, recon.shape[1]), dtype=np.float64)
        out[ctrl_pos] = recon
    assert out is not None
    return out


def apply_pred_control_genes(
    arrays: TxClArrays,
    indices: np.ndarray,
    true_genes: np.ndarray,
    pred_genes: np.ndarray,
    *,
    mode: str = "decoded",
) -> np.ndarray:
    """Choose predicted control basal after ``predicted_genes_for_indices``.

    ``decoded`` / ``generated`` leave control rows as produced (reconstructed
    observed latents vs ODE-from-noise). ``observed`` copies real ``X`` onto
    control rows so ``pred_delta = D(z_pert) - X_ctrl``.
    """
    if mode in ("decoded", "generated"):
        return pred_genes
    if mode != "observed":
        raise ValueError(
            f"pred_control mode must be one of {PRED_CONTROL_MODES}, got {mode!r}"
        )
    control = arrays.encoding.control_pert_index
    if control is None:
        return pred_genes
    is_ctrl = np.asarray(arrays.p[np.asarray(indices, dtype=np.int64)]) == int(control)
    out = np.array(pred_genes, copy=True, dtype=np.float64)
    out[is_ctrl] = np.asarray(true_genes, dtype=np.float64)[is_ctrl]
    return out


def score_gene_strata(
    arrays: TxClArrays,
    indices: np.ndarray,
    true_genes: np.ndarray,
    pred_genes: np.ndarray,
    task_ids: np.ndarray,
    *,
    control_pert_index: int | None,
    pred_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Macro Pearson and Pearson-Δ over strata.

    Pearson-Δ uses split basals:

    ``pearson(real_pert - real_control, pred_pert - D(observed_control_SE))``

    Real basal is observed control genes. Predicted basal is the task
    decoder applied to the *observed* control SE (reconstruction, not
    generated control). Predicted perts are ODE samples. Control strata
    must be present in these rows. Train/finetune pools always include
    them; held-out STATE ``test`` rows do not, so the caller injects
    same-context controls. A pool that still lacks a basal on either side
    contributes plain Pearson instead of being dropped.

    ``pred_indices`` lets the predicted rows be a different set from the
    observed ones, which is what fixed-K generation produces. Both sides are
    still reduced to one mean per stratum, so only the estimator's sample count
    changes.

    Residual Pearson-Δ subtracts each pool's mean delta. A pool needs at least
    ``MIN_RESIDUAL_POOL`` scored perturbations; smaller pools contribute NaN
    rather than a hard 0.0 from a degenerate residual.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if not (idx.size == true_genes.shape[0] == task_ids.shape[0]):
        raise ValueError("indices, genes and task_ids must have equal length")
    pred_idx = idx if pred_indices is None else np.asarray(pred_indices, dtype=np.int64)
    if pred_idx.size != pred_genes.shape[0]:
        raise ValueError("predicted rows must align with their own indices")
    keys = _stratum_keys(arrays, idx)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)

    true_means: dict[tuple[int, ...], np.ndarray] = {}
    task_by_key: dict[tuple[int, ...], int] = {}
    for j, row in enumerate(uniq):
        mask = inverse == j
        key = tuple(int(v) for v in row)
        tids = np.unique(task_ids[mask])
        if tids.size != 1:
            raise ValueError(f"stratum {key} spans task ids {tids.tolist()}")
        true_means[key] = true_genes[mask].mean(axis=0, dtype=np.float64)
        task_by_key[key] = int(tids[0])

    pred_means: dict[tuple[int, ...], np.ndarray] = {}
    pred_uniq, pred_inverse = np.unique(
        _stratum_keys(arrays, pred_idx), axis=0, return_inverse=True
    )
    for j, row in enumerate(pred_uniq):
        pred_means[tuple(int(v) for v in row)] = pred_genes[
            pred_inverse == j
        ].mean(axis=0, dtype=np.float64)
    missing = set(true_means) - set(pred_means)
    if missing:
        raise ValueError(
            f"{len(missing)} strata have observed cells but no prediction; "
            f"first={sorted(missing)[0]}"
        )

    # Basal pool key excludes perturbation: (context, z...).
    real_basal: dict[tuple[int, ...], np.ndarray] = {}
    pred_basal: dict[tuple[int, ...], np.ndarray] = {}
    if control_pert_index is not None:
        for key, mean in true_means.items():
            if key[1] == int(control_pert_index):
                pool = (key[0], *key[2:])
                real_basal[pool] = mean
                pred_basal[pool] = pred_means[key]

    rows: list[dict[str, Any]] = []
    true_deltas: dict[tuple[int, ...], np.ndarray] = {}
    pred_deltas: dict[tuple[int, ...], np.ndarray] = {}
    pool_of: dict[tuple[int, ...], tuple[int, ...]] = {}
    for key in sorted(true_means):
        if control_pert_index is not None and key[1] == int(control_pert_index):
            continue
        yt = true_means[key]
        yp = pred_means[key]
        pool = (key[0], *key[2:])
        p = pearson_vector(yt, yp)
        pd = float("nan")
        has_basal = pool in real_basal and pool in pred_basal
        if has_basal:
            td = yt - real_basal[pool]
            ppd = yp - pred_basal[pool]
            pd = pearson_vector(td, ppd)
            true_deltas[key] = td
            pred_deltas[key] = ppd
            pool_of[key] = pool
        rows.append(
            {
                "key": list(key),
                "task_id": task_by_key[key],
                "pearson": p,
                "pearson_delta": pd,
                "pearson_delta_residual": float("nan"),
                "selection_score": pd if np.isfinite(pd) else p,
                "has_basal": has_basal,
            }
        )

    residual_by_key: dict[tuple[int, ...], float] = {}
    pools: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for key, pool in pool_of.items():
        pools.setdefault(pool, []).append(key)
    for ks in pools.values():
        if len(ks) < MIN_RESIDUAL_POOL:
            for k in ks:
                residual_by_key[k] = float("nan")
            continue
        true_mat = np.stack([true_deltas[k] for k in ks], axis=0)
        pred_mat = np.stack([pred_deltas[k] for k in ks], axis=0)
        true_res = true_mat - true_mat.mean(axis=0, keepdims=True)
        pred_res = pred_mat - pred_mat.mean(axis=0, keepdims=True)
        if float(np.linalg.norm(pred_res)) <= 1e-12:
            for k in ks:
                residual_by_key[k] = 0.0
            continue
        for i, k in enumerate(ks):
            residual_by_key[k] = pearson_vector(true_res[i], pred_res[i])
    for row in rows:
        key = tuple(int(v) for v in row["key"])
        if key in residual_by_key:
            row["pearson_delta_residual"] = residual_by_key[key]

    pearson = np.asarray([r["pearson"] for r in rows], dtype=np.float64)
    delta = np.asarray([r["pearson_delta"] for r in rows], dtype=np.float64)
    residual = np.asarray(
        [r["pearson_delta_residual"] for r in rows], dtype=np.float64
    )
    mean_p = float(np.nanmean(pearson)) if np.isfinite(pearson).any() else float("nan")
    mean_pd = float(np.nanmean(delta)) if np.isfinite(delta).any() else float("nan")
    n_residual = int(np.isfinite(residual).sum())
    mean_res = (
        float(np.nanmean(residual)) if n_residual else float("nan")
    )
    effective = np.asarray([r["selection_score"] for r in rows], dtype=np.float64)
    n_fallback = int(sum(not bool(r["has_basal"]) for r in rows))
    selection_metric = (
        "pearson_delta"
        if n_fallback == 0
        else "pearson_delta_with_pearson_fallback"
    )
    selection_score = (
        float(np.nanmean(effective)) if np.isfinite(effective).any() else float("nan")
    )
    per_task: dict[str, dict[str, float | int]] = {}
    for tid in sorted(set(int(r["task_id"]) for r in rows)):
        rr = [r for r in rows if int(r["task_id"]) == tid]
        tp = np.asarray([r["pearson"] for r in rr], dtype=np.float64)
        td = np.asarray([r["pearson_delta"] for r in rr], dtype=np.float64)
        tr = np.asarray(
            [r["pearson_delta_residual"] for r in rr], dtype=np.float64
        )
        per_task[str(tid)] = {
            "n_strata": len(rr),
            "n_delta_strata": int(np.isfinite(td).sum()),
            "pearson": float(np.nanmean(tp)) if np.isfinite(tp).any() else float("nan"),
            "pearson_delta": (
                float(np.nanmean(td)) if np.isfinite(td).any() else float("nan")
            ),
            "n_residual_strata": int(np.isfinite(tr).sum()),
            "pearson_delta_residual": (
                float(np.nanmean(tr)) if np.isfinite(tr).any() else float("nan")
            ),
            "selection_score": float(
                np.nanmean(
                    np.asarray([r["selection_score"] for r in rr], dtype=np.float64)
                )
            ),
            "n_pearson_fallback_strata": int(
                sum(not bool(r["has_basal"]) for r in rr)
            ),
        }
    return {
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "pearson_macro": mean_p,
        "pearson_delta_macro": mean_pd,
        "pearson_delta_residual_macro": mean_res,
        "n_strata": len(rows),
        "n_delta_strata": int(np.isfinite(delta).sum()),
        "n_residual_strata": n_residual,
        "n_pearson_fallback_strata": n_fallback,
        "per_task": per_task,
        "rows": rows,
    }


def old_new_balanced_selection(result: dict[str, Any], *, new_task_id: int) -> dict[str, Any]:
    """Average the old-task stratum macro with the new-task stratum macro.

    ``score_gene_strata`` is unweighted over every ``(c, p)`` stratum, so a
    pretrain task with thousands of genes dominates λ-grid. This re-score
    takes the mean Pearson-Δ (with the same Pearson fallback) over old
    strata (``task_id < new_task_id``) and over new strata
    (``task_id == new_task_id``), then averages those two numbers. STATE
    test is not involved; the caller still passes train/finetune rows.
    """
    if int(new_task_id) < 1:
        raise ValueError(f"new_task_id must be >= 1, got {new_task_id}")
    rows = result.get("rows") or []
    old_scores: list[float] = []
    new_scores: list[float] = []
    for row in rows:
        tid = int(row["task_id"])
        score = float(row["selection_score"])
        if tid < int(new_task_id):
            old_scores.append(score)
        elif tid == int(new_task_id):
            new_scores.append(score)

    def _macro(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanmean(arr))

    old_macro = _macro(old_scores)
    new_macro = _macro(new_scores)
    parts = [x for x in (old_macro, new_macro) if np.isfinite(x)]
    balanced = float(np.mean(parts)) if parts else float("nan")
    return {
        "selection_metric": "old_new_balanced_pearson_delta",
        "selection_score": balanced,
        "old_task_macro": old_macro,
        "new_task_macro": new_macro,
        "n_old_strata": len(old_scores),
        "n_new_strata": len(new_scores),
    }


def evaluate_model_gene_space(
    model: ConditionalFlowMatching,
    arrays: TxClArrays,
    decoders: TaskDecoderBank,
    indices: np.ndarray,
    true_genes: np.ndarray,
    task_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    ode_method: str,
    ode_step_size: float | None,
    ode_rtol: float,
    ode_atol: float,
    pred_control: str = "decoded",
    flow_source: str = "noise",
    source_indices: np.ndarray | None = None,
    samples_per_stratum: int = 0,
    score_arrays: TxClArrays | None = None,
    score_task_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    control_pert_index = arrays.encoding.control_pert_index
    pred_idx, pred_tids, pred_src = stratum_generation_plan(
        arrays,
        indices,
        task_ids,
        samples_per_stratum=samples_per_stratum,
        control_pert_index=control_pert_index,
    )
    pred_genes = predicted_genes_for_indices(
        model,
        arrays,
        decoders,
        pred_idx,
        pred_tids,
        device=device,
        batch_size=batch_size,
        seed=seed,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        pred_control=pred_control,
        flow_source=flow_source,
        source_indices=source_indices,
    )
    true_for_pred = np.zeros_like(pred_genes)
    carried = pred_src >= 0
    true_for_pred[carried] = np.asarray(true_genes, dtype=np.float64)[
        pred_src[carried]
    ]
    pred_genes = apply_pred_control_genes(
        arrays, pred_idx, true_for_pred, pred_genes, mode=pred_control
    )
    scoring_arrays = arrays if score_arrays is None else score_arrays
    scoring_task_ids = task_ids if score_task_ids is None else score_task_ids
    return score_gene_strata(
        scoring_arrays,
        indices,
        true_genes,
        pred_genes,
        scoring_task_ids,
        control_pert_index=control_pert_index,
        pred_indices=pred_idx,
    )

