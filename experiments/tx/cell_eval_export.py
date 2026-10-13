"""Build real/predicted AnnData pairs for post-hoc Cell-Eval scoring."""

from __future__ import annotations

from dataclasses import dataclass
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from trajectory_replay.data.pca_decode import PcaGeneDecoder
from trajectory_replay.data.tx_dataset import TxEncoding, encode_labels
from trajectory_replay.data.tx_strata import (
    default_aggregate_covariate,
    iter_aggregate_covariate_slices,
    iter_export_slices,
    iter_observed_strata,
    subsample_indices_per_stratum_np,
)
from trajectory_replay.training.flow_matching import _z_batch_from_key
from trajectory_replay.holdout.data import stratum_seed
from experiments.tx.tx_common import contexts_missing_in_phase_controls

# Bump when export semantics change (see submit_cell_eval_bucket.sh skip logic).
CELL_EVAL_EXPORT_PROTOCOL = 10

# STATE-style cap on decoded / exported gene values (log1p-normalized space).
CELL_EVAL_GENE_MAX = 14.0


def subsample_indices_per_stratum(
    c: np.ndarray,
    p: np.ndarray,
    *,
    z_np: dict[str, np.ndarray] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    max_per_stratum: int,
    seed: int,
) -> np.ndarray:
    """Same cell draws as ``subsample_per_stratum``; returns indices into ``c`` / ``p``."""
    return subsample_indices_per_stratum_np(
        c,
        p,
        z_np=z_np,
        covariate_col_order=covariate_col_order,
        max_per_stratum=max_per_stratum,
        seed=seed,
    )


def clip_gene_matrix_for_celleval(
    x: np.ndarray,
    *,
    gene_max: float = CELL_EVAL_GENE_MAX,
) -> np.ndarray:
    """Clip gene matrices to valid log1p range for Cell-Eval (STATE uses max 14)."""
    arr = np.asarray(x, dtype=np.float32)
    return np.clip(arr, 0.0, float(gene_max))


def read_raw_gene_matrix(adata: ad.AnnData) -> np.ndarray:
    """Read stored log-normalized expression from ``adata.X``."""
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return clip_gene_matrix_for_celleval(np.asarray(x, dtype=np.float64))


def decode_obsm_to_genes(
    adata: ad.AnnData,
    decoder: PcaGeneDecoder,
    obsm_key: str,
) -> np.ndarray:
    """Decode PCA embeddings to gene space (same representation as VCB eval)."""
    if obsm_key not in adata.obsm:
        raise KeyError(f"obsm[{obsm_key!r}] missing from AnnData")
    x_pca = np.asarray(adata.obsm[obsm_key])
    return clip_gene_matrix_for_celleval(decoder.decode(x_pca))


def gene_matrix_for_adata(
    adata: ad.AnnData,
    decoder: PcaGeneDecoder,
    obsm_key: str,
    *,
    use_raw_x: bool,
) -> np.ndarray:
    """Gene matrix for Cell-Eval: raw ``adata.X`` or linear PCA decode."""
    if use_raw_x:
        return read_raw_gene_matrix(adata)
    return decode_obsm_to_genes(adata, decoder, obsm_key)


def observed_gene_matrix_for_cells(
    adata: ad.AnnData,
    cell_mask: np.ndarray,
    decoder: PcaGeneDecoder,
    obsm_key: str,
    *,
    use_raw_x: bool = True,
) -> np.ndarray:
    """Observed gene matrix for a cell subset (used for STATE-style control preds)."""
    return gene_matrix_for_adata(
        adata[cell_mask].copy(), decoder, obsm_key, use_raw_x=use_raw_x
    )


@dataclass(frozen=True)
class CellEvalPair:
    adata_real: ad.AnnData
    adata_pred: ad.AnnData
    context_label: str
    slice_label: str
    n_strata: int
    aggregate_covariate: str | None = None
    pool_contexts: bool = False


def covariate_arrays_from_adata(
    adata: ad.AnnData,
    encoding: TxEncoding,
) -> dict[str, np.ndarray] | None:
    """Obs covariate columns as string arrays for ``encode_labels``."""
    cov_order = encoding.covariate_col_order
    if not cov_order:
        return None
    missing = [col for col in cov_order if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"obs missing covariate column(s) {missing!r} required by encoding")
    return {col: adata.obs[col].astype(str).to_numpy() for col in cov_order}


def _control_pert_index(
    encoding: TxEncoding,
    control_perturbation: str | None,
) -> int | None:
    if not control_perturbation:
        return encoding.control_pert_index
    return encoding.perturbation_to_idx.get(control_perturbation)


def _stratum_pert_label(
    view_scored: ad.AnnData,
    *,
    perturbation_col: str,
    encoding: TxEncoding,
    ci: int,
    pj: int,
    c_np: np.ndarray,
    p_np: np.ndarray,
    fallback: str | None = None,
) -> str:
    """Use obs perturbation strings so real/pred Cell-Eval labels match exactly."""
    m = (c_np == ci) & (p_np == pj)
    if m.any():
        return str(view_scored.obs.iloc[int(np.where(m)[0][0])][perturbation_col])
    if fallback is not None:
        return fallback
    return str(encoding.perturbation_categories[pj])


def _pert_label_series(labels: list[str]) -> pd.Series:
    """Avoid fixed-width numpy strings truncating labels like ``non-targeting``."""
    return pd.Series(labels, dtype="object")


@torch.no_grad()
def build_cell_eval_pair(
    model: nn.Module,
    adata_phase: ad.AnnData,
    *,
    encoding: TxEncoding,
    context_idx: int | None,
    covariate_indices: dict[str, int] | None = None,
    decoder: PcaGeneDecoder,
    device: torch.device,
    eval_seed: int,
    context_col: str,
    perturbation_col: str,
    pert_col: str,
    max_per_stratum: int,
    subsample_seed: int,
    obsm_key: str,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    control_perturbation: str | None = None,
    control_phase_adata: ad.AnnData | None = None,
    use_observed_controls_only: bool = True,
    use_raw_x: bool = True,
    aggregate_covariate: str | None = None,
    pool_contexts: bool = False,
) -> CellEvalPair:
    """One export slice: observed cells vs ODE+decode predictions.

    When ``pool_contexts`` is True (STATE-style), all contexts/donors in the phase
    are included for the slice's ``aggregate_covariate`` category; ODE sampling
    still uses each cell's true context embedding.
    """
    cov_order = encoding.covariate_col_order
    if pool_contexts:
        if not aggregate_covariate or not covariate_indices:
            raise ValueError("pool_contexts requires aggregate_covariate and covariate_indices")
        agg_col = aggregate_covariate
        if agg_col not in covariate_indices:
            raise KeyError(f"aggregate covariate {agg_col!r} missing from covariate_indices")
        zi = covariate_indices[agg_col]
        slice_label = encoding.covariate_categories[agg_col][zi]
        ctx_mask = adata_phase.obs[agg_col].astype(str).to_numpy() == slice_label
        context_label = slice_label
    else:
        if context_idx is None:
            raise ValueError("context_idx is required when pool_contexts is False")
        context_label = encoding.context_categories[context_idx]
        ctx_mask = adata_phase.obs[context_col].astype(str).to_numpy() == context_label
        if covariate_indices:
            for col, zi in covariate_indices.items():
                cov_label = encoding.covariate_categories[col][zi]
                ctx_mask &= adata_phase.obs[col].astype(str).to_numpy() == cov_label
        slice_label = context_label
        if covariate_indices:
            for col in cov_order:
                if col in covariate_indices:
                    slice_label = (
                        f"{slice_label}__"
                        f"{encoding.covariate_categories[col][covariate_indices[col]]}"
                    )

    if not ctx_mask.any():
        raise ValueError(
            f"No cells for export slice label={slice_label!r} "
            f"pool_contexts={pool_contexts} cov={covariate_indices}"
        )

    view = adata_phase[ctx_mask].copy()
    contexts = view.obs[context_col].astype(str).to_numpy()
    perturbations = view.obs[perturbation_col].astype(str).to_numpy()
    cov_arrays = covariate_arrays_from_adata(view, encoding)
    c_np, p_np, z_np = encode_labels(contexts, perturbations, encoding, covariate_arrays=cov_arrays)
    sel = subsample_indices_per_stratum(
        c_np,
        p_np,
        z_np=z_np or None,
        covariate_col_order=cov_order,
        max_per_stratum=max_per_stratum,
        seed=subsample_seed,
    )
    view = view[sel].copy()
    c_np = c_np[sel]
    p_np = p_np[sel]
    if z_np:
        z_np = {col: arr[sel] for col, arr in z_np.items()}

    c_t = torch.from_numpy(c_np).long()
    p_t = torch.from_numpy(p_np).long()
    z_t = {col: torch.from_numpy(z_np[col]).long() for col in cov_order} if z_np else None
    strata = iter_observed_strata(c_t, p_t, z=z_t, covariate_col_order=cov_order)
    if not strata:
        raise ValueError(f"No strata for export slice label={slice_label!r}")

    ctrl_p_idx = _control_pert_index(encoding, control_perturbation)
    scored_p = {key[1] for key in strata}
    if ctrl_p_idx is not None:
        scored_p.add(ctrl_p_idx)

    cell_mask = np.isin(p_np, list(scored_p))
    view_scored = view[cell_mask].copy()
    c_np_scored = c_np[cell_mask]
    p_np_scored = p_np[cell_mask]
    z_np_scored = {col: arr[cell_mask] for col, arr in z_np.items()} if z_np else None
    z_t_scored = (
        {col: torch.from_numpy(z_np_scored[col]).long() for col in cov_order}
        if z_np_scored
        else None
    )

    adata_real = ad.AnnData(
        X=gene_matrix_for_adata(view_scored, decoder, obsm_key, use_raw_x=use_raw_x),
        obs=view_scored.obs.copy(),
        var=view_scored.var.copy(),
    )
    adata_real.obs[pert_col] = (
        adata_real.obs[perturbation_col].astype(str).astype(object).to_numpy()
    )

    pred_keys = list(strata)
    if ctrl_p_idx is not None and not any(key[1] == ctrl_p_idx for key in pred_keys):
        if np.any(p_np_scored == ctrl_p_idx):
            for key in strata:
                if key[1] == ctrl_p_idx:
                    pred_keys.append(key)
                    break
            else:
                ctrl_key: tuple[int, ...] = (int(c_np_scored[p_np_scored == ctrl_p_idx][0]), ctrl_p_idx)
                if cov_order and z_np_scored:
                    ctrl_key = ctrl_key + tuple(
                        int(z_np_scored[col][p_np_scored == ctrl_p_idx][0]) for col in cov_order
                    )
                pred_keys.append(ctrl_key)

    gen = torch.Generator(device=device)
    pred_chunks: list[np.ndarray] = []
    pert_labels: list[str] = []
    for key in pred_keys:
        ci, pj = key[0], key[1]
        m_np = (c_np_scored == ci) & (p_np_scored == pj)
        if z_np_scored:
            for i, col in enumerate(cov_order):
                m_np &= z_np_scored[col] == key[2 + i]
        n = int(m_np.sum())
        if n == 0:
            continue
        if use_observed_controls_only and ctrl_p_idx is not None and pj == ctrl_p_idx:
            pred_chunks.append(
                observed_gene_matrix_for_cells(
                    view_scored, m_np, decoder, obsm_key, use_raw_x=use_raw_x
                )
            )
        else:
            gen.manual_seed(stratum_seed(eval_seed, *key))
            cb = torch.full((n,), ci, dtype=torch.long, device=device)
            pb = torch.full((n,), pj, dtype=torch.long, device=device)
            zb = _z_batch_from_key(key, n, device, cov_order)
            xh = model.sample(
                cb,
                pb,
                zb,
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
                generator=gen,
            )
            pred_chunks.append(
                clip_gene_matrix_for_celleval(decoder.decode(xh.detach().cpu().numpy()))
            )
        pert_name = _stratum_pert_label(
            view_scored,
            perturbation_col=perturbation_col,
            encoding=encoding,
            ci=ci,
            pj=pj,
            c_np=c_np_scored,
            p_np=p_np_scored,
            fallback=control_perturbation or encoding.control_perturbation,
        )
        pert_labels.extend([pert_name] * n)

    if not pred_chunks:
        raise ValueError(f"No predictions generated for export slice label={slice_label!r}")

    adata_pred = ad.AnnData(
        X=np.concatenate(pred_chunks, axis=0),
        obs=pd.DataFrame({pert_col: _pert_label_series(pert_labels)}),
        var=view_scored.var.copy(),
    )

    ctrl_label = control_perturbation or encoding.control_perturbation
    if ctrl_p_idx is not None and ctrl_label:
        present_real = set(adata_real.obs[pert_col].astype(str).to_numpy())
        if ctrl_label not in present_real:
            adata_real, adata_pred = _append_control_stratum(
                adata_real=adata_real,
                adata_pred=adata_pred,
                control_phase_adata=control_phase_adata,
                context_label=context_label,
                context_idx=context_idx,
                covariate_indices=covariate_indices,
                encoding=encoding,
                control_perturbation=ctrl_label,
                context_col=context_col,
                perturbation_col=perturbation_col,
                pert_col=pert_col,
                max_per_stratum=max_per_stratum,
                subsample_seed=subsample_seed,
                obsm_key=obsm_key,
                model=model,
                decoder=decoder,
                device=device,
                eval_seed=eval_seed,
                ode_method=ode_method,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                ode_options=ode_options,
                use_observed_controls_only=use_observed_controls_only,
                use_raw_x=use_raw_x,
                pool_contexts=pool_contexts,
            )

    return CellEvalPair(
        adata_real=adata_real,
        adata_pred=adata_pred,
        context_label=context_label,
        slice_label=slice_label,
        n_strata=len(pred_keys),
        aggregate_covariate=aggregate_covariate,
        pool_contexts=pool_contexts,
    )


def select_control_cells_for_stratum(
    adata: ad.AnnData,
    *,
    context_label: str,
    control_perturbation: str,
    context_col: str,
    perturbation_col: str,
    encoding: TxEncoding,
    covariate_indices: dict[str, int] | None = None,
    pool_contexts: bool = False,
) -> ad.AnnData | None:
    """Pick in-slice PBS/control cells (same covariates; optional context pooling).

    Matches STATE / cell-eval: ``split_anndata_on_celltype`` then ``MetricsEvaluator``
    expects controls within each exported AnnData slice — never another cell type.
    When ``pool_contexts`` is True, controls are taken from every context/donor in the
    slice category (same cell type).
    """
    if control_perturbation not in encoding.perturbation_to_idx:
        raise KeyError(f"Control perturbation {control_perturbation!r} not in encoding")

    ctx = adata.obs[context_col].astype(str).to_numpy()
    perts = adata.obs[perturbation_col].astype(str).to_numpy()
    ctrl_mask = perts == control_perturbation
    if not pool_contexts:
        ctrl_mask &= ctx == context_label
    if covariate_indices:
        for col, zi in covariate_indices.items():
            cov_label = encoding.covariate_categories[col][zi]
            ctrl_mask &= adata.obs[col].astype(str).to_numpy() == cov_label
    if not ctrl_mask.any():
        return None
    return adata[ctrl_mask].copy()


def export_slice_has_controls(
    control_pool: ad.AnnData | None,
    *,
    context_label: str,
    control_perturbation: str,
    context_col: str,
    perturbation_col: str,
    encoding: TxEncoding,
    covariate_indices: dict[str, int] | None = None,
    pool_contexts: bool = False,
) -> bool:
    """True when eval/reference pool has in-slice controls for cell-eval export."""
    if control_pool is None or control_perturbation is None:
        return False
    picked = select_control_cells_for_stratum(
        control_pool,
        context_label=context_label,
        control_perturbation=control_perturbation,
        context_col=context_col,
        perturbation_col=perturbation_col,
        encoding=encoding,
        covariate_indices=covariate_indices,
        pool_contexts=pool_contexts,
    )
    return picked is not None and picked.n_obs > 0


def merge_control_reference_adata(
    eval_phase_adata: ad.AnnData,
    reference_adata: ad.AnnData | None,
) -> ad.AnnData:
    """Eval phase first, then optional OOD/finetune reference cells for control borrowing."""
    if reference_adata is None:
        return eval_phase_adata
    if reference_adata.n_obs == 0:
        return eval_phase_adata
    return ad.concat([eval_phase_adata, reference_adata], join="outer", index_unique=None)


@torch.no_grad()
def _append_control_stratum(
    *,
    adata_real: ad.AnnData,
    adata_pred: ad.AnnData,
    control_phase_adata: ad.AnnData | None,
    context_label: str,
    context_idx: int | None,
    covariate_indices: dict[str, int] | None = None,
    encoding: TxEncoding,
    control_perturbation: str,
    context_col: str,
    perturbation_col: str,
    pert_col: str,
    max_per_stratum: int,
    subsample_seed: int,
    obsm_key: str,
    model: nn.Module,
    decoder: PcaGeneDecoder,
    device: torch.device,
    eval_seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    ode_options: dict[str, float] | None,
    use_observed_controls_only: bool = True,
    use_raw_x: bool = True,
    pool_contexts: bool = False,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Add observed control cells when the export slice lacks in-slice PBS."""
    if control_phase_adata is None:
        raise ValueError(
            f"Control perturbation {control_perturbation!r} missing from eval slice "
            f"and no control_phase_adata was provided."
        )
    if control_perturbation not in encoding.perturbation_to_idx:
        raise KeyError(f"Control perturbation {control_perturbation!r} not in encoding")

    ctrl_p_idx = encoding.perturbation_to_idx[control_perturbation]
    ctrl_view = select_control_cells_for_stratum(
        control_phase_adata,
        context_label=context_label,
        control_perturbation=control_perturbation,
        context_col=context_col,
        perturbation_col=perturbation_col,
        encoding=encoding,
        covariate_indices=covariate_indices,
        pool_contexts=pool_contexts,
    )
    if ctrl_view is None or ctrl_view.n_obs == 0:
        cov_suffix = ""
        if covariate_indices:
            labels = [
                encoding.covariate_categories[col][zi]
                for col, zi in covariate_indices.items()
            ]
            cov_suffix = f", covariates={labels!r}"
        raise ValueError(
            f"No control cells for slice={context_label!r}, "
            f"pert={control_perturbation!r} in eval/reference phases{cov_suffix}."
        )

    cov_order = encoding.covariate_col_order
    if pool_contexts and max_per_stratum > 0:
        contexts = ctrl_view.obs[context_col].astype(str).to_numpy()
        perturbations = ctrl_view.obs[perturbation_col].astype(str).to_numpy()
        cov_arrays = covariate_arrays_from_adata(ctrl_view, encoding)
        c_np, p_np, z_np = encode_labels(
            contexts, perturbations, encoding, covariate_arrays=cov_arrays
        )
        sel = subsample_indices_per_stratum(
            c_np,
            p_np,
            z_np=z_np or None,
            covariate_col_order=cov_order,
            max_per_stratum=max_per_stratum,
            seed=subsample_seed,
        )
        ctrl_view = ctrl_view[sel].copy()
    elif max_per_stratum > 0 and ctrl_view.n_obs > max_per_stratum:
        if context_idx is None:
            raise ValueError("context_idx required to subsample controls when pool_contexts is False")
        n = max_per_stratum
        seed_key = (context_idx, ctrl_p_idx)
        if cov_order and covariate_indices:
            seed_key = seed_key + tuple(covariate_indices[col] for col in cov_order)
        rng = np.random.default_rng(stratum_seed(subsample_seed, *seed_key))
        pick = rng.choice(ctrl_view.n_obs, size=n, replace=False)
        ctrl_view = ctrl_view[pick].copy()

    n = int(ctrl_view.n_obs)
    real_append = gene_matrix_for_adata(ctrl_view, decoder, obsm_key, use_raw_x=use_raw_x)
    ctrl_obs = ctrl_view.obs.copy()
    ctrl_obs[pert_col] = control_perturbation
    adata_real = ad.AnnData(
        X=np.vstack([np.asarray(adata_real.X), real_append]),
        obs=pd.concat([adata_real.obs, ctrl_obs], ignore_index=True),
        var=adata_real.var.copy(),
    )

    if use_observed_controls_only:
        pred_append = real_append.copy()
    else:
        gen = torch.Generator(device=device)
        contexts = ctrl_view.obs[context_col].astype(str).to_numpy()
        perturbations = ctrl_view.obs[perturbation_col].astype(str).to_numpy()
        cov_arrays = covariate_arrays_from_adata(ctrl_view, encoding)
        c_np, p_np, z_np = encode_labels(
            contexts, perturbations, encoding, covariate_arrays=cov_arrays
        )
        pred_rows: list[np.ndarray] = []
        strata = iter_observed_strata(
            torch.from_numpy(c_np).long(),
            torch.from_numpy(p_np).long(),
            z={col: torch.from_numpy(z_np[col]).long() for col in cov_order} if z_np else None,
            covariate_col_order=cov_order,
        )
        for key in strata:
            ci, pj = key[0], key[1]
            m_np = (c_np == ci) & (p_np == pj)
            if z_np:
                for i, col in enumerate(cov_order):
                    m_np &= z_np[col] == key[2 + i]
            n_key = int(m_np.sum())
            if n_key == 0:
                continue
            gen.manual_seed(stratum_seed(eval_seed, *key))
            cb = torch.full((n_key,), ci, dtype=torch.long, device=device)
            pb = torch.full((n_key,), pj, dtype=torch.long, device=device)
            zb = _z_batch_from_key(key, n_key, device, cov_order)
            xh = model.sample(
                cb,
                pb,
                zb,
                method=ode_method,
                rtol=ode_rtol,
                atol=ode_atol,
                options=ode_options,
                generator=gen,
            )
            pred_rows.append(
                clip_gene_matrix_for_celleval(decoder.decode(xh.detach().cpu().numpy()))
            )
        pred_append = np.concatenate(pred_rows, axis=0) if pred_rows else real_append.copy()
    adata_pred = ad.AnnData(
        X=np.vstack([np.asarray(adata_pred.X), pred_append]),
        obs=pd.concat(
            [
                adata_pred.obs,
                pd.DataFrame(
                    {pert_col: _pert_label_series([control_perturbation] * n)}
                ),
            ],
            ignore_index=True,
        ),
        var=adata_pred.var.copy(),
    )
    return adata_real, adata_pred


def contexts_missing_controls_for_export_slices(
    adata: ad.AnnData,
    encoding: TxEncoding,
    control_pert: int,
    *,
    context_col: str,
    perturbation_col: str,
    context_labels: tuple[str, ...] | None = None,
    aggregate_covariate: str | None = None,
) -> frozenset[int]:
    """Context indices where an export slice lacks in-phase controls."""
    ctrl_label = encoding.control_perturbation
    if ctrl_label is None:
        return frozenset()
    cov_order = encoding.covariate_col_order
    if not cov_order:
        return frozenset()

    if aggregate_covariate and aggregate_covariate in cov_order:
        missing: set[int] = set()
        for _ci, cov_map, slice_label in iter_aggregate_covariate_slices(
            adata,
            encoding,
            aggregate_covariate=aggregate_covariate,
            context_col=context_col,
            context_labels=context_labels,
        ):
            if not export_slice_has_controls(
                adata,
                context_label=slice_label,
                control_perturbation=ctrl_label,
                context_col=context_col,
                perturbation_col=perturbation_col,
                encoding=encoding,
                covariate_indices=cov_map or None,
                pool_contexts=True,
            ):
                for ctx_label in set(adata.obs[context_col].astype(str).unique()):
                    if context_labels and ctx_label not in context_labels:
                        continue
                    missing.add(encoding.context_to_idx[ctx_label])
        return frozenset(missing)

    missing = set()
    for ci, cov_map, _ in iter_export_slices(
        adata,
        encoding,
        context_col=context_col,
        covariate_col_order=cov_order,
        context_labels=context_labels,
    ):
        ctx_label = encoding.context_categories[ci]
        mask = adata.obs[context_col].astype(str).to_numpy() == ctx_label
        for col, zi in cov_map.items():
            cov_label = encoding.covariate_categories[col][zi]
            mask &= adata.obs[col].astype(str).to_numpy() == cov_label
        perts = adata.obs[perturbation_col].astype(str).to_numpy()
        if not (mask & (perts == ctrl_label)).any():
            missing.add(ci)
    return frozenset(missing)


def contexts_missing_in_phase_adata(
    adata: ad.AnnData,
    encoding: TxEncoding,
    control_pert: int,
    *,
    context_col: str,
    perturbation_col: str,
    context_labels: tuple[str, ...] | None = None,
    aggregate_covariate: str | None = None,
) -> frozenset[int]:
    """Context indices in ``adata`` that lack in-phase control cells (per export pool)."""
    cov_order = encoding.covariate_col_order
    if cov_order:
        return contexts_missing_controls_for_export_slices(
            adata,
            encoding,
            control_pert,
            context_col=context_col,
            perturbation_col=perturbation_col,
            context_labels=context_labels,
            aggregate_covariate=aggregate_covariate,
        )
    contexts = adata.obs[context_col].astype(str).to_numpy()
    perturbations = adata.obs[perturbation_col].astype(str).to_numpy()
    c_np, p_np, _ = encode_labels(contexts, perturbations, encoding)
    return contexts_missing_in_phase_controls(
        torch.from_numpy(c_np).long(),
        torch.from_numpy(p_np).long(),
        control_pert,
    )


def load_control_reference_adata(
    h5ad_path,
    *,
    split_col: str,
    reference_phases: tuple[str, ...],
) -> ad.AnnData | None:
    """Load and concatenate cells from reference split phase(s) for control borrowing."""
    parts: list[ad.AnnData] = []
    for phase in reference_phases:
        try:
            parts.append(load_phase_adata(h5ad_path, split_col=split_col, phase=phase))
        except ValueError:
            continue
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ad.concat(parts, join="outer", index_unique=None)


def load_phase_adata(
    h5ad_path,
    *,
    split_col: str,
    phase: str,
) -> ad.AnnData:
    adata = ad.read_h5ad(h5ad_path)
    if split_col not in adata.obs.columns:
        raise KeyError(f"obs[{split_col!r}] missing")
    mask = adata.obs[split_col].astype(str).to_numpy() == phase
    if not mask.any():
        raise ValueError(f"No cells for phase {phase!r} in {h5ad_path}")
    return adata[mask].copy()


def context_indices_in_adata(
    adata: ad.AnnData,
    encoding: TxEncoding,
    context_col: str,
    *,
    context_labels: tuple[str, ...] | None = None,
) -> list[int]:
    present = set(adata.obs[context_col].astype(str).unique())
    labels = context_labels or encoding.context_categories
    out: list[int] = []
    for label in labels:
        if label not in present:
            continue
        out.append(encoding.context_to_idx[label])
    if not out:
        raise ValueError(f"No encoded contexts present in adata; obs has {sorted(present)}")
    return out
