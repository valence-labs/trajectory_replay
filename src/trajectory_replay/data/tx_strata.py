"""Stratum keys and subsampling for transcriptomics (c, p, optional covariates)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

StratumKey = tuple[int, ...]  # (c, p, *covariate_indices)
ControlPoolKey = tuple[int, ...]  # (c, *covariate_indices)


def stratum_seed(base_seed: int, *indices: int) -> int:
    """Deterministic seed from base + integer indices (c, p, covariates, ...)."""
    h = int(base_seed)
    for i, idx in enumerate(indices):
        mult = 100_003 if i == 0 else 1_009 * (10 ** min(i - 1, 3))
        h += int(idx) * mult
    return h


def parse_covariate_cols(spec: str | None) -> tuple[str, ...]:
    """Parse comma-separated covariate column names (empty → no covariates)."""
    if not spec or not str(spec).strip():
        return ()
    cols = tuple(c.strip() for c in str(spec).split(",") if c.strip())
    if len(cols) != len(set(cols)):
        raise ValueError(f"duplicate covariate columns in {spec!r}")
    return cols


def _covariate_arrays(
    z: dict[str, torch.Tensor] | None,
    covariate_col_order: tuple[str, ...],
) -> list[np.ndarray]:
    if not covariate_col_order:
        return []
    if z is None:
        raise ValueError("covariate tensors required when covariate_col_order is non-empty")
    return [z[col].detach().cpu().numpy() for col in covariate_col_order]


def iter_observed_strata(
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    skip_contexts: frozenset[int] | None = None,
    skip_perturbations: frozenset[int] | None = None,
    skip_combos: frozenset[tuple[int, int]] | None = None,
) -> list[StratumKey]:
    """Unique (c, p, *z) strata present in tensors, after optional filters."""
    c_np = c.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    z_nps = _covariate_arrays(z, covariate_col_order)
    cols = [c_np, p_np, *z_nps]
    uniq = np.unique(np.stack(cols, axis=1), axis=0)
    strata: list[StratumKey] = []
    for row in uniq:
        ci, pj = int(row[0]), int(row[1])
        if skip_combos is not None and (ci, pj) in skip_combos:
            continue
        if skip_contexts is not None and ci in skip_contexts:
            continue
        if skip_perturbations is not None and pj in skip_perturbations:
            continue
        key: StratumKey = (ci, pj)
        if z_nps:
            key = key + tuple(int(v) for v in row[2:])
        strata.append(key)
    return strata


def stratum_mask(
    c: torch.Tensor,
    p: torch.Tensor,
    key: StratumKey,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
) -> torch.Tensor:
    """Boolean mask for cells matching a full stratum key."""
    ci, pj = key[0], key[1]
    m = (c == ci) & (p == pj)
    if covariate_col_order:
        if z is None:
            raise ValueError("covariate tensors required")
        for i, col in enumerate(covariate_col_order):
            m = m & (z[col] == key[2 + i])
    return m


def subsample_per_stratum(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    max_per_stratum: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Keep up to ``max_per_stratum`` cells per observed stratum."""
    if max_per_stratum <= 0:
        return x, c, p, z

    n = len(x)
    c_np = c.cpu().numpy()
    p_np = p.cpu().numpy()
    z_nps = _covariate_arrays(z, covariate_col_order)
    cols = [c_np, p_np, *z_nps]

    strata: dict[StratumKey, list[int]] = {}
    stacked = np.stack(cols, axis=1)
    for i in range(n):
        row = stacked[i]
        key: StratumKey = (int(row[0]), int(row[1]))
        if z_nps:
            key = key + tuple(int(v) for v in row[2:])
        strata.setdefault(key, []).append(i)

    picked: list[int] = []
    for key in sorted(strata.keys()):
        idxs = np.asarray(strata[key], dtype=np.int64)
        rng = np.random.default_rng(stratum_seed(seed, *key))
        if len(idxs) <= max_per_stratum:
            picked.extend(idxs.tolist())
        else:
            picked.extend(rng.choice(idxs, size=max_per_stratum, replace=False).tolist())

    sel = torch.as_tensor(picked, dtype=torch.long)
    z_out = None
    if z is not None:
        z_out = {col: z[col][sel] for col in covariate_col_order}
    return x[sel], c[sel], p[sel], z_out


def stratified_subsample(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Sample up to ``max_samples`` with equal quota per stratum."""
    n = len(x)
    if max_samples >= n:
        return x, c, p, z

    rng = np.random.default_rng(seed)
    c_np = c.cpu().numpy()
    p_np = p.cpu().numpy()
    z_nps = _covariate_arrays(z, covariate_col_order)
    cols = [c_np, p_np, *z_nps]

    strata: dict[StratumKey, list[int]] = {}
    stacked = np.stack(cols, axis=1)
    for i in range(n):
        row = stacked[i]
        key: StratumKey = (int(row[0]), int(row[1]))
        if z_nps:
            key = key + tuple(int(v) for v in row[2:])
        strata.setdefault(key, []).append(i)

    n_strata = len(strata)
    base = max_samples // n_strata
    remainder = max_samples % n_strata
    picked: list[int] = []
    for j, key in enumerate(sorted(strata.keys())):
        quota = base + (1 if j < remainder else 0)
        idxs = np.array(strata[key], dtype=np.int64)
        if len(idxs) <= quota:
            picked.extend(idxs.tolist())
        else:
            picked.extend(rng.choice(idxs, size=quota, replace=False).tolist())

    if len(picked) < max_samples:
        used = set(picked)
        rest = [i for i in range(n) if i not in used]
        need = max_samples - len(picked)
        if rest:
            extra = rng.choice(np.array(rest, dtype=np.int64), size=min(need, len(rest)), replace=False)
            picked.extend(extra.tolist())

    sel = torch.as_tensor(picked[:max_samples], dtype=torch.long)
    z_out = None
    if z is not None:
        z_out = {col: z[col][sel] for col in covariate_col_order}
    return x[sel], c[sel], p[sel], z_out


def control_indices_by_prefix(
    c: torch.Tensor,
    p: torch.Tensor,
    control_pert: int,
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
) -> dict[ControlPoolKey, torch.Tensor]:
    """Map (c, *covariate_indices) → row indices of control cells."""
    c = c.long()
    p = p.long()
    pools: dict[ControlPoolKey, list[int]] = {}
    for i in range(c.shape[0]):
        if int(p[i].item()) != control_pert:
            continue
        key: ControlPoolKey = (int(c[i].item()),)
        if covariate_col_order:
            if z is None:
                raise ValueError("covariate tensors required")
            key = key + tuple(int(z[col][i].item()) for col in covariate_col_order)
        pools.setdefault(key, []).append(i)
    out: dict[ControlPoolKey, torch.Tensor] = {}
    for key, idxs in pools.items():
        if not idxs:
            raise ValueError(f"No control cells for pool key {key} (control_pert={control_pert})")
        out[key] = torch.tensor(idxs, dtype=torch.long)
    return out


def control_pool_key_for_stratum(
    key: StratumKey,
    covariate_col_order: tuple[str, ...],
) -> ControlPoolKey:
    """Drop perturbation index from a full stratum key."""
    if not covariate_col_order:
        return (key[0],)
    return (key[0],) + key[2:]


def stratum_row_to_dict(
    key: StratumKey,
    encoding: object,
    *,
    covariate_col_order: tuple[str, ...] = (),
) -> dict[str, int | str]:
    """Serialize a stratum key to metric row fields (c/p labels + covariate labels)."""
    from trajectory_replay.data.tx_dataset import TxEncoding

    if not isinstance(encoding, TxEncoding):
        raise TypeError(f"encoding must be TxEncoding, got {type(encoding)!r}")
    row: dict[str, int | str] = {
        "c": key[0],
        "p": key[1],
        "context_label": encoding.context_categories[key[0]],
        "perturbation_label": encoding.perturbation_categories[key[1]],
    }
    for i, col in enumerate(covariate_col_order):
        zi = key[2 + i]
        row[f"z_{col}"] = zi
        row[f"{col}_label"] = encoding.covariate_categories[col][zi]
    return row


def subsample_indices_per_stratum_np(
    c: np.ndarray,
    p: np.ndarray,
    *,
    z_np: dict[str, np.ndarray] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    max_per_stratum: int,
    seed: int,
) -> np.ndarray:
    """NumPy variant for cell-eval export; returns indices into input arrays."""
    if max_per_stratum <= 0:
        return np.arange(len(c), dtype=np.int64)

    z_arrays = [z_np[col] for col in covariate_col_order] if covariate_col_order else []
    cols = [c, p, *z_arrays]
    strata: dict[StratumKey, list[int]] = {}
    stacked = np.stack(cols, axis=1)
    for i in range(len(c)):
        row = stacked[i]
        key: StratumKey = (int(row[0]), int(row[1]))
        if z_arrays:
            key = key + tuple(int(v) for v in row[2:])
        strata.setdefault(key, []).append(i)

    picked: list[int] = []
    for key in sorted(strata.keys()):
        idxs = np.asarray(strata[key], dtype=np.int64)
        rng = np.random.default_rng(stratum_seed(seed, *key))
        if len(idxs) <= max_per_stratum:
            picked.extend(idxs.tolist())
        else:
            picked.extend(rng.choice(idxs, size=max_per_stratum, replace=False).tolist())
    return np.asarray(picked, dtype=np.int64)


def iter_export_slices(
    adata,
    encoding: object,
    *,
    context_col: str,
    covariate_col_order: tuple[str, ...],
    context_labels: Iterable[str] | None = None,
) -> list[tuple[int, dict[str, int], str]]:
    """Unique (context_idx, covariate_idx_map, path_suffix) for cell-eval export."""
    from trajectory_replay.data.tx_dataset import TxEncoding

    if not isinstance(encoding, TxEncoding):
        raise TypeError(f"encoding must be TxEncoding, got {type(encoding)!r}")

    ctx_filter = set(context_labels) if context_labels else None
    slices: list[tuple[int, dict[str, int], str]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()

    obs = adata.obs
    contexts = obs[context_col].astype(str).to_numpy()
    for i in range(len(contexts)):
        ctx_label = contexts[i]
        if ctx_filter is not None and ctx_label not in ctx_filter:
            continue
        ci = encoding.context_to_idx[ctx_label]
        z_map: dict[str, int] = {}
        z_key_parts: list[int] = []
        for col in covariate_col_order:
            label = str(obs[col].iloc[i])
            zi = encoding.covariate_to_idx[col][label]
            z_map[col] = zi
            z_key_parts.append(zi)
        dedupe_key = (ci, tuple(z_key_parts))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        suffix = encoding.context_categories[ci]
        for col in covariate_col_order:
            suffix = f"{suffix}__{encoding.covariate_categories[col][z_map[col]]}"
        slices.append((ci, z_map, suffix))
    return sorted(slices, key=lambda t: (t[0], tuple(t[1].get(c, -1) for c in covariate_col_order)))


def default_aggregate_covariate(
    *,
    context_col: str,
    covariate_col_order: tuple[str, ...],
) -> str:
    """Default Cell-Eval export aggregation column (STATE-style pooling axis)."""
    if "cell_type" in covariate_col_order:
        return "cell_type"
    if context_col == "cell_line":
        return "cell_line"
    if covariate_col_order:
        return covariate_col_order[0]
    return context_col


def iter_aggregate_covariate_slices(
    adata,
    encoding: object,
    *,
    aggregate_covariate: str,
    context_col: str,
    context_labels: Iterable[str] | None = None,
) -> list[tuple[int | None, dict[str, int], str]]:
    """Export slices pooling contexts within each ``aggregate_covariate`` category.

    Returns ``(context_idx, covariate_idx_map, slice_label)``:

    - When aggregating a model covariate (e.g. ``cell_type``), ``context_idx`` is
      ``None`` and all donors/contexts in the phase are pooled per category.
    - When aggregating ``context_col`` itself (Replogle ``cell_line``), one slice
      per context label (same as legacy context-only export).
    """
    from trajectory_replay.data.tx_dataset import TxEncoding

    if not isinstance(encoding, TxEncoding):
        raise TypeError(f"encoding must be TxEncoding, got {type(encoding)!r}")

    cov_order = encoding.covariate_col_order
    ctx_filter = set(context_labels) if context_labels else None
    obs = adata.obs

    if aggregate_covariate not in cov_order:
        slices: list[tuple[int | None, dict[str, int], str]] = []
        for ci, label in enumerate(encoding.context_categories):
            if ctx_filter is not None and label not in ctx_filter:
                continue
            if label not in set(obs[context_col].astype(str).unique()):
                continue
            slices.append((ci, {}, label))
        return sorted(slices, key=lambda t: t[2])

    seen: set[int] = set()
    slices = []
    col = aggregate_covariate
    contexts = obs[context_col].astype(str).to_numpy()
    agg_vals = obs[col].astype(str).to_numpy()
    for i in range(len(contexts)):
        if ctx_filter is not None and contexts[i] not in ctx_filter:
            continue
        label = str(agg_vals[i])
        zi = encoding.covariate_to_idx[col][label]
        if zi in seen:
            continue
        seen.add(zi)
        slices.append((None, {col: zi}, label))
    return sorted(slices, key=lambda t: (t[1][col], t[2]))
