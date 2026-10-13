"""Holdout masks, index remapping, and replay-buffer sampling for finetuning experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def remap_perturbation_indices(p: torch.Tensor, held_out: int) -> torch.Tensor:
    """Map full perturbation ids (excluding ``held_out``) to contiguous [0, n_p-2]."""
    return p - (p > held_out).long()


def remap_context_indices(c: torch.Tensor, held_out: int) -> torch.Tensor:
    """Map full context ids (excluding ``held_out``) to contiguous [0, n_c-2]."""
    return c - (c > held_out).long()


def mask_excluding_perturbation(
    x: torch.Tensor, c: torch.Tensor, p: torch.Tensor, held_out: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = p != held_out
    return x[m], c[m], remap_perturbation_indices(p[m], held_out)


def mask_only_perturbation(
    x: torch.Tensor, c: torch.Tensor, p: torch.Tensor, held_out: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = p == held_out
    return x[m], c[m], p[m]


def mask_excluding_context(
    x: torch.Tensor, c: torch.Tensor, p: torch.Tensor, held_out: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = c != held_out
    return x[m], remap_context_indices(c[m], held_out), p[m]


def mask_only_context(
    x: torch.Tensor, c: torch.Tensor, p: torch.Tensor, held_out: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = c == held_out
    return x[m], c[m], p[m]


def mask_excluding_combo(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    held_context: int,
    held_pert: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = ~((c == held_context) & (p == held_pert))
    return x[m], c[m], p[m]


def mask_only_combo(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    held_context: int,
    held_pert: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = (c == held_context) & (p == held_pert)
    return x[m], c[m], p[m]


def mask_replay_pool(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    holdout_mode: str,
    held_out_pert: int,
    held_out_context: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pretrain support with full (c, p) indices (no embedding remapping)."""
    if holdout_mode == "perturbation":
        m = p != held_out_pert
    elif holdout_mode == "context":
        assert held_out_context is not None
        m = c != held_out_context
    elif holdout_mode == "combo":
        assert held_out_context is not None
        m = ~((c == held_out_context) & (p == held_out_pert))
    else:
        raise ValueError(f"Unknown holdout_mode {holdout_mode!r}")
    return x[m], c[m], p[m]


def stratified_subsample(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    max_samples: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample up to ``max_samples`` points with equal quota per (c, p) stratum."""
    n = len(x)
    if max_samples is None or max_samples >= n:
        return x, c, p

    rng = np.random.default_rng(seed)
    strata: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        key = (int(c[i].item()), int(p[i].item()))
        strata.setdefault(key, []).append(i)

    n_strata = len(strata)
    base = max_samples // n_strata
    remainder = max_samples % n_strata
    picked: list[int] = []
    stratum_keys = sorted(strata.keys())
    for j, key in enumerate(stratum_keys):
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
    return x[sel], c[sel], p[sel]


def support_context_indices(n_contexts: int, held_out_context: int) -> list[int]:
    """Context indices seen in pretrain (exclude held-out context row)."""
    return [c for c in range(n_contexts) if c != held_out_context]


def sample_uniform_support_conditions(
    n: int,
    *,
    support_contexts: list[int],
    n_perturbations: int,
    device: torch.device,
    cpu_gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n`` pairs ``(c, p)`` uniformly over support contexts and all perturbations."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not support_contexts:
        raise ValueError("support_contexts is empty")
    if n_perturbations <= 0:
        raise ValueError(f"n_perturbations must be positive, got {n_perturbations}")

    support_t = torch.tensor(support_contexts, dtype=torch.long)
    ctx_pick = torch.randint(len(support_contexts), (n,), generator=cpu_gen)
    p = torch.randint(n_perturbations, (n,), generator=cpu_gen)
    c = support_t[ctx_pick].to(device=device)
    return c, p.to(device=device)


@dataclass(frozen=True)
class SupportStratumDistribution:
    """Empirical (c, p[, z...]) distribution on pretrain support (one row per stratum)."""

    c: torch.Tensor
    p: torch.Tensor
    probs: torch.Tensor
    z: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.c.shape != self.p.shape or self.c.shape != self.probs.shape:
            raise ValueError("c, p, probs must have the same length")
        if len(self.c) == 0:
            raise ValueError("empty support stratum distribution")
        if not torch.isclose(self.probs.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError(f"probs must sum to 1, got {float(self.probs.sum())}")
        if self.z is not None:
            for arr in self.z.values():
                if arr.shape != self.c.shape:
                    raise ValueError("each z column must match c/p length")


def build_support_stratum_distribution(
    c: torch.Tensor,
    p: torch.Tensor,
    support_contexts: list[int],
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
) -> SupportStratumDistribution:
    """Empirical P(c, p[, z...]) over cells with context in ``support_contexts``."""
    support_set = set(support_contexts)
    c_np = c.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    mask = np.isin(c_np, list(support_set))
    if not mask.any():
        raise ValueError("no (c, p) strata on support contexts in provided tensors")

    cols = [c_np[mask], p_np[mask]]
    if covariate_col_order:
        if z is None:
            raise ValueError("z required when covariate_col_order is non-empty")
        cols.extend(z[col].detach().cpu().numpy()[mask] for col in covariate_col_order)
    stacked = np.stack(cols, axis=1)
    uniq, counts = np.unique(stacked, axis=0, return_counts=True)

    c_out = torch.from_numpy(uniq[:, 0].astype(np.int64))
    p_out = torch.from_numpy(uniq[:, 1].astype(np.int64))
    w = torch.from_numpy(counts.astype(np.float64))
    probs = (w / w.sum()).to(torch.float32)
    z_out: dict[str, torch.Tensor] | None = None
    if covariate_col_order:
        z_out = {
            col: torch.from_numpy(uniq[:, 2 + j].astype(np.int64))
            for j, col in enumerate(covariate_col_order)
        }
    return SupportStratumDistribution(c=c_out, p=p_out, probs=probs, z=z_out)


REPLAY_CONDITION_SAMPLINGS = ("proportional", "observed_uniform", "uniform")
OBSERVED_STRATUM_SAMPLINGS = ("proportional", "observed_uniform")


def stratum_sampling_probs(
    stratum_dist: SupportStratumDistribution,
    sampling: str,
) -> torch.Tensor:
    """CPU probabilities for one draw over observed support strata."""
    if sampling == "proportional":
        return stratum_dist.probs.cpu()
    if sampling == "observed_uniform":
        n_strata = int(stratum_dist.c.shape[0])
        return torch.full((n_strata,), 1.0 / n_strata, dtype=torch.float32)
    raise ValueError(
        f"sampling {sampling!r} is not an observed-stratum mode; "
        f"use one of {OBSERVED_STRATUM_SAMPLINGS}"
    )


def sample_support_conditions(
    n: int,
    *,
    support_contexts: list[int],
    n_perturbations: int,
    device: torch.device,
    cpu_gen: torch.Generator,
    sampling: str = "proportional",
    stratum_dist: SupportStratumDistribution | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Sample ``n`` replay conditions on pretrain support.

    ``uniform``: independent uniform c (support) and p (all perturbation ids),
    including combinations that never appear in the support pool.
    ``proportional``: observed ``(c, p[, z...])`` with probability ∝ cell count.
    ``observed_uniform``: the same observed strata, equal weight each, so a fat
    control pool is one vote like any knockdown.

    When ``stratum_dist.z`` is set, returns ``(c, p, z)``; otherwise ``(c, p)``.
    """
    if sampling == "uniform":
        c, p = sample_uniform_support_conditions(
            n,
            support_contexts=support_contexts,
            n_perturbations=n_perturbations,
            device=device,
            cpu_gen=cpu_gen,
        )
        if stratum_dist is not None and stratum_dist.z is not None:
            raise ValueError(
                "uniform sampling cannot sample covariates; use "
                "proportional or observed_uniform"
            )
        return c, p
    if sampling not in OBSERVED_STRATUM_SAMPLINGS:
        raise ValueError(
            f"unknown sampling {sampling!r} (use {REPLAY_CONDITION_SAMPLINGS})"
        )
    if stratum_dist is None:
        raise ValueError(f"{sampling} sampling requires stratum_dist")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    idx = torch.multinomial(
        stratum_sampling_probs(stratum_dist, sampling),
        n,
        replacement=True,
        generator=cpu_gen,
    )
    c = stratum_dist.c[idx].to(device=device)
    p = stratum_dist.p[idx].to(device=device)
    if stratum_dist.z is None:
        return c, p
    z = {col: arr[idx].to(device=device, dtype=torch.long) for col, arr in stratum_dist.z.items()}
    return c, p, z


def build_support_flat_index(
    c: torch.Tensor,
    support_contexts: list[int],
) -> torch.Tensor:
    """Row indices of all cells whose context is in ``support_contexts``."""
    c_np = c.detach().cpu().numpy()
    mask = np.isin(c_np, list(support_contexts))
    picked = np.where(mask)[0]
    if picked.size == 0:
        raise ValueError("no cells on support contexts in provided tensor")
    return torch.from_numpy(picked.astype(np.int64))


def build_stratum_cell_index(
    c: torch.Tensor,
    p: torch.Tensor,
    support_contexts: list[int],
    *,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
) -> dict[tuple[int, ...], torch.Tensor]:
    """Map each observed support stratum to cell row indices."""
    c_np = c.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    mask = np.isin(c_np, list(support_contexts))
    if not mask.any():
        raise ValueError("no strata on support contexts in provided tensors")

    row_idx = np.where(mask)[0]
    cols = [c_np[mask], p_np[mask]]
    if covariate_col_order:
        if z is None:
            raise ValueError("z required when covariate_col_order is non-empty")
        cols.extend(z[col].detach().cpu().numpy()[mask] for col in covariate_col_order)

    stacked = np.stack(cols, axis=1)
    uniq, inverse = np.unique(stacked, axis=0, return_inverse=True)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for local_i, stratum_id in enumerate(inverse):
        key = tuple(int(v) for v in uniq[stratum_id])
        buckets.setdefault(key, []).append(int(row_idx[local_i]))
    return {k: torch.tensor(v, dtype=torch.long) for k, v in buckets.items()}


def _keys_from_sampled_conditions(
    c_cond: torch.Tensor,
    p_cond: torch.Tensor,
    *,
    z_cond: dict[str, torch.Tensor] | None,
    covariate_col_order: tuple[str, ...],
) -> list[tuple[int, ...]]:
    c_np = c_cond.detach().cpu().numpy()
    p_np = p_cond.detach().cpu().numpy()
    if not covariate_col_order:
        return [(int(c_np[i]), int(p_np[i])) for i in range(c_np.shape[0])]
    if z_cond is None:
        raise ValueError("covariate conditions required")
    cols = [c_np, p_np] + [z_cond[col].detach().cpu().numpy() for col in covariate_col_order]
    stacked = np.stack(cols, axis=1)
    return [tuple(int(v) for v in row) for row in stacked]


def _pick_cells_for_stratum_keys(
    keys: list[tuple[int, ...]],
    *,
    stratum_cells: dict[tuple[int, ...], torch.Tensor],
    support_flat_idx: torch.Tensor,
    cpu_gen: torch.Generator,
) -> torch.Tensor:
    """Pick one cell row index per stratum key (fallback to any support cell)."""
    picked = torch.empty(len(keys), dtype=torch.long)
    by_key: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(keys):
        by_key.setdefault(key, []).append(i)
    for key, positions in by_key.items():
        count = len(positions)
        pool = stratum_cells.get(key)
        if pool is not None and len(pool) > 0:
            js = torch.randint(len(pool), (count,), generator=cpu_gen)
            rows = pool[js]
        else:
            js = torch.randint(len(support_flat_idx), (count,), generator=cpu_gen)
            rows = support_flat_idx[js]
        for pos, row in zip(positions, rows.tolist(), strict=True):
            picked[pos] = row
    return picked


def sample_support_cells(
    n: int,
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    support_contexts: list[int],
    n_perturbations: int,
    device: torch.device,
    cpu_gen: torch.Generator,
    sampling: str,
    stratum_dist: SupportStratumDistribution | None,
    stratum_cells: dict[tuple[int, ...], torch.Tensor],
    support_flat_idx: torch.Tensor,
    z: dict[str, torch.Tensor] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    max_resample: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    """Sample ``n`` real cells from a pretrain support pool.

    ``(c, p[, z...])`` are drawn with the same rules as ``sample_support_conditions``; each
    draw picks a uniform random cell among rows matching that stratum. Empty strata
    (possible under ``uniform`` c×p sampling) fall back to a random support cell.
    """
    del max_resample  # batched sampling; uniform misses fall back once per draw
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    cond_out = sample_support_conditions(
        n,
        support_contexts=support_contexts,
        n_perturbations=n_perturbations,
        device=torch.device("cpu"),
        cpu_gen=cpu_gen,
        sampling=sampling,
        stratum_dist=stratum_dist,
    )
    z_cond: dict[str, torch.Tensor] | None = None
    if len(cond_out) == 3:
        c_cond, p_cond, z_cond = cond_out
    else:
        c_cond, p_cond = cond_out

    keys = _keys_from_sampled_conditions(
        c_cond,
        p_cond,
        z_cond=z_cond,
        covariate_col_order=covariate_col_order,
    )
    idx = _pick_cells_for_stratum_keys(
        keys,
        stratum_cells=stratum_cells,
        support_flat_idx=support_flat_idx,
        cpu_gen=cpu_gen,
    )
    out = (
        x[idx].to(device),
        c[idx].to(device=device, dtype=torch.long),
        p[idx].to(device=device, dtype=torch.long),
    )
    if covariate_col_order:
        if z is None:
            raise ValueError("z required when covariate_col_order is non-empty")
        z_out = {col: z[col][idx].to(device=device, dtype=torch.long) for col in covariate_col_order}
        return (*out, z_out)
    return out


def stratum_seed(base_seed: int, *indices: int) -> int:
    """Deterministic seed for a stratum (c, p, covariates, ...)."""
    from trajectory_replay.data.tx_strata import stratum_seed as _seed

    return _seed(base_seed, *indices)


def subsample_per_stratum(
    x: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
    *,
    max_per_stratum: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep up to ``max_per_stratum`` cells per (c, p), with a fixed seed per stratum.

    Each stratum uses ``seed + c * 100_003 + p * 1_009``, so the same (c, p) draws the
    same cells whenever that stratum is present, independent of global pool size or
    eval label.
    """
    if max_per_stratum <= 0:
        return x, c, p

    strata: dict[tuple[int, int], list[int]] = {}
    for i in range(len(x)):
        key = (int(c[i].item()), int(p[i].item()))
        strata.setdefault(key, []).append(i)

    picked: list[int] = []
    for ci, pi in sorted(strata.keys()):
        idxs = np.asarray(strata[(ci, pi)], dtype=np.int64)
        rng = np.random.default_rng(stratum_seed(seed, ci, pi))
        if len(idxs) <= max_per_stratum:
            picked.extend(idxs.tolist())
        else:
            picked.extend(rng.choice(idxs, size=max_per_stratum, replace=False).tolist())

    sel = torch.as_tensor(picked, dtype=torch.long)
    return x[sel], c[sel], p[sel]
