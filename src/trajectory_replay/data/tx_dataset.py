"""AnnData transcriptomics → (x, c, p, z) tensors for conditional flow matching.

Maps a context obs column → ``c`` and a perturbation obs column → ``p``.
Optional ``covariate_cols`` (e.g. ``cell_type``) → ``z`` embeddings per column.
Uses ``obsm['X_pca']`` (or another matrix key) as the data representation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import torch
from torch.utils.data import TensorDataset

from trajectory_replay.data.tx_strata import parse_covariate_cols

SplitPhase = Literal["train", "finetune", "hold_out"]


@dataclass(frozen=True)
class TxEncoding:
    """Stable categorical maps for context (c), perturbation (p), and covariates."""

    context_categories: tuple[str, ...]
    perturbation_categories: tuple[str, ...]
    context_to_idx: dict[str, int]
    perturbation_to_idx: dict[str, int]
    covariate_col_order: tuple[str, ...] = ()
    covariate_categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    covariate_to_idx: dict[str, dict[str, int]] = field(default_factory=dict)
    control_perturbation: str | None = None

    @property
    def n_contexts(self) -> int:
        return len(self.context_categories)

    @property
    def n_perturbations(self) -> int:
        return len(self.perturbation_categories)

    @property
    def n_covariates(self) -> int:
        return len(self.covariate_col_order)

    @property
    def control_pert_index(self) -> int | None:
        if self.control_perturbation is None:
            return None
        return self.perturbation_to_idx.get(self.control_perturbation)

    def covariate_sizes(self) -> dict[str, int]:
        return {col: len(self.covariate_categories[col]) for col in self.covariate_col_order}

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, d: dict[str, Any]) -> TxEncoding:
        """Load encoding JSON (current or legacy donor/cytokine field names)."""
        if "context_categories" in d:
            return cls(
                context_categories=tuple(d["context_categories"]),
                perturbation_categories=tuple(d["perturbation_categories"]),
                context_to_idx=dict(d["context_to_idx"]),
                perturbation_to_idx=dict(d["perturbation_to_idx"]),
                covariate_col_order=tuple(d.get("covariate_col_order") or ()),
                covariate_categories={
                    k: tuple(v) for k, v in (d.get("covariate_categories") or {}).items()
                },
                covariate_to_idx={
                    k: dict(v) for k, v in (d.get("covariate_to_idx") or {}).items()
                },
                control_perturbation=d.get("control_perturbation"),
            )
        return cls(
            context_categories=tuple(d["donor_categories"]),
            perturbation_categories=tuple(d["cytokine_categories"]),
            context_to_idx=dict(d["donor_to_idx"]),
            perturbation_to_idx=dict(d["cytokine_to_idx"]),
            control_perturbation=d.get("control_cytokine"),
        )


@dataclass
class TxDataBundle:
    """Tensors and metadata for one filtered AnnData slice."""

    x: torch.Tensor
    c: torch.Tensor
    p: torch.Tensor
    encoding: TxEncoding
    data_dim: int
    split_col: str
    phases: tuple[str, ...]
    n_cells: int
    obsm_key: str
    z: dict[str, torch.Tensor] = field(default_factory=dict)


def _repo_root() -> Path:
    # .../src/trajectory_replay/data/tx_dataset.py -> repo root
    return Path(__file__).resolve().parents[3]


def default_h5ad_path() -> Path:
    return _repo_root() / "data_dir" / "pbmc_parse_with_splits.h5ad"


def build_encoding_from_h5ad(
    h5ad_path: Path | str,
    *,
    context_col: str = "donor",
    perturbation_col: str = "cytokine",
    covariate_cols: tuple[str, ...] = (),
    control_perturbation: str = "PBS",
) -> TxEncoding:
    """Category maps from all cells in the h5ad (stable indices across splits)."""
    path = Path(h5ad_path)
    adata = ad.read_h5ad(path)
    required = [context_col, perturbation_col, *covariate_cols]
    for col in required:
        if col not in adata.obs.columns:
            raise KeyError(f"obs[{col!r}] missing; columns: {list(adata.obs.columns)}")
    covariate_arrays = {
        col: adata.obs[col].astype(str).to_numpy() for col in covariate_cols
    }
    return build_encoding(
        adata.obs[context_col].astype(str).to_numpy(),
        adata.obs[perturbation_col].astype(str).to_numpy(),
        covariate_arrays=covariate_arrays,
        covariate_col_order=covariate_cols,
        control_perturbation=control_perturbation,
    )


def context_index(encoding: TxEncoding, label: str) -> int:
    if label not in encoding.context_to_idx:
        raise KeyError(f"Unknown context {label!r}; known: {encoding.context_categories}")
    return encoding.context_to_idx[label]


def perturbation_index(encoding: TxEncoding, label: str) -> int:
    if label not in encoding.perturbation_to_idx:
        raise KeyError(f"Unknown perturbation {label!r}; known: {encoding.perturbation_categories}")
    return encoding.perturbation_to_idx[label]


def build_encoding(
    contexts: np.ndarray,
    perturbations: np.ndarray,
    *,
    covariate_arrays: dict[str, np.ndarray] | None = None,
    covariate_col_order: tuple[str, ...] = (),
    control_perturbation: str = "PBS",
) -> TxEncoding:
    context_cats = tuple(sorted({str(x) for x in contexts}))
    pert_cats = tuple(sorted({str(x) for x in perturbations}))
    cov_cats: dict[str, tuple[str, ...]] = {}
    cov_to_idx: dict[str, dict[str, int]] = {}
    if covariate_arrays is None:
        covariate_arrays = {}
    for col in covariate_col_order:
        if col not in covariate_arrays:
            raise KeyError(f"covariate column {col!r} missing from covariate_arrays")
        cats = tuple(sorted({str(x) for x in covariate_arrays[col]}))
        cov_cats[col] = cats
        cov_to_idx[col] = {d: i for i, d in enumerate(cats)}
    return TxEncoding(
        context_categories=context_cats,
        perturbation_categories=pert_cats,
        context_to_idx={d: i for i, d in enumerate(context_cats)},
        perturbation_to_idx={k: i for i, k in enumerate(pert_cats)},
        covariate_col_order=covariate_col_order,
        covariate_categories=cov_cats,
        covariate_to_idx=cov_to_idx,
        control_perturbation=control_perturbation if control_perturbation in pert_cats else None,
    )


def encode_labels(
    contexts: np.ndarray,
    perturbations: np.ndarray,
    encoding: TxEncoding,
    *,
    covariate_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    c = np.array([encoding.context_to_idx[str(d)] for d in contexts], dtype=np.int64)
    p = np.array([encoding.perturbation_to_idx[str(k)] for k in perturbations], dtype=np.int64)
    z: dict[str, np.ndarray] = {}
    if encoding.covariate_col_order:
        if covariate_arrays is None:
            raise ValueError("covariate_arrays required when encoding has covariates")
        for col in encoding.covariate_col_order:
            z[col] = np.array(
                [encoding.covariate_to_idx[col][str(v)] for v in covariate_arrays[col]],
                dtype=np.int64,
            )
    return c, p, z


def read_obsm_matrix(adata: ad.AnnData, key: str = "X_pca") -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"obsm[{key!r}] not found; available: {list(adata.obsm.keys())}")
    x = np.asarray(adata.obsm[key], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"obsm[{key!r}] must be 2D, got shape {x.shape}")
    return x


def load_tx_data(
    h5ad_path: Path | str,
    *,
    split_col: str = "split1",
    phases: tuple[str, ...] = ("train",),
    context_col: str = "donor",
    perturbation_col: str = "cytokine",
    covariate_cols: tuple[str, ...] = (),
    obsm_key: str = "X_pca",
    control_perturbation: str = "PBS",
    max_samples: int | None = None,
    max_samples_per_stratum: int | None = None,
    subsample_seed: int = 0,
    encoding: TxEncoding | None = None,
    only_control_cells: bool = False,
) -> TxDataBundle:
    """Load h5ad, filter by split phase(s), return float x and long c, p, z.

    ``context_col`` / ``perturbation_col`` name ``adata.obs`` columns (defaults match
    the PBMC h5ad: donor → c, cytokine → p). ``covariate_cols`` adds optional z columns.

    If ``encoding`` is provided (e.g. from pretrain), categories are fixed to that
    mapping; unknown labels raise ``KeyError``.
    """
    path = Path(h5ad_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    adata = ad.read_h5ad(path)
    required = [split_col, context_col, perturbation_col, *covariate_cols]
    for col in required:
        if col not in adata.obs.columns:
            raise KeyError(f"obs[{col!r}] missing; columns: {list(adata.obs.columns)}")

    split_labels = adata.obs[split_col].astype(str).to_numpy()
    mask = np.isin(split_labels, list(phases))
    if not mask.any():
        raise ValueError(
            f"No cells for split_col={split_col!r} phases={phases}; "
            f"available: {sorted(set(split_labels))}"
        )

    contexts = adata.obs[context_col].astype(str).to_numpy()[mask]
    perturbations = adata.obs[perturbation_col].astype(str).to_numpy()[mask]
    covariate_arrays = {
        col: adata.obs[col].astype(str).to_numpy()[mask] for col in covariate_cols
    }
    x_np = read_obsm_matrix(adata, obsm_key)[mask]

    if encoding is None:
        enc = build_encoding(
            contexts,
            perturbations,
            covariate_arrays=covariate_arrays,
            covariate_col_order=covariate_cols,
            control_perturbation=control_perturbation,
        )
    else:
        enc = encoding
        if enc.covariate_col_order != covariate_cols:
            raise ValueError(
                f"covariate_cols={covariate_cols!r} != encoding.covariate_col_order="
                f"{enc.covariate_col_order!r}"
            )
        for ctx in set(contexts):
            if ctx not in enc.context_to_idx:
                raise KeyError(f"Unknown context {ctx!r} (not in encoding)")
        for pert in set(perturbations):
            if pert not in enc.perturbation_to_idx:
                raise KeyError(f"Unknown perturbation {pert!r} (not in encoding)")
        for col in covariate_cols:
            for val in set(covariate_arrays[col]):
                if val not in enc.covariate_to_idx[col]:
                    raise KeyError(f"Unknown covariate {col}={val!r} (not in encoding)")

    c_np, p_np, z_np = encode_labels(
        contexts, perturbations, enc, covariate_arrays=covariate_arrays
    )

    x = torch.from_numpy(x_np)
    c = torch.from_numpy(c_np).long()
    p = torch.from_numpy(p_np).long()
    z = {col: torch.from_numpy(z_np[col]).long() for col in enc.covariate_col_order}

    if only_control_cells:
        if enc.control_pert_index is None:
            raise ValueError("only_control_cells requires encoding.control_pert_index")
        keep = p == int(enc.control_pert_index)
        if not bool(keep.any()):
            raise ValueError(
                f"No control cells for phases={phases!r} "
                f"control_pert_index={enc.control_pert_index}"
            )
        x, c, p = x[keep], c[keep], p[keep]
        if z:
            z = {col: z[col][keep] for col in z}

    if max_samples_per_stratum is not None and max_samples_per_stratum > 0:
        from trajectory_replay.data.tx_strata import subsample_per_stratum as _sub_stratum

        x, c, p, z = _sub_stratum(
            x,
            c,
            p,
            z=z or None,
            covariate_col_order=enc.covariate_col_order,
            max_per_stratum=max_samples_per_stratum,
            seed=subsample_seed,
        )
    elif max_samples is not None and max_samples > 0 and x.shape[0] > max_samples:
        from trajectory_replay.data.tx_strata import stratified_subsample as _sub_global

        x, c, p, z = _sub_global(
            x,
            c,
            p,
            z=z or None,
            covariate_col_order=enc.covariate_col_order,
            max_samples=max_samples,
            seed=subsample_seed,
        )

    return TxDataBundle(
        x=x,
        c=c,
        p=p,
        z=z,
        encoding=enc,
        data_dim=int(x.shape[1]),
        split_col=split_col,
        phases=phases,
        n_cells=int(x.shape[0]),
        obsm_key=obsm_key,
    )


def make_tx_tensor_dataset(bundle: TxDataBundle) -> TensorDataset:
    """Build a ``TensorDataset`` with (x, c, p, *z_cols) in encoding column order."""
    tensors: list[torch.Tensor] = [bundle.x, bundle.c, bundle.p]
    for col in bundle.encoding.covariate_col_order:
        tensors.append(bundle.z[col])
    return TensorDataset(*tensors)


def unpack_tx_batch(
    batch: tuple[torch.Tensor, ...],
    covariate_col_order: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    """Unpack a training batch into x, c, p, and optional covariate dict."""
    x1, c, p = batch[0], batch[1], batch[2]
    if not covariate_col_order:
        return x1, c, p, None
    z = {col: batch[3 + i] for i, col in enumerate(covariate_col_order)}
    return x1, c, p, z


def save_encoding(path: Path, encoding: TxEncoding) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encoding.to_jsonable(), indent=2), encoding="utf-8")


def load_encoding(path: Path | str) -> TxEncoding:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return TxEncoding.from_jsonable(d)


def parse_phases(s: str) -> tuple[str, ...]:
    phases = tuple(p.strip() for p in s.split(",") if p.strip())
    if not phases:
        raise ValueError("At least one split phase required")
    return phases


__all__ = [
    "SplitPhase",
    "TxEncoding",
    "TxDataBundle",
    "build_encoding",
    "build_encoding_from_h5ad",
    "context_index",
    "default_h5ad_path",
    "encode_labels",
    "load_encoding",
    "load_tx_data",
    "make_tx_tensor_dataset",
    "parse_covariate_cols",
    "parse_phases",
    "perturbation_index",
    "read_obsm_matrix",
    "save_encoding",
    "unpack_tx_batch",
]
