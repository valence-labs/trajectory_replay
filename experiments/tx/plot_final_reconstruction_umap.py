#!/usr/bin/env python3
"""Observed versus frozen-affine PCA reconstruction UMAPs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from experiments.tx.train_cl import DEFAULTS
from trajectory_replay.data.tx_cl_data import load_gene_rows
from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, load_stream_obs, stream_for
from trajectory_replay.data.tx_preprocess import pca_sidecar_names
from trajectory_replay.data.tx_state_splits import REPLOGLE_FEWSHOT_CELL_LINE
from trajectory_replay.data.tx_task_decoder import TaskDecoderBank


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PanelSpec:
    dataset: str
    label: str
    pca_dim: int
    context_tasks: tuple[tuple[str, int], ...]


PANELS = (
    PanelSpec(
        dataset="replogle",
        label="REPLOGLE — HepG2 test",
        pca_dim=50,
        context_tasks=((REPLOGLE_FEWSHOT_CELL_LINE, 1),),
    ),
    PanelSpec(
        dataset="pbmc",
        label="PBMC — all donors",
        pca_dim=100,
        context_tasks=(
            ("Donor2", 0),
            ("Donor3", 0),
            ("Donor5", 0),
            ("Donor6", 0),
            ("Donor7", 0),
            ("Donor8", 0),
            ("Donor10", 0),
            ("Donor11", 0),
            ("Donor1", 1),
            ("Donor4", 2),
            ("Donor9", 3),
            ("Donor12", 4),
        ),
    ),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9,
        }
    )


def _joint_umap(
    true: np.ndarray,
    recon: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    matrix = np.concatenate([true, recon], axis=0).astype(np.float32, copy=False)
    joint = ad.AnnData(X=matrix)
    n_components = min(50, joint.n_vars - 1, joint.n_obs - 1)
    sc.pp.pca(joint, n_comps=n_components, random_state=seed)
    sc.pp.neighbors(joint, n_neighbors=15, random_state=seed)
    sc.tl.umap(joint, random_state=seed)
    return np.asarray(joint.obsm["X_umap"], dtype=float)


def _per_cell_pearson(true: np.ndarray, recon: np.ndarray) -> np.ndarray:
    true_centered = true - true.mean(axis=1, keepdims=True)
    recon_centered = recon - recon.mean(axis=1, keepdims=True)
    numerator = np.sum(true_centered * recon_centered, axis=1)
    denominator = np.sqrt(
        np.sum(true_centered**2, axis=1)
        * np.sum(recon_centered**2, axis=1)
    )
    return numerator / np.maximum(denominator, 1e-12)


def _load_panel(
    spec: PanelSpec,
    *,
    n_cells: int,
    seed: int,
) -> dict[str, object]:
    defaults = DEFAULTS[spec.dataset]
    stream = stream_for(spec.dataset)
    h5ad_path = Path(defaults["h5ad_pca"])
    obs = load_stream_obs(h5ad_path, stream)
    context = np.asarray(obs["context"]).astype(str)
    split = np.asarray(obs["split"]).astype(str)
    rng = np.random.default_rng(seed)
    base, remainder = divmod(n_cells, len(spec.context_tasks))
    selected: list[np.ndarray] = []
    split_by_context: dict[str, str] = {}
    for position, (context_label, task_id) in enumerate(spec.context_tasks):
        pool_split = (
            stream.task(0).train_split
            if spec.dataset == "pbmc" and task_id == 0
            else TEST_SPLIT
        )
        split_by_context[context_label] = pool_split
        eligible = np.flatnonzero(
            (context == context_label) & (split == pool_split)
        )
        requested = base + (1 if position < remainder else 0)
        if eligible.size < requested:
            raise ValueError(
                f"{spec.dataset}/{context_label}: requested {requested} cells, "
                f"only {eligible.size} available"
            )
        selected.append(rng.choice(eligible, size=requested, replace=False))
    indices = np.sort(np.concatenate(selected))
    task_by_context = dict(spec.context_tasks)
    task_ids = np.asarray(
        [task_by_context[label] for label in context[indices]],
        dtype=np.int16,
    )

    _, embedding_name = pca_sidecar_names(spec.pca_dim)
    embedding_path = Path(defaults["pca"]).parent / embedding_name
    embedding = np.load(embedding_path, mmap_mode="r")
    latent = np.asarray(embedding[indices], dtype=np.float64)

    decoder_dir = (
        Path(defaults["bucket"])
        / f"decoders_pca{spec.pca_dim}_affine"
    )
    bank = TaskDecoderBank.load_dir(decoder_dir)
    recon = bank.decode(latent, task_ids).astype(np.float32)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        true = load_gene_rows(adata, indices)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    cell_pearson = _per_cell_pearson(true, recon)
    residual = float(np.sum((true - recon) ** 2))
    total = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    metrics = {
        "n_cells": int(n_cells),
        "per_cell_reconstruction_pearson_mean": float(
            np.nanmean(cell_pearson)
        ),
        "per_cell_reconstruction_pearson_median": float(
            np.nanmedian(cell_pearson)
        ),
        "decoder_r2": float(1.0 - residual / max(total, 1e-12)),
    }
    return {
        "spec": spec,
        "true": true,
        "recon": recon,
        "metrics": metrics,
        "sources": {
            "h5ad": str(h5ad_path),
            "embedding": str(embedding_path),
            "decoder_dir": str(decoder_dir),
        },
        "split_by_context": split_by_context,
    }


def render(panels: list[dict[str, object]], out: Path, *, seed: int) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 3.15))
    for panel_index, (ax, panel) in enumerate(
        zip(axes, panels, strict=True)
    ):
        spec = panel["spec"]
        if not isinstance(spec, PanelSpec):
            raise TypeError("Invalid panel specification")
        true = np.asarray(panel["true"])
        recon = np.asarray(panel["recon"])
        coords = _joint_umap(
            true,
            recon,
            seed=seed + 100 * panel_index,
        )
        n_true = true.shape[0]
        ax.scatter(
            coords[:n_true, 0],
            coords[:n_true, 1],
            s=5,
            color="#457B9D",
            alpha=0.42,
            linewidths=0,
            label="Observed",
            rasterized=True,
        )
        ax.scatter(
            coords[n_true:, 0],
            coords[n_true:, 1],
            s=5,
            color="#E76F51",
            alpha=0.42,
            linewidths=0,
            label="Reconstructed",
            rasterized=True,
        )
        metrics = panel["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("Invalid panel metrics")
        ax.set_title(
            f"{spec.label}\n"
            f"PCA-{spec.pca_dim} affine, "
            f"cell $r={float(metrics['per_cell_reconstruction_pearson_mean']):.3f}$",
            pad=5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        if panel_index == 0:
            ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        markerscale=1.8,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=400,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.03,
    )
    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.03,
    )
    plt.close(fig)


def write_summary(panels: list[dict[str, object]], out: Path, *, seed: int) -> None:
    payload = {
        "description": (
            "Observed cells and reconstructions from observed PCA latents using "
            "the frozen task-specific affine decoder; no CFM generation."
        ),
        "sampling_seed": seed,
        "panels": [
            {
                "dataset": spec.dataset,
                "contexts": [context for context, _ in spec.context_tasks],
                "split_by_context": panel["split_by_context"],
                "pca_dim": spec.pca_dim,
                "decoder_task_by_context": dict(spec.context_tasks),
                "metrics": panel["metrics"],
                "sources": panel["sources"],
            }
            for panel in panels
            if isinstance((spec := panel["spec"]), PanelSpec)
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results_dir" / "tx_results" / "figures",
    )
    args = parser.parse_args()
    panels = [
        _load_panel(
            spec,
            n_cells=int(args.n_cells),
            seed=int(args.seed) + index,
        )
        for index, spec in enumerate(PANELS)
    ]
    out = args.out_dir / "tx_observed_vs_affine_reconstruction_umap.png"
    render(panels, out, seed=int(args.seed))
    write_summary(
        panels,
        args.out_dir / "tx_observed_vs_affine_reconstruction_umap.json",
        seed=int(args.seed),
    )
    print(out)


if __name__ == "__main__":
    main()
