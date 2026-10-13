#!/usr/bin/env python3
"""Build every Replogle CL input from the raw panel, in one place.

Input is ``<out-dir>/replogle.h5ad``: the 6642-gene panel, already
``normalize_total(1e4)`` + ``log1p``, with 2000 genes flagged ``var.highly_variable``
and no split column. Everything downstream is derived here, so the dump is
reproducible from that one file.

Stages (all idempotent; existing outputs are skipped unless ``--force``):

``adata``  ``adata.h5ad``      2000-HVG matrix + ``obs['split']`` (train/finetune/test)
``pca``    ``pca_*`` / ``pca{d}_*``  PCA fit on task-0 train cells; unlabeled files are 256-PC
``se``     ``se.npy``          Arc SE-600M on the **full panel**
``sci``    ``scimilarity.npy`` SCimilarity on the **full panel**

Embeddings are encoded from the panel because more genes is strictly more
signal for a frozen encoder, while perturbation prediction and all metrics stay
in the 2000-HVG space of ``adata.h5ad``. Every sidecar shares that file's cell
order, checked against a fingerprint recorded in ``manifest.json``.

``se`` and ``sci`` need their own interpreters (upstream pins Python <3.13), so
those stages re-exec: ``se`` through the ``state`` CLI, ``sci`` through this same
file under the SCimilarity venv. Nothing else is a separate script.

    uv run python scripts/build_replogle.py                    # everything
    uv run python scripts/build_replogle.py --stages adata,pca  # no GPU needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from trajectory_replay.data.tx_preprocess import (
    DEFAULT_PCA_DIM,
    fit_pca_basis,
    pca_sidecar_names,
    project_pca,
    reconstruction_pearson,
    save_pca_basis,
)

ALL_STAGES = ("adata", "pca", "se", "sci")

PANEL_NAME = "replogle.h5ad"
ADATA_NAME = "adata.h5ad"
SE_NAME = "se.npy"
SCI_NAME = "scimilarity.npy"
SCI_UNNORM_NAME = "scimilarity_unnorm.npy"
MANIFEST_NAME = "manifest.json"

KEEP_OBS = ("gene", "sgID_AB", "cell_line")
CONTEXT_COL = "cell_line"
PERTURBATION_COL = "gene"


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def cell_fingerprint(obs_names: np.ndarray) -> str:
    """Stable digest of cell order, so sidecars can prove they line up."""
    h = hashlib.sha256()
    h.update(np.asarray(obs_names).astype("U").tobytes())
    return h.hexdigest()[:16]


def read_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(out_dir: Path, **entries) -> dict:
    manifest = read_manifest(out_dir)
    manifest.update(entries)
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def require_fingerprint(out_dir: Path) -> str:
    fp = read_manifest(out_dir).get("cell_fingerprint")
    if not fp:
        raise SystemExit("No cell_fingerprint in manifest.json; run --stages adata first.")
    return str(fp)


def check_sidecar(path: Path, n_cells: int, label: str) -> dict:
    if n_cells <= 0:
        raise SystemExit(f"No adata cell count in manifest.json; run --stages adata first.")
    arr = np.load(path, mmap_mode="r")
    if arr.shape[0] != n_cells:
        raise SystemExit(
            f"{label} has {arr.shape[0]:,} rows but adata.h5ad has {n_cells:,} cells."
        )
    return {"path": path.name, "shape": [int(x) for x in arr.shape], "dtype": str(arr.dtype)}


# --------------------------------------------------------------------------- #
# stage: adata
# --------------------------------------------------------------------------- #
def derive_splits(cell_line: np.ndarray, gene: np.ndarray) -> np.ndarray:
    """Label cells train / finetune / test from the STATE hepg2 few-shot spec.

    HepG2 is the held-out context: its STATE test genes are never trainable,
    everything else on HepG2 is the few-shot pool. ``non-targeting`` stays
    trainable so the held-out context keeps its own basal.
    """
    from trajectory_replay.data.tx_state_splits import (
        REPLOGLE_CONTROL_PERTURBATION,
        REPLOGLE_FEWSHOT_CELL_LINE,
        expected_replogle_phase,
        load_replogle_state_test_genes,
    )

    test_genes = load_replogle_state_test_genes()
    split = np.full(cell_line.shape, "train", dtype=object)
    fewshot = cell_line == REPLOGLE_FEWSHOT_CELL_LINE
    split[fewshot] = "finetune"
    split[
        fewshot
        & np.isin(gene, list(test_genes))
        & (gene != REPLOGLE_CONTROL_PERTURBATION)
    ] = "test"

    # The vectorized form above must agree with the scalar spec on every
    # (context, perturbation) pair present in the data.
    codes, pairs = pd.MultiIndex.from_arrays([cell_line, gene]).factorize()
    expected = np.array(
        [expected_replogle_phase(str(c), str(g), test_genes=test_genes) for c, g in pairs],
        dtype=object,
    )[codes]
    bad = expected != split
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"split mismatch for ({cell_line[i]}, {gene[i]}): "
            f"{split[i]!r} != {expected[i]!r} ({int(bad.sum()):,} cells)"
        )
    return split.astype(str)


def stage_adata(panel_path: Path, out_dir: Path, *, chunk: int) -> dict:
    print(f"[adata] reading {panel_path}", flush=True)
    panel = ad.read_h5ad(panel_path, backed="r")
    try:
        if "highly_variable" not in panel.var.columns:
            raise SystemExit(f"{panel_path.name} has no var['highly_variable'].")
        hv = np.flatnonzero(panel.var["highly_variable"].values.astype(bool))
        missing = [c for c in KEEP_OBS if c not in panel.obs.columns]
        if missing:
            raise SystemExit(f"{panel_path.name} missing obs columns {missing}.")

        obs = panel.obs[list(KEEP_OBS)].copy()
        cell_line = np.asarray(obs[CONTEXT_COL]).astype(str)
        gene = np.asarray(obs[PERTURBATION_COL]).astype(str)
        obs["split"] = derive_splits(cell_line, gene)
        counts = {k: int(v) for k, v in obs["split"].value_counts().items()}
        print(
            f"[adata] {panel.n_obs:,} cells x {panel.n_vars:,} panel genes "
            f"-> {hv.size:,} HVGs\n[adata] splits {counts}",
            flush=True,
        )

        blocks = []
        for start in range(0, panel.n_obs, chunk):
            stop = min(start + chunk, panel.n_obs)
            blk = panel.X[start:stop]
            blocks.append(blk[:, hv] if sp.issparse(blk) else np.asarray(blk)[:, hv])
            print(f"[adata]   rows {stop:,}/{panel.n_obs:,}", flush=True)
        X = sp.vstack(blocks, format="csr") if sp.issparse(blocks[0]) else np.vstack(blocks)
        del blocks

        var = panel.var.iloc[hv].copy()
        obs_names = np.asarray(panel.obs_names).astype(str)
        panel_n_genes = int(panel.n_vars)
    finally:
        if getattr(panel, "file", None) is not None:
            panel.file.close()

    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obs_names = obs_names
    for col in (*KEEP_OBS, "split"):
        out.obs[col] = out.obs[col].astype("category")
    dest = out_dir / ADATA_NAME
    print(f"[adata] writing {dest}", flush=True)
    out.write_h5ad(dest)

    return {
        "panel": {
            "path": panel_path.name,
            "n_cells": int(out.n_obs),
            "n_genes": panel_n_genes,
        },
        "adata": {
            "path": ADATA_NAME,
            "n_cells": int(out.n_obs),
            "n_genes": int(out.n_vars),
            "split_counts": counts,
            "obs_columns": list(out.obs.columns),
        },
        "cell_fingerprint": cell_fingerprint(obs_names),
    }


# --------------------------------------------------------------------------- #
# stage: pca
# --------------------------------------------------------------------------- #
def stage_pca(out_dir: Path, *, n_comps: int, seed: int) -> dict:
    """PCA fit on task-0 train cells only, in the HVG space used for metrics.

    Fitting on anything else leaks: HepG2 is task 1, so its cells (and their
    contribution to the gene means) must not shape the basis task 0 sees.
    """
    from trajectory_replay.data.tx_cl_tasks import TEST_SPLIT, stream_for

    adata_path = out_dir / ADATA_NAME
    if not adata_path.is_file():
        raise SystemExit(f"{adata_path} missing; run --stages adata first.")

    print(f"[pca] reading {adata_path}", flush=True)
    adata = ad.read_h5ad(adata_path)
    split = np.asarray(adata.obs["split"]).astype(str)
    context = np.asarray(adata.obs[CONTEXT_COL]).astype(str)

    task0 = stream_for("replogle").task(0)
    fit_idx = np.flatnonzero(
        np.isin(context, list(task0.contexts)) & (split == task0.train_split)
    )
    if fit_idx.size == 0:
        raise SystemExit(f"No cells for task 0 {task0.contexts} split={task0.train_split}.")
    future = tuple(c for c in np.unique(context) if c not in task0.contexts)
    if (split[fit_idx] == TEST_SPLIT).any():
        raise AssertionError("test cells in the PCA fit set")
    if np.isin(context[fit_idx], list(future)).any():
        raise AssertionError("future-task contexts in the PCA fit set")
    print(
        f"[pca] task0={task0.name} contexts={task0.contexts} "
        f"fit={fit_idx.size:,}/{adata.n_obs:,} excluded={future} n_comps={n_comps}",
        flush=True,
    )

    basis = fit_pca_basis(
        adata, fit_idx, n_comps=n_comps, seed=seed, fit_phases=(task0.train_split,)
    )
    x_pca = project_pca(adata, basis)
    basis_name, emb_name = pca_sidecar_names(n_comps)
    basis_path = save_pca_basis(out_dir / basis_name, basis)
    emb_path = out_dir / emb_name
    np.save(emb_path, np.asarray(x_pca, dtype=np.float32))

    recon_all = float(reconstruction_pearson(adata, basis, x_pca, seed=seed))
    test_idx = np.flatnonzero(split == TEST_SPLIT)
    recon_test = (
        float(reconstruction_pearson(adata[test_idx], basis, x_pca[test_idx], seed=seed))
        if test_idx.size
        else float("nan")
    )
    print(
        f"[pca] cum var={float(basis.variance_ratio.sum()):.4f} "
        f"recon all={recon_all:.4f} test={recon_test:.4f}",
        flush=True,
    )
    return {
        "pca": {
            "basis_path": basis_path.name,
            "embedding_path": emb_path.name,
            "n_comps": int(basis.n_comps),
            "n_genes": int(basis.n_genes),
            "n_fit_cells": int(basis.n_fit_cells),
            "task0_contexts": list(task0.contexts),
            "task0_train_split": task0.train_split,
            "excluded_contexts": list(future),
            "cumulative_variance_ratio": float(basis.variance_ratio.sum()),
            "recon_pearson_all": recon_all,
            "recon_pearson_test": recon_test,
        }
    }


# --------------------------------------------------------------------------- #
# stage: se (Arc SE-600M, separate venv)
# --------------------------------------------------------------------------- #
def resolve_checkpoint(model_dir: Path) -> Path:
    for name in ("se600m_epoch16.ckpt", "se600m_epoch15.ckpt"):
        if (model_dir / name).is_file():
            return model_dir / name
    ckpts = sorted(model_dir.glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"No *.ckpt under {model_dir}. See third_party/README.md.")
    return ckpts[-1]


def stage_se(panel_path: Path, out_dir: Path, *, model_dir: Path, batch_size: int) -> dict:
    state_bin = REPO_ROOT / "third_party/arc-state/.venv/bin/state"
    if not state_bin.is_file():
        raise SystemExit(f"{state_bin} missing. Run ./third_party/setup_arc_state.sh")
    dest = out_dir / SE_NAME
    cmd = [
        str(state_bin), "emb", "transform",
        "--model-folder", str(model_dir),
        "--checkpoint", str(resolve_checkpoint(model_dir)),
        "--input", str(panel_path.resolve()),
        "--output", str(dest),
        "--embed-key", "X_state",
        "--batch-size", str(batch_size),
    ]
    print(f"[se] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT / "third_party/arc-state")
    n_cells = int(read_manifest(out_dir).get("adata", {}).get("n_cells") or 0)
    return {"se": {**check_sidecar(dest, n_cells, SE_NAME), "encoded_from": panel_path.name}}


# --------------------------------------------------------------------------- #
# stage: sci (SCimilarity, separate venv -> this file re-execs itself)
# --------------------------------------------------------------------------- #
def encode_scimilarity(
    panel_path: Path,
    dest: Path,
    *,
    model_dir: Path,
    buffer_size: int,
    skip_l2_normalize: bool = False,
) -> None:
    """Run inside the SCimilarity venv. ``X`` is already log-normalized."""
    from scimilarity.cell_embedding import CellEmbedding
    from scimilarity.nn_models import Encoder
    from scimilarity.utils import align_dataset

    if skip_l2_normalize:
        _orig_forward = Encoder.forward

        def _forward_no_l2(self, x):
            for layer in self.network:
                x = layer(x)
            return x

        Encoder.forward = _forward_no_l2
        print("[sci] skipping Encoder F.normalize (last Linear output)", flush=True)
    else:
        _orig_forward = None

    try:
        ce = CellEmbedding(model_path=str(model_dir), use_gpu=True)
        print(f"[sci] encoder n_genes={ce.n_genes} latent={ce.latent_dim}", flush=True)
        adata = ad.read_h5ad(panel_path)
        overlap = int(adata.var.index.isin(ce.gene_order).sum())
        print(
            f"[sci] {adata.n_obs:,} cells x {adata.n_vars:,} genes; "
            f"overlap {overlap:,}/{ce.n_genes:,}",
            flush=True,
        )
        aligned = align_dataset(adata, ce.gene_order, keep_obsm=False)
        del adata
        emb = np.ascontiguousarray(
            ce.get_embeddings(aligned.X, buffer_size=buffer_size), dtype=np.float32
        )
    finally:
        if _orig_forward is not None:
            Encoder.forward = _orig_forward
    norms = np.linalg.norm(emb, axis=1)
    print(
        f"[sci] row-norm mean={float(norms.mean()):.4f} "
        f"std={float(norms.std()):.4f} min={float(norms.min()):.4f} "
        f"max={float(norms.max()):.4f}",
        flush=True,
    )
    np.save(dest, emb)
    print(f"[sci] wrote {dest} {emb.shape}", flush=True)


def stage_sci(
    panel_path: Path,
    out_dir: Path,
    *,
    model_dir: Path,
    buffer_size: int,
    skip_l2_normalize: bool = False,
) -> dict:
    dest = out_dir / (SCI_UNNORM_NAME if skip_l2_normalize else SCI_NAME)
    required = ("encoder.ckpt", "gene_order.tsv", "layer_sizes.json", "label_ints.csv")
    missing = [f for f in required if not (model_dir / f).is_file()]
    if missing:
        raise SystemExit(
            f"SCimilarity model {model_dir} missing {missing}. "
            "Run: sbatch sbatch/third_party/download_scimilarity_model.sbatch"
        )
    sci_python = REPO_ROOT / "third_party/scimilarity/.venv/bin/python"
    if not sci_python.is_file():
        raise SystemExit(f"{sci_python} missing. Run ./third_party/setup_scimilarity.sh")

    cmd = [
        str(sci_python), str(Path(__file__).resolve()),
        "--worker-sci",
        "--out-dir", str(out_dir),
        "--sci-model-dir", str(model_dir),
        "--sci-buffer-size", str(buffer_size),
    ]
    if skip_l2_normalize:
        cmd.append("--sci-skip-l2-normalize")
    print(f"[sci] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    n_cells = int(read_manifest(out_dir).get("adata", {}).get("n_cells") or 0)
    return {
        "sci_unnorm" if skip_l2_normalize else "sci": {
            **check_sidecar(dest, n_cells, dest.name),
            "encoded_from": panel_path.name,
            "skip_l2_normalize": bool(skip_l2_normalize),
        }
    }


# --------------------------------------------------------------------------- #
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data_dir/replogle")
    p.add_argument("--stages", default="all", help=f"comma list of {ALL_STAGES} or 'all'")
    p.add_argument("--force", action="store_true", help="rebuild stages whose outputs exist")
    p.add_argument("--n-comps", type=int, default=DEFAULT_PCA_DIM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=100_000, help="cells per read when subsetting HVGs")
    p.add_argument("--se-model-dir", type=Path, default=REPO_ROOT / "data/arc_state/SE-600M")
    p.add_argument("--se-batch-size", type=int, default=256)
    p.add_argument("--sci-model-dir", type=Path, default=REPO_ROOT / "data/scimilarity/model")
    p.add_argument("--sci-buffer-size", type=int, default=4096)
    p.add_argument(
        "--sci-skip-l2-normalize",
        action="store_true",
        help="Write scimilarity_unnorm.npy from the last Linear (no F.normalize).",
    )
    p.add_argument("--worker-sci", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> None:
    args = make_parser().parse_args()
    out_dir = args.out_dir.resolve()
    panel_path = out_dir / PANEL_NAME

    if args.worker_sci:
        dest = out_dir / (SCI_UNNORM_NAME if args.sci_skip_l2_normalize else SCI_NAME)
        encode_scimilarity(
            panel_path,
            dest,
            model_dir=args.sci_model_dir,
            buffer_size=args.sci_buffer_size,
            skip_l2_normalize=bool(args.sci_skip_l2_normalize),
        )
        return

    stages = ALL_STAGES if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(",") if s.strip()
    )
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}; choose from {ALL_STAGES}.")
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, pca_emb_name = pca_sidecar_names(args.n_comps)
    outputs = {
        "adata": out_dir / ADATA_NAME,
        "pca": out_dir / pca_emb_name,
        "se": out_dir / SE_NAME,
        "sci": out_dir / (SCI_UNNORM_NAME if args.sci_skip_l2_normalize else SCI_NAME),
    }
    for stage in stages:
        if outputs[stage].exists() and not args.force:
            print(f"[{stage}] {outputs[stage].name} exists; skipping (use --force)", flush=True)
            # Sidecars can be produced out of band (e.g. a standalone encode job),
            # so adopt them into the manifest rather than leaving them unrecorded.
            if stage in ("se", "sci") and stage not in read_manifest(out_dir):
                n_cells = int(read_manifest(out_dir).get("adata", {}).get("n_cells") or 0)
                update_manifest(
                    out_dir,
                    **{
                        stage: {
                            **check_sidecar(outputs[stage], n_cells, outputs[stage].name),
                            "encoded_from": panel_path.name,
                        }
                    },
                )
            continue
        if stage == "adata":
            entries = stage_adata(panel_path, out_dir, chunk=args.chunk)
        elif stage == "pca":
            entries = stage_pca(out_dir, n_comps=args.n_comps, seed=args.seed)
        elif stage == "se":
            require_fingerprint(out_dir)
            entries = stage_se(
                panel_path, out_dir, model_dir=args.se_model_dir, batch_size=args.se_batch_size
            )
        else:
            require_fingerprint(out_dir)
            entries = stage_sci(
                panel_path,
                out_dir,
                model_dir=args.sci_model_dir,
                buffer_size=args.sci_buffer_size,
                skip_l2_normalize=bool(args.sci_skip_l2_normalize),
            )
        update_manifest(out_dir, **entries)

    print(f"\n{json.dumps(read_manifest(out_dir), indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
