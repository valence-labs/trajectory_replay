# Third-party tooling

Vendored upstream packages that are **not** CL baselines and must stay out of the
root project `.venv` (Python version / license / heavy deps differ).

| Package | Role | Pin file | Clone path |
| --- | --- | --- | --- |
| [ArcInstitute/state](https://github.com/ArcInstitute/state) (`arc-state`) | SE-600M cell embeddings (`state emb transform`) | [`arc-state.PIN.txt`](arc-state.PIN.txt) | `third_party/arc-state/` |
| [Genentech/scimilarity](https://github.com/Genentech/scimilarity) | Frozen cell embeddings (encoder-only) | [`scimilarity.PIN.txt`](scimilarity.PIN.txt) | `third_party/scimilarity/` (venv only) |

CL comparison code lives under [`baselines/`](../baselines/README.md) instead.

## Arc STATE (SE embeddings)

Upstream requires **Python ≥3.11,<3.13** (we use 3.12 here). Root project is 3.13+.

### One-shot setup

```bash
./third_party/setup_arc_state.sh
```

This clones (if needed), checks out the pin, and runs `uv sync --python 3.12`
inside `third_party/arc-state/` (creates `.venv` there only).

### Manual clone / re-pin

```bash
git clone https://github.com/ArcInstitute/state.git third_party/arc-state
cd third_party/arc-state
git checkout "$(cat ../arc-state.PIN.txt)"
uv sync --python 3.12
uv run state --help
```

### Encode cells with SE-600M

Download weights + Replogle/Nadig expression once, then:

```bash
# From third_party/arc-state/
uv run state emb transform \
  --model-folder /path/to/SE-600M \
  --checkpoint /path/to/SE-600M/se600m_epoch16.ckpt \
  --input /path/to/replogle.h5ad \
  --output /path/to/replogle_se.h5ad
```

HF assets:
- Model: [`arcinstitute/SE-600M`](https://huggingface.co/arcinstitute/SE-600M) — use `se600m_epoch16.ckpt` (+ `protein_embeddings.pt`)
- Data (local): place the Replogle AnnData file at `data_dir/replogle.h5ad`
  (same file as in [`arcinstitute/Replogle-Nadig-Preprint`](https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint))

Cluster helper (downloads SE-600M if missing; uses local `replogle.h5ad` by default):

```bash
sbatch sbatch/third_party/encode_se.sbatch

# Other local h5ad (e.g. PBMC)
sbatch --export=ALL,INPUT_H5AD=data_dir/pbmc_parse_with_splits.h5ad \
  sbatch/third_party/encode_se.sbatch

# Optional: also pull the full HF preprint pack
sbatch --export=ALL,DOWNLOAD_DATA=1 sbatch/third_party/encode_se.sbatch

# Resolve paths only
sbatch --export=ALL,DRY_RUN=1,SKIP_DOWNLOAD=1 sbatch/third_party/encode_se.sbatch
```

Output: `results_dir/arc_state/se/replogle_se.h5ad` with `obsm["X_state"]` (SE-600M dim **2048**).

### License note

Arc STATE code is CC BY-NC-SA 4.0; model weights have a separate non-commercial
license. Cite the [State paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2)
if you use this.

## SCimilarity (cell embeddings)

Isolated Python 3.12 venv; root project stays 3.13. Encoder-only — do not
download the 60 GB kNN index.

```bash
./third_party/setup_scimilarity.sh
```

Put `encoder.ckpt`, `gene_order.tsv`, `layer_sizes.json`, and `label_ints.csv`
from [Zenodo 10685499](https://zenodo.org/records/10685499) in
`data/scimilarity/model/`. Then:

```bash
STAGES=sci sbatch sbatch/tx/build_replogle.sbatch
```

That writes `data_dir/replogle/scimilarity.npy` in the same cell order as
`data_dir/replogle/adata.h5ad`. The unified builder encodes the 6642-gene panel,
not the 2000-gene CL dump — SCimilarity `align_dataset` requires ≥5000
overlapping genes. There is no counts layer, so the encoder sees the panel's
existing log-normalized `X` (zero-filled to the encoder gene list). Cite
[Heimberg et al., Nature 2024](https://doi.org/10.1038/s41586-024-08411-y).

Oneshot / sequential CL:

```bash
REPRESENTATION=scimilarity sbatch sbatch/tx/train_se_oneshot_hepg2.sbatch
# or
sbatch sbatch/tx/train_scimilarity_oneshot_hepg2.sbatch

REPRESENTATION=scimilarity DATASET=replogle STRATEGY=standard \
  sbatch sbatch/tx/train_cl.sbatch
```
