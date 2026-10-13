State Embedding model checkpoints are meant to be used with the [Arc State](https://pypi.org/project/arc-state/) repository.

State is distributed via [`uv`](https://docs.astral.sh/uv).

## Running from pypi

```bash
uv tool install arc-state
```

To generate embeddings given an AnnData:

```code
state emb --model-folder <SE-600M_MODEL_PATH> --input <INPUT_ANNDATA>.h5ad --output <OUTPUT_PATH>.h5ad
```

## Running from source

```bash
# Clone repo
git clone github.com:arcinstitute/state
cd state

# Initialize venv
uv venv

# Install
uv tool install -e .
```

To generate embeddings given an AnnData:

```code
state emb --model-folder <SE-600M_MODEL_PATH> --input <INPUT_ANNDATA>.h5ad --output <OUTPUT_PATH>.h5ad
```

For model licenses please see `MODEL_ACCEPTABLE_USE_POLICY.md`, `MODEL_LICENSE.md`, and `LICENSE.md`.