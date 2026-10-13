# MNIST perturbation-holdout experiments

Minimal MNIST track mirroring the transcriptomics pretrain → finetune → replay workflow.
Goal: test whether **field replay** stays robust when finetuning does **not** use digit-specific conditioning.

## Conceptual mapping

| Transcriptomics | MNIST analogue |
|-----------------|----------------|
| Context `c` (donor) | Fixed `c=0` (single dummy context) |
| Perturbation `p` (cytokine) | Digit label `p ∈ {0,…,9}` |
| Pretrain cells | All digits except held-out (default: `8`) |
| Finetune cells | Held-out digit only |
| New embedding at finetune | Held-out digit row `pert_emb[8]` exists from init; trained at finetune (no table expansion) |
| Generative replay | `train_flow_matching_generative_replay(..., replay_mode="endpoint")` |
| Field replay | `train_flow_matching_generative_replay(..., replay_mode="field")` |

## Data preparation

Run the notebook:

```bash
jupyter notebook notebooks/mnist/prepare_mnist_splits.ipynb
```

Outputs (`data_dir/mnist/processed/`):

| File | Contents |
|------|----------|
| `pretrain.pt` | Seen digits (`p != held_out`): `x`, `c`, `p` |
| `finetune.pt` | Held-out digit only: `x`, `c`, `p` |
| `manifest.json` | Held-out digit, counts, normalization |

Only the MNIST **train** (60k) and **test** (10k) IDX files are loaded, concatenated into one 70k pool, then split by digit into pretrain vs finetune. There is no separate evaluation test set.

## Experiment matrix

Two finetune **methods** × two **conditioning** modes:

| Run | Finetune strategy | Conditioning |
|-----|-------------------|--------------|
| A | `generative_replay` | **Conditional**: train held-out row `pert_emb[8]` (full table from init) |
| B | `generative_field_replay` | **Conditional**: train held-out row `pert_emb[8]` (full table from init) |
| C | `generative_replay` | **Unconditional**: no embeddings; vector field `v(x, t)` only |
| D | `generative_field_replay` | **Unconditional**: no embeddings; vector field `v(x, t)` only |

**Conditional** (tx default): model has all `n_perturbations=10` embedding rows from init. Pretrain data excludes digit 8 but keeps true `p` labels; row 8 stays at init. Finetune trains row `8` + vector field via `configure_finetune` (no `expand_perturbation_embedding`). Finetune batches use `(c=0, p=8)`.

**Unconditional**: pass `--unconditional` at **both** pretrain and finetune. The model has no `context_emb` / `pert_emb`; the vector field depends only on state and time. Replay rolls out the same unconditional ref field (conditions in the data loader are ignored). Use a matching unconditional pretrain checkpoint.

## File layout

```
experiments/mnist/
├── README.md                      # this file
├── mnist_common.py                # load splits, model factory, eval helpers
├── train_mnist_pretrain.py        # pretrain on seen digits
├── train_mnist_finetune.py        # finetune: generative_replay | generative_field_replay
└── plot_mnist_eval.py             # visual eval grids (real vs generated)

src/trajectory_replay/data/
└── mnist_dataset.py               # shared loader used by notebook + scripts
```

Reuse from existing code:

- Model: `ConditionalFlowMatching` with `use_conditioning=True/False`
- Training: `train_flow_matching`, `train_flow_matching_generative_replay`
- Holdout plotting: pretrain checkpoints use true digit indices directly (full embedding table)
- Results I/O: `trajectory_replay.io.results`

## Pipeline stages

### 1. Pretrain (`train_mnist_pretrain.py`)

```
load pretrain.pt  →  (x, c, p) with p ≠ held_out (true digit labels, no remapping)
model = ConditionalFlowMatching(..., n_perturbations=10, use_conditioning=not --unconditional)
train_flow_matching(model, loader, ...)
save checkpoint + ODE samples for seen digits
```

CLI:

```bash
# Conditional (default)
python experiments/mnist/train_mnist_pretrain.py \
  --held-out-digit 8 \
  --epochs 100 \
  --run-name mnist_pretrain_d8

# Unconditional pipeline
python experiments/mnist/train_mnist_pretrain.py \
  --unconditional \
  --epochs 100 \
  --run-name mnist_pretrain_uncond_d8
```

Pretrain uses **true** perturbation indices (digit = `p`); the held-out digit is excluded from data only, not from the embedding table (same protocol as tx context holdout).

### 2. Finetune (`train_mnist_finetune.py`)

```
load pretrain checkpoint (matching use_conditioning; conditional requires n_perturbations=10)
if conditional:
    configure_finetune(holdout_mode="perturbation", held_out_pert=held_out)
else:
    train vector field only (no embedding table)

model_ref = frozen copy of pretrain model
support_contexts = [0]
generative replay on seen digits (true strata for conditional; same field for unconditional)

train_flow_matching_generative_replay(..., replay_mode="endpoint" | "field")
```

CLI:

```bash
# Conditional field replay
python experiments/mnist/train_mnist_finetune.py \
  --pretrain-run mnist_pretrain_d8 \
  --finetune-strategy generative_field_replay \
  --generative-replay-fraction 0.25 \
  --finetune-epochs 50 \
  --run-name mnist_ft_field_cond_d8

# Unconditional field replay (requires unconditional pretrain)
python experiments/mnist/train_mnist_finetune.py \
  --pretrain-run mnist_pretrain_uncond_d8 \
  --unconditional \
  --finetune-strategy generative_field_replay \
  --run-name mnist_ft_field_uncond_d8
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--unconditional` | No context/pert embeddings; `v(x, t)` only |
| `--finetune-train-vf` | Train UNet (default: on) |
| `--finetune-train-old-embeddings` | Unfreeze seen digit rows (conditional only) |
| `--generative-replay-fraction` | Batch mix: `(1-r)` OOD digit 8, `r` support replay |

### 3. Visual evaluation (`plot_mnist_eval.py`)

For each checkpoint, sample via ODE integration and write per-digit grids (real, generated, side-by-side) plus an overview figure.

- **Conditional**: generates one panel per perturbation index `p=0..9` (all digit conditions).
- **Unconditional**: fresh unconditional samples per digit; real images grouped by true label in the data split.

```bash
# Finetuned conditional model — all digits, reals from pretrain+finetune pools
python experiments/mnist/plot_mnist_eval.py \
  --run-name mnist_ft_field_cond_d8 \
  --data-split both \
  --n-per-digit 8

# Pretrain only (seen digits have reals; held-out digit 8 is gen-only)
python experiments/mnist/plot_mnist_eval.py \
  --run-name mnist_pretrain_d8 \
  --data-split pretrain

# Unconditional finetune (conditioning inferred from checkpoint)
python experiments/mnist/plot_mnist_eval.py \
  --run-name mnist_ft_field_uncond_d8 \
  --unconditional \
  --data-split both
```

Outputs land in `results_dir/<run-name>/eval_plots/` (`real_digit_*.png`, `gen_digit_*.png`, `compare_digit_*.png`, `overview_all_digits.png`).

## Expected outcomes (hypotheses)

1. **Conditional + field replay**: digit 8 improves with limited forgetting on seen digits.
2. **Conditional + generative replay**: digit 8 improves but more forgetting.
3. **Unconditional + field replay**: if replay still prevents forgetting, regularization is **trajectory-level**, not embedding-level.
4. **Unconditional + generative replay**: likely weaker on digit 8 (no condition signal for OOD data).

## Quick reference commands

```bash
# 1. Prepare data
jupyter execute notebooks/mnist/prepare_mnist_splits.ipynb

# 2. Pretrain (conditional + unconditional)
python experiments/mnist/train_mnist_pretrain.py --run-name mnist_pretrain_d8
python experiments/mnist/train_mnist_pretrain.py --unconditional --run-name mnist_pretrain_uncond_d8

# 3. Finetune grid (4 runs)
for strategy in generative_replay generative_field_replay; do
  python experiments/mnist/train_mnist_finetune.py \
    --pretrain-run mnist_pretrain_d8 \
    --finetune-strategy "$strategy" \
    --run-name "mnist_ft_${strategy}_cond_d8"
  python experiments/mnist/train_mnist_finetune.py \
    --pretrain-run mnist_pretrain_uncond_d8 \
    --unconditional \
    --finetune-strategy "$strategy" \
    --run-name "mnist_ft_${strategy}_uncond_d8"
done
```
