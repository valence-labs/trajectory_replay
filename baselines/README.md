# Baselines

Third-party reference implementations used for image continual-learning comparisons.
Each baseline gets its **own UV environment** (do not merge into the root project:
our code targets Python ≥3.13; Masip’s stack is Python 3.9 + Torch 2.0).

Non-CL tooling (e.g. Arc STATE SE embeddings) lives under [`third_party/`](../third_party/README.md), not here.

## Masip et al. (generative distillation)

Paper: [Continual Learning of Diffusion Models with Generative Distillation](https://proceedings.mlr.press/v274/masip25a.html)  
Code: [Atenrev/diffusion_continual_learning](https://github.com/Atenrev/diffusion_continual_learning)

Pinned commit: see [`diffusion_continual_learning.PIN.txt`](diffusion_continual_learning.PIN.txt).

### Clone

```bash
git clone https://github.com/Atenrev/diffusion_continual_learning.git \
  baselines/diffusion_continual_learning
cd baselines/diffusion_continual_learning
git checkout "$(cat ../diffusion_continual_learning.PIN.txt)"
```

### UV environment

Tracked overlay (copied into the clone): [`diffusion_continual_learning.pyproject.toml`](diffusion_continual_learning.pyproject.toml).

```bash
cd baselines/diffusion_continual_learning
cp ../diffusion_continual_learning.pyproject.toml ./pyproject.toml   # if re-cloning
uv sync
uv run python train_cl.py -h
```

This uses a lean `pyproject.toml` (not the frozen conda `requirements.txt`, which
pulls `mkl-*` / `brotlipy` that break outside conda). Torch wheels come from the
PyTorch cu118 index.

Paper CL launches:

- `scripts/launch_cl_fmnist_experiments.sh`
- `scripts/launch_cl_cifar10_experiments.sh`

### Cluster (SLURM)

One seed per job; UV env is the Masip clone’s, not the root project’s.

```bash
# Minimal FMNIST grid: naive / cumulative / no_distillation / full_generation_distillation × seeds 42,69,1714
./sbatch/baselines/submit_masip_fmnist_min.sh

# Same grid on Split-CIFAR10 (12 jobs)
./sbatch/baselines/submit_masip_cifar10_min.sh

# Same grid on Split-ImageNet-100, in SD-VAE latent space (12 jobs)
./sbatch/baselines/submit_masip_imagenet100_min.sh

# FM-budget-matched ImageNet-100 re-run (large UNet, ~100k steps/exp; 12 jobs)
./sbatch/baselines/submit_masip_imagenet100_fm_budget.sh

# Dry-run
DRY_RUN=1 ./sbatch/baselines/submit_masip_fmnist_min.sh

# Single job
sbatch --export=ALL,STRATEGY=full_generation_distillation,DATASET=split_fmnist,SEED=42,LAMBD=0.75 \
  sbatch/baselines/train_masip_cl.sbatch
```

`LAMBD=0.75` matches paper Table B.4 (GD, 2 teacher steps). Unused for naive / cumulative / no_distillation.

`DATASET` drives everything dataset-specific in `train_masip_cl.sbatch`:

| | `split_fmnist` | `split_cifar10` | `split_imagenet100` |
| --- | --- | --- | --- |
| generator | `ddim_medium.json` | `ddim_medium_3ch.json` | `ddim_medium_4ch_latent.json` |
| solver | `cnn.json` | `cnn_3ch.json` | none (generator-only) |
| strategy configs | `configs/strategy/` (100 epochs, batch 128) | `configs/strategy/cifar/` (200 epochs, batch 64) | `configs/strategy/imagenet100/` (200 epochs, batch 128) |
| `no_distillation` config | `diffusion_no_distill.json` | `diffusion_no_distill_preliminary_10.json` (paper: 10 teacher steps) | `diffusion_no_distill.json` |
| `full_generation_distillation` config | `diffusion_full_gen_distill.json` | `diffusion_full_gen_distill_10.json` (paper: 10 teacher steps) | `diffusion_full_gen_distill.json` |
| generation steps | 10 | 20 | 20 |
| KLD classifier | `weights/cnn_fmnist/` | `weights/cnn_cifar10/` | `torchvision:resnet50` (CMC-100 masked argmax) |

`DRY_RUN=1` on `train_masip_cl.sbatch` itself prints the resolved configs and exits
before training, which is the quickest way to check a new dataset/strategy pair.

### ImageNet-100 runs in latent space

`split_imagenet100` does **not** train on pixels. It reads the same cached frozen
SD-VAE latents as the flow-matching pipeline (`LATENT_DIR`, `4×32×32`, scaled by
`0.18215`), trains the diffusion model there, and decodes samples back to
256×256 RGB before FID and the ResNet50 KLD probe. FID/KLD are therefore directly
comparable to the latent-FM rows of the ImageNet table.

**Class-conditional (default for ImageNet submit):** `CLASS_CONDITIONAL=1` adds
`num_class_embeds` (+ null for CFG), label dropout `0.1`, guidance scale `1.5`,
and `N_EXPERIENCES=3` (34/33/33) to match the locked FM protocol. Task class
sets are identical to FM (`Avalanche NC` shuffle for 3 tasks; frozen JSON for
the 5×20 ablation) — do **not** rechunk the 5×20 concat. Replay /
distillation sample labels from previously seen classes; eval FID samples labels
over all classes seen so far. Disable with
`CLASS_CONDITIONAL=0 N_EXPERIENCES=5` for the unconditional Masip 5×20 ablation.

Two things this required beyond the pixel datasets:

- **Sampling stays unnormalized.** `generate_samples` uses `output_type="torch_raw"`
  so latents skip the pixel `[-1,1] → [0,1]` remap, and the metric hook decodes them.
- **The DDIM clip range is on the latent scale** (`clip_sample_range: 8.0` instead of
  `1.0`). The vendored scheduler uses zero terminal SNR, so `alphas_cumprod[999] == 0`
  and the x0 prediction at the first inference step divides by zero — clipping is the
  only thing keeping it finite, so it cannot simply be disabled. At 8.0 it clips
  ~2e-6 of real latent values (std 0.83) while still bounding that step.

Smoke test (GPU, ~2 min): `sbatch sbatch/baselines/smoke_masip_latent_imagenet100.sbatch`.

Defaults (overridable via env):

| Env | Default |
|-----|---------|
| `OUTPUT_DIR` | `$REPO_ROOT/results_dir/baselines/masip` |
| `DATA_CACHE` | `$REPO_ROOT/data_dir/cache` (`AVALANCHE_HOME` / `TORCH_HOME`) |
| `KEEP_CHECKPOINTS` | `0` (set `1` to skip upstream checkpoint cleanup) |
| `SOLVER_TYPE` | `cnn` (set `None` for FID/KLD-only, cheaper) |

### Local patches to the clone

Re-apply after any re-clone:

```bash
for p in baselines/patches/*.patch; do
  patch -p1 -d baselines/diffusion_continual_learning < "$p"
done
```

- `keep_checkpoints.patch` — honour `KEEP_CHECKPOINTS=1` instead of always deleting checkpoints.
- `disable_fused_sdpa.patch` — honour `DISABLE_FUSED_SDPA` (default `1`). **Required on H100.**
  torch 2.0.1's flash / mem-efficient attention kernels miscompile on sm_90, so the UNet
  backward fails with `GET was unable to find an engine to execute this computation`.
  A100 (sm_80) works with or without it. The math fallback keeps `cudnn.deterministic=True`
  and costs essentially nothing here (attention only runs at 8x8 and 4x4).
- `class_conditional_imagenet.patch` — class-conditional UNet + CFG for ImageNet-100
  (label dropout, guidance, old-class replay labels, `n_experiences`).
- `harmonize_imagenet100_class_orders.patch` — Avalanche NC class splits for
  `N_EXPERIENCES≠5` (e.g. 3 → 34/33/33) identical to FM; do not rechunk the 5×20 table.
- `fm_matched_step_budget.patch` — honour optional
  ``target_steps_per_experience`` in strategy JSONs (used by
  `baselines/configs/imagenet100_fm_budget/` + `submit_masip_imagenet100_fm_budget.sh`
  to match the locked FM ~100k Adam-steps/task recipe, including shrinking
  epochs as cumulative's union grows).

To pin A100s instead: `sbatch --gres=gpu:a100:1 …`.

### Outputs

Defaults are in the table above. Point `OUTPUT_DIR` at
`results_dir/baselines/masip/` when you want the runs next to the other results.
