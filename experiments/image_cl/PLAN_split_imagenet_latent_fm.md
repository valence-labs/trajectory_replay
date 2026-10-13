# Split-ImageNet-100 Latent Flow Matching — Experiment Plan

## Objective

Extend `experiments/image_cl/` (currently Split-FMNIST / Split-CIFAR10 unconditional
CFM) to a third dataset: unconditional continual flow matching on **Split-ImageNet-100**
(the standard 100-class CIL subset), trained in **latent space** rather than pixel
space.

## CL protocol

- Dataset pool: **ImageNet-100** (the 100-class subset standard in class-incremental
  learning work — iCaRL / LUCIR / PODNet / DER / FOSTER all use variants of it), not
  full ImageNet-1k. Label it explicitly as "ImageNet-100" in any write-up.
- Split: **3 tasks × ~33 classes/task** (34/33/33; 100 is not divisible by 3) —
  ImageNet image_cl default. Pass ``--n-experiences 5`` for the older Masip-
  matched **5 × 20** protocol. FMNIST/CIFAR stay at 5 tasks.
- Unconditional (no class labels fed to the model — matches existing `naive` /
  `generative` / `trajectory` strategies already in `train_cl.py`).
- Avalanche's `SplitImageNet` (`avalanche-lib`, pinned at `0.3.1` in
  `baselines/diffusion_continual_learning/pyproject.toml`) is built for full
  ImageNet-1k; it likely does **not** accept an arbitrary 100-class subset directly.
  Plan to: (1) filter the full ImageNet-1k `ImageFolder` down to a standard
  ImageNet-100 class list (e.g. the CMC/MoCo list), then (2) build the 5×20 split
  either with Avalanche's generic `nc_benchmark` generator or a manual class-order
  permutation (matching the `AVALANCHE_SPLIT_CIFAR10_TASKS` pattern used for
  FMNIST/CIFAR-10) rather than assuming `SplitImageNet(n_experiences=5)` already
  restricts to 100 classes.

## Latent space

- **Frozen, off-the-shelf pretrained VAE** — `stabilityai/sd-vae-ft-mse` (or
  equivalent SD-family KL-VAE). Do **not** train a VAE on ImageNet: that would leak
  future-task class information into the encoder before task 0 even starts, which
  undermines the CL setup. The off-the-shelf VAE was trained on OpenImages/LAION,
  never saw ImageNet train/val/test, and is known to reconstruct ImageNet-domain
  images well (same VAE used by DiT for ImageNet latent diffusion, rFID ~0.7-1.0).
- Resolution: images → 256×256 RGB → encode → **32×32×4 latents**, scaled by SD's
  `0.18215` factor before training.
- Encode once per task, cache latents to disk (`torch.save` alongside
  `x_train`/`x_val`/`x_test` in the `TaskSplit`-equivalent). This is a one-time
  preprocessing cost, not part of the per-epoch training loop.

## Model

- Reuse `DiffusersUNetFlowMatching` (same class as FMNIST/CIFAR-10), new backbone
  config analogous to `DEFAULT_DDIM_MEDIUM_3CH` but `in_channels=out_channels=4`,
  `sample_size=32`.
- Flow matching objective / training loop (`train_unconditional_fm`,
  `train_unconditional_generative_replay`, `train_unconditional_trajectory_replay`)
  is unchanged — it operates on whatever tensor shape it's given.

## Training config

- Epochs: **~150-200/task**. At batch 256, ~23.4k train images/task → ~91
  steps/epoch, so 200 epochs ≈ 18k updates/task. (Previously batch 128 ≈ 37k
  updates at 200 ep — raise epochs / use a step budget if you need that count.)
  Same order of thinking as CIFAR-10's per-task step budget, scaled to
  20-class-per-task diversity instead of 2.
  Per-epoch compute is cheap at this model/latent scale, so this is sized for
  decent FID convergence, not minimal wall-clock.
- Batch size: 128.
- Expected wall-clock: ~15-20 min/task base (`naive`, no λ-sweep) → **~1.5-1.75h
  for all 5 tasks sequentially, one seed**.
- `--lambda-mode equal` (fixed λ=(n-1)/n, no sweep) for the first validation pass —
  the default grid sweep (5 candidates) applies to tasks 1-4, multiplying their
  cost ~5× (task0 + 4×5 candidates ≈ ~6h/seed/strategy for `generative`/
  `trajectory`). Full reported run (`--lambda-mode grid`, all 3 strategies, 3
  seeds) is roughly **~1.5-2 days of H100 time** total — switch to grid mode only
  for the runs you intend to report.

## Rollout stages

1. **Smoke test** (`--smoke`, 2 epochs) — validate plumbing: Avalanche split
   builds, VAE encode/cache shapes, checkpointing, FID/KLD scaffolding. Catch
   shape/config bugs cheaply before a real run.
2. **Single-seed validation run** — full ~100 epochs/task, `--lambda-mode equal`,
   `naive` + one replay strategy. Confirms real throughput and FID trend.
3. **Full reported run** — `--lambda-mode grid`, all strategies (`naive` /
   `generative` / `trajectory`), 3 seeds.

## Evaluation notes

- FID/KLD sampling currently runs entirely in pixel space for FMNIST/CIFAR-10.
  For latent FM, the ODE-sampled output is a **latent** — add a VAE **decode**
  step before computing Inception features / FID.
- Stream-KLD stays **in scope** — it's the metric that actually diagnoses
  per-class forgetting under replay, which is the point of the CL comparison;
  FID alone only measures visual quality.
- Masip's FMNIST/CIFAR-10 KLD classifiers (`weights/cnn_fmnist`,
  `weights/cnn_cifar10`) are custom-trained because no off-the-shelf classifier
  existed for those datasets. ImageNet doesn't have that problem: use a
  standard **pretrained** classifier (e.g.
  `torchvision.models.resnet50(weights=IMAGENET1K_V2)`) covering all 1000
  classes. This is a frozen, evaluation-only tool — no training needed, and no
  CL-leakage concern (same role as the ImageNet-pretrained Inception-v3 already
  used for FID on the other datasets: it never touches gradients or the model
  being continually trained).

## Implementation tasks (mirrors existing FMNIST/CIFAR-10 pattern)

- [ ] `trajectory_replay/data/split_imagenet.py`: wraps Avalanche
  `SplitImageNet(n_experiences=3, seed=...)`, produces the same `TaskSplit`-style
  train/val/test tensors (or cached latent tensors) used by `build_image_cl_tasks`.
- [ ] Add `avalanche-lib` as a dependency of the main package (currently only in
  `baselines/diffusion_continual_learning/pyproject.toml`).
- [ ] Frozen VAE wrapper (encode/decode + `0.18215` scaling) — one-time cache
  step, called before/outside `train_one_task`.
- [ ] Register `split_imagenet` in `image_cl_common.py`:
  `DATASET_RESULTS_SLUG`, `IMAGE_CL_DATASET_DEFAULTS` (epochs=200, batch=256),
  `image_cl_in_channels` (→4), `image_cl_unet_kwargs` (new latent backbone config),
  `build_image_cl_tasks` dispatch, `--dataset` choices.
- [x] Relax the `n_experiences != 5` hard-fail pattern (currently only in
  `build_split_cifar10_tasks`) — `split_imagenet.py` accepts `n_experiences=3`
  with nearly-equal 34/33/33 chunks (remainder on the first tasks).
  natively since it delegates to Avalanche rather than a hardcoded table.
- [ ] Decode step in the FID/KLD sampling path for latent models.
- [ ] `resolve_kld_clf_path`-equivalent for `split_imagenet`: load a pretrained
  `torchvision` ImageNet classifier (no training) instead of a `weights/cnn_*`
  path; adapt the KLD class-histogram code to the 1000-way label space
  restricted to classes seen so far.
- [ ] New `sbatch/image_cl/` job script(s) for the ImageNet dataset, single H100.

## Open questions to confirm before/while implementing

- Does Avalanche's `SplitImageNet` require a manually-downloaded ImageNet copy
  (torchvision won't auto-download it) — confirm the data path convention used
  elsewhere in this repo/cluster.
- Confirm `avalanche-lib==0.3.1`'s `SplitImageNet` signature matches the `0.2.1`
  docs referenced earlier (param names/defaults may have shifted).
