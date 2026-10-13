# Trajectory Replay

Code for **Trajectory Replay: Continuous-Time Mixture Transport for Lifelong Generative Models**.

Trajectory replay updates a flow-matching model on a new task by matching the historical velocity along its own trajectories. Generative replay, by contrast, simulates those trajectories to completion and trains on the resulting samples. A stability-plasticity weight \(\lambda\) sets how much of the update is spent on the historical field. Equal-task weighting uses \(\lambda_k = k/(k+1)\) when task \(k\) arrives. Adaptive weighting picks \(\lambda\) on validation data from the tasks seen so far.

The sections below follow the paper: controlled mixture recovery (§4.1), the five-task Gaussian ring (§4.2), class-incremental images (§4.3), single-cell perturbation streams (§4.4), and RxRx1 microscopy (§4.5). Appendix D.4 is the synthetic perturbation-shift experiment.

## Setup

Python 3.13. From the repository root:

```bash
uv sync
```

Paper runs expect a CUDA GPU. Outputs go to `results_dir/` and datasets to `data_dir/`, both at the repository root and both gitignored. On this cluster they are symlinks:

```text
data_dir    -> ~/sandbox/tx_analysis_data
results_dir -> ~/sandbox/tx_analysis_results
```

Elsewhere, create those two directories yourself. Commands below are run from the repository root.

SLURM launchers under `sbatch/` use this cluster's QoS `hooke-predict` and partitions `cpu`, `def`, or `hopper`. Change those directives before submitting somewhere else. `DRY_RUN=1` prints the jobs without submitting them.

Optional extras:

```bash
uv sync --extra cell-eval    # DE Overlap used in §4.4
mkdir -p external
git clone https://github.com/valence-labs/vcb.git -b main external/vcb
uv sync --extra vcb          # optional; the vcb extra is a path dependency
```

## §4.1 Controlled mixture recovery

Two isotropic Gaussians in \(\mathbb{R}^d\), \(\sigma = 0.35\), means at \(\pm 1.5\) on the first coordinate. A model pretrained on the historical mode is updated on the incoming mode. At \(d = 2\), \(\lambda \in \{0.1, 0.3, 0.5, 0.7, 0.9\}\) with 10,000 samples per task. The shared task-0 checkpoint trains for 400 epochs (Adam, \(10^{-3}\), batch 256, seed 0). Each update uses the same budget for seeds \(\{0, 1, 2\}\). Sampling is Euler with step 0.05, and evaluation draws 4,000 samples.

```bash
uv run python experiments/sim_mixture/build_datasets.py --dim 2 --samples-per-task 10000

uv run python experiments/sim_mixture/train_pretrain.py \
  --dim 2 --epochs 400 --train-seed 0 --device cuda \
  --run-name pretrain_task0_seed0

# one update; --strategy is naive, generative, trajectory, or real
uv run python experiments/sim_mixture/train_finetune.py \
  --dim 2 --epochs 400 --strategy trajectory --lambda-mix 0.5 \
  --train-seed 0 --device cuda \
  --pretrain-checkpoint results_dir/sim_mixture/dim2/pretrain_task0_seed0/checkpoint.pt
```

The paper grid (naive, generative, and trajectory replay, five \(\lambda\) values, three seeds) plus the joint-training oracle:

```bash
EPOCHS=400 SEEDS="0 1 2" \
  PRETRAIN_CHECKPOINT=results_dir/sim_mixture/dim2/pretrain_task0_seed0/checkpoint.pt \
  ./sbatch/sim_mixture/submit_ft_lambda_grid.sh

EPOCHS=400 ALL_LAMBDAS=1 SEEDS="0 1 2" \
  ./sbatch/sim_mixture/submit_oracle_mixture.sh
```

Figure 7 repeats the \(\lambda = 0.5\) update at \(d \in \{128, 256\}\) with 100,000 samples per task, 513 update epochs, and an evaluation of 2,000 samples every 10 epochs:

```bash
uv run python experiments/sim_mixture/build_datasets.py --dim 128 --samples-per-task 100000
uv run python experiments/sim_mixture/build_datasets.py --dim 256 --samples-per-task 100000

DIMS="128 256" PRETRAIN_EPOCHS=400 EPOCHS=513 EVAL_EVERY=10 \
  N_GEN_EVAL_CURVE=2000 SEEDS="0 1 2" \
  ./sbatch/sim_mixture/submit_nfe_grid.sh
```

Runs land in `results_dir/sim_mixture/dim<d>/`.

## §4.2 Sequential Gaussian ring

Five tasks in \(\mathbb{R}^2\), \(\sigma = 0.35\), means on a circle of radius 3. Each task has 10,000 / 2,000 / 5,000 train / validation / test points. Every task, including task 0, gets 20,000 optimizer steps (Adam \(10^{-3}\), batch 256, gradient clip 1.0). Seeds are \(\{0, 1, 2\}\). Replay is constant \(\lambda \in \{0.25, 0.5, 0.75, 0.9\}\), equal-task, or adaptive. The adaptive sweep is \(\{0.25, 0.5, 2/3, 0.75, 0.8, 0.9\}\), chosen by energy distance on 4,000 validation samples. Test metrics use 10,000 samples.

```bash
uv run python experiments/sim_sequential/build_datasets.py

uv run python experiments/sim_sequential/train_cl.py \
  --strategy trajectory --lambda-mode equal --seed 0 --device cuda

# naive, plus generative and trajectory replay under constant, equal, and adaptive λ
./sbatch/sim_sequential/submit_campaign.sh
```

Data and figures go to `results_dir/sim_sequential/`.

## §4.3 Class-incremental image generation

Five tasks of two classes on Fashion-MNIST and CIFAR-10, and three tasks (34/33/33) on ImageNet-100. Seeds are \(\{42, 69, 1714\}\), with class order from Avalanche. A deterministic 10% of each task's training split is validation. Reported metrics use 10,000 samples (FID, and KLD on FMNIST and CIFAR-10). Adaptive \(\lambda\) is chosen from 2,048 validation samples.

| | Fashion-MNIST | CIFAR-10 | ImageNet-100 |
| --- | --- | --- | --- |
| State | \(1\times 32\times 32\) | \(3\times 32\times 32\) | \(4\times 32\times 32\) SD-VAE latent |
| UNet | medium, 35.8M | medium, 35.8M | large, 106.4M, class-conditional, CFG 1.5 |
| Epochs / batch / lr | 100 / 128 / \(2\times 10^{-4}\) | 200 / 64 / \(2\times 10^{-4}\) | 667 (~100k steps) / 256 / \(10^{-4}\), clip 1.0 |
| Eval Euler step | 0.1 (10 NFE) | 0.05 (20 NFE) | 0.02 (50 NFE) |
| Replay Euler step | 0.1 | 0.05 | 0.033 (30 NFE) |
| Adaptive \(\lambda\) | 0.1, 0.25, 0.5, 0.75, 0.9 | 0.25, 0.5, 0.75 | 0.25, 0.5, 0.75 |

FMNIST and CIFAR-10 download into `data_dir/torchvision`. ImageNet-100 is trained on cached `stabilityai/sd-vae-ft-mse` latents. The training default is `~/sandbox/tx_analysis_data/imagenet100/latents_sdvae_ft_mse_256`; set `LATENT_DIR` to use another cache. Encode a torchvision ImageNet tree with:

```bash
uv run python scripts/encode_imagenet_vae_latents.py \
  --image-root data_dir/imagenet --out-dir data_dir/imagenet100/latents
```

KLD on FMNIST and CIFAR-10 uses the Masip classifiers at `baselines/diffusion_continual_learning/weights/cnn_{fmnist,cifar10}/resnet.pth`. Clone that repo first ([baselines/README.md](baselines/README.md)).

A one-GPU smoke test:

```bash
uv run python experiments/image_cl/train_cl.py \
  --dataset split_fmnist --strategy trajectory --seed 42 --smoke \
  --run-name image_cl/fmnist/smoke_traj_s42
```

Paper ODE jobs:

```bash
# Fashion-MNIST: naive / generative / trajectory × adaptive λ × 3 seeds
./sbatch/image_cl/submit_fmnist_min.sh
STRATEGIES="generative trajectory" LAMBDA_MODE=equal \
  ./sbatch/image_cl/submit_fmnist_min.sh

# CIFAR-10 uses the three-value λ grid (run suffix g3)
LAMBDA_GRID="0.25 0.5 0.75" RUN_SUFFIX=g3 \
  ./sbatch/image_cl/submit_cifar10_min.sh
STRATEGIES="generative trajectory" LAMBDA_MODE=equal \
  ./sbatch/image_cl/submit_cifar10_min.sh

# ImageNet-100: joint upper bound and fine-tuning, then replay
STRATEGIES="naive cumulative" ./sbatch/image_cl/submit_imagenet100_min.sh
STRATEGIES="generative trajectory" LAMBDA_MODE=grid RUN_SUFFIX=g3 \
  REPLAY_ODE_STEP_SIZE=0.03333333333333333 \
  ./sbatch/image_cl/submit_imagenet100_min.sh
STRATEGIES="generative trajectory" LAMBDA_MODE=equal \
  REPLAY_ODE_STEP_SIZE=0.03333333333333333 \
  ./sbatch/image_cl/submit_imagenet100_min.sh
```

MeanFlow replay (\(N \in \{1, 4\}\), both \(\lambda\) policies, three seeds):

```bash
./sbatch/image_cl/submit_fmnist_meanflow.sh
./sbatch/image_cl/submit_cifar10_meanflow.sh
LATENT_DIR=data_dir/imagenet100/latents \
  ./sbatch/image_cl/submit_imagenet100_meanflow.sh
```

Runs land in `results_dir/image_cl/<fmnist|cifar10|imagenet100>/`.

The diffusion rows of Table 1 are matched runs of [Masip et al.](https://github.com/Atenrev/diffusion_continual_learning). After the clone in [baselines/README.md](baselines/README.md):

```bash
./sbatch/baselines/submit_masip_fmnist_min.sh
./sbatch/baselines/submit_masip_cifar10_min.sh
./sbatch/baselines/submit_masip_imagenet100_fm_budget.sh
```

LoRA and gradual fine-tuning (Table 3) use `./sbatch/image_cl/submit_lora_gft.sh`.

## §4.4 Single-cell perturbation streams

Replogle-Nadig is a two-task stream: task 0 is Jurkat, K562, and RPE1, and task 1 is HepG2. Parse-PBMC is a five-task donor stream: eight pretraining donors, then Donor1, Donor4, Donor9, and Donor12. Both use the 2,000 highly variable genes, library-size normalization to \(10^4\), and a \(\log(1+x)\) transform. Flow matching runs in a PCA basis fit on task-0 training cells only (50 components on Replogle, 100 on PBMC). A frozen per-task affine decoder maps latents back to genes. Seeds are \(\{42, 69, 1714\}\). Adam uses learning rate \(10^{-4}\) and batch size 1024. The teacher is Euler with step 0.05. Adaptive \(\lambda\) is chosen from \(\{0.25, 0.5, 0.75\}\) by balanced Pearson \(\Delta\) with 200 generated cells per stratum. Final metrics use 500 generated cells per non-control stratum.

| | Replogle-Nadig | Parse-PBMC |
| --- | --- | --- |
| Backbone | MLP, width 2048, depth 4 | residual MLP, width 2048, depth 6 |
| Task 0 / later steps | 100,000 / 50,000 | 300,000 / 150,000 |
| Conditioning | cell line, gene | donor, cytokine, cell type |

Expected inputs, produced by `notebooks/bio_data/` and `experiments/tx/prepare_cl_decoders.py`:

```text
data_dir/replogle/adata.h5ad
data_dir/replogle/pca50_embedding.npy
data_dir/replogle/pca50_basis.npz
results_dir/tx_results/replogle/decoders_pca50_affine/

data_dir/pbmc/adata.h5ad
data_dir/pbmc/pca100_embedding.npy
data_dir/pbmc/pca100_basis.npz
results_dir/tx_results/pbmc/decoders_pca100_affine/
```

The final scripts refuse to train if the decoder bank is missing.

```bash
# replay matrix: real, generative, trajectory × ODE and MeanFlow N=1,4
# × equal and adaptive λ × 3 seeds
DRY_RUN=1 ./sbatch/tx/submit_final_replogle_replay.sh
./sbatch/tx/submit_final_replogle_replay.sh
./sbatch/tx/submit_final_pbmc_replay.sh

# fine-tuning, LoRA, gradual fine-tuning, and two-stage adaptation
./sbatch/tx/submit_final_replogle_baselines.sh
./sbatch/tx/submit_final_pbmc_baselines.sh
```

Pearson \(\Delta\) is written by the training job (`final_metrics.json`, `heldout_curve.json` under `results_dir/tx_results/`). DE Overlap is the separate CellEval job those scripts submit, which needs `uv sync --extra cell-eval`.

## §4.5 RxRx1 microscopy

Three U2OS acquisition batches, in order U2OS-01, U2OS-02, U2OS-03, over 1,071 perturbations. Inputs are native 6×96×96 crops in \([-1, 1]\), with uniform dequantization and random flips at training time. The backbone is a Config-B ADM U-Net (56.2M parameters) conditioned on perturbation and experiment. Each task is 25,000 Adam steps at \(10^{-4}\), batch 64, gradient clip 0.5, and a power-function EMA. Seeds are \(\{42, 69, 1714\}\). Replay uses Euler step 0.05. Adaptive \(\lambda\) is \(\{0.25, 0.5, 0.75\}\), chosen by macro validation FID. Evaluation generates one image per treated test condition (2,942 images) and reports pooled Inception FID and DINOv2 Fréchet distance from the EMA weights.

Metadata defaults to `data_dir/rxrx1/rxrx1_metadata.csv`. Override with `RXRX1_METADATA_PATH`. Image paths are a column of that table.

```bash
# Full flow-matching matrix, one shared task-0 checkpoint per seed:
# naive, cumulative, real, and both replay families under equal and adaptive λ.
# Replay jobs include the ODE teacher and MeanFlow at N=1 and N=4.
./sbatch/px/submit_cl_all.sh

# The paper's MeanFlow rows are the narrower slice: trajectory replay,
# equal-task λ, N=1 and N=4.
STRATEGIES=trajectory LAMBDA_MODES=equal SEEDS="42 69 1714" \
  ./sbatch/px/submit_cl_mean_flow.sh
```

Weights & Biases projects default to `trajectory-replay-px` and `trajectory-replay-px-cl`. Set `WANDB_PROJECT` to change them, or `WANDB_MODE=offline` to skip upload. The submitit launcher accepts `--launcher.partition`, `--launcher.qos`, and `--launcher.wckey` (`TR_SLURM_PARTITION`, `TR_SLURM_QOS`, `TR_SLURM_WCKEY`).

The diffusion distillation row is a matched Masip run on the same ADM, budget, and EMA:

```bash
./sbatch/px/submit_masip_all.sh
```

Runs land in `results_dir/px/`.

## Appendix D.4 Synthetic perturbation shift

Six contexts on a hexagon and four shared perturbation offsets in \(\mathbb{R}^2\). Pretraining sees contexts 0–4. The update introduces context 5 and withholds perturbation 3, so the test cell is \((c^\star, y^\star) = (5, 3)\). Pretraining is 1,000 epochs and the update is 2,000 (Adam \(3\times 10^{-4}\), batch 1024). Seeds are \(\{10, 20, 30, 40, 50\}\) and \(\lambda \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}\).

```bash
DRY_RUN=1 ./sbatch/sim_conditional/reproduce_stage_b_paper.sh
./sbatch/sim_conditional/reproduce_stage_b_paper.sh
```

## Layout

```text
src/trajectory_replay/   models, training, data, metrics
experiments/             entry points for each paper section
sbatch/                  SLURM launchers
configs/                 transcriptomics run tables
notebooks/               dataset preparation
baselines/               Masip diffusion continual learning
third_party/             Arc STATE and SCimilarity embedders
tests/
```

Arc STATE embeddings and SCimilarity are optional and are not used by the paper tables. Setup is in [third_party/README.md](third_party/README.md). Arc STATE weights are under a separate non-commercial license.

## License

Apache License 2.0. Third-party components keep their own licenses. UNet blocks under `src/trajectory_replay/model/unet/` are adapted from [openai/guided-diffusion](https://github.com/openai/guided-diffusion). The RxRx1 transformer is adapted from [facebookresearch/DiT](https://github.com/facebookresearch/DiT) (MIT).
