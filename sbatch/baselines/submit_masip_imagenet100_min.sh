#!/usr/bin/env bash
# Masip Split-ImageNet-100 diffusion baseline grid (latent-space protocol).
#   naive / cumulative (joint) / no_distillation / full_generation_distillation
# × seeds 42, 69, 1714 → 12 jobs.
#
# Locked recipe to match the class-cond FM ImageNet table:
#   CLASS_CONDITIONAL=1, N_EXPERIENCES=3 (34/33/33), CFG 1.5, label dropout 0.1
#   Eval DDIM GENERATION_STEPS=50 (match FM Euler ~50 NFEs)
#   Distill replay teacher_steps=30 in imagenet100 strategy JSONs (match FM replay)
#
# Notes:
#   - Trains on the same frozen SD-VAE 4x32x32 latents and the same Avalanche
#     class splits as FM (3×34/33/33, not a rechunk of the 5×20 table); samples
#     are decoded to 256x256 before FID/KLD, so numbers are comparable.
#   - Generator-only (SOLVER_TYPE=None); KLD via torchvision ResNet50-V2.
#   - Default walltime 48h (cumulative grows with seen data); raise for distill.
#
# From repo root:
#   ./sbatch/baselines/submit_masip_imagenet100_min.sh
#   DRY_RUN=1 ./sbatch/baselines/submit_masip_imagenet100_min.sh
#   # Paper-style unconditional 5×20 ablation:
#   CLASS_CONDITIONAL=0 N_EXPERIENCES=5 ./sbatch/baselines/submit_masip_imagenet100_min.sh
#   SEEDS="42" STRATEGIES="cumulative" ./sbatch/baselines/submit_masip_imagenet100_min.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export DATASET=split_imagenet100
# Eval NFE (FID / sample dump). Replay NFE for distill is teacher_steps in
# configs/strategy/imagenet100/diffusion_{no_distill,full_gen_distill}.json.
export GENERATION_STEPS="${GENERATION_STEPS:-50}"
export JOB_PREFIX="${JOB_PREFIX:-masip-in100}"
export SOLVER_TYPE="${SOLVER_TYPE:-None}"
export TIME="${TIME:-48:00:00}"
export QOS="${QOS:-YOUR_QOS}"
export LATENT_DIR="${LATENT_DIR:-${REPO_ROOT}/data_dir/imagenet100/latents_sdvae_ft_mse_256}"
export VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"
# Match FM ImageNet class-cond protocol by default.
export CLASS_CONDITIONAL="${CLASS_CONDITIONAL:-1}"
export N_EXPERIENCES="${N_EXPERIENCES:-3}"
export LABEL_DROPOUT="${LABEL_DROPOUT:-0.1}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"

exec "${SCRIPT_DIR}/submit_masip_fmnist_min.sh"
