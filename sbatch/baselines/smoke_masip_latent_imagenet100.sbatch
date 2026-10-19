#!/bin/bash
#SBATCH --job-name=masip-in100-smoke
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch/logs/masip_in100_latent_smoke_%j.log

# Smoke test for the latent-space Masip ImageNet-100 baseline:
# benchmark from SD-VAE latents -> 4ch UNet/DDIM -> raw latent samples ->
# VAE decode -> 256x256 FID + ResNet50 masked-argmax KLD.
#
#   sbatch sbatch/baselines/smoke_masip_latent_imagenet100.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

MASIP_ROOT="${MASIP_ROOT:-${REPO_ROOT}/baselines/diffusion_continual_learning}"
LATENT_DIR="${LATENT_DIR:-${REPO_ROOT}/data_dir/imagenet100/latents_sdvae_ft_mse_256}"
VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"

DATA_CACHE="${DATA_CACHE:-${REPO_ROOT}/data_dir/cache}"
export TORCH_HOME="${TORCH_HOME:-${DATA_CACHE}/torch}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data_dir/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

# Required on H100: torch 2.0.1 fused attention kernels miscompile on sm_90.
export DISABLE_FUSED_SDPA="${DISABLE_FUSED_SDPA:-1}"

unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

echo "Job ${SLURM_JOB_ID:-local} Masip latent smoke at $(date -Is)"
echo "  MASIP_ROOT=${MASIP_ROOT}"
echo "  LATENT_DIR=${LATENT_DIR} VAE_ID=${VAE_ID}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

cd "${MASIP_ROOT}"
uv sync
uv run python scripts/smoke_latent_imagenet100.py \
  --latent-dir "${LATENT_DIR}" \
  --vae-id "${VAE_ID}"

echo "Finished at $(date -Is)"
