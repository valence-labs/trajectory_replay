#!/bin/bash
#SBATCH --job-name=icl-kld-verify
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=sbatch/logs/imagenet100_kld_verify_%j.log

# Verify ResNet50-V2 CMC-100 masked KLD probe on decoded val latents.
#
#   sbatch sbatch/image_cl/verify_imagenet100_kld.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

LATENT_DIR="${LATENT_DIR:-${HOME}/sandbox/tx_analysis_data/imagenet100/latents_sdvae_ft_mse_256}"
VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"
N="${N:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"

export HF_HOME="${HF_HOME:-${HOME}/sandbox/tx_analysis_data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

echo "Job ${SLURM_JOB_ID:-local} imagenet100 KLD verify at $(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

uv sync
uv run python scripts/verify_imagenet100_kld_clf.py \
  --latent-dir "${LATENT_DIR}" \
  --vae-id "${VAE_ID}" \
  --n "${N}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda

echo "Finished at $(date -Is)"
