#!/bin/bash
#SBATCH --job-name=imagenet-vae-encode
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=sbatch/logs/imagenet_vae_encode_%j.log

# Encode ImageNet-100 (torchvision layout) → frozen SD-VAE latents.
#
#   sbatch sbatch/image_cl/encode_imagenet_latents.sbatch
#   # Full ImageNet-1k:
#   sbatch --export=ALL,IMAGE_ROOT=$HOME/sandbox/tx_analysis_data/imagenet \
#     sbatch/image_cl/encode_imagenet_latents.sbatch
#   # Force re-encode:
#   sbatch --export=ALL,OVERWRITE=1 sbatch/image_cl/encode_imagenet_latents.sbatch
#   # Skip re-encode; only write orig|recon QA grids (uses existing .pt):
#   sbatch --export=ALL,DECODE_SAMPLES=16 sbatch/image_cl/encode_imagenet_latents.sbatch
#
# Env overrides: IMAGE_ROOT, OUT_DIR, VAE_ID, IMAGE_SIZE, BATCH_SIZE,
# NUM_WORKERS, DTYPE (float16|float32), OVERWRITE (0|1), DECODE_SAMPLES,
# HF_HOME, SPLITS.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

IMAGE_ROOT="${IMAGE_ROOT:-${HOME}/sandbox/tx_analysis_data/imagenet100}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
OUT_DIR="${OUT_DIR:-${IMAGE_ROOT}/latents_sdvae_ft_mse_${IMAGE_SIZE}}"
VAE_ID="${VAE_ID:-stabilityai/sd-vae-ft-mse}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DTYPE="${DTYPE:-float16}"
OVERWRITE="${OVERWRITE:-0}"
DECODE_SAMPLES="${DECODE_SAMPLES:-16}"
SPLITS="${SPLITS:-train val}"

# Keep HF downloads off the login-node home defaults / GPFS-friendly cache.
export HF_HOME="${HF_HOME:-${HOME}/sandbox/tx_analysis_data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

# Drop Cursor/login-node UV cache paths leaked via --export=ALL.
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

echo "Job ${SLURM_JOB_ID:-local} imagenet VAE encode at $(date -Is)"
echo "  host=$(hostname)"
echo "  IMAGE_ROOT=${IMAGE_ROOT}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  VAE_ID=${VAE_ID} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE}"
echo "  DTYPE=${DTYPE} OVERWRITE=${OVERWRITE} DECODE_SAMPLES=${DECODE_SAMPLES}"
echo "  SPLITS=${SPLITS}"
echo "  HF_HOME=${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

extra=()
if [[ "${OVERWRITE}" == "1" ]]; then
  extra+=(--overwrite)
fi
# shellcheck disable=SC2206
split_arr=(${SPLITS})

uv sync
uv run python scripts/encode_imagenet_vae_latents.py \
  --image-root "${IMAGE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --vae-id "${VAE_ID}" \
  --image-size "${IMAGE_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --dtype "${DTYPE}" \
  --device cuda \
  --splits "${split_arr[@]}" \
  --save-decode-samples "${DECODE_SAMPLES}" \
  "${extra[@]}"

echo "done at $(date -Is)"
ls -lh "${OUT_DIR}" || true
cat "${OUT_DIR}/meta.json" || true
