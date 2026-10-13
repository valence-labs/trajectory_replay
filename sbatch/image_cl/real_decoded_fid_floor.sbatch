#!/bin/bash
#SBATCH --job-name=icl-real-fid
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/image_cl_real_fid_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:h100:1

# Task-0 per-class decoded-real vs decoded-real FID floor (full decode, no subsample).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

: "${SEED:=42}"
: "${TASK_ID:=0}"
: "${SPLIT_SEED:=0}"
: "${POOLED_REPEATS:=5}"
: "${DECODE_BATCH_SIZE:=64}"
: "${OUTPUT:=results_dir/image_cl/imagenet100/pilot/real_decoded_fid_floor_task00_pooled.json}"
# Skip per-class by default for the pooled stability check (set SKIP_PER_CLASS=0 to include).
: "${SKIP_PER_CLASS:=1}"

unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"
export HF_HOME="${HF_HOME:-${HOME}/sandbox/tx_analysis_data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

extra=()
if [[ -n "${LATENT_DIR:-}" ]]; then
  extra+=(--latent-dir "${LATENT_DIR}")
fi
if [[ "${SKIP_PER_CLASS}" != "0" ]]; then
  extra+=(--skip-per-class)
fi

echo "Job ${SLURM_JOB_ID:-local} pooled decoded-real FID floor at $(date -Is)"
echo "  TASK_ID=${TASK_ID} SPLIT_SEED=${SPLIT_SEED} POOLED_REPEATS=${POOLED_REPEATS} OUTPUT=${OUTPUT}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

uv sync
uv run python experiments/image_cl/real_decoded_fid_floor.py \
  --seed "${SEED}" \
  --task-id "${TASK_ID}" \
  --split-seed "${SPLIT_SEED}" \
  --pooled-repeats "${POOLED_REPEATS}" \
  --decode-batch-size "${DECODE_BATCH_SIZE}" \
  --output "${OUTPUT}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
