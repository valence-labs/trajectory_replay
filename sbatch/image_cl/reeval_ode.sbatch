#!/bin/bash
#SBATCH --job-name=icl-reeval-ode
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/logs/image_cl_reeval_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Re-evaluate finished image_cl checkpoints under alternate ODE settings (no retraining).
#
#   sbatch sbatch/image_cl/reeval_ode.sbatch
#   RUN_DIRS="results_dir/image_cl/imagenet100/naive_s42" TASK_ID=2 STREAM=1 \
#     SETTINGS="euler:0.05 euler:0.02" sbatch --export=ALL sbatch/image_cl/reeval_ode.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/sbatch/logs"

: "${DISABLE_FUSED_SDPA:=1}"
export DISABLE_FUSED_SDPA

DEFAULT_RUNS=(
  results_dir/image_cl/imagenet100/pilot/pilot_uncond_t1_s42
  results_dir/image_cl/imagenet100/pilot/pilot_cc_t1_s42
  results_dir/image_cl/imagenet100/pilot/pilot_uncond_large_t1_s42
  results_dir/image_cl/imagenet100/pilot/pilot_cc_large_t1_s42
  results_dir/image_cl/imagenet100/pilot/pilot_uncond_large_clip_t1_s42
  results_dir/image_cl/imagenet100/pilot/pilot_cc_large_clip_t1_s42
)
if [[ -n "${RUN_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  RUNS=(${RUN_DIRS})
else
  RUNS=("${DEFAULT_RUNS[@]}")
fi

: "${SETTINGS:=euler:0.05 euler:0.02 euler:0.01 dopri5}"
: "${SAVE_GRIDS:=0}"
: "${TASK_ID:=0}"
: "${STREAM:=0}"
: "${OUT_NAME:=reeval_ode.json}"
# shellcheck disable=SC2206
SETTINGS_ARR=(${SETTINGS})

extra=()
if [[ "${SAVE_GRIDS}" == "1" ]]; then
  extra+=(--save-grids)
fi
if [[ "${STREAM}" == "1" ]]; then
  extra+=(--stream)
fi
extra+=(--task-id "${TASK_ID}")
extra+=(--out-name "${OUT_NAME}")

# Drop Cursor/login-node UV cache paths leaked via --export=ALL.
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data_dir/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

echo "Job ${SLURM_JOB_ID:-local} image_cl ODE re-eval at $(date -Is)"
echo "  RUNS=${RUNS[*]}"
echo "  SETTINGS=${SETTINGS}"
echo "  TASK_ID=${TASK_ID} STREAM=${STREAM} OUT_NAME=${OUT_NAME}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || true

uv sync
uv run python experiments/image_cl/reeval_ode_fid.py \
  --run-dir "${RUNS[@]}" \
  --settings "${SETTINGS_ARR[@]}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
