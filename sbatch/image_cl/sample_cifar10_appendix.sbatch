#!/bin/bash
#SBATCH --job-name=cifar10-samples
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/logs/cifar10_samples_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

extra=()
if [[ -n "${TAG:-}" ]]; then
  extra+=(--tag "${TAG}")
fi
if [[ -n "${ODE_STEP_SIZE:-}" ]]; then
  extra+=(--ode-step-size "${ODE_STEP_SIZE}")
fi

"${REPO_ROOT}/.venv/bin/python" \
  experiments/image_cl/sample_cifar10_appendix_pools.py \
  --seed "${SEED:-42}" \
  --n-samples "${N_SAMPLES:-512}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --device cuda \
  "${extra[@]}"
