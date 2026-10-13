#!/bin/bash
#SBATCH --job-name=imagenet-examples
#SBATCH --qos=normal
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/imagenet_examples_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:h100:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
mkdir -p sbatch/logs
unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

"${REPO_ROOT}/.venv/bin/python" \
  experiments/image_cl/plot_imagenet100_appendix_examples.py \
  --seed "${SEED:-42}" \
  --cfg "${CFG_SCALE:-2.0}" \
  --ode-step-size "${ODE_STEP_SIZE:-0.01}" \
  --n-per-class "${N_PER_CLASS:-9}" \
  --batch-size "${BATCH_SIZE:-18}" \
  --device cuda
