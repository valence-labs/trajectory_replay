#!/bin/bash
#SBATCH --job-name=simcond-b-pre
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/logs/sim_conditional_stage_b_pretrain_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${N_CONTEXTS:=6}"
: "${N_PERTURBATIONS:=4}"
: "${EPOCHS:=1000}"
: "${BATCH_SIZE:=1024}"
: "${LR:=3e-4}"
: "${TRAIN_SEED:=0}"
: "${RUN_NAME:=pretrain_seed${TRAIN_SEED}}"
: "${DEVICE:=cuda}"
: "${N_GEN_EVAL:=500}"

echo "Job ${SLURM_JOB_ID:-local} Stage B pretrain at $(date -Is)"
echo "DIM=${DIM} C=${N_CONTEXTS} P=${N_PERTURBATIONS} RUN_NAME=${RUN_NAME} EPOCHS=${EPOCHS}"

unset UV_CACHE_DIR UV_TOOL_DIR UV_PYTHON_INSTALL_DIR || true
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}"

uv sync
uv run python experiments/sim_conditional/stage_b_shift/train_pretrain.py \
  --dim "${DIM}" \
  --n-contexts "${N_CONTEXTS}" \
  --n-perturbations "${N_PERTURBATIONS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  --run-name "${RUN_NAME}"

echo "Finished at $(date -Is)"
