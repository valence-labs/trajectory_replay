#!/bin/bash
#SBATCH --job-name=simmix-oracle
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/sim_mixture_oracle_%j.log
#SBATCH --partition=def
#SBATCH --gres=gpu:1

# Train FM from scratch on a reference mixture at fixed λ.
#
#   LAMBDA_MIX=0.5 ./sbatch/sim_mixture/submit_oracle_mixture.sh

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${LAMBDA_MIX:=0.5}"
: "${DATA_DIR:=}"
: "${EPOCHS:=200}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${HIDDEN_DIM:=256}"
: "${DEPTH:=4}"
: "${TIME_EMB_DIM:=64}"
: "${DEVICE:=cuda}"
: "${N_GEN_EVAL:=4000}"
: "${RUN_NAME:=}"

extra=()
if [[ -n "${DATA_DIR}" ]]; then
  extra+=(--data-dir "${DATA_DIR}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  extra+=(--run-name "${RUN_NAME}")
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "Oracle mixture λ=${LAMBDA_MIX}  DIM=${DIM}  DEVICE=${DEVICE}"

uv sync
uv run python experiments/sim_mixture/train_on_mixture.py \
  --dim "${DIM}" \
  --lambda-mix "${LAMBDA_MIX}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --time-emb-dim "${TIME_EMB_DIM}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
