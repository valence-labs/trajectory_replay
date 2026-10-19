#!/bin/bash
#SBATCH --job-name=simcond-au-pre
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/sim_conditional_stage_a_uncond_pretrain_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${RUN_NAME:=pretrain_seed0}"
: "${EPOCHS:=200}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cuda}"
: "${N_GEN_EVAL:=2000}"

echo "Job ${SLURM_JOB_ID:-local} Stage A uncond pretrain at $(date -Is)"
echo "DIM=${DIM} RUN_NAME=${RUN_NAME} EPOCHS=${EPOCHS} DEVICE=${DEVICE}"

uv sync
uv run python experiments/sim_conditional/stage_a_uncond_leaveout/train_pretrain.py \
  --dim "${DIM}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  --run-name "${RUN_NAME}"

echo "Finished at $(date -Is)"
