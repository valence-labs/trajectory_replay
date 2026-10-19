#!/bin/bash
#SBATCH --job-name=simmix-pretrain
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/sim_mixture_pretrain_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

# Unconditional MoG pretrain on task 0 (dim=2 by default).
#
#   ./sbatch/sim_mixture/submit_pretrain.sh
#   sbatch --export=ALL,EPOCHS=200,DEVICE=cuda sbatch/sim_mixture/train_pretrain.sbatch

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"

mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${RUN_NAME:=pretrain_task0}"
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

extra_data=()
if [[ -n "${DATA_DIR}" ]]; then
  extra_data+=(--data-dir "${DATA_DIR}")
fi

echo "Job ${SLURM_JOB_ID:-local} starting at $(date -Is)"
echo "SIM_ROOT=$(pwd)  DIM=${DIM}  RUN_NAME=${RUN_NAME}  DEVICE=${DEVICE}"
echo "  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  LR=${LR}  HIDDEN_DIM=${HIDDEN_DIM}"

uv sync
uv run python experiments/sim_mixture/train_pretrain.py \
  --dim "${DIM}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --time-emb-dim "${TIME_EMB_DIM}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  --run-name "${RUN_NAME}" \
  "${extra_data[@]}"

echo "Finished at $(date -Is)"
echo "Results: $(pwd)/results_dir/sim_mixture/dim${DIM}/${RUN_NAME}"
