#!/bin/bash
#SBATCH --job-name=simcond-a-ft
#SBATCH --qos=YOUR_QOS
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/logs/sim_conditional_stage_a_ft_%j.log
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-.}"
cd "${SIM_ROOT}"
mkdir -p "${SIM_ROOT}/sbatch/logs"

: "${DIM:=2}"
: "${STRATEGY:=trajectory}"
: "${LAMBDA_MIX:=0.5}"
: "${EPOCHS:=100}"
: "${BATCH_SIZE:=256}"
: "${LR:=1e-3}"
: "${TRAIN_SEED:=0}"
: "${DEVICE:=cuda}"
: "${N_GEN_EVAL:=2000}"
: "${ODE_METHOD:=euler}"
: "${ODE_STEP_SIZE:=0.05}"
: "${TRAJ_KAPPA:=none}"
: "${PRETRAIN_CHECKPOINT:=}"
: "${RUN_NAME:=}"

extra=()
if [[ -n "${PRETRAIN_CHECKPOINT}" ]]; then
  extra+=(--pretrain-checkpoint "${PRETRAIN_CHECKPOINT}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  extra+=(--run-name "${RUN_NAME}")
fi
if [[ "${STRATEGY}" == "trajectory" ]]; then
  extra+=(--traj-kappa "${TRAJ_KAPPA}")
fi

echo "Job ${SLURM_JOB_ID:-local} Stage A FT at $(date -Is)"
echo "STRATEGY=${STRATEGY} LAMBDA_MIX=${LAMBDA_MIX} EPOCHS=${EPOCHS} DEVICE=${DEVICE}"

uv sync
uv run python experiments/sim_conditional/stage_a_leaveout/train_finetune.py \
  --dim "${DIM}" \
  --strategy "${STRATEGY}" \
  --lambda-mix "${LAMBDA_MIX}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --n-gen-eval "${N_GEN_EVAL}" \
  --ode-method "${ODE_METHOD}" \
  --ode-step-size "${ODE_STEP_SIZE}" \
  "${extra[@]}"

echo "Finished at $(date -Is)"
